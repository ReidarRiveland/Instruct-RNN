import numpy as np
import random
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import os
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, TransformerDecoderLayer, TransformerDecoder

from Task import Task, construct_batch
from LangModule import get_batch, toNumerals
task_list = Task.TASK_LIST
tuning_dirs = Task.TUNING_DIRS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def popvec(nn_final_out):
    act_sum = np.sum(nn_final_out)
    temp_cos = np.sum(np.multiply(nn_final_out, np.cos(tuning_dirs)))/act_sum
    temp_sin = np.sum(np.multiply(nn_final_out, np.sin(tuning_dirs)))/act_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)

def get_dist(original_dist):
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

def isCorrect(nn_out, nn_target, target_dirs): 
    nn_out = nn_out.detach().to('cpu').numpy()
    if not isinstance(nn_target, np.ndarray):
        nn_target = nn_target.to('cpu').numpy()
    isCorrect = np.empty(target_dirs.size, dtype=bool)
    criterion = (2*np.pi)/10 

    for i in range(target_dirs.size):
        isFixed = all(np.where(nn_target[i, :, 0] == 0.85, nn_out[i, :, 0] > 0.5, True))
        if np.isnan(target_dirs[i]): 
            isDir = all((nn_out[i, 114:119, :].flatten() < 0.15))
        else:
            is_response = np.max(nn_out[i, -1, 1:]) > 0.6
            loc = popvec(nn_out[i, -1, 1:])
            dist = get_dist(loc - target_dirs[i])        
            isDir = dist < criterion and is_response
        isCorrect[i] = isDir and isFixed
    return isCorrect

def masked_MSE_Loss(outputs, targets, mask):
    mask_applied = torch.mul(torch.pow((outputs - targets), 2), mask)
    #average along time dimension?
    avg_applied = torch.mean(torch.mean(mask_applied, 2), 1)
    return torch.mean(avg_applied)

def mask_input_rule(in_tensor, batch_len, seq_len): 
    mask = torch.zeros((batch_len, seq_len, len(task_list))).to(device)
    masked_input = torch.cat((in_tensor[:, :, 0:1], mask, in_tensor[:, :, len(task_list)+1:]), axis=2)
    return masked_input

def del_input_rule(in_tensor): 
    new_input = torch.cat((in_tensor[:, :, 0:1], in_tensor[:, :, len(task_list)+1:]), axis=2)
    return new_input


class CogModule():
    def __init__(self, model_dict):
        self.model_dict = model_dict
        for model in model_dict.values(): 
            model.to(device)
        self.total_task_list = []
        self.total_loss_dict = defaultdict(list)
        self.total_correct_dict = defaultdict(list)
        self.task_sorted_loss = {}
        self.task_sorted_correct = {}
        self.holdout_task = None
        self.holdout_instruct = None
        #set_seed()

    def reset_data(self): 
        self.total_task_list = []
        self.total_loss_dict = defaultdict(list)
        self.total_correct_dict = defaultdict(list)
        self.task_sorted_loss = {}
        self.task_sorted_correct = {}
        self.holdout_task = None
        self.holdout_instruct = None

    def _get_lang_input(self, model, batch_len, task_type, instruct_mode): 
        tokenizer_type = model.langMod.tokenizer_type
        batch_instruct, _ = get_batch(batch_len, None, tokenizer_type, task_type=task_type, instruct_mode=instruct_mode)
        return batch_instruct

    def _get_model_resp(self, model, batch_len, in_data, task_type, instruct_mode, holdout_task): 
        #change this
        h0 = model.initHidden(batch_len, 0.1).to(device)
        if model.isLang: 
            ins = del_input_rule(torch.Tensor(in_data)).to(device)
            instruct = self._get_lang_input(model, batch_len, task_type, instruct_mode)
            out, hid = model(instruct, ins, h0)
        else: 
            ins = torch.Tensor(in_data).to(device)
            if instruct_mode == 'masked' or ((task_type == holdout_task) and (model.instruct_mode == 'masked')): 
                ins = mask_input_rule(ins, batch_len, 120).to(device)
            out, hid = model(ins, h0)
        return out, hid

    def sort_perf_by_task(self): 
        for model_type in self.model_dict.keys():
            loss_temp_dict=defaultdict(list)
            for task, loss in zip(self.total_task_list, self.total_loss_dict[model_type]): 
                loss_temp_dict[task].append(loss)    
            
            correct_temp_dict=defaultdict(list)
            for task, correct in zip(self.total_task_list, self.total_correct_dict[model_type]): 
                correct_temp_dict[task].append(correct)

            self.task_sorted_correct[model_type] = correct_temp_dict
            self.task_sorted_loss[model_type] = loss_temp_dict


    def train(self, data, epochs, weight_decay = 0.0, lr = 0.001, holdout_task = None, instruct_mode = None, freeze_langModel = False): 
        self.holdout_task = holdout_task
        opt_dict = {}
        for model_type, model in self.model_dict.items(): 
            if (model.isLang and not model.tune_langModel) or (model.tune_langModel and freeze_langModel): 
                opt_dict[model_type] = optim.Adam(model.rnn.parameters(), lr=lr, weight_decay=weight_decay)
            else: 
                opt_dict[model_type] = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                print(model_type)
            model.train()
        
        ins_tensor = data[0]
        tar_tensor = data[1]
        mask_tensor = data[2]
        tar_dir_vec = data[3]
        task_type_vec = data[4]

        batch_len = ins_tensor.shape[1]
        batch_num = ins_tensor.shape[0]

        correct_array = np.empty((batch_len, batch_num), dtype=bool)

        holdout_instruct, _ = get_batch(batch_len, holdout_task, None, task_type=holdout_task, instruct_mode='single')

        for i in range(epochs):
            print('epoch', i)
            for j in range(batch_num): 
                task_type = task_type_vec[j]
                task_index = task_list.index(task_type)
                tar = torch.Tensor(tar_tensor[j, :, :, :]).to(device)
                mask = torch.Tensor(mask_tensor[j, :, :, :]).to(device)
                tar_dir = tar_dir_vec[j]
                if j%50 == 0: 
                    print(task_type)

                for model_type, model in self.model_dict.items(): 
                    opt = opt_dict[model_type]
                    opt.zero_grad()
                    if task_type == holdout_task and instruct_mode == 'single' and model.isLang: 
                        h0 = model.initHidden(batch_len, 0.1).to(device)
                        if model.embedderStr is not 'SBERT': 
                            instruct = toNumerals(model.embedderStr, holdout_instruct)
                        else: 
                            instruct = holdout_instruct
                        ins = del_input_rule(torch.Tensor(ins_tensor[j, :, :, :])).to(device)
                        print(instruct)
                        out, _ = model(instruct, ins, h0)
                    else: 
                        out, _ = self._get_model_resp(model, batch_len, ins_tensor[j, :, :, :], task_type, instruct_mode, holdout_task)
            
                    loss = masked_MSE_Loss(out, tar, mask) 
                
                    loss.backward()
                    opt.step()

                    if j%50 == 0: 
                        print(j, ':', model_type, ":", "{:.2e}".format(loss.item()))                
                        self.total_loss_dict[model_type].append(loss.item())
                    self.total_correct_dict[model_type].append(np.mean(isCorrect(out, tar, tar_dir)))
                self.total_task_list.append(task_type)
        self.sort_perf_by_task()
        return opt_dict

    def plot_learning_curve(self, mode, task_type=None, smoothing = 2):
        assert mode in ['loss', 'correct'], "mode must be 'loss' or 'correct', entered: %r" %mode
        if mode == 'loss': 
            y_label = 'MSE Loss'
            y_lim = (0, 250)
            title = 'MSE Loss over training'
            perf_dict = self.task_sorted_loss
        if mode == 'correct': 
            y_label = 'Fraction Correct/Batch'
            y_lim = (0, 1.15)
            title = 'Fraction Correct Response over training'
            perf_dict = self.task_sorted_correct

        if task_type is None: 
            fig, axn = plt.subplots(4,4, sharey = True)
            plt.suptitle(title)
            for i, ax in enumerate(axn.flat):
                ax.set_ylim(y_lim)
                if i > len(set(self.total_task_list)):
                    fig.delaxes(ax)
                    continue

                cur_task = Task.TASK_LIST[i]
                for model_type in self.model_dict.keys():
                    if cur_task not in self.total_task_list:
                        continue

                    smoothed_perf = gaussian_filter1d(perf_dict[model_type][cur_task], sigma=smoothing)
                    ax.plot(smoothed_perf)
                ax.set_title(cur_task)
        else:
            assert task_type in self.total_task_list, "model has no training examples on entered task_type: %r" %task_type
            fig, ax = plt.subplots(1,1)
            plt.suptitle(title)
            ax.set_ylim(y_lim)
            for model_type in self.model_dict.keys():            
                smoothed_perf = gaussian_filter1d(perf_dict[model_type][task_type], sigma=smoothing)
                ax.plot(smoothed_perf)
            ax.set_title(task_type)
        fig.legend(self.model_dict.keys())
        fig.text(0.5, 0.04, 'Batches', ha='center')
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
        fig.show()

    def get_performance(self, model, instruct_mode = None): 
        model.eval()
        batch_len = 250
        with torch.no_grad():
            perf_dict = dict.fromkeys(task_list)
            perf_dict_masked = dict.fromkeys(task_list)
            for task_type in task_list: 
                trial = construct_batch(task_type, batch_len)
                ins = trial.inputs
                out, _ = self._get_model_resp(model, batch_len, ins, task_type, instruct_mode, holdout_task=self.holdout_task) 
                #masked_out, _ = self._get_model_resp(model, batch_len, ins, task_type, instruct_mode = 'masked', holdout_task=self.holdout_task)
                perf_dict[task_type] = np.mean(isCorrect(out, trial.targets, trial.target_dirs))
                #perf_dict_masked[task_type] = np.mean(isCorrect(masked_out, trial.targets, trial.target_dirs))
        return perf_dict #perf_dict_masked

    def plot_trained_performance(self, model_dict=None, instruct_mode = None):
        if model_dict is None: 
            model_dict = self.model_dict
        barWidth = 0.1
        for i, model in enumerate(model_dict.values()):  
            perf_dict = self.get_performance(model, instruct_mode)
            keys = list(perf_dict.keys())
            values = list(perf_dict.values())
            len_values = len(task_list)
            if i == 0:
                r = np.arange(len_values)
            else:
                r = [x + barWidth for x in r]
            plt.bar(r, values, width =barWidth, label = list(model_dict.keys())[i])

        plt.ylim(0, 1.15)
        plt.title('Trained Performance')
        plt.xlabel('Task Type', fontweight='bold')
        plt.ylabel('Percentage Correct')
        r = np.arange(len_values)
        plt.xticks([r + barWidth for r in range(len_values)], task_list)
        plt.legend()
        plt.show()

    def plot_response(self, model, task_type, instruct_mode=None):
        task = construct_batch(task_type, 1)
        tar = task.targets
        ins = task.inputs
        h0 = model.initHidden(1, 0.1).to(device)

        if task_type == self.holdout_task and instruct_mode == 'single' and model.isLang: 
            if model.embedderStr is not 'SBERT': 
                instruct = toNumerals(model.embedderStr, self.holdout_instruct)
            else: 
                instruct = self.holdout_instruct
            ins = del_input_rule(ins)
            out, _ = model(instruct, torch.Tensors(ins).to(device), h0)
        else: 
            out, _ = self._get_model_resp(model, 1, ins, task_type, instruct_mode, self.holdout_task)
        
        out = out.squeeze().detach().to('cpu').numpy()
        to_plot = (ins.squeeze().T, tar.squeeze().T, out.T)

        fig, axn = plt.subplots(3,1, sharex = True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        ylabels = ('Input', 'Target', 'Output')
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title('%r Trial Response' %task_type)
            if i == 4: 
                ax.set_xlabel('time (DELTA_T=%r ms)'%Task.DELTA_T)
        plt.show()

    def plot_k_shot_learning(self, ks, task_type, model_dict=None, include_legend = False):
        if model_dict is None: 
            model_dict = self.model_dict
        barWidth = 0.1
        for i, model_name in enumerate(list(model_dict.keys())): 
            per_correct = [self.task_sorted_correct[model_name][task_type][k] for k in ks]
            len_values = len(ks)
            if i == 0:
                r = np.arange(len_values)
            else:
                r = [x + barWidth for x in r]
            plt.bar(r, per_correct, width =barWidth, label = list(model_dict.keys())[i])

        plt.ylim(0, 1.15)
        plt.title('Few-Shot Learning Performance')
        plt.xlabel('Number of Training Batches', fontweight='bold')
        plt.ylabel('Percentage Correct')
        r = np.arange(len_values)
        plt.xticks([r + barWidth for r in range(len_values)], ks)
        if include_legend: 
            plt.legend()
        plt.show()

    def save_models(self, holdout_task, foldername):
        pickle.dump(self.total_correct_dict, open(foldername+'/'+holdout_task+'_training_correct_dict', 'wb'))
        pickle.dump(self.total_loss_dict, open(foldername+'/'+holdout_task+'_training_loss_dict', 'wb'))
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'_'+model_name+'.pt'
            filename = filename.replace(' ', '_')
            torch.save(model.state_dict(), filename)

    def load_models(self, holdout_task, foldername):
        if 'Model1 Masked' in self.model_dict.keys(): 
            del(self.model_dict['Model1 Masked'])
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'_'+model_name+'.pt'
            filename = foldername.replace(' ', '_')
            model.load_state_dict(torch.load(filename))
        self.add_masked_model1()
        self.reset_data()

    def add_masked_model1(self): 
        net_masked = simpleNet(77, 128, 1, instruct_mode='masked').to(device)
        net_masked.load_state_dict(self.model_dict['Model1'].state_dict())
        models = list(self.model_dict.values())
        models.insert(1, net_masked)
        model_names = list(self.model_dict.keys())
        model_names.insert(1, 'Model1 Masked')
        self.model_dict = dict(zip(model_names, models))


class simpleNet(nn.Module): 
    def __init__(self, in_dim, hid_dim, num_layers, instruct_mode=None):
        super(simpleNet, self).__init__()
        self.tune_langModel = None
        self.instruct_mode = instruct_mode
        self.in_dim = in_dim
        self.out_dim = 33
        self.isLang = False
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(self.in_dim, hid_dim, self.num_layers, batch_first=True)
        self.W_out = nn.Linear(hid_dim, self.out_dim)
        self.weights_init()

    def weights_init(self):
        for n, p in self.named_parameters():
            if 'weight_ih' in n:
                for ih in p.chunk(3, 0):
                    torch.nn.init.normal_(ih, std = 1/np.sqrt(self.in_dim))
            elif 'weight_hh' in n:
                for hh in p.chunk(3, 0):
                    hh.data.copy_(torch.eye(self.hid_dim)*0.5)
            elif 'W_out' in n:
                torch.nn.init.normal_(p, std = 0.4/np.sqrt(self.hid_dim))

    def forward(self, x, h): 
        rnn_out, hid = self.rnn(x, h)
        motor_out = self.W_out(rnn_out)
        out = torch.sigmoid(motor_out)
        return out, hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)

class instructNet(nn.Module): 
    def __init__(self, langMod, hid_dim, num_layers, drop_p = 0.0, instruct_mode=None, tune_langModel = False): 
        super(instructNet, self).__init__()
        self.instruct_mode = instruct_mode
        self.tune_langModel = tune_langModel
        self.sensory_in_dim = 65
        self.isLang = True 
        self.hid_dim = hid_dim
        self.embedderStr = langMod.embedderStr
        self.langModel = langMod.langModel.eval()
        self.langMod = langMod
        self.num_layers = num_layers
        self.lang_embed_dim = langMod.langModel.out_dim
        self.rnn = simpleNet(self.sensory_in_dim+self.lang_embed_dim, hid_dim, self.num_layers)

        if tune_langModel:
            self.langModel.train()
            for param in self.langModel.parameters(): 
                param.requires_grad = True
        else: 
            for param in self.langModel.parameters(): 
                param.requires_grad = False
            self.langModel.eval()

    def forward(self, instruction_tensor, x, h):
        embedded_instruct = self.langModel(instruction_tensor)
        seq_blocked = embedded_instruct.unsqueeze(1).repeat(1, 120, 1)
        rnn_ins = torch.cat((seq_blocked, x.type(torch.float32)), 2)
        outs, hid = self.rnn(rnn_ins, h)
        return outs, hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)
