from Data import data_streamer
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter1d
import umap
from sklearn.manifold import TSNE

import pickle
import random
from collections import defaultdict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from Task import Task, construct_batch
from LangModule import get_batch, swaps
from Data import data_streamer


task_list = Task.TASK_LIST
tuning_dirs = Task.TUNING_DIRS


def gpu_to_np(t):
    """removes tensor from gpu and converts to np.array""" 
    if t.get_device() == 0: 
        t = t.detach().to('cpu').numpy()
    elif t.get_device() == -1: 
        t = t.detach().numpy()
    return t

def popvec(act_vec):
    """Population vector decoder that reads the orientation of activity in vector of activities
    Args:      
        act_vec (np.array): output tensor of neural network model; shape: (features, 1)

    Returns:
        float: decoded orientation of activity (in radians)
    """

    act_sum = np.sum(act_vec)
    temp_cos = np.sum(np.multiply(act_vec, np.cos(tuning_dirs)))/act_sum
    temp_sin = np.sum(np.multiply(act_vec, np.sin(tuning_dirs)))/act_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)

def get_dist(angle1, angle2):
    """Returns the true distance between two angles mod 2pi
    Args:      
        angle1, angle2 (float): angles in radians

    Returns:
        float: distance between given angles mod 2pi
    """
    dist = angle1-angle2
    return np.minimum(abs(dist),2*np.pi-abs(dist))

def isCorrect(nn_out, nn_target, target_dirs): 
    """Determines whether a given neural network response is correct, computed by batch
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_size, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_size, seq_len, features)
        target_dirs (np.array): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        np.array: weighted loss of neural network response; shape: (batch)
    """
    batch_size = nn_out.shape[0]
    if type(nn_out) == torch.Tensor: 
        nn_out = gpu_to_np(nn_out)
    if type(nn_target) == torch.Tensor: 
        nn_target = gpu_to_np(nn_target)

    isCorrect = np.empty(batch_size, dtype=bool)
    criterion = (2*np.pi)/10

    for i in range(batch_size):
        #checks response maintains fixataion
        isFixed = all(np.where(nn_target[i, :, 0] == 0.85, nn_out[i, :, 0] > 0.5, True))

        #checks trials that requiring repressing responses
        if np.isnan(target_dirs[i]): 
            isDir = all((nn_out[i, 114:119, :].flatten() < 0.15))
        
        #checks responses are coherent and in the correct direction
        else:
            is_response = np.max(nn_out[i, -1, 1:]) > 0.6
            loc = popvec(nn_out[i, -1, 1:])
            dist = get_dist(loc, target_dirs[i])        
            isDir = dist < criterion and is_response
        isCorrect[i] = isDir and isFixed
    return isCorrect

def masked_MSE_Loss(nn_out, nn_target, mask):
    """MSE loss (averaged over features then time) function with special weighting mask that prioritizes loss in response epoch 
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_num, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_num, seq_len, features)
        mask (Tensor): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: weighted loss of neural network response; shape: (1x1)
    """

    mask_applied = torch.mul(torch.pow((nn_out - nn_target), 2), mask)
    avg_applied = torch.mean(torch.mean(mask_applied, 2), 1)
    return torch.mean(avg_applied)

def comp_one_hot(task_type): 
    if task_type == 'Go': 
        comp_vec = Task._rule_one_hot('RT Go')+(Task._rule_one_hot('Anti Go')-Task._rule_one_hot('Anti RT Go'))
    if task_type == 'RT Go':
        comp_vec = Task._rule_one_hot('Go')+(Task._rule_one_hot('Anti RT Go')-Task._rule_one_hot('Anti Go'))
    if task_type == 'Anti Go':
        comp_vec = Task._rule_one_hot('Anti RT Go')+(Task._rule_one_hot('Go')-Task._rule_one_hot('RT Go'))
    if task_type == 'Anti RT Go':
        comp_vec = Task._rule_one_hot('Anti Go')+(Task._rule_one_hot('RT Go')-Task._rule_one_hot('Go'))
    if task_type == 'DM':
        comp_vec = Task._rule_one_hot('MultiDM') + (Task._rule_one_hot('Anti DM') - Task._rule_one_hot('Anti MultiDM'))
    if task_type == 'Anti DM': 
        comp_vec = Task._rule_one_hot('Anti MultiDM') + (Task._rule_one_hot('DM') - Task._rule_one_hot('MultiDM'))
    if task_type == 'MultiDM': 
        comp_vec = Task._rule_one_hot('DM') + (Task._rule_one_hot('Anti MultiDM')-Task._rule_one_hot('Anti DM'))
    if task_type == 'Anti MultiDM': 
        comp_vec = Task._rule_one_hot('Anti DM') + (Task._rule_one_hot('MultiDM')-Task._rule_one_hot('DM'))
    if task_type == 'COMP1': 
        comp_vec = Task._rule_one_hot('COMP2') + (Task._rule_one_hot('MultiCOMP1')-Task._rule_one_hot('MultiCOMP2'))
    if task_type == 'COMP2': 
        comp_vec = Task._rule_one_hot('COMP1') + (Task._rule_one_hot('MultiCOMP2')-Task._rule_one_hot('MultiCOMP1'))
    if task_type == 'MultiCOMP1': 
        comp_vec = Task._rule_one_hot('MultiCOMP2') + (Task._rule_one_hot('COMP1')-Task._rule_one_hot('COMP2'))
    if task_type == 'MultiCOMP2': 
        comp_vec = Task._rule_one_hot('MultiCOMP1') + (Task._rule_one_hot('COMP2')-Task._rule_one_hot('COMP1'))
    if task_type == 'DMS': 
        comp_vec = Task._rule_one_hot('DMC') + (Task._rule_one_hot('DNMS')-Task._rule_one_hot('DNMC'))
    if task_type == 'DMC': 
        comp_vec = Task._rule_one_hot('DMS') + (Task._rule_one_hot('DNMC')-Task._rule_one_hot('DNMS'))
    if task_type == 'DNMS': 
        comp_vec = Task._rule_one_hot('DNMC') + (Task._rule_one_hot('DMS')-Task._rule_one_hot('DMC'))
    if task_type == 'DNMC': 
        comp_vec = Task._rule_one_hot('DNMS') + (Task._rule_one_hot('DMC')-Task._rule_one_hot('DMS'))

    return comp_vec


def one_hot_input_rule(in_tensor, task_type, shuffled=False): 
    if shuffled: index = Task.SHUFFLED_TASK_LIST.index(task_type) 
    else: index = Task.TASK_LIST.index(task_type)
    one_hot = torch.zeros(len(Task.TASK_LIST))
    one_hot[index] = 1
    one_hot_tensor= one_hot.unsqueeze(0).repeat(in_tensor.shape[0], in_tensor.shape[1], 1).to(in_tensor.get_device())

    inputs = torch.cat((in_tensor[:, :, 0:1], one_hot_tensor, in_tensor[:, :, 1:]), axis=2)
    return inputs


def mask_input_rule(in_tensor, lang_dim=None): 
    """Masks the one-hot rule information in an input batch 
    Args:      
        in_tensor (Tensor): input tensor for a batch of trials; shape: (batch_size, seq_len, features) 
    
    Returns:
        Tensor: identical to in_tensor with one-hot rule information zero'd out; shape: (batch_size, seq_len, features) 
    """
    if lang_dim is not None: 
        mask = torch.zeros((in_tensor.shape[0], in_tensor.shape[1], lang_dim)).to(device)
        masked_input = torch.cat((mask, in_tensor), axis=2)
    else: 
        mask = torch.zeros((in_tensor.shape[0], in_tensor.shape[1], len(task_list))).to(device)
        masked_input = torch.cat((in_tensor[:, :, 0:1], mask, in_tensor[:, :, 1:]), axis=2)
    return masked_input


def comp_input_rule(in_tensor, task_type): 
    """Replaces one-hot input rule with a analagous linear combination of rules for related tasks
    Args:      
        in_tensor (Tensor): input tensor for a batch of trials; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: identical input data as in_tensor with combinational input rule; shape: (batch_num, seq_len, features-#tasks) 
    """
    comp_vec = comp_one_hot(task_type)
    comp_tensor = torch.Tensor(comp_vec).unsqueeze(0).repeat(in_tensor.shape[0], in_tensor.shape[1], 1).to(in_tensor.get_device())
    comp_input = torch.cat((in_tensor[:, :, 0:1], comp_tensor, in_tensor[:, :, 1:]), axis=2)
    return comp_input

def use_shuffled_one_hot(in_tensor, task_type): 
    shuffled_one_hot = torch.Tensor(Task._rule_one_hot(task_type, shuffled=True)).unsqueeze(0).repeat(in_tensor.shape[0], in_tensor.shape[1], 1).to(in_tensor.get_device())
    shuffled_one_hot_input = torch.cat((in_tensor[:, :, 0:1], shuffled_one_hot, in_tensor[:, :, 1:]), axis=2)
    return shuffled_one_hot_input

def swap_input_rule(in_tensor, task_type): 
    """Swaps one-hot rule inputs for given tasks 
    'Go' <--> 'Anti DM' 
    'Anti RT Go' <--> 'DMC'
    'RT Go' <--> 'COMP2'
    Args:      
        in_tensor (Tensor): input tensor for a batch of trials; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: identical input data as in_tensor with swapped input rule; shape: (batch_num, seq_len, features) 
    """

    assert task_type in ['Go', 'Anti DM', 'Anti RT Go', 'DMC', 'RT Go', 'COMP2']
    swapped = [x for x in swaps if task_type in x][0]
    swap_task = swapped[swapped.index(task_type)-1]
    swapped_one_hot = torch.Tensor(Task._rule_one_hot(swap_task)).unsqueeze(0).repeat(in_tensor.shape[0], in_tensor.shape[1], 1).to(in_tensor.get_device())
    swapped_input = torch.cat((in_tensor[:, :, 0:1], swapped_one_hot, in_tensor[:, :, len(task_list)+1:]), axis=2)
    return swapped_input


class CogModule():
    def __init__(self, model_dict):
        self.model_dict = model_dict
        for model in self.model_dict.values(): 
            if model is not None: model.to(device)
        self.total_task_list = []
        self.total_loss_dict = defaultdict(list)
        self.total_correct_dict = defaultdict(list)
        self.task_sorted_loss = {}
        self.task_sorted_correct = {}
        self.opt_dict = None

    def reset_data(self): 
        self.total_task_list = []
        self.total_loss_dict = defaultdict(list)
        self.total_correct_dict = defaultdict(list)
        self.task_sorted_loss = {}
        self.task_sorted_correct = {}
    
    @staticmethod
    def _get_lang_input(model, batch_len, task_type, instruct_mode): 
        tokenizer = model.langModel.tokenizer
        batch_instruct, _ = get_batch(batch_len, tokenizer, task_type=task_type, instruct_mode=instruct_mode)
        return batch_instruct

    @staticmethod
    def _get_model_resp(model, batch_len, ins, task_type): 
        h0 = model.initHidden(batch_len, 0.1).to(device)
        if model.isLang: 
            if model.instruct_mode == 'masked': 
                ins = mask_input_rule(ins, model.langModel.out_dim)
                out, hid = model.rnn(ins, h0)
            else: 
                instruct = CogModule._get_lang_input(model, batch_len, task_type, model.instruct_mode)
                out, hid = model(instruct, ins, h0)
        else: 
            instruct = None
            if model.instruct_mode == 'masked': 
                ins = mask_input_rule(ins).to(device)
            if model.instruct_mode == 'comp': 
                ins = comp_input_rule(ins, task_type)
            if model.instruct_mode == 'instruct_swap': 
                ins = swap_input_rule(ins, task_type)
            if model.instruct_mode == 'shuffled_one_hot':
                ins = use_shuffled_one_hot(ins, task_type)
            else: 
                ins = one_hot_input_rule(ins, task_type)

            out, hid = model(ins, h0)
        return out, hid

    def save_training_data(self, holdout_task,  foldername, name): 
        holdout_task = holdout_task.replace(' ', '_')
        pickle.dump(self.task_sorted_correct, open(foldername+'/'+holdout_task+'/'+name+'_training_correct_dict', 'wb'))
        pickle.dump(self.task_sorted_loss, open(foldername+'/'+holdout_task+'/'+name+'_training_loss_dict', 'wb'))

    def load_training_data(self, holdout_task, foldername, name): 
        holdout_task = holdout_task.replace(' ', '_')
        self.task_sorted_correct = pickle.load(open(foldername+'/'+holdout_task+'/'+name+'_training_correct_dict', 'rb'))
        self.task_sorted_loss = pickle.load(open(foldername+'/'+holdout_task+'/'+name+'_training_loss_dict', 'rb'))

    def sort_perf_by_task(self, holdout=None): 
        for model_type in self.model_dict.keys():
            loss_temp_dict=defaultdict(list)

            if holdout is None: 
                tasks = self.total_task_list
            else: 
                tasks = [holdout] * len(self.total_loss_dict[model_type])

            for task, loss in zip(tasks, self.total_loss_dict[model_type]): 
                loss_temp_dict[task].append(loss)    
            
            correct_temp_dict=defaultdict(list)
            for task, correct in zip(tasks, self.total_correct_dict[model_type]): 
                correct_temp_dict[task].append(correct)

            self.task_sorted_correct[model_type] = correct_temp_dict
            self.task_sorted_loss[model_type] = loss_temp_dict

    def save_models(self, holdout_task, foldername, name):
        self.save_training_data(holdout_task, foldername, name)
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'/'+holdout_task+'_'+model_name+'.pt'
            filename = filename.replace(' ', '_')
            torch.save(model.state_dict(), filename)

    def load_models(self, holdout_task, foldername):
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'/'+holdout_task+'_'+model_name+'.pt'
            filename = filename.replace(' ', '_')
            model.load_state_dict(torch.load(filename))
    
    def init_optimizers(self, weight_decay, lr, milestones, freeze_langModel, langLR, langWeightDecay):
        self.opt_dict = {}
        for model_type, model in self.model_dict.items(): 
            if (model.isLang and not model.tune_langModel) or (model.tune_langModel and freeze_langModel): 
                optimizer = optim.Adam(model.rnn.parameters(), lr=lr, weight_decay=weight_decay)
                self.opt_dict[model_type] =(optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5))
            else: 
                if (langWeightDecay is not None or langLR is not None) and model.isLang: 
                    print('LangWeightDecay')
                    optimizer = optim.Adam([
                            {'params' : model.rnn.parameters()},
                            {'params' : model.langModel.model.parameters(), 'lr': langLR, 'weight_decay':langWeightDecay}
                        ], lr=lr, weight_decay=weight_decay)
                else: 
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                self.opt_dict[model_type] = (optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5))
                print(model_type)
            model.train()
        
    def train(self, streamer, epochs, scheduler = True, weight_decay = 0.0, lr = 0.001, milestones = [], 
                freeze_langModel = False, langLR = None, langWeightDecay=None): 
        #torch.autograd.set_detect_anomaly
        self.init_optimizers(weight_decay, lr, milestones, freeze_langModel, langLR, langWeightDecay)
        batch_len = streamer.batch_len 

        for model_type, model in self.model_dict.items(): 
            opt = self.opt_dict[model_type][0]
            opt_scheduler = self.opt_dict[model_type][1]
            for i in range(epochs):
                print('epoch', i)
                streamer.permute_task_order()
                for j, data in enumerate(streamer.get_batch()): 
                    ins, tar, mask, tar_dir, task_type = data

                    opt.zero_grad()
                    out, _ = self._get_model_resp(model, batch_len, torch.Tensor(ins).to(device), task_type)

                    loss = masked_MSE_Loss(out, torch.Tensor(tar).to(device), torch.Tensor(mask).to(device)) 
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
                    opt.step()

                    frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                    self.total_loss_dict[model_type].append(loss.item())
                    self.total_correct_dict[model_type].append(frac_correct)
                    if j%50 == 0:
                        print(task_type)
                        print(j, ':', model_type, ":", "{:.2e}".format(loss.item()))
                        self.sort_perf_by_task()
                        print('Frac Correct ' + str(frac_correct) + '\n')
                    self.total_task_list.append(task_type) 
                if scheduler: 
                    opt_scheduler.step()    


    def _get_performance(self, model, num_batches): 
        model.eval()
        batch_len = 128
        with torch.no_grad():
            perf_dict = dict.fromkeys(task_list)
            for task_type in task_list:
                for _ in range(num_batches): 
                    mean_list = [] 
                    ins, targets, _, target_dirs, _ = construct_batch(task_type, batch_len)
                    out, _ = self._get_model_resp(model, batch_len, torch.Tensor(ins).to(device), task_type) 
                    mean_list.append(np.mean(isCorrect(out, torch.Tensor(targets).to(device), target_dirs)))
                perf_dict[task_type] = np.mean(mean_list)
        return perf_dict 

    def _plot_trained_performance(self, instruct_mode = None):
        barWidth = 0.1
        for i, model in enumerate(self.model_dict.values()):  
            perf_dict = self._get_performance(model, 5)
            keys = list(perf_dict.keys())
            values = list(perf_dict.values())
            len_values = len(task_list)
            if i == 0:
                r = np.arange(len_values)
            else:
                r = [x + barWidth for x in r]
            plt.bar(r, values, width =barWidth, label = list(self.model_dict.keys())[i])

        plt.ylim(0, 1.15)
        plt.title('Trained Performance')
        plt.xlabel('Task Type', fontweight='bold')
        plt.ylabel('Percentage Correct')
        r = np.arange(len_values)
        plt.xticks([r + barWidth for r in range(len_values)], task_list)
        plt.legend()
        plt.show()

    def plot_response(self, model_name, task_type, task = None, trial_num = 0, instruct = None, instruct_mode=None, show=True):
        model = self.model_dict[model_name]
        model.eval()
        with torch.no_grad(): 
            if task == None: 
                task = construct_batch(task_type, 1)

            tar = task.targets
            ins = task.inputs
            h0 = model.initHidden(1, 0.1).to(device)

            if instruct is not None: 
                instruct = [instruct]
                ins = torch.Tensor(ins).to(device)
                out, hid = model(instruct, ins, h0)
            else: 
                out, hid, instruct = self._get_model_resp(model, ins.shape[0], torch.Tensor(ins).to(device), task_type, instruct_mode)
            
            correct = isCorrect(out, torch.Tensor(tar), task.target_dirs)[trial_num]


            out = out.detach().cpu().numpy()[trial_num, :, :]
            hid = hid.detach().cpu().numpy()[trial_num, :, :]



            fix = ins[trial_num, :, 0:1]            
            num_rules = len(Task.TASK_LIST)
            rule_vec = ins[trial_num, :, 1:num_rules+1]
            mod1 = ins[trial_num, :, 1+num_rules:1+num_rules+Task.STIM_DIM]
            mod2 = ins[trial_num, :, 1+num_rules+Task.STIM_DIM:1+num_rules+(2*Task.STIM_DIM)]

            #to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, rule_vec.squeeze().T, tar.squeeze().T, out.squeeze().T]
            to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, tar[trial_num, :, :].T, out.squeeze().T]
            gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 2, 5, 5])
            ylabels = ['fix.', 'mod. 1', 'mod. 2',  'Target', 'Response']

            if model.isLang: 
                embedded_instruct = model.langModel(self._get_lang_input(model, 1, task_type, instruct_mode))
                task_info = embedded_instruct.repeat(120, 1).cpu().numpy()
                task_info_str = 'Instruction Embedding'
            else: 
                task_info = rule_vec
                task_info_str = 'Task one-hot'

            to_plot.insert(3, task_info.T)
            ylabels.insert(3, task_info_str)

            fig, axn = plt.subplots(6,1, sharex = True, gridspec_kw=gs_kw)
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, ax in enumerate(axn.flat):
                sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
                #sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=False, vmin=0, vmax=1, cbar_ax=None)

                ax.set_ylabel(ylabels[i])
                if i == 0: 
                    ax.set_title(r"$\textbf{Decision Making (DM) Trial Info}$")
                if i == 5: 
                    ax.set_xlabel('time')
            #plt.tight_layout()

            if show: 
                plt.show()
            return correct, instruct[trial_num]
