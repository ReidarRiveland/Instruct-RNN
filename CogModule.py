import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib import rc

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter1d

import pickle
import random
from collections import defaultdict


from Task import Task, construct_batch
from LangModule import get_batch, toNumerals
from RNNs import simpleNet

task_list = Task.TASK_LIST
tuning_dirs = Task.TUNING_DIRS


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
    avg_applied = torch.mean(torch.mean(mask_applied, 2), 1)
    return torch.mean(avg_applied)

def mask_input_rule(in_tensor, batch_len, seq_len): 
    mask = torch.zeros((batch_len, seq_len, len(task_list)))
    masked_input = torch.cat((in_tensor[:, :, 0:1], mask, in_tensor[:, :, len(task_list)+1:]), axis=2)
    return masked_input

def del_input_rule(in_tensor): 
    new_input = torch.cat((in_tensor[:, :, 0:1], in_tensor[:, :, len(task_list)+1:]), axis=2)
    return new_input

class CogModule():
    ALL_STYLE_DICT = {'Model1': ('blue', None), 'SIF':('brown', None), 'BoW': ('orange', None), 'GPT_cat': ('red', '^'), 'GPT train': ('red', '.'), 
                            'BERT_cat': ('green', '^'), 'BERT train': ('green', '+'), 'S-Bert_cat': ('purple', '^'), 'S-Bert train': ('purple', '.'), 
                            'InferSent train': ('yellow', '.'), 'InferSent_cat': ('yellow', '^'), 'Transformer': ('pink', '.')}
    COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'GPT': 'red', 'BERT': 'green', 'S-Bert': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
    MODEL_MARKER_DICT = {'SIF':None, 'BoW':None, 'cat': '^', 'train': '.', 'Transformer':'.'}
    MARKER_DICT = {'^': 'task categorization', '.': 'end-to-end'}
    NAME_TO_PLOT_DICT = {'Model1': 'One-Hot Task Vec.', 'SIF':'SIF', 'BoW': 'BoW', 'GPT_cat': 'GPT (task cat.)', 'GPT train': 'GPT (end-to-end)', 
                            'BERT_cat': 'BERT (task cat.)', 'BERT train': 'BERT (end-to-end)', 'S-Bert_cat': 'S-BERT (task cat.)', 'S-Bert train': 'S-BERT (end-to-end)', 
                            'InferSent train': 'InferSent (end-to-end)', 'InferSent_cat': 'InferSent (task cat.)', 'Transformer': 'Transformer (end-to-end)'}

    def __init__(self, model_dict):
        self.model_dict = model_dict
        for model in self.model_dict.values(): 
            if model is not None: model.to(device)
        self.total_task_list = []
        self.total_loss_dict = defaultdict(list)
        self.total_correct_dict = defaultdict(list)
        self.task_sorted_loss = {}
        self.task_sorted_correct = {}
        self.holdout_task = None
        self.holdout_instruct = None

    def get_model_patches(self): 
        Patches = []
        Markers = []
        color_dict = self.COLOR_DICT.copy()
        for model_name in self.model_dict.keys(): 
            architecture_type = list(self.COLOR_DICT.keys())[np.where([model_name.startswith(key) for key in self.COLOR_DICT.keys()])[0][0]]
            try:
                color = color_dict.pop(architecture_type)
            except:
                continue
            if architecture_type == 'Model1': architecture_type = 'One-Hot Vec.'
            patch = mpatches.Patch(color=color, label=architecture_type)
            Patches.append(patch)

        for model_name in self.model_dict.keys(): 
            if model_name in ['Model1', 'BoW', 'SIF']: 
                continue
            where_array = np.array([model_name.find(key) for key in self.MODEL_MARKER_DICT.keys()])
            marker = self.MODEL_MARKER_DICT[list(self.MODEL_MARKER_DICT.keys())[np.where(where_array >= 0)[0][0]]]
            if any([marker == m.get_marker() for m in Markers]): 
                continue
            mark = Line2D([0], [0], marker=marker, color='w', label=self.MARKER_DICT[marker], markerfacecolor='grey', markersize=10)
            Markers.append(mark)

        return Patches, Markers

    def reset_data(self): 
        self.total_task_list = []
        self.total_loss_dict = defaultdict(list)
        self.total_correct_dict = defaultdict(list)
        self.task_sorted_loss = {}
        self.task_sorted_correct = {}
        self.holdout_task = None
        self.holdout_instruct = None

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

    def save_models(self, holdout_task, foldername):
        self.save_training_data(holdout_task, foldername, holdout_task)
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'/'+holdout_task+'_'+model_name+'.pt'
            filename = filename.replace(' ', '_')
            torch.save(model.state_dict(), filename)

    def load_models(self, holdout_task, foldername):
        self.load_training_data(holdout_task, foldername, holdout_task)
        if 'Model1 Masked' in self.model_dict.keys(): 
            del(self.model_dict['Model1 Masked'])
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'/'+holdout_task+'_'+model_name+'.pt'
            filename = filename.replace(' ', '_')
            model.load_state_dict(torch.load(filename))

    def add_masked_model1(self): 
        in_dim = self.model_dict['Model1'].in_dim
        net_masked = simpleNet(in_dim, 128, 1, instruct_mode='masked').to(device)
        net_masked.load_state_dict(self.model_dict['Model1'].state_dict())
        models = list(self.model_dict.values())
        models.insert(1, net_masked)
        model_names = list(self.model_dict.keys())
        model_names.insert(1, 'Model1 Masked')
        self.model_dict = dict(zip(model_names, models))

    def _get_lang_input(self, model, batch_len, task_type, instruct_mode): 
        tokenizer = model.langModel.tokenizer
        batch_instruct, _ = get_batch(batch_len, tokenizer, task_type=task_type, instruct_mode=instruct_mode)
        return batch_instruct

    def _get_model_resp(self, model, batch_len, ins, task_type, instruct_mode, holdout_task): 
        h0 = model.initHidden(batch_len, 0.1).to(device)
        if model.isLang: 
            ins = del_input_rule(ins)
            instruct = self._get_lang_input(model, batch_len, task_type, instruct_mode)
            out, hid = model(instruct, ins, h0)
        else: 
            if instruct_mode == 'masked' or ((task_type == holdout_task) and (model.instruct_mode == 'masked')): 
                ins = mask_input_rule(ins, batch_len, 120)
            out, hid = model(ins, h0)
        return out, hid

    def train(self, data, epochs, weight_decay = 0.0, lr = 0.001, holdout_task = None, instruct_mode = None, freeze_langModel = False): 
        self.holdout_task = holdout_task
        opt_dict = {}
        for model_type, model in self.model_dict.items(): 
            if (model.isLang and not model.tune_langModel) or (model.tune_langModel and freeze_langModel): 
                optimizer = optim.Adam(model.rnn.parameters(), lr=lr, weight_decay=weight_decay)
                opt_dict[model_type] =(optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.5))
            else: 
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                opt_dict[model_type] = (optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.5))
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

        for i in range(epochs):
            print('epoch', i)
            for j in range(batch_num): 
                task_type = task_type_vec[j]
                task_index = task_list.index(task_type)
                tar = torch.Tensor(tar_tensor[j, :, :, :]).to(device)
                mask = torch.Tensor(mask_tensor[j, :, :, :]).to(device)
                ins = torch.Tensor(ins_tensor[j, :, :, :]).to(device)
                tar_dir = tar_dir_vec[j]
                if j%50 == 0: 
                    print(task_type)

                for model_type, model in self.model_dict.items(): 
                    opt = opt_dict[model_type][0]
                    opt.zero_grad()
                    out, _ = self._get_model_resp(model, batch_len, ins, task_type, instruct_mode, holdout_task)
                    loss = masked_MSE_Loss(out, tar, mask) 
                    loss.backward()
                    opt.step()

                    if j%50 == 0: 
                        print(j, ':', model_type, ":", "{:.2e}".format(loss.item()))                
                    self.total_loss_dict[model_type].append(loss.item())
                    self.total_correct_dict[model_type].append(np.mean(isCorrect(out, tar, tar_dir)))
                self.total_task_list.append(task_type)                
            self.sort_perf_by_task()
            for model_type in self.model_dict.keys(): 
                opt_dict[model_type][1].step()
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
            y_lim = (-0.05, 1.15)
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
                for model_name in self.model_dict.keys():
                    if cur_task not in self.total_task_list:
                        continue
                    smoothed_perf = gaussian_filter1d(perf_dict[model_name][cur_task], sigma=smoothing)
                    ax.plot(smoothed_perf, color = self.ALL_STYLE_DICT[model_name][0], marker = self.ALL_STYLE_DICT[model_name][1], markersize = 5, markevery=2)
                ax.set_title(cur_task)
        else:
            fig, ax = plt.subplots(1,1)
            plt.suptitle(title)
            ax.set_ylim(y_lim)
            for model_name in self.model_dict.keys():    
                smoothed_perf = gaussian_filter1d(perf_dict[model_name][task_type], sigma=smoothing)
                ax.plot(smoothed_perf, color = self.ALL_STYLE_DICT[model_name][0], marker = self.ALL_STYLE_DICT[model_name][1], markersize = 5, markevery=2)
            ax.set_title(task_type + ' holdout')

        Patches, Markers = self.get_model_patches()
        arch_legend = plt.legend(handles=Patches, title = r"$\textbf{Language Module}$", bbox_to_anchor = (0.9, 0.25), loc = 'lower center')
        ax = plt.gca().add_artist(arch_legend)
        plt.legend(handles= Markers, title = r"$\textbf{Transformer Fine-Tuning}$", bbox_to_anchor = (0.9, 0.25), loc = 'upper center')
        fig.text(0.5, 0.04, 'Batches', ha='center')
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
        fig.show()

    def _get_performance(self, model, instruct_mode = None): 
        model.eval()
        batch_len = 250
        with torch.no_grad():
            perf_dict = dict.fromkeys(task_list)
            perf_dict_masked = dict.fromkeys(task_list)
            for task_type in task_list: 
                trial = construct_batch(task_type, batch_len)
                ins = trial.inputs
                out, _ = self._get_model_resp(model, batch_len, torch.Tensor(ins).to(device), task_type, instruct_mode, holdout_task=self.holdout_task) 
                perf_dict[task_type] = np.mean(isCorrect(out, trial.targets, trial.target_dirs))
        return perf_dict 

    def _plot_trained_performance(self, instruct_mode = None):
        barWidth = 0.1
        for i, model in enumerate(self.model_dict.values()):  
            perf_dict = self._get_performance(model, instruct_mode)
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

    def plot_response(self, model_name, task_type, instruct = None, instruct_mode=None):
        model = self.model_dict[model_name]
        model.eval()
        with torch.no_grad(): 
            task = construct_batch(task_type, 1)
            tar = task.targets
            ins = task.inputs
            h0 = model.initHidden(1, 0.1).to(device)

            if instruct is not None: 
                if model.embedderStr is not 'SBERT': 
                    instruct = toNumerals(model.embedderStr, instruct)
                else: 
                    instruct = [instruct]
def plot_multi_learning_curves(model_dict, tasks, foldername, smoothing=1): 
    cog = CogModule(model_dict)
    fig, axn = plt.subplots(2,2, sharey = True, sharex=True)
    plt.suptitle('Holdout Learning Curves')
    for i, task in enumerate(tasks): 
        ax = axn.flat[i]
        cog.load_training_data(task, foldername, 'holdout')
        for model_name in model_dict.keys(): 
            smoothed_perf = gaussian_filter1d(cog.task_sorted_correct[model_name][task], sigma=smoothing)
            ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_name][0], marker=cog.ALL_STYLE_DICT[model_name][1], alpha=1, markersize=5, markevery=3)
        ax.set_title(task + ' Holdout')
    Patches, Markers = cog.get_model_patches()
    label_plot(fig, Patches, Markers, legend_loc=(1.3, 0.5))
    fig.show()


plot_multi_learning_curves(model_dict, ['Anti Go', 'Anti DM', 'MultiDM', 'DMC'], foldername, 1)

                out, hid = self._get_model_resp(model, 1, torch.Tensor(ins).to(device), task_type, instruct_mode, self.holdout_task)
            
            out = out.squeeze().detach().cpu().numpy()
            hid = hid.squeeze().detach().cpu().numpy()
            
            ylabels = ['Input', 'Hidden', 'Output', 'Target']
            ins = del_input_rule(torch.Tensor(ins)).numpy()
            to_plot = [ins.squeeze().T, hid.T, out.T, tar.squeeze().T]

            if model.isLang: 
                embedded_instruct = model.langModel(self._get_lang_input(model, 1, task_type, instruct_mode))
                task_info = embedded_instruct.repeat(120, 1).cpu().numpy()
                task_info_str = 'Instruction Vec.'
            else: 
                one_hot = np.zeros(len(task_list))
                one_hot[task_list.index(task_type)] = 1
                task_info = one_hot.repeat(120, 1)
                task_info_str = 'Task one-hot'

            ylabels.insert(1, task_info_str)
            to_plot.insert(1, task_info.T)

            gs_kw = dict(width_ratios=[1], height_ratios=[65, 20, 120, 33, 33])


            fig, axn = plt.subplots(5,1, sharex = True, gridspec_kw=gs_kw)

            for i, ax in enumerate(axn.flat):
                sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax)
                ax.set_ylabel(ylabels[i])
                if i == 0: 
                    ax.set_title(model_name + ' %r Trial Response' %task_type)
                if i == len(to_plot): 
                    ax.set_xlabel('time')
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
            plt.bar(r, per_correct, width =barWidth, label = list(model_dict.keys())[i], color = self.ALL_STYLE_DICT[model_name][0])

        plt.ylim(0, 1.15)
        plt.title('Few-Shot Learning Performance')
        plt.xlabel('Number of Training Batches', fontweight='bold')
        plt.ylabel('Percentage Correct')
        r = np.arange(len_values)
        plt.xticks([r + barWidth for r in range(len_values)], ks)
        if include_legend: 
            plt.legend()
        plt.show()

    def plot_task_rep(self, model_name, epoch, num_trials = 250, dim = 2, instruct_mode = None, tasks = task_list, avg_rep = True): 
        model = self.model_dict[model_name]
        if not next(model.rnn.parameters()).is_cuda:
            model.to(device)

        assert epoch in ['input', 'stim', 'response', 'prep'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
        
        task_reps = []
        for task in task_list: 
            trials = construct_batch(task, num_trials)
            tar = trials.targets
            ins = trials.inputs

            out, hid = self._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task, instruct_mode, None)

            hid = hid.detach().cpu().numpy()
            epoch_state = []

            for i in range(num_trials): 
                if epoch.isnumeric(): 
                    epoch_index = int(epoch)
                    epoch_state.append(hid[i, epoch_index, :])
                if epoch == 'stim': 
                    epoch_index = np.where(tar[i, :, 0] == 0.85)[0][-1]
                    epoch_state.append(hid[i, epoch_index, :])
                if epoch == 'response':
                    epoch_state.append(hid[i, -1, :])
                if epoch == 'input':
                    epoch_state.append(hid[i, 0, :])
                if epoch == 'prep': 
                    epoch_index = np.where(ins[0, :, 18:]>0.25)[0][0]-1
                    epoch_state.append(hid[i, epoch_index, :])
            
            if avg_rep: 
                epoch_state = [np.mean(np.stack(epoch_state), axis=0)]
            
            task_reps += epoch_state


        embedded = PCA(n_components=dim).fit_transform(task_reps)

        if avg_rep: 
            to_plot = np.stack([embedded[task_list.index(task), :] for task in tasks])
            task_indices = np.array([task_list.index(task) for task in tasks]).astype(int)
            marker_size = 100
        else: 
            to_plot = np.stack([embedded[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials, :] for task in tasks]).reshape(len(tasks)*num_trials, dim)
            task_indices = np.array([[task_list.index(task)]*num_trials for task in tasks]).astype(int).flatten()
            marker_size = 25

        cmap = matplotlib.cm.get_cmap('tab20')

        if dim ==3: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:,2], c = cmap(task_indices), cmap=cmap, s=marker_size)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')

        else:             
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.scatter(to_plot[:, 0], to_plot[:, 1], c=cmap(task_indices), cmap=cmap, s=marker_size)
            plt.xlabel("PC 1", fontsize = 18)
            plt.ylabel("PC 2", fontsize = 18)

        plt.title("PCA Embedding for Task Rep.", fontsize=18)
        digits = np.arange(len(tasks))
        Patches = [mpatches.Patch(color=cmap(d), label=task_list[d]) for d in set(task_indices)]


        plt.legend(handles=Patches)
        plt.show()


        return dict(zip(tasks, to_plot))

    def get_hid_traj(self, models, tasks, dim, instruct_mode):
        models = [self.model_dict[model] for model in models]
        for model in models: 
            if not next(model.rnn.parameters()).is_cuda:
                model.to(device)
 
        task_info_list = []
        for task in tasks: 
            trial = construct_batch(task, 1)
            task_info_list.append(trial.inputs)

        model_task_state_dict = {}
        for model_name, model in self.model_dict.items(): 
            tasks_dict = {}
            for i, task in enumerate(tasks): 
                out, hid = self._get_model_resp(model, 1, torch.Tensor(task_info_list[i]).to(device), task, instruct_mode, None)
                embedded = PCA(n_components=dim).fit_transform(hid.squeeze().detach().cpu())
                tasks_dict[task] = embedded
            model_task_state_dict[model_name] = tasks_dict
        return model_task_state_dict

    def plot_hid_traj(self, models, tasks, dim, instruct_mode = None): 
        if dim == 2: 
            fig, ax = plt.subplots()
        else: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        fig.suptitle('RNN Hidden State Trajectory')
        model_task_state_dict = self.get_hid_traj(models, tasks, dim, instruct_mode)

        data_array = np.empty((len(models)* len(tasks)*dim), dtype=list)

        for i in range(data_array.shape[0]): 
            data_array[i] = [] 
        data_array = np.reshape(data_array, (len(models), len(tasks), dim))

        plot_list = []
        style_list = ['-', '--']
        for i in range(len(models)):
            for j in range (len(tasks)): 
                model_name = models[i]
                if dim == 2: 
                    plot_list.append(plt.plot([],[], label = model_name, color = self.ALL_STYLE_DICT[models[i]][0], linestyle = style_list[j]))
                else:
                    embedding_data = model_task_state_dict[models[i]][tasks[j]]
                    plot_list.append(plt.plot(embedding_data[0, 0:1],embedding_data[1, 0:1], embedding_data[2, 0:1], color = self.ALL_STYLE_DICT[models[i]][0], linestyle = style_list[j]))

        plot_array = np.array(plot_list).reshape((len(models), len(tasks)))

        def init():
            ax.set_xlim(-10, 10)
            ax.set_xlabel('PC 1')
            ax.set_ylim(-10, 10)
            ax.set_ylabel('PC 2')
            if dim == 3: 
                ax.set_zlim(-10, 10)
                ax.set_zlabel('PC 3')
            return tuple(plot_array.flatten())


        def update(i): 
            for j, model_name in enumerate(models): 
                for k, task in enumerate(tasks):
                    embedding_data = model_task_state_dict[model_name][task]
                    if dim ==3: 
                        plot_array[j][k].set_data(embedding_data[0:i, 0], embedding_data[0:i, 1])
                        plot_array[j][k].set_3d_properties(embedding_data[0:i, 2])
                    else: 
                        data_array[j][k][0].append(embedding_data[i, 0])
                        data_array[j][k][1].append(embedding_data[i, 1])
                        plot_array[j][k].set_data(data_array[j][k][0], data_array[j][k][1])
            return tuple(plot_array.flatten())



        ani = animation.FuncAnimation(fig, update, frames=119,
                            init_func=init, blit=True)

        Patches, _ = self.get_model_patches()

        for i, task in enumerate(tasks):
            Patches.append(Line2D([0], [0], linestyle=style_list[i], color='grey', label=task, markerfacecolor='grey', markersize=10))
        plt.legend(handles=Patches)
        plt.show()

        ax.clear()
