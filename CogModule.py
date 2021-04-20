import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
import umap
from sklearn.manifold import TSNE

import pickle
import random
from collections import defaultdict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from Task import Task, construct_batch
from LangModule import get_batch, toNumerals, swaps
from RNNs import simpleNet

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
    nn_out = gpu_to_np(nn_out)
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

def mask_input_rule(in_tensor): 
    """Masks the one-hot rule information in an input batch 
    Args:      
        in_tensor (Tensor): input tensor for a batch of trials; shape: (batch_size, seq_len, features) 
    
    Returns:
        Tensor: identical to in_tensor with one-hot rule information zero'd out; shape: (batch_size, seq_len, features) 
    """

    mask = torch.zeros((in_tensor.shape[0], in_tensor.shape[1], len(task_list))).to(device)
    masked_input = torch.cat((in_tensor[:, :, 0:1], mask, in_tensor[:, :, len(task_list)+1:]), axis=2)
    return masked_input

def del_input_rule(in_tensor): 
    """Deletes the one-hot rule information in an input batch 
    Args:      
        in_tensor (Tensor): input tensor for a batch of trials; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: identical input data as in_tensor with one-hot rule information deleted; shape: (batch_num, seq_len, features-#tasks) 
    """

    new_input = torch.cat((in_tensor[:, :, 0:1], in_tensor[:, :, len(task_list)+1:]), axis=2)
    return new_input

def comp_input_rule(in_tensor, task_type): 
    """Replaces one-hot input rule with a analagous linear combination of rules for related tasks
    Args:      
        in_tensor (Tensor): input tensor for a batch of trials; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: identical input data as in_tensor with combinational input rule; shape: (batch_num, seq_len, features-#tasks) 
    """

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
    
    comp_tensor = torch.Tensor(comp_vec).unsqueeze(0).repeat(in_tensor.shape[0], in_tensor.shape[1], 1).to(in_tensor.get_device())
    comp_input = torch.cat((in_tensor[:, :, 0:1], comp_tensor, in_tensor[:, :, len(task_list)+1:]), axis=2)
    return comp_input

def use_shuffled_one_hot(in_tensor, task_type): 
    shuffled_one_hot = torch.Tensor(Task._rule_one_hot(task_type, shuffled=True)).unsqueeze(0).repeat(in_tensor.shape[0], in_tensor.shape[1], 1).to(in_tensor.get_device())
    shuffled_one_hot_input = torch.cat((in_tensor[:, :, 0:1], shuffled_one_hot, in_tensor[:, :, len(task_list)+1:]), axis=2)
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

def strip_model_name(model_name): 
    try:
        stripped_name = model_name[:model_name.index('_seed')]
    except: 
        stripped_name = model_name
    return stripped_name

class CogModule():
    ALL_STYLE_DICT = {'Model1': ('blue', None), 'Model1shuffled': ('blue', '+'), 'SIF':('brown', None), 'BoW': ('orange', None), 'GPT_cat': ('red', '^'), 'GPT train': ('red', '.'), 
                            'BERT_cat': ('green', '^'), 'BERT train': ('green', '+'), 'S-Bert_cat': ('purple', '^'), 'S-Bert train': ('purple', '.'), 'S-Bert' : ('purple', None), 
                            'InferSent train': ('yellow', '.'), 'InferSent_cat': ('yellow', '^'), 'Transformer': ('pink', '.')}
    COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'GPT': 'red', 'BERT': 'green', 'S-Bert': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
    MODEL_MARKER_DICT = {'SIF':None, 'BoW':None, 'shuffled':'+', 'cat': '^', 'train': '.', 'Transformer':'.'}
    MARKER_DICT = {'^': 'task categorization', '.': 'end-to-end', '+':'shuffled'}
    NAME_TO_PLOT_DICT = {'Model1': 'One-Hot Task Vec.','Model1shuffled': 'Shuffled One-Hot', 'SIF':'SIF', 'BoW': 'BoW', 'GPT_cat': 'GPT (task cat.)', 'GPT train': 'GPT (end-to-end)', 
                            'BERT_cat': 'BERT (task cat.)', 'BERT train': 'BERT (end-to-end)', 'S-Bert_cat': 'S-BERT (task cat.)', 'S-Bert train': 'S-BERT (end-to-end)',  
                            'S-Bert': 'S-BERT (raw)', 'InferSent train': 'InferSent (end-to-end)', 'InferSent_cat': 'InferSent (task cat.)', 'Transformer': 'Transformer (end-to-end)'}

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
            print(strip_model_name(model_name))
            if strip_model_name(model_name) in ['Model1', 'BoW', 'SIF', 'S-Bert']: 
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
        try:
            self.load_training_data(holdout_task, foldername, holdout_task)
        except: 
            pass
        for model_name, model in self.model_dict.items():
            filename = foldername+'/'+holdout_task+'/'+holdout_task+'_'+model_name+'.pt'
            filename = filename.replace(' ', '_')
            model.load_state_dict(torch.load(filename))

    def _get_lang_input(self, model, batch_len, task_type, instruct_mode): 
        tokenizer = model.langModel.tokenizer
        batch_instruct, _ = get_batch(batch_len, tokenizer, task_type=task_type, instruct_mode=instruct_mode)
        return batch_instruct

    def _get_model_resp(self, model, batch_len, ins, task_type, instruct_mode): 
        h0 = model.initHidden(batch_len, 0.1).to(device)
        if model.isLang: 
            if instruct_mode == 'masked': 
                masked = torch.zeros(ins.shape[0], ins.shape[1], model.langModel.out_dim).to(device)
                ins = torch.cat((del_input_rule(ins), masked), dim=2)
                out, hid = model.rnn(ins, h0)
            else: 
                ins = del_input_rule(ins)
                ####ISSUE HERE - how to properly pass instruct_mode
                #instruct = self._get_lang_input(model, batch_len, task_type, model.instruct_mode)
                instruct = self._get_lang_input(model, batch_len, task_type, instruct_mode)
                #print(instruct)
                out, hid = model(instruct, ins, h0)
        else: 
            instruct = None
            if instruct_mode == 'masked': 
                ins = mask_input_rule(ins).to(device)
            if instruct_mode == 'comp': 
                ins = comp_input_rule(ins, task_type)
            if instruct_mode == 'instruct_swap': 
                ins = swap_input_rule(ins, task_type)
            if model.instruct_mode == 'shuffled_one_hot':
                ins = use_shuffled_one_hot(ins, task_type)

            out, hid = model(ins, h0)
        return out, hid, instruct

    def train(self, data, epochs, scheduler = True, weight_decay = 0.0, lr = 0.001, milestones = [], 
                        holdout_task = None, instruct_mode = None, freeze_langModel = False, langLR = None, langWeightDecay=None): 
        #torch.autograd.set_detect_anomaly
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
        
        ins_tensor = data[0]
        tar_tensor = data[1]
        mask_tensor = data[2]
        tar_dir_vec = data[3]
        task_type_vec = data[4]

        batch_len = ins_tensor.shape[1]
        batch_num = ins_tensor.shape[0]
        correct_array = np.empty((batch_len, batch_num), dtype=bool)
        for model_type, model in self.model_dict.items(): 
            opt = self.opt_dict[model_type][0]
            opt_scheduler = self.opt_dict[model_type][1]
            for i in range(epochs):
                print('epoch', i)
                index_list = list(np.arange(batch_num))
                np.random.shuffle(index_list)
                for j in range(batch_num): 
                    index = index_list[j]
                    task_type = task_type_vec[index]
                    task_index = task_list.index(task_type)
                    tar = torch.Tensor(tar_tensor[index, :, :, :]).to(device)
                    mask = torch.Tensor(mask_tensor[index, :, :, :]).to(device)
                    ins = torch.Tensor(ins_tensor[index, :, :, :]).to(device)
                    tar_dir = tar_dir_vec[index]

                    opt.zero_grad()
                    out, _, _ = self._get_model_resp(model, batch_len, ins, task_type, instruct_mode)

                    loss = masked_MSE_Loss(out, tar, mask) 
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



    def plot_learning_curve(self, mode, task_type=None, smoothing = 2):
        assert mode in ['loss', 'correct'], "mode must be 'loss' or 'correct', entered: %r" %mode
        if mode == 'loss': 
            y_label = 'MSE Loss'
            y_lim = (0, 0.05)
            title = 'MSE Loss over training'
            perf_dict = self.task_sorted_loss
        if mode == 'correct': 
            y_label = 'Fraction Correct'
            y_lim = (-0.05, 1.15)
            title = 'Fraction Correct Response over training'
            perf_dict = self.task_sorted_correct

        if task_type is None: 
            fig, axn = plt.subplots(4,4, sharey = True, figsize=(16, 18))
            plt.suptitle(title)
            for i, ax in enumerate(axn.flat):
                ax.set_ylim(y_lim)

                cur_task = Task.TASK_LIST[i]
                for model_name in self.model_dict.keys():
                    smoothed_perf = gaussian_filter1d(perf_dict[model_name][cur_task], sigma=smoothing)
                    ax.plot(smoothed_perf, color = self.ALL_STYLE_DICT[strip_model_name(model_name)][0], marker = self.ALL_STYLE_DICT[strip_model_name(model_name)][1], linewidth= 1.0, markersize = 5, markevery=20)
                ax.set_title(cur_task)
        else:
            fig, ax = plt.subplots(1,1)
            plt.suptitle(title)
            ax.set_ylim(y_lim)
            for model_name in self.model_dict.keys():    
                smoothed_perf = gaussian_filter1d(perf_dict[model_name][task_type], sigma=smoothing)
                ax.plot(smoothed_perf, color = self.ALL_STYLE_DICT[strip_model_name(model_name)][0], marker = self.ALL_STYLE_DICT[strip_model_name(model_name)][1], markersize = 5, markevery=3)
            ax.set_title(task_type + ' holdout')

        Patches, Markers = self.get_model_patches()
        arch_legend = plt.figlegend(handles=Patches+Markers, title = r"$\textbf{Language Module}$", loc='center right')

        fig.text(0.5, 0.04, 'Training Examples', ha='center')
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
        fig.show()

    def _get_performance(self, model, instruct_mode, num_batches): 
        model.eval()
        batch_len = 128
        with torch.no_grad():
            perf_dict = dict.fromkeys(task_list)
            for task_type in task_list:
                for _ in range(num_batches): 
                    mean_list = [] 
                    trial = construct_batch(task_type, batch_len)
                    ins = trial.inputs
                    out, _, _ = self._get_model_resp(model, batch_len, torch.Tensor(ins).to(device), task_type, instruct_mode) 
                    mean_list.append(np.mean(isCorrect(out, torch.Tensor(trial.targets).to(device), trial.target_dirs)))
                perf_dict[task_type] = np.mean(mean_list)
        return perf_dict 

    def _plot_trained_performance(self, instruct_mode = None):
        barWidth = 0.1
        for i, model in enumerate(self.model_dict.values()):  
            perf_dict = self._get_performance(model, instruct_mode, 3)
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
                if model.embedderStr is not 'SBERT': 
                    instruct = toNumerals(model.embedderStr, instruct)
                else: 
                    instruct = [instruct]
                ins = del_input_rule(torch.Tensor(ins)).to(device)
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

    def plot_task_rep(self, model_name, epoch, reduction_method = 'PCA', num_trials = 250, dim = 2, instruct_mode = None, holdout_task = None, tasks = task_list, avg_rep = False, Title=''): 
        if instruct_mode == 'comp': 
            assert holdout_task != None 
                
        model = self.model_dict[model_name]
        # if not next(model.rnn.parameters()).is_cuda:
        #     model.to(device)

        assert epoch in ['input', 'stim', 'response', 'prep'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
        
        task_reps = []
        for task in task_list: 
            trials = construct_batch(task, num_trials)
            tar = trials.targets
            ins = trials.inputs

            if instruct_mode == 'comp': 
                if task == holdout_task: 
                    out, hid = self._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task, 'comp')
                else: 
                    out, hid = self._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task, None)
            else: 
                out, hid = self._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task, instruct_mode)


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
                    epoch_index = np.where(ins[i, :, 18:]>0.25)[0][0]-1
                    epoch_state.append(hid[i, epoch_index, :])
            
            if avg_rep: 
                epoch_state = [np.mean(np.stack(epoch_state), axis=0)]
            
            task_reps += epoch_state


        if reduction_method == 'PCA': 
            embedder = PCA(n_components=dim)
        elif reduction_method == 'UMAP':
            embedder = umap.UMAP()
        elif reduction_method == 'tSNE': 
            embedder = TSNE(n_components=2)

        embedded = embedder.fit_transform(task_reps)
        if reduction_method == 'PCA': 
            explained_variance = embedder.explained_variance_ratio_
        else: 
            explained_variance = None

        if avg_rep: 
            to_plot = np.stack([embedded[task_list.index(task), :] for task in tasks])
            task_indices = np.array([task_list.index(task) for task in tasks]).astype(int)
            marker_size = 100
        else: 
            to_plot = np.stack([embedded[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials, :] for task in tasks]).reshape(len(tasks)*num_trials, dim)
            task_indices = np.array([[task_list.index(task)]*num_trials for task in tasks]).astype(int).flatten()
            marker_size = 25

        cmap = matplotlib.cm.get_cmap('tab20')
        Patches = []
        if dim ==3: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = [to_plot[:, 0], to_plot[:, 1], to_plot[:,2], cmap(task_indices), cmap, marker_size]
            ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:,2], c = cmap(task_indices), cmap=cmap, s=marker_size)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')

        else:             
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = [to_plot[:, 0], to_plot[:, 1], cmap(task_indices), cmap, marker_size]
            # for i, color in enumerate(['Red', 'Blue', 'Green', 'Yellow']): 
            #     start = i*num_trials
            #     stop = start+num_trials
            #     plt.scatter(to_plot[:, 0][start:stop], to_plot[:, 1][start:stop], color=listset(task_indices), s=marker_size)
            #     Patches.append(mpatches.Patch(color = cmap(task_indices), label = task_list[list(set(task_indices))[i]]))
            ax.scatter(to_plot[:, 0], to_plot[:, 1], c = cmap(task_indices), cmap=cmap, s=marker_size)
            plt.xlabel("PC 1", fontsize = 18)
            plt.ylabel("PC 2", fontsize = 18)

        #plt.suptitle(r"$\textbf{PCA Embedding for Task Representation$", fontsize=18)
        plt.title(Title)
        digits = np.arange(len(tasks))
        plt.tight_layout()
        Patches = [mpatches.Patch(color=cmap(d), label=task_list[d]) for d in set(task_indices)]
        scatter.append(Patches)
        plt.legend(handles=Patches)
        plt.show()


        return explained_variance, scatter

    def get_hid_traj(self, models, tasks, dim, instruct_mode):
        models = [self.model_dict[model] for model in models]
        for model in models: 
            if not next(model.parameters()).is_cuda:
                model.to(device)
 
        task_info_list = []
        for task in tasks: 
            trial = construct_batch(task, 1)
            task_info_list.append(trial.inputs)

        model_task_state_dict = {}
        for model_name, model in self.model_dict.items(): 
            tasks_dict = {}
            for i, task in enumerate(tasks): 
                out, hid = self._get_model_resp(model, 1, torch.Tensor(task_info_list[i]).to(device), task, instruct_mode)
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



