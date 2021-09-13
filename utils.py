from functools import reduce
import numpy as np
import pickle
from numpy.linalg import norm

import torch
import itertools

from task import Task
task_list = Task.TASK_LIST
swapped_task_list = Task.SWAPPED_TASK_LIST
tuning_dirs = Task.TUNING_DIRS

train_instruct_dict = pickle.load(open('Instructions/train_instruct_dict2', 'rb'))
test_instruct_dict = pickle.load(open('Instructions/test_instruct_dict2', 'rb'))


task_swaps_map = {'Go': 'Go_Anti_DM', 
                'Anti Go': 'Anti_Go_MultiCOMP1', 
                'RT Go': 'RT_Go_DNMC', 
                'Anti RT Go': 'Anti_RT_Go_DMC',
                'DM': 'DM_MultiCOMP2',
                'Anti DM': 'Go_Anti_DM',
                'MultiDM': 'MultiDM_DNMS',
                'Anti MultiDM': 'Anti_MultiDM_COMP1',
                'COMP1': 'Anti_MultiDM_COMP1', 
                'COMP2': 'COMP2_DMS',
                'MultiCOMP1': 'Anti_Go_MultiCOMP1',
                'MultiCOMP2': 'DM_MultiCOMP2',
                'DMS': 'COMP2_DMS', 
                'DNMS': 'MultiDM_DNMS', 
                'DMC': 'Anti_RT_Go_DMC', 
                'DNMC': 'RT_Go_DNMC'}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

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
    act_sum = np.sum(act_vec, axis=1)
    temp_cos = np.sum(np.multiply(act_vec, np.cos(tuning_dirs)), axis=1)/act_sum
    temp_sin = np.sum(np.multiply(act_vec, np.sin(tuning_dirs)), axis=1)/act_sum
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

    criterion = (2*np.pi)/10

    #checks response maintains fixataion
    isFixed = np.all(np.where(nn_target[:, :, 0] == 0.85, nn_out[:, :, 0] > 0.5, True), axis=1)

    #checks for repressed responses
    isRepressed = np.all(nn_out[:, 114:119, :].reshape(batch_size, -1) < 0.15, axis = 1)

    #checks is responses are in the correct direction 
    is_response = np.max(nn_out[:, -1, 1:]) > 0.6
    loc = popvec(nn_out[:, -1, 1:])
    dist = get_dist(loc, np.nan_to_num(target_dirs))        
    isDir = np.logical_and(dist < criterion, is_response)

    #checks if responses were correctly repressed or produced 
    correct_reponse = np.where(np.isnan(target_dirs), isRepressed, isDir)

    #checks if correct responses also maintained fixation 
    is_correct = np.logical_and(correct_reponse, isFixed)
    return is_correct

def shuffle_instruction(instruct): 
    instruct = instruct.split()
    shuffled = np.random.permutation(instruct)
    instruct = ' '.join(list(shuffled))
    return instruct

def sort_vocab(): 
    combined_instruct= {key: list(train_instruct_dict[key]) + list(test_instruct_dict[key]) for key in train_instruct_dict}
    all_sentences = list(itertools.chain.from_iterable(combined_instruct.values()))
    sorted_vocab = sorted(list(set(' '.join(all_sentences).split(' '))))
    return sorted_vocab

def get_instructions(batch_size, task_type, instruct_mode):
        assert instruct_mode in ['', 'swap', 'shuffled', 'validation']

        if instruct_mode == 'swap': 
            instruct_dict = dict(zip(swapped_task_list, train_instruct_dict.values()))
        elif instruct_mode == 'validation': 
            instruct_dict = test_instruct_dict
        else: 
            instruct_dict = train_instruct_dict

        instructs = np.random.choice(instruct_dict[task_type], size=batch_size)
        if instruct_mode == 'shuffled': 
            instructs = list(map(shuffle_instruction, instructs))

        return list(instructs)

def two_line_instruct(instruct): 
    split_instruct = instruct.split()
    split_instruct.insert(int(np.ceil(len(split_instruct)/2)), '\n')
    return ' '.join(split_instruct)


def one_hot_input_rule(batch_size, task_type, shuffled=False): 
    if shuffled: index = Task.SHUFFLED_TASK_LIST.index(task_type) 
    else: index = Task.TASK_LIST.index(task_type)
    one_hot = np.zeros((1, len(Task.TASK_LIST)))
    one_hot[:,index] = 1
    one_hot= np.repeat(one_hot, batch_size, axis=0)
    return one_hot

def mask_input_rule(batch_size, lang_dim=None): 
    if lang_dim is not None: 
        mask = np.zeros((batch_size, lang_dim))
    else: 
        mask = np.zeros((batch_size, len(task_list)))
    return mask

def comp_input_rule(batch_size, task_type): 
    if task_type == 'Go': 
        comp_vec = one_hot_input_rule(batch_size, 'RT Go')+(one_hot_input_rule(batch_size, 'Anti Go')-one_hot_input_rule(batch_size, 'Anti RT Go'))
    if task_type == 'RT Go':
        comp_vec = one_hot_input_rule(batch_size, 'Go')+(one_hot_input_rule(batch_size, 'Anti RT Go')-one_hot_input_rule(batch_size, 'Anti Go'))
    if task_type == 'Anti Go':
        comp_vec = one_hot_input_rule(batch_size, 'Anti RT Go')+(one_hot_input_rule(batch_size, 'Go')-one_hot_input_rule(batch_size, 'RT Go'))
    if task_type == 'Anti RT Go':
        comp_vec = one_hot_input_rule(batch_size, 'Anti Go')+(one_hot_input_rule(batch_size, 'RT Go')-one_hot_input_rule(batch_size, 'Go'))
    if task_type == 'DM':
        comp_vec = one_hot_input_rule(batch_size, 'MultiDM') + (one_hot_input_rule(batch_size, 'Anti DM') - one_hot_input_rule(batch_size, 'Anti MultiDM'))
    if task_type == 'Anti DM': 
        comp_vec = one_hot_input_rule(batch_size, 'Anti MultiDM') + (one_hot_input_rule(batch_size, 'DM') - one_hot_input_rule(batch_size, 'MultiDM'))
    if task_type == 'MultiDM': 
        comp_vec = one_hot_input_rule(batch_size, 'DM') + (one_hot_input_rule(batch_size, 'Anti MultiDM')-one_hot_input_rule(batch_size, 'Anti DM'))
    if task_type == 'Anti MultiDM': 
        comp_vec = one_hot_input_rule(batch_size, 'Anti DM') + (one_hot_input_rule(batch_size, 'MultiDM')-one_hot_input_rule(batch_size, 'DM'))
    if task_type == 'COMP1': 
        comp_vec = one_hot_input_rule(batch_size, 'COMP2') + (one_hot_input_rule(batch_size, 'MultiCOMP1')-one_hot_input_rule(batch_size, 'MultiCOMP2'))
    if task_type == 'COMP2': 
        comp_vec = one_hot_input_rule(batch_size, 'COMP1') + (one_hot_input_rule(batch_size, 'MultiCOMP2')-one_hot_input_rule(batch_size, 'MultiCOMP1'))
    if task_type == 'MultiCOMP1': 
        comp_vec = one_hot_input_rule(batch_size, 'MultiCOMP2') + (one_hot_input_rule(batch_size, 'COMP1')-one_hot_input_rule(batch_size, 'COMP2'))
    if task_type == 'MultiCOMP2': 
        comp_vec = one_hot_input_rule(batch_size, 'MultiCOMP1') + (one_hot_input_rule(batch_size, 'COMP2')-one_hot_input_rule(batch_size, 'COMP1'))
    if task_type == 'DMS': 
        comp_vec = one_hot_input_rule(batch_size, 'DMC') + (one_hot_input_rule(batch_size, 'DNMS')-one_hot_input_rule(batch_size, 'DNMC'))
    if task_type == 'DMC': 
        comp_vec = one_hot_input_rule(batch_size, 'DMS') + (one_hot_input_rule(batch_size, 'DNMC')-one_hot_input_rule(batch_size, 'DNMS'))
    if task_type == 'DNMS': 
        comp_vec = one_hot_input_rule(batch_size, 'DNMC') + (one_hot_input_rule(batch_size, 'DMS')-one_hot_input_rule(batch_size, 'DMC'))
    if task_type == 'DNMC': 
        comp_vec = one_hot_input_rule(batch_size, 'DNMS') + (one_hot_input_rule(batch_size, 'DMC')-one_hot_input_rule(batch_size, 'DMS'))

    return comp_vec

def swap_input_rule(batch_size, task_type): 
    """Swaps one-hot rule inputs for given tasks 
    'Go' <--> 'Anti DM' 
    'Anti RT Go' <--> 'DMC'
    'RT Go' <--> 'COMP2'
    """
    index = Task.TASK_LIST.index(task_type)
    swapped_one_hot = one_hot_input_rule(batch_size, swapped_task_list[index])
    return swapped_one_hot

def get_input_rule(batch_size, task_type, instruct_mode, lang_dim = None): 
    if instruct_mode == 'swap': 
        task_rule = swap_input_rule(batch_size, task_type)
    elif instruct_mode == 'comp': 
        task_rule = comp_input_rule(batch_size, task_type)
    elif instruct_mode == 'masked': 
        task_rule = mask_input_rule(batch_size, lang_dim)
    elif instruct_mode == ' shuffled': 
        task_rule = one_hot_input_rule(batch_size, task_type, shuffled=True)
    else: 
        task_rule = one_hot_input_rule(batch_size, task_type)
    
    return torch.Tensor(task_rule)

