from collections import defaultdict

from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

import torch
import torch.multiprocessing as mp
import itertools


from task import Task
task_list = Task.TASK_LIST
tuning_dirs = torch.Tensor(Task.TUNING_DIRS)


train_instruct_dict = pickle.load(open('Instructions/train_instruct_dict2', 'rb'))
test_instruct_dict = pickle.load(open('Instructions/test_instruct_dict2', 'rb'))


swapped_task_list = ['Anti DM', 'COMP2', 'Anti Go', 'DMC', 'DM', 'Go', 'MultiDM', 'Anti MultiDM', 'COMP1',
                             'RT Go', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'Anti RT Go', 'DNMC']

task_group_colors = defaultdict(dict)
task_group_colors['Go'] = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange'}
task_group_colors['Decision Making'] = { 'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow'}
task_group_colors['Comparison'] = { 'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold'}
task_group_colors['Delay'] = { 'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

ALL_STYLE_DICT = {'Model1': ('blue', None), 'Model1shuffled': ('blue', '+'), 'SIF':('brown', None), 'BoW': ('orange', None), 'GPT_cat': ('red', '^'), 'GPT train': ('red', '.'), 
                        'BERT_cat': ('green', '^'), 'BERT train': ('green', '.'), 'S-Bert_cat': ('purple', '^'), 'S-Bert train': ('purple', '.'), 'S-Bert' : ('purple', None), 
                        'InferSent train': ('yellow', '.'), 'InferSent_cat': ('yellow', '^'), 'Transformer': ('pink', '.')}
COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'GPT': 'red', 'BERT': 'green', 'S-Bert': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
MODEL_MARKER_DICT = {'SIF':None, 'BoW':None, 'shuffled':'+', 'cat': '^', 'train': '.', 'Transformer':'.'}
MARKER_DICT = {'^': 'task categorization', '.': 'end-to-end', '+':'shuffled'}
NAME_TO_PLOT_DICT = {'Model1': 'One-Hot Task Vec.','Model1shuffled': 'Shuffled One-Hot', 'SIF':'SIF', 'BoW': 'BoW', 'GPT_cat': 'GPT (task cat.)', 'GPT train': 'GPT (end-to-end)', 
                        'BERT_cat': 'BERT (task cat.)', 'BERT train': 'BERT (end-to-end)', 'S-Bert_cat': 'S-BERT (task cat.)', 'S-Bert train': 'S-BERT (end-to-end)',  
                        'S-Bert': 'S-BERT (raw)', 'InferSent train': 'InferSent (end-to-end)', 'InferSent_cat': 'InferSent (task cat.)', 'Transformer': 'Transformer (end-to-end)'}


swaps= [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['COMP2', 'RT Go']]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def popvec(act_vec, tuning_dirs):
    """Population vector decoder that reads the orientation of activity in vector of activities
    Args:      
        act_vec (np.array): output tensor of neural network model; shape: (features, 1)

    Returns:
        float: decoded orientation of activity (in radians)
    """

    act_sum = torch.sum(act_vec)
    temp_cos = torch.sum(torch.multiply(act_vec, torch.cos(tuning_dirs)))/act_sum
    temp_sin = torch.sum(torch.multiply(act_vec, torch.sin(tuning_dirs)))/act_sum
    loc = torch.atan2(temp_sin, temp_cos)
    return loc % 2*np.pi

def get_dist(angle1, angle2):
    """Returns the true distance between two angles mod 2pi
    Args:      
        angle1, angle2 (float): angles in radians

    Returns:
        float: distance between given angles mod 2pi
    """
    dist = angle1-angle2
    return torch.minimum(abs(dist),2*np.pi-abs(dist))


def isCorrect(nn_out, nn_target, target_dirs, tuning_dirs): 
    """Determines whether a given neural network response is correct, computed by batch
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_size, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_size, seq_len, features)
        target_dirs (np.array): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        np.array: weighted loss of neural network response; shape: (batch)
    """
    batch_size = nn_out.shape[0]

    isCorrect = torch.empty(batch_size)
    criterion = (2*np.pi)/10

    for i in range(batch_size):
        #checks response maintains fixataion
        isFixed = all(torch.where(nn_target[i, :, 0] == 0.85, nn_out[i, :, 0] > 0.5, True))

        #checks trials that requiring repressing responses
        if np.isnan(target_dirs[i]): 
            isDir = all((nn_out[i, 114:119, :].flatten() < 0.2))
        
        #checks responses are coherent and in the correct direction
        else:
            is_response = torch.max(nn_out[i, -1, 1:]) > 0.6
            loc = popvec(nn_out[i, -1, 1:], tuning_dirs)
            dist = get_dist(loc, target_dirs[i])        
            isDir = dist < criterion and is_response
        isCorrect[i] = isDir and isFixed

    return isCorrect






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
        assert instruct_mode in [None, 'instruct_swap', 'shuffled', 'validation']

        if instruct_mode == 'instruct_swap': 
            instruct_dict = dict(zip(swapped_task_list, train_instruct_dict.values()))
        elif instruct_mode == 'validation': 
            instruct_dict = test_instruct_dict
        else: 
            instruct_dict = train_instruct_dict

        instructs = np.random.choice(instruct_dict[task_type], size=batch_size)
        if instruct_mode == 'shuffled': 
            instructs = list(map(shuffle_instruction, instructs))

        return list(instructs)




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
    assert task_type in ['Go', 'Anti DM', 'Anti RT Go', 'DMC', 'RT Go', 'COMP2']
    swapped_index = swapped_task_list.index(task_type)
    swapped_one_hot = one_hot_input_rule(batch_size, task_list[swapped_index])
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







def task_cmap(array): 
    all_task_dict = {}
    for task_colors in task_group_colors.values(): 
        all_task_dict.update(task_colors)
    color_list = []

    for index in array: 
        color_list.append(all_task_dict[task_list[index]])

    return color_list


def strip_model_name(model_name): 
    try:
        stripped_name = model_name[:model_name.index('_seed')]
    except: 
        stripped_name = model_name
    return stripped_name

def get_model_patches(model_list): 
    Patches = []
    Markers = []
    color_dict = COLOR_DICT.copy()
    for model_name in model_list: 
        architecture_type = list(COLOR_DICT.keys())[np.where([model_name.startswith(key) for key in COLOR_DICT.keys()])[0][0]]
        try:
            color = color_dict.pop(architecture_type)
        except:
            continue
        if architecture_type == 'Model1': architecture_type = 'One-Hot Vec.'
        patch = mpatches.Patch(color=color, label=architecture_type)
        Patches.append(patch)

    for model_name in model_list: 
        print(strip_model_name(model_name))
        if strip_model_name(model_name) in ['Model1', 'BoW', 'SIF', 'S-Bert']: 
            continue
        where_array = np.array([model_name.find(key) for key in MODEL_MARKER_DICT.keys()])
        marker = MODEL_MARKER_DICT[list(MODEL_MARKER_DICT.keys())[np.where(where_array >= 0)[0][0]]]
        if any([marker == m.get_marker() for m in Markers]): 
            continue
        mark = Line2D([0], [0], marker=marker, color='w', label=MARKER_DICT[marker], markerfacecolor='grey', markersize=10)
        Markers.append(mark)

    return Patches, Markers

def _label_plot(fig, Patches, Markers, legend_loc = (0.9, 0.3)): 
    arch_legend = plt.legend(handles=Patches, title = r"$\textbf{Language Module}$", bbox_to_anchor = legend_loc, loc = 'lower center')
    ax = plt.gca().add_artist(arch_legend)
    plt.legend(handles= Markers, title = r"$\textbf{Transformer Fine-Tuning}$", bbox_to_anchor = legend_loc, loc = 'upper center')
    fig.text(0.5, 0.04, 'Training Examples', ha='center')
    fig.text(0.04, 0.5, 'Fraction Correct', va='center', rotation='vertical')

