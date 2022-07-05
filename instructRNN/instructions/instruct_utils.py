import numpy as np
import pickle
import itertools
import torch
from collections import Counter
import os

from instructRNN.tasks.tasks import *

def inv_instruct_dict(instruct_dict):
    inv_dict = {}
    for task, instructs in instruct_dict.items(): 
        for instruct in instructs: 
            inv_dict[instruct] = task
    return inv_dict

try:
    INSTRUCT_PATH = os.environ['MODEL_FOLDER']+'/instructs/'
except KeyError:
    INSTRUCT_PATH = '6.20models/instructs/'
    
train_instruct_dict = pickle.load(open(INSTRUCT_PATH+'train_instruct_dict', 'rb'))
test_instruct_dict = pickle.load(open(INSTRUCT_PATH+'test_instruct_dict', 'rb'))
inv_train_instruct_dict = inv_instruct_dict(train_instruct_dict)
 
 
def get_all_sentences():
    combined_instruct= {task: list(train_instruct_dict[task]) for task in TASK_LIST}
    all_sentences = list(itertools.chain.from_iterable(combined_instruct.values()))
    return all_sentences

def sort_vocab(): 
    #combined_instruct= {key: list(train_instruct_dict[key]) + list(test_instruct_dict[key]) for key in train_instruct_dict}
    all_sentences = get_all_sentences()
    sorted_vocab = sorted(list(set(' '.join(all_sentences).split(' '))))
    return sorted_vocab

def count_vocab(): 
    combined_instruct= {key: list(train_instruct_dict[key]) for key in train_instruct_dict}
    all_sentences = list(itertools.chain.from_iterable(combined_instruct.values()))
    counts = Counter(list(itertools.chain.from_iterable([sentence.split() for sentence in all_sentences])))
    return counts, sorted(counts.keys())

def shuffle_instruction(instruct): 
    instruct = instruct.split()
    shuffled = np.random.permutation(instruct)
    instruct = ' '.join(list(shuffled))
    return instruct

def get_swap_task(task):
    swap_label = INV_SWAPS_DICT[task]
    pos = SWAPS_DICT[swap_label].index(task)
    swap_index = (pos+1)%len(SWAPS_DICT[swap_label])
    return SWAPS_DICT[swap_label][swap_index]

def get_instruction_dict(instruct_mode): 
    assert instruct_mode in [None, 'swap', 'validation']
    if instruct_mode == 'swap': 
        swap_dict = {}
        for task in TASK_LIST: 
            swap_dict[task] = train_instruct_dict[get_swap_task(task)]

    elif instruct_mode == 'validation': 
        return test_instruct_dict

    else: 
        return train_instruct_dict

def get_instructions(batch_size, task_type, instruct_mode=None):
        instruct_dict = get_instruction_dict(instruct_mode = instruct_mode)
        instructs = np.random.choice(instruct_dict[task_type], size=batch_size)
        if instruct_mode == 'shuffled': 
            instructs = list(map(shuffle_instruction, instructs))

        return list(instructs)

def make_one_hot(size, index): 
    one_hot = np.zeros((1, size))
    one_hot[:,index] = 1
    return one_hot

def one_hot_input_rule(batch_size, task_type, shuffled=False): 
    if shuffled: index = Task.SHUFFLED_TASK_LIST.index(task_type) 
    else: index = TASK_LIST.index(task_type)
    one_hot = make_one_hot(len(TASK_LIST), index)
    one_hot= np.repeat(one_hot, batch_size, axis=0)
    return one_hot

def get_comp_rep(batch_size, task_type): 
    ref_tasks = construct_trials(task_type, None).comp_ref_tasks
    comp_rep = one_hot_input_rule(batch_size, ref_tasks[0])-one_hot_input_rule(batch_size, ref_tasks[1]) \
                        +one_hot_input_rule(batch_size, ref_tasks[2])
    return comp_rep

def get_comp_rule(batch_size, task_type, instruct_mode=None): 
    if instruct_mode == 'swap': 
        swapped_task = get_swap_task(task_type)
        task_rule = get_comp_rep(batch_size, swapped_task)
    elif instruct_mode == 'masked': 
        task_rule = np.zeros((batch_size, 11))
    else: 
        task_rule = get_comp_rep(batch_size, task_type)
    
    return torch.Tensor(task_rule)

def get_input_rule(batch_size, task_type, instruct_mode=None): 
    if instruct_mode == 'swap': 
        swapped_task = get_swap_task(task_type)
        task_rule = one_hot_input_rule(batch_size, swapped_task)
    elif instruct_mode == 'masked': 
        task_rule = np.zeros((batch_size, len(TASK_LIST)))
    else: 
        task_rule = one_hot_input_rule(batch_size, task_type)
    
    return torch.Tensor(task_rule)

def get_task_info(batch_len, task_type, info_type, instruct_mode=None): 
    if info_type=='lang': 
        return get_instructions(batch_len, task_type, instruct_mode = instruct_mode)
    elif info_type=='comp': 
        return get_comp_rule(batch_len, task_type, instruct_mode = instruct_mode)
    else: 
        return get_input_rule(batch_len, task_type, instruct_mode = instruct_mode)


