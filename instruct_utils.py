import numpy as np
import pickle
import itertools
from tasks import Task
import torch
from tasks import TASK_LIST
from tasks_utils import get_swap_task
import os 

from collections import Counter
task_list = TASK_LIST

#cur_folder = os.getenv('MODEL_FOLDER')

train_instruct_dict = pickle.load(open('6.4models/train_instruct_dict', 'rb'))

test_instruct_dict = pickle.load(open('Instructions/test_instruct_dict', 'rb'))

inv_train_instruct_dict = inv_train_instruct_dict = dict(zip(list(itertools.chain(*[list(instructions) for instructions in train_instruct_dict.values()])), 
                                            list(itertools.chain(*[[task]*15 for task in TASK_LIST]))))

def sort_vocab(): 
    #combined_instruct= {key: list(train_instruct_dict[key]) + list(test_instruct_dict[key]) for key in train_instruct_dict}
    combined_instruct= {key: list(train_instruct_dict[key]) for key in train_instruct_dict}
    all_sentences = list(itertools.chain.from_iterable(combined_instruct.values()))
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

def get_instruction_dict(instruct_mode): 
    assert instruct_mode in [None, 'swap', 'shuffled', 'validation']
    if instruct_mode == 'swap': 
        swap_dict = {}
        for task in TASK_LIST: 
            swap_dict[task] = train_instruct_dict[get_swap_task(task)]

    elif instruct_mode == 'shuffled': 
        shuffle_dict = {}
        for task in TASK_LIST: 
            shuffled_instructs = [shuffle_instruction(instruct) for instruct in train_instruct_dict[task]]
            shuffle_dict[task] = shuffled_instructs

    elif instruct_mode == 'validation': 
        return test_instruct_dict

    else: 
        return train_instruct_dict

def get_instructions(batch_size, task_type, instruct_mode):
        instruct_dict = get_instruction_dict(instruct_mode = instruct_mode)
        instructs = np.random.choice(instruct_dict[task_type], size=batch_size)
        if instruct_mode == 'shuffled': 
            instructs = list(map(shuffle_instruction, instructs))

        return list(instructs)

def one_hot_input_rule(batch_size, task_type, shuffled=False): 
    if shuffled: index = Task.SHUFFLED_TASK_LIST.index(task_type) 
    else: index = TASK_LIST.index(task_type)
    one_hot = np.zeros((1, len(TASK_LIST)))
    one_hot[:,index] = 1
    one_hot= np.repeat(one_hot, batch_size, axis=0)
    return one_hot

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

# def swap_input_rule(batch_size, task_type): 
#     index = Task.TASK_LIST.index(task_type)
#     swapped_one_hot = one_hot_input_rule(batch_size, swapped_task_list[index])
#     return swapped_one_hot

def get_input_rule(batch_size, task_type, instruct_mode): 
    # if instruct_mode == 'swap': 
    #     task_rule = swap_input_rule(batch_size, task_type)
    if instruct_mode == 'comp': 
        task_rule = comp_input_rule(batch_size, task_type)
    elif instruct_mode == 'masked': 
        task_rule = np.zeros((batch_size, 20))
    elif instruct_mode == ' shuffled': 
        task_rule = one_hot_input_rule(batch_size, task_type, shuffled=True)
    else: 
        task_rule = one_hot_input_rule(batch_size, task_type)
    
    return torch.Tensor(task_rule)

def get_task_info(batch_len, task_type, is_instruct, device='cpu', instruct_mode=None,): 
    if is_instruct: 
        return get_instructions(batch_len, task_type, instruct_mode)
    else: 
        return get_input_rule(batch_len, task_type, instruct_mode).to(device)
