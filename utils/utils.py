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

from collections import Counter

training_lists_dict={
'single_holdouts' :  [[item] for item in Task.TASK_LIST.copy()+['Multitask']],
'dual_holdouts' : [['RT Go', 'Anti Go'], ['Anti MultiDM', 'DM'], ['COMP1', 'MultiCOMP2'], ['DMC', 'DNMS']],
'aligned_holdouts' : [['Anti DM', 'Anti MultiDM'], ['COMP1', 'MultiCOMP1'], ['DMS', 'DNMS'],['Go', 'RT Go']],
'swap_holdouts' : [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['RT Go', 'DNMC'], ['DM', 'MultiCOMP2'], ['MultiDM', 'DNMS'], ['Anti MultiDM', 'COMP1'], ['COMP2', 'DMS'], ['Anti Go', 'MultiCOMP1']]
}

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
                'DNMC': 'RT_Go_DNMC',
                'Multitask':'Multitask'}

all_swaps = list(set(task_swaps_map.values()))

task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'goldenrod', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'bow20Net': ('yellow', None), 'bowNet': ('orange', None), 'gptNet': ('red', None), 'gptNet_tuned': ('red', 'v'), 'bertNet_tuned': ('green', 'v'),
                    'bertNet': ('green', None), 'bertNet_layer_11': ('green', '.'), 'sbertNet': ('purple', None), 'sbertNet_tuned': ('purple', 'v'), 'sbertNet_layer_11': ('purple', '.')}

def get_holdout_file_name(holdouts): 
    if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
    else: holdout_file = holdouts[0]
    holdout_file = holdout_file.replace(' ', '_')
    return holdout_file

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


def load_holdout_data(foldername, model_list): 
    #for each model name in the dict entry 
    data_dict = dict.fromkeys(model_list)

    for model_name in model_list: 
        model_data_dict = {}
        for mode in ['swap', '']:
            holdout_data = np.empty((2, 5, len(Task.TASK_LIST), 100))
            #training_data = np.empty((len(seeds), 4, 100))

            for i in range(5):
                seed_name = 'seed' + str(i)

                for j, task in enumerate(task_list):
                    holdout_file = task.replace(' ', '_')
                    task_file = task_swaps_map[task]
                    holdout_file += '_'

                    try:
                        holdout_data[0, i, j, :] = pickle.load(open(foldername+'/'+task_file+'/'+model_name+'/'+mode+holdout_file+seed_name+'_holdout_correct', 'rb'))
                        holdout_data[1, i, j, :] = pickle.load(open(foldername+'/'+task_file+'/'+model_name+'/'+mode+holdout_file+seed_name+'_holdout_loss', 'rb'))
                    except FileNotFoundError: 
                        print('No training data for '+ model_name + ' '+seed_name+' '+mode+task)
                        #print(foldername+'/'+task_file+'/'+model_name+'/'+holdout_file+seed_name)
                        continue 
            model_data_dict[mode] = holdout_data
        data_dict[model_name] = model_data_dict
    return data_dict

def load_training_data(foldername, model_list): 
    #for each model name in the dict entry 
    data_dict = dict.fromkeys(model_list)

    for model_name in model_list: 
        training_data = np.full((2, 5, len(all_swaps), len(task_list), 2000), np.NaN)
        for i in range(5):
            seed_name = 'seed' + str(i)
            for j, task_file in enumerate(all_swaps+['Multitask']):
                try:
                    correct_dict = pickle.load(open(foldername+'/'+task_file+'/'+model_name+'/'+seed_name+'_training_correct', 'rb'))
                    loss_dict = pickle.load(open(foldername+'/'+task_file+'/'+model_name+'/'+seed_name+'_training_loss', 'rb'))
                except FileNotFoundError: 
                    print('No folder for '+ foldername+'/'+task_file+'/'+model_name+'/'+seed_name)
                    continue

                for k, task in enumerate(task_list): 
                    try:
                        num_examples = len(correct_dict[task])
                        training_data[0, i, j, k,:num_examples] = correct_dict[task]
                        training_data[1, i, j, k, :num_examples] = loss_dict[task]
                    except: 
                        print('No training data for '+ model_name + ' '+seed_name+' '+task)
                        print(foldername+'/'+task_file+'/'+model_name+'/'+seed_name)
                        continue 
        data_dict[model_name] = training_data
    return data_dict

def load_context_training_data(foldername, model_list, train_mode=''): 
    #for each model name in the dict entry 
    data_dict = dict.fromkeys(model_list)

    for model_name in model_list: 
        training_data = np.full((2, 5, len(all_swaps)+1, len(task_list), 20000), np.NaN)
        for i in range(5):
            seed_name = 'seed' + str(i)
            for j, task_file in enumerate(all_swaps+['Multitask']):
                file_prefix = foldername+'/'+task_file+'/'+model_name+'/contexts/'+seed_name+'/'

                for k, task in enumerate(task_list): 

                    try:
                        correct_dict = pickle.load(open(file_prefix+task+'_'+train_mode+'context_correct_data20', 'rb'))
                        loss_dict = pickle.load(open(file_prefix+task+'_'+train_mode+'context_loss_data20', 'rb'))
                        num_examples = len(correct_dict[task])
                        training_data[0, i, j, k,:num_examples] = correct_dict[task]
                        training_data[1, i, j, k, :num_examples] = loss_dict[task]
                    except FileNotFoundError: 
                        print('No training data for '+ model_name + ' '+seed_name+' '+task)
                        print(foldername+'/'+task_file+'/'+model_name+'/'+seed_name)
                        continue 
        data_dict[model_name] = training_data
    return data_dict
