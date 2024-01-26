from instructRNN.analysis.model_analysis import *
from instructRNN.tasks.tasks import TASK_LIST
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.models.full_models import make_default_model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans 

import torch
import torch.nn as nn

##############RULE CLUSTER####################

def get_directions(rule_reps):
    directions_list = []
    directions_labels = []

    for key, values in DICH_DICT.items():
        directions_list.extend([rule_reps[TASK_LIST.index(dich_pair[1])]-rule_reps[TASK_LIST.index(dich_pair[0])] for dich_pair in values])
        directions_labels.extend(values)

    directions_array = np.array(directions_list)
    return directions_labels, directions_array

def get_held_in_indices(swap_label): 
    int_list = list(range(50))
    [int_list.remove(x) for x in [TASK_LIST.index(task) for task in SWAPS_DICT[swap_label]]]
    return int_list
    
def get_holdout_rule_reps(model, swap_label, seed): 
    held_in_indices = get_held_in_indices(swap_label)    

    if hasattr(model, 'langModel'): 
        rule_reps = get_instruct_reps(model.langModel)
    elif hasattr(model, 'rule_encoder'): 
        rule_reps = get_rule_reps(model, use_rule_encoder=True)
    else: 
        rule_reps = get_rule_reps(model)
        
    return rule_reps[held_in_indices, ...]

def gen_task_transitions(rule_reps, num_transitions):
    transitions = []
    for _ in range(num_transitions): 
        task_draw = np.random.randint(rule_reps.shape[0], size=2)
        instruct_draw = np.random.randint(rule_reps.shape[1], size=2)
        transitions.append(rule_reps[task_draw[0], instruct_draw[0], : ]-rule_reps[task_draw[1], instruct_draw[1], :])
    return np.array(transitions)

def get_combo_mat(rule_reps, clusters):
    comb_mat = np.empty((len(rule_reps), len(clusters), 64))
    for i, rep in enumerate(rule_reps.mean(1)): 
        for j, direc in enumerate(clusters): 
            comb_mat[i, j, :] = rep+direc
    
    return comb_mat
    
def eval_task_combos(model, combos, num_repeats = 25, tasks=TASK_LIST): 
    all_correct_array=np.empty((len(tasks), num_repeats, combos.shape[0], combos.shape[1]))
    with torch.no_grad(): 
        for i, task in enumerate(tasks): 
            print(f'processing task {task}')
            for j in range(num_repeats): 
                print(j)
                rules = torch.tensor(combos.reshape(-1, 64)).to(model.__device__)
                ins, targets, _, target_dirs, _ = construct_trials(task, rules.shape[0])
                out, _ = model(torch.Tensor(ins).to(model.__device__), info_embedded=rules)
                corrects = isCorrect(out, torch.Tensor(targets), target_dirs)
                reshaped_correct = corrects.reshape(combos.shape[0], combos.shape[1])
                all_correct_array[i, j , :, :] = reshaped_correct

    return all_correct_array


def get_holdout_combo_perfs(model_name, seeds=range(5), n_components=25, num_repeats=25, num_transitions=800, save=True):
    all_correct_array=np.zeros((len(seeds), len(TASK_LIST), num_repeats, 45, n_components))

    model = make_default_model(model_name)
    for seed in seeds:
        print(f'\n PROCESSING SEED {seed} \n')

        for swap_label, holdouts in SWAPS_DICT.items(): 
            
            model.load_model(f'NN_simData/swap_holdouts/{swap_label}/{model_name}', suffix=f'_seed{seed}')
            rule_reps = get_holdout_rule_reps(model, swap_label, seed)
            task_transition_data = gen_task_transitions(rule_reps, num_transitions=num_transitions)

            k_means = KMeans(n_clusters=n_components)
            k_means.fit(task_transition_data)
            clusters = k_means.cluster_centers_

            combo_mat = get_combo_mat(rule_reps, clusters)
            evals = eval_task_combos(model, combo_mat, num_repeats = num_repeats, tasks=holdouts)
            all_correct_array[seed, [TASK_LIST.index(task) for task in holdouts], ...] = evals

    if save:
        np.save(f'rule_combo_perf/{model_name}_n{n_components}_transitions{num_transitions}', all_correct_array)

    return all_correct_array




