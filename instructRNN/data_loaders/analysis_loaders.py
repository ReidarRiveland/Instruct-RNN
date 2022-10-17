import numpy as np 
from instructRNN.tasks.tasks import *

def load_multi_ccgp(model_name, seeds=range(5), layer='task'):
    task_holdout_array = np.full((len(seeds), len(TASK_LIST)), np.nan)
    dich_holdout_array = np.full((len(seeds), len(DICH_DICT)), np.nan)

    for i, seed in enumerate(seeds):
        task_load_str = '7.20models/multitask_holdouts/CCGP_scores/'+model_name+'/layer'+layer+'_task_multi_seed'+str(seed)+'.npy'
        dich_load_str = '7.20models/multitask_holdouts/CCGP_scores/'+model_name+'/layer'+layer+'_dich_multi_seed'+str(seed)+'.npy'
        task_arr = np.load(open(task_load_str, 'rb'))
        dich_arr = np.load(open(dich_load_str, 'rb'))
        task_holdout_array[i, :] = task_arr
        dich_holdout_array[i, :] = dich_arr

    return task_holdout_array, dich_holdout_array

def load_holdout_ccgp(folder_name, model_name, layer_list, seeds, verbose=False): 
    task_holdout_array = np.full((len(seeds), len(layer_list), len(TASK_LIST)), np.nan)        

    for i, seed in enumerate(seeds):
        for j, layer in enumerate(layer_list):
            try:
                task_load_str = folder_name+'/CCGP_scores/'+model_name+'/layer'+layer+'_task_holdout_seed'+str(seed)+'.npy'
                task_arr = np.load(open(task_load_str, 'rb'))
                task_holdout_array[i, j, :] = task_arr
            except FileNotFoundError:
                if verbose: 
                    print('no data for layer {} for model {} seed {}'.format(layer, model_name, seed))
                    print(task_load_str)

    return task_holdout_array

def load_holdout_dim_measures(folder_name, model_name, layer_list, seeds=range(5), verbose=False): 
    if 'swap' in folder_name: 
        exp_dict = SWAPS_DICT
    var_exp_array = np.full((len(seeds), len(layer_list), len(exp_dict), 25), np.nan)
    thresholds_array = np.full((len(seeds), len(layer_list), len(exp_dict)), np.nan)

    for i, seed in enumerate(seeds):
        for j, layer in enumerate(layer_list):
            try:
                load_str = folder_name+'/dim_measures/'+model_name+'/layer'+layer
                var_exp = np.load(open(load_str+'_var_exp_arr_seed'+str(seed)+'.npy', 'rb'))
                threshold = np.load(open(load_str+'_thresholds_seed'+str(seed)+'.npy', 'rb'))
                var_exp_array[i, j, ...] = var_exp
                thresholds_array[i, j, :] = threshold
            except FileNotFoundError:
                if verbose: 
                    print('no data for layer {} for model {} seed {}'.format(layer, model_name, seed))
                    print(load_str)

    return var_exp_array, thresholds_array

def load_dim_measures(folder_name, model_name, layer_list, seeds, verbose=False): 
    task_holdout_array = np.full((len(seeds), len(layer_list), len(TASK_LIST)), np.nan)

    for i, seed in enumerate(seeds):
        for j, layer in enumerate(layer_list):
            try:
                task_load_str = folder_name+'/CCGP_scores/'+model_name+'/layer'+layer+'_task_holdout_seed'+str(seed)+'.npy'
                task_arr = np.load(open(task_load_str, 'rb'))
                task_holdout_array[i, j, :] = task_arr
            except FileNotFoundError:
                if verbose: 
                    print('no data for layer {} for model {} seed {}'.format(layer, model_name, seed))
                    print(task_load_str)

    return task_holdout_array


def load_cluster_measures(folder_name, model_list, seeds=range(5), verbose=False): 
    if 'swap' in folder_name: 
        exp_dict = SWAPS_DICT
    num_cluster_array = np.full((len(seeds), len(model_list), len(exp_dict), 10), np.nan)

    for i, seed in enumerate(seeds):
        for j, model_name in enumerate(model_list):
            try:
                load_str = folder_name+'/cluster_measures/'+model_name+'/optim_clusters_seed'+str(seed)
                clusters = np.load(open(load_str+'.npy', 'rb'))
                num_cluster_array[i, j, ...] = clusters
            except FileNotFoundError:
                if verbose: 
                    print('no data for model {} seed {}'.format( model_name, seed))
                    print(load_str)

    return num_cluster_array


def load_val_perf(model_list, seeds=range(5), verbose=False): 
    val_perf_arr= np.full((len(seeds), len(model_list), len(TASK_LIST)), np.nan)

    for i, seed in enumerate(seeds):
        for j, model_name in enumerate(model_list):
            try:
                load_str = '7.20models/multitask_holdouts/val_perf/'+model_name+'/'+model_name+'_val_perf_seed'+str(seed)
                clusters = np.load(open(load_str+'.npy', 'rb'))
                val_perf_arr[i, j, ...] = clusters
            except FileNotFoundError:
                if verbose: 
                    print('no data for model {} seed {}'.format( model_name, seed))
                    print(load_str)

    return val_perf_arr
