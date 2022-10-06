import numpy as np
from instructRNN.tasks.tasks import *

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
