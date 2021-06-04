import numpy as np
from collections import defaultdict

from numpy.core.fromnumeric import size

from Task import Task, construct_batch
default_task_dict = dict.fromkeys(Task.TASK_LIST, 1/len(Task.TASK_LIST))

import multiprocessing

pooler = multiprocessing.Pool()

def make_data(task_dict = default_task_dict, batch_size = 128, num_batches = 500, holdouts=[], save_file=None):
    """Creates trial data
    Args:      
        task_dict (dict): dictionary where keys are tasks type and values are probabilities of the task occuring in the training set, 
        probabilities must add up to one, default is equal probabilities
        batch_size (int): size of batches used for training 
        num_batches (int): number of batches in training data
        holdouts (list): list of tasks to be held out of training 

    Returns:
        tuple: input_data (np.array); size: (num_batches, batch_size, 120, input_dim)
                target_data (np.array); size: (num_batches, batch_size, 120, input_dim)
                masks_data (np.array); size: (num_batches, batch_size, 120, output_dim)
                target_dirs(np.array); size: batch_size
                trial_type(np.array); size: batch_size
    """


    if len(holdouts) > 0: 
        holdout_list = [task for task in task_dict.keys() if task not in holdouts]
        task_dict = dict.fromkeys(holdout_list, 1/len(holdout_list))

    batches_per_task = int(np.ceil(num_batches/len(task_dict.keys())))
    trial_type = (list(task_dict.keys())*batches_per_task)[:num_batches]

    args = zip(trial_type, [batch_size]*num_batches)
    data = pooler.starmap(construct_batch, args)
    data_array = np.array(data)
    print('data_made')
    if save_file != None: 
        np.save('training_data/' + save_file, data_array)
        return
    else: 
        return np.stack(data_array[:, 0]), np.stack(data_array[:, 1]), np.stack(data_array[:, 2]), np.stack(data_array[:, 3]), np.stack(data_array[:, 4])
