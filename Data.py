import numpy as np
from collections import defaultdict

from Task import Task, construct_batch
default_task_dict = dict.fromkeys(Task.TASK_LIST, 1/len(Task.TASK_LIST))


def make_data(task_dict = default_task_dict, batch_size = 128, num_batches = 500, holdouts=[]):
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
    input_data = np.empty((num_batches, batch_size, 120, Task.INPUT_DIM), dtype=np.float32)
    target_data = np.empty((num_batches, batch_size, 120, Task.OUTPUT_DIM), dtype=np.float32)
    masks_data = np.empty((num_batches, batch_size, 120, Task.OUTPUT_DIM), dtype=np.float32)
    target_dirs = np.empty((num_batches, batch_size))
    
    if len(holdouts) > 0: 
        holdout_list = [task for task in task_dict.keys() if task not in holdouts]
        task_dict = dict.fromkeys(holdout_list, 1/len(holdout_list))
    
    batches_per_task = int(np.ceil(num_batches/len(task_dict.keys())))
    trial_type = np.array((list(task_dict.keys())*batches_per_task)[:num_batches])
    for i in range(num_batches):
        trial = construct_batch(trial_type[i], batch_size)
        input_data[i, :, :, :] = trial.inputs
        target_data[i,: , :, :] = trial.targets
        masks_data[i, :, :, :] = trial.masks
        target_dirs[i, :] = trial.target_dirs

    return input_data, target_data, masks_data, target_dirs, trial_type



