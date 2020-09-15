import numpy as np
from collections import defaultdict

from Task import Task, construct_batch
default_task_dict = dict.fromkeys(Task.TASK_LIST, 1/len(Task.TASK_LIST))

def make_data(task_dict = default_task_dict, BATCH_LEN = 128, NUM_BATCHES = 500, holdouts=None):
    input_data = np.empty((NUM_BATCHES, BATCH_LEN, 120, Task.INPUT_DIM), dtype=np.float32)
    target_data = np.empty((NUM_BATCHES, BATCH_LEN, 120, Task.OUTPUT_DIM), dtype=np.float32)
    masks_data = np.empty((NUM_BATCHES, BATCH_LEN, 120, Task.OUTPUT_DIM), dtype=np.float32)
    target_dirs = np.empty((NUM_BATCHES, BATCH_LEN))
    if holdouts is not None: 
        holdout_list = [task for task in Task.TASK_LIST if task not in holdouts]
        task_dict = dict.fromkeys(holdout_list, 1/len(holdout_list))
    
    trial_type = np.random.choice(list(task_dict.keys()), p=list(task_dict.values()), size=NUM_BATCHES)

    for i in range(NUM_BATCHES):
        task = construct_batch(trial_type[i], BATCH_LEN)
        input_data[i, :, :, :] = task.inputs
        target_data[i,: , :, :] = task.targets
        masks_data[i, :, :, :] = task.masks
        target_dirs[i, :] = task.target_dirs

    return input_data, target_data, masks_data, target_dirs, trial_type



