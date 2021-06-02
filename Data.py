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
    trial_type = (list(task_dict.keys())*batches_per_task)[:num_batches]
    trial_indices = np.array([Task.TASK_LIST.index(task) for task in trial_type])    
    
    for i in range(num_batches):
        trial = construct_batch(trial_type[i], batch_size)
        input_data[i, :, :, :] = trial.inputs
        target_data[i,: , :, :] = trial.targets
        masks_data[i, :, :, :] = trial.masks
        target_dirs[i, :] = trial.target_dirs

    return input_data, target_data, masks_data, target_dirs, trial_indices

input_data, target_data, masks_data, target_dirs, trial_indices = make_data(num_batches=500)

np.savez('training_data/Anti_Go', input_data=input_data, target_data=target_data, masks_data=masks_data, target_dirs=target_dirs, trial_indices=trial_indices)

np.save('training_data/Multitask_data/input_data', input_data)
np.save('training_data/Multitask_data/target_data', target_data)
np.save('training_data/Multitask_data/masks_data', masks_data)
np.save('training_data/Multitask_data/target_dirs', target_dirs)
np.save('training_data/Multitask_data/type_indices', trial_indices)

for i in range(16): 
    with np.load('training_data/Anti_Go/128batch.npz') as data: 
        data['input_data'].shape

trial_indices.shape

data_folder = 'training_data/Multitask_data'

memmap_task_types = np.load(data_folder+ '/type_indices.npy')
memmap_task_types[1]

from mmap import mmap
memmap_task_types = np.lib.format.open_memmap(data_folder+ '/type_indices.npy', dtype = 'int', mode = 'r', shape = 500)
memmap_task_types[16]


class data_streamer(): 
    def __init__(self, data_folder, num_batches): 
        self.num_batches = num_batches
        self.data_folder = data_folder
        self.memmap_inputs = np.lib.format.open_memmap(self.data_folder+'/input_data.npy', dtype = 'float32', mode = 'r', shape = (num_batches, 128, 120, 65))
        self.memmap_target = np.lib.format.open_memmap(self.data_folder+'/target_data.npy', dtype = 'float32', mode = 'r', shape = (num_batches, 128, 120, 33))
        self.memmap_masks = np.lib.format.open_memmap(self.data_folder+'/masks_data.npy', dtype = 'float32', mode = 'r', shape = (num_batches, 128, 120, 33))
        self.memmap_target_dirs = np.lib.format.open_memmap(self.data_folder+ '/target_dirs.npy', dtype = 'float32', mode = 'r', shape = (num_batches, 128))
        self.memmap_task_types = np.lib.format.open_memmap(self.data_folder+ '/type_indices.npy', dtype = 'int', mode = 'r', shape = num_batches)
        self.task_order = None
        self.permute_task_order()

    def permute_task_order(self): 
        self.task_order = np.random.permutation(self.memmap_task_types.copy())

    def get_data(self): 
        for i in self.task_order: 
            yield self.memmap_inputs[i, ].copy(), self.memmap_target[i, ].copy(), self.memmap_masks[i, ].copy(), self.memmap_target_dirs[i, ].copy(), Task.TASK_LIST[self.memmap_task_types[i, ]]


streamer = data_streamer('training_data/Multitask_data', 500)
streamer.task_order

for i in range(5):
    print(i)
    streamer.permute_task_order()
    for data in streamer.get_data(): 
        ins, tar, mask, tar_dir, task_type = data
        print(ins.shape)
