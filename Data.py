import numpy as np
from collections import defaultdict

from numpy.core.fromnumeric import size
from numpy.core.records import array

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
    return np.stack(data_array[:, 0]), np.stack(data_array[:, 1]), np.stack(data_array[:, 2]), np.stack(data_array[:, 3]), np.stack(data_array[:, 4])

for task in ['Go']: 
    task_file = task.replace(' ', '_')
    print(task_file)
    input_data, target_data, masks_data, target_dirs, trial_indices = make_data(holdouts=[task], batch_size=128, num_batches=1000)

#     np.save('training_data/' + task_file+'/task_testing/input_data', input_data)
#     np.save('training_data/' + task_file+'/task_testing/target_data', target_data)
#     np.save('training_data/' + task_file+'/task_testing/masks_data', masks_data)
#     np.save('training_data/' + task_file+'/task_testing/target_dirs', target_dirs)
#     np.save('training_data/' + task_file+'/task_testing/type_indices', trial_indices)
    np.save('training_data/' + task_file+'/task_testing/input_data', input_data)
    np.save('training_data/' + task_file+'/task_testing/target_data', target_data)
    np.save('training_data/' + task_file+'/task_testing/masks_data', masks_data)
    np.save('training_data/' + task_file+'/task_testing/target_dirs', target_dirs)
    np.save('training_data/' + task_file+'/task_testing/type_indices', trial_indices)


class data_streamer(): 
    def __init__(self, data_folder): 
        if 'holdout_training' in data_folder: 
            self.num_batches = 500
            self.num_batches = 1000
            self.epoch_len = 500
            self.batch_len = 128
        elif 'task_testing' in data_folder: 
            self.num_batches = 100
            self.epoch_len = 100
            self.batch_len = 256
        else: 
            raise Exception("invalid data folder")
        self.data_folder = data_folder
        self.memmap_inputs = np.lib.format.open_memmap(self.data_folder+'/input_data.npy', dtype = 'float32', mode = 'r', shape = (self.num_batches, self.batch_len, 120, 65))
        self.memmap_target = np.lib.format.open_memmap(self.data_folder+'/target_data.npy', dtype = 'float32', mode = 'r', shape = (self.num_batches, self.batch_len, 120, 33))
        self.memmap_masks = np.lib.format.open_memmap(self.data_folder+'/masks_data.npy', dtype = 'float32', mode = 'r', shape = (self.num_batches, self.batch_len, 120, 33))
        self.memmap_target_dirs = np.lib.format.open_memmap(self.data_folder+ '/target_dirs.npy', dtype = 'float32', mode = 'r', shape = (self.num_batches, self.batch_len))
        self.memmap_task_types = np.lib.format.open_memmap(self.data_folder+ '/type_indices.npy', dtype = 'int', mode = 'r', shape = self.num_batches)
        self.task_order = None
        self.permute_task_order()

    def permute_task_order(self): 
        self.task_order = np.random.permutation(self.num_batches)
        self.task_order = np.random.choice(np.arange(self.num_batches), size=self.epoch_len)

    def get_data(self): 
        for i in self.task_order: 
            yield self.memmap_inputs[i, ].copy(), self.memmap_target[i, ].copy(), self.memmap_masks[i, ].copy(), self.memmap_target_dirs[i, ].copy(), Task.TASK_LIST[self.memmap_task_types[i, ]]
