from genericpath import exists
import numpy as np
import torch
import os

from instructRNN.tasks.tasks import TASK_LIST, construct_trials
location = str(os.pathlib.Path(__file__).parent.absolute())


class TaskDataSet():
    DEFAULT_TASK_DICT = dict.fromkeys(TASK_LIST, 1/len(TASK_LIST)) 
    def __init__(self, exp_file, stream= True, batch_len=128, num_batches=500, holdouts=[], set_single_task = None): 
        __len__ = num_batches
        self.stream = stream
        self.batch_len = batch_len
        self.num_batches = num_batches
        self.data_folder = exp_file
        self.holdouts = holdouts

        if set_single_task is None: 
            self.task_ratio_dict = self.DEFAULT_TASK_DICT.copy()
        else: 
            assert not bool(holdouts), 'cannot have holdouts and set a single task'
            self.task_ratio_dict={set_single_task:1}

        self.trial_types = None
        self.stream_order = None
        self.memmap_dict = {}

        self.__init_task_distribution__()
        if not stream: 
            self.__make_memmaps__()
            self.in_data, self.tar_data, self.mask_data, self.tar_dirs = self.__populate_data__()
        self.shuffle_stream_order()

    def data_to_device(self, device): 
        self.in_data = self.in_data.to(device)
        self.tar_data = self.tar_data.to(device)
        self.mask_data = self.mask_data.to(device)
    
    def __make_memmaps__(self): 
        for task in self.task_ratio_dict.keys(): 
            task_data_path = location +'/'+self.data_folder+'/'+task
            if not os.path.exists(task_data_path):
                print('\n no training data for {task} discovered at {data_load_path} \n'.format(task=task, data_load_path=self.data_folder))
                build_training_data(self.data_folder, task)

            self.memmap_dict[task] = (np.lib.format.open_memmap(task_data_path+'/input_data.npy', dtype = 'float32', mode = 'r', shape = (10000, 120, 65)),
                    np.lib.format.open_memmap(task_data_path+'/target_data.npy', dtype = 'float32', mode = 'r', shape = (10000, 120, 33)),
                    np.lib.format.open_memmap(task_data_path+'/masks_data.npy', dtype = 'int', mode = 'r', shape = (10000, 120, 33)),
                    np.lib.format.open_memmap(task_data_path+'/target_dirs.npy', dtype = 'float32', mode = 'r', shape = (10000)))


    def __init_task_distribution__(self):
        #rescale sampling ratio of tasks to deal with holdouts 
        if len(self.holdouts) > 0 and 'Multitask' not in self.holdouts: 
            for task in self.holdouts: 
                del self.task_ratio_dict[task]
            raw_ratios=list(self.task_ratio_dict.values())
            new_ratios =[ratio/sum(raw_ratios) for ratio in raw_ratios]
            self.task_ratio_dict = dict(zip(self.task_ratio_dict.keys(), new_ratios))

        #make a list of tasks type of size num_batches with ratios of each task determined by task_dict 
        batches_per_task = [int(np.floor(self.num_batches*ratio)) for ratio in self.task_ratio_dict.values()] 
        self.trial_types = sum([[task]*batches for task, batches in zip(self.task_ratio_dict.keys(), batches_per_task)], [])
        while len(self.trial_types) < self.num_batches:
            self.trial_types.append(list(self.task_ratio_dict.keys())[-(self.num_batches-len(self.trial_types))])

    def __populate_data__(self): 
        tmp_in_data = np.empty((self.num_batches, self.batch_len, 120, 65), dtype=np.float32)
        tmp_tar_data = np.empty((self.num_batches, self.batch_len, 120, 33), dtype=np.float32)
        tmp_mask_data = np.empty((self.num_batches, self.batch_len, 120, 33), dtype=np.int)
        tmp_tar_dirs = np.empty((self.num_batches, self.batch_len), dtype = np.float32)
        for index, task in enumerate(self.trial_types):
            if index % 50 == 0: 
                print('populating data ' + str(index))
            batch_indices = list(np.random.choice(np.arange(10000), size=self.batch_len))
            memmaps = self.memmap_dict[task]
            tmp_in_data[index, ...] = memmaps[0][batch_indices, ].copy()
            tmp_tar_data[index, ...]=memmaps[1][batch_indices, ].copy() 
            tmp_mask_data[index, ...]=memmaps[2][batch_indices, ].copy()
            tmp_tar_dirs[index] = memmaps[3][batch_indices, ].copy()
        
        return (torch.tensor((tmp_in_data), dtype=torch.float16), torch.tensor((tmp_tar_data), dtype=torch.float16), 
                            torch.tensor((tmp_mask_data), dtype=torch.float16), torch.tensor((tmp_tar_dirs), dtype=torch.float16))

    def shuffle_stream_order(self): 
        self.stream_order = np.random.permutation(np.arange(self.num_batches))

    def stream_batch(self): 
        for i in self.stream_order: 
            if not self.stream: 
                yield self.in_data[i, ...], self.tar_data[i, ...], self.mask_data[i, ...], self.tar_dirs[i, ...], self.trial_types[i]
            else:
                yield construct_trials(self.trial_types[i], self.batch_len, return_tensor=True)


def build_training_data(foldername, task):
    print('Building training data for ' + task + '...')
    path = foldername +'/'+ task
    if os.path.exists(path):
        pass
    else: 
        os.makedirs(path)
    input_data, target_data, masks_data, target_dirs, trial_indices = construct_trials(task, 10000)
    np.save(path+'/input_data', input_data)
    np.save(path+'/target_data', target_data)
    np.save(path+'/masks_data', masks_data)
    np.save(path +'/target_dirs', target_dirs)
    np.save(path +'/type_indices', trial_indices)
