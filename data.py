import numpy as np
import torch
from task import Task, construct_batch
task_list = Task.TASK_LIST
default_task_dict = dict.fromkeys(Task.TASK_LIST, 1/len(Task.TASK_LIST))


class TaskDataSet(): 
    def __init__(self, data_folder, batch_len=128, num_batches=500, task_ratio_dict = None, holdouts=[]): 
        __len__ = num_batches
        self.batch_len = batch_len
        self.num_batches = num_batches
        self.data_folder = data_folder
        if task_ratio_dict is None: self.task_ratio_dict = default_task_dict.copy()
        else: self.task_ratio_dict=task_ratio_dict
        self.holdouts = holdouts

        self.trial_types = None
        self.stream_order = None
        self.memmap_dict = {}

        self.__make_memmaps__()
        self.__init_task_distribution__()
        self.in_data, self.tar_data, self.mask_data, self.tar_dirs = self.__populate_data__()

        self.shuffle_stream_order()

    def data_to_device(self, device): 
        self.in_data = self.in_data.to(device)
        self.tar_data = self.tar_data.to(device)
        self.mask_data = self.mask_data.to(device)
    
    def __make_memmaps__(self): 
        for task in self.task_ratio_dict.keys(): 
            task_file = task.replace(' ', '_')
            self.memmap_dict[task] = (np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/input_data.npy', dtype = 'float32', mode = 'r', shape = (8000, 120, 65)),
                    np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/target_data.npy', dtype = 'float32', mode = 'r', shape = (8000, 120, 33)),
                    np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/masks_data.npy', dtype = 'int', mode = 'r', shape = (8000, 120, 33)),
                    np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/target_dirs.npy', dtype = 'float32', mode = 'r', shape = (8000)))
    
    def __init_task_distribution__(self):
        #rescale sampling ratio of tasks to deal with holdouts 
        if len(self.holdouts) > 0: 
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
            batch_indices = list(np.random.choice(np.arange(8000), size=self.batch_len))
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
            yield self.in_data[i, ...], self.tar_data[i, ...], self.mask_data[i, ...], self.tar_dirs[i, ...], self.trial_types[i]
