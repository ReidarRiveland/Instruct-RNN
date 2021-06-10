import numpy as np
from task import Task
task_list = Task.TASK_LIST
default_task_dict = dict.fromkeys(Task.TASK_LIST, 1/len(Task.TASK_LIST))

class data_streamer(): 
    def __init__(self, batch_len=128, num_batches=500, task_ratio_dict = default_task_dict.copy(), holdouts=[]): 
        self.batch_len = batch_len
        self.num_batches = num_batches
        self.data_folder = 'training_data'
        self.memmap_dict = {}
        self.task_order = None
        self.task_ratio_dict = task_ratio_dict


        #rescale sampling ratio of tasks to deal with holdouts 
        if len(holdouts) > 0: 
            for task in holdouts: 
                del task_ratio_dict[task]
            raw_ratios=list(task_ratio_dict.values())
            new_ratios =[ratio/sum(raw_ratios) for ratio in raw_ratios]
            self.task_ratio_dict = dict(zip(task_ratio_dict.keys(), new_ratios))

        #make a list of tasks type of size num_batches with ratios of each task determined by task_dict 
        batches_per_task = [int(np.floor(num_batches*ratio)) for ratio in self.task_ratio_dict.values()] 
        self.trial_types = sum([[task]*batches for task, batches in zip(self.task_ratio_dict.keys(), batches_per_task)], [])
        while len(self.trial_types) < num_batches:
            self.trial_types.append(task_list[-(num_batches-len(self.trial_types))])


        self.make_memmaps()
        self.permute_task_order()
        
    def make_memmaps(self): 
        for task in self.task_ratio_dict.keys(): 
            task_file = task.replace(' ', '_')
            self.memmap_dict[task] = (np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/input_data.npy', dtype = 'float32', mode = 'r', shape = (8000, 120, 65)),
                    np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/target_data.npy', dtype = 'float32', mode = 'r', shape = (8000, 120, 33)),
                    np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/masks_data.npy', dtype = 'int', mode = 'r', shape = (8000, 120, 33)),
                    np.lib.format.open_memmap(self.data_folder+'/'+task_file+'/target_dirs.npy', dtype = 'float32', mode = 'r', shape = (8000)))

    def permute_task_order(self): 
        self.task_order = np.random.permutation(self.trial_types)

    def get_batch(self): 
        for task in self.task_order: 
            batch_indices = list(np.random.choice(np.arange(8000), size=self.batch_len))
            memmaps = self.memmap_dict[task]
            yield memmaps[0][batch_indices, ].copy(), memmaps[1][batch_indices, ].copy(), memmaps[2][batch_indices, ].copy(), memmaps[3][batch_indices, ].copy(), task

