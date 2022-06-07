from dataclasses import dataclass
import pickle
from tasks_utils import SWAPS_DICT

from attr import frozen
from torch import arange
from tasks import Task
import numpy as np

@dataclass(frozen=True)
class HoldoutDataFrame(): 
    file_path: str
    exp_type: str 
    model_name: str
    perf_type: str
    mode: str = ''
    seeds: range = range(5)

    def __post_init__(self):
        self.load_data()

    def get_k_shot(self, k, task=None): 
        if task is None: 
            return self.data[:, :, k]
        else: 
            return self.data[:,Task.TASK_LIST.index(task),k]

    def get_seed(self, seed: int): 
        return self.data[seed, ...]

    def get_task(self, task:str): 
        return self.data[:, Task.TASK_LIST.index(task), :]

    def load_data(self): 
        if self.exp_type == 'swaps': 
            training_sets = SWAPS_DICT
        data = np.full((5, len(Task.TASK_LIST), 100), np.nan) #seeds, task, num_batches        
        for i in self.seeds:
            seed_name = 'seed' + str(i)
            for label, tasks in training_sets.items():
                for task in tasks: 
                    load_path = self.file_path+'/'+self.exp_type+'/'+label+'/'+self.model_name+'/'\
                                    +self.mode+task+'_'+seed_name
                    try:
                        data[i, Task.TASK_LIST.index(task), :] = pickle.load(open(load_path+'_holdout_' + self.perf_type, 'rb'))[task]
                    except FileNotFoundError: 
                        print('No holdout data for '+ load_path)
        super().__setattr__('data', data)

@dataclass(frozen=True)
class TrainingDataFrame(): 
    file_path: str
    model_name: str
    perf_type: str
    def __post_init__(self):
        self.load_data()

    def load_data(self): 
        data = np.full((2, 5, len(Task.TASK_LIST), 2000), np.NaN)
        for i in range(5):
            seed_name = 'seed' + str(i)
            load_path = self.file_path+'/'+self.model_name+'/'+seed_name
            try:
                data_dict = pickle.load(open(load_path+'_training_'+self.perf_type, 'rb'))
            except FileNotFoundError: 
                print('No folder for '+ load_path)
                
            for k, task in enumerate(Task.TASK_LIST): 
                try:
                    num_examples = len(data_dict[task])
                    data[i, k,:num_examples] = data_dict[task]
                except KeyError: 
                    print('No training data for '+ self.model_name + ' '+seed_name+' '+task)
        super().__setattr__('data', data)

