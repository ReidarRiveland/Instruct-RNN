from dataclasses import dataclass
import pickle
import numpy as np
import warnings

from instructRNN.tasks.tasks import TASK_LIST, SWAPS_DICT, ALIGNED_DICT, FAMILY_DICT

@dataclass(frozen=True)
class HoldoutDataFrame(): 
    file_path: str
    exp_type: str 
    model_name: str
    perf_type: str = 'correct'
    mode: str = ''
    seeds: range = range(5)

    verbose: bool = True


    def __post_init__(self):
        self.load_data()

    def get_k_shot(self, k, task=None): 
        if task is None: 
            return self.data[:, :, k]
        else: 
            return self.data[:,TASK_LIST.index(task),k]

    def get_seed(self, seed: int): 
        return self.data[seed, ...]

    def get_task(self, task:str): 
        return self.data[:, TASK_LIST.index(task), :]

    def avg_seeds(self, task=None, k_shot=slice(0, 100)): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            i = TASK_LIST.index(task) if task in TASK_LIST else slice(0, len(TASK_LIST))

            data = self.data[:, i, k_shot]
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            return mean, std

    def avg_tasks(self, seeds=range(5), k_shot=slice(0, 100)): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = self.data[seeds, :, k_shot]
            _mean = np.nanmean(data, axis=1)
            std = np.nanstd(_mean, axis=0)
            mean = np.nanmean(_mean, axis=0)

            return mean, std

    def load_data(self): 
        if self.exp_type == 'swap': 
            training_sets = SWAPS_DICT
        elif self.exp_type == 'aligned': 
            training_sets = ALIGNED_DICT
        elif self.exp_type == 'family': 
            training_sets = FAMILY_DICT

        data = np.full((5, len(TASK_LIST), 100), np.nan) #seeds, task, num_batches        
        for i in self.seeds:
            seed_name = 'seed' + str(i)
            for label, tasks in training_sets.items():
                for task in tasks: 
                    load_path = self.file_path+'/'+self.exp_type+'_holdouts/'+label+'/'+self.model_name+'/holdouts/'\
                                    +self.mode+task+'_'+seed_name
                    try:
                        data[i, TASK_LIST.index(task), :] = pickle.load(open(load_path+'_' + self.perf_type, 'rb'))
                    except FileNotFoundError: 
                        if self.verbose:
                            print('No holdout data for '+ load_path)
        super().__setattr__('data', data)

@dataclass(frozen=True)
class TrainingDataFrame(): 
    file_path: str
    exp_type: str 
    holdout_file: str
    model_name: str
    perf_type: str = 'correct'
    seeds: range = range(5)
    verbose: bool = True


    def __post_init__(self):
        self.load_data()

    def avg_seeds(self, task=None, k_shot=slice(0, 2000)): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            i = TASK_LIST.index(task) if task in TASK_LIST else slice(0, len(TASK_LIST))

            data = self.data[:, i, k_shot]
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            return mean, std

    def load_data(self): 
        data = np.full((5, len(TASK_LIST), 2000), np.NaN)
        for i in range(5):
            seed_name = 'seed' + str(i)
            load_path = self.file_path+'/'+self.exp_type+'_holdouts/'+self.holdout_file+'/'+self.model_name+'/'+seed_name
            try:
                data_dict = pickle.load(open(load_path+'_training_'+self.perf_type, 'rb'))
            except FileNotFoundError: 
                if self.verbose:
                    print('No folder for '+ load_path)
                
            for k, task in enumerate(TASK_LIST): 
                try:
                    num_examples = len(data_dict[task])
                    data[i, k,:num_examples] = data_dict[task]
                except KeyError: 
                    if self.verbose: 
                        print('No training data for '+ self.model_name + ' '+seed_name+' '+task)
        super().__setattr__('data', data)





