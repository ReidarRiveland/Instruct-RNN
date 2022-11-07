from dataclasses import dataclass
import pickle
import numpy as np
import warnings

from instructRNN.tasks.tasks import TASK_LIST, SWAPS_DICT, ALIGNED_DICT, FAMILY_DICT

# @dataclass
# class HoldoutDataFrame(): 
#     file_path: str
#     exp_type: str 
#     model_name: str
#     perf_type: str = 'correct'
#     mode: str = ''
#     seeds: range = range(5)
#     layer_list: list = []
#     verbose: bool = True


#     def __post_init__(self):
#         if self.model_name == 'simpleNet' and self.mode == 'combined': 
#             self.mode = ''
            
#         self.load_data()

#     def get_k_shot(self, k, task=None): 
#         if task is None: 
#             return self.data[:, :, k]
#         else: 
#             return self.data[:,TASK_LIST.index(task),k]

#     def get_seed(self, seed: int): 
#         return self.data[seed, ...]

#     def get_task(self, task:str): 
#         return self.data[:, TASK_LIST.index(task), :]

#     def avg_seeds(self, task=None, k_shot=None): 
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             i = TASK_LIST.index(task) if task in TASK_LIST else slice(0, len(TASK_LIST))

#             if k_shot is None: 
#                 k_shot=slice(0, self.num_batches)

#             data = self.data[:, i, k_shot]
#             mean = np.nanmean(data, axis=0)
#             std = np.nanstd(data, axis=0)
#             return mean, std

#     def avg_tasks(self, seeds=range(5), k_shot=None): 
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             if k_shot is None: 
#                 k_shot=slice(0, self.num_batches)


#             data = self.data[seeds, :, k_shot]
#             _mean = np.nanmean(data, axis=1)
#             std = np.nanstd(_mean, axis=0)
#             mean = np.nanmean(_mean, axis=0)

#             return mean, std

#     def load_data(self): 
#         if self.exp_type == 'swap': 
#             training_sets = SWAPS_DICT
#         elif self.exp_type == 'aligned': 
#             training_sets = ALIGNED_DICT
#         elif self.exp_type == 'family': 
#             training_sets = FAMILY_DICT

#         if 'input' in self.mode or 'comp' in self.mode:
#             self.num_batches = 1500
#         else: 
#             self.num_batches = 100

#         data = np.full((5, len(TASK_LIST), self.num_batches), np.nan) #seeds, task, num_batches        
#         for i in self.seeds:
#             seed_name = 'seed' + str(i)
#             for label, tasks in training_sets.items():
#                 for task in tasks: 
#                     load_path = self.file_path+'/'+self.exp_type+'_holdouts/'+label+'/'+self.model_name+'/holdouts/'\
#                                     +self.mode+task+'_'+seed_name
#                     try:
#                         data[i, TASK_LIST.index(task), :] = pickle.load(open(load_path+'_' + self.perf_type, 'rb'))
#                     except FileNotFoundError: 
#                         if self.verbose:
#                             print('No holdout data for '+ load_path)
#         super().__setattr__('data', data)


@dataclass
class PerfDataFrame(): 
    file_path: str
    exp_type: str 
    model_name: str
    perf_type: str = 'correct'
    mode: str = ''
    holdout_file: str = ''

    seeds: range = range(5)
    layer_list: tuple = ()
    verbose: bool = False

    def __post_init__(self):
        if self.exp_type == 'swap': 
            self.exp_dict = SWAPS_DICT
        elif self.exp_type == 'family': 
            self.exp_dict = FAMILY_DICT

        ###load modes
        if self.mode == 'multi_CCGP':
            load_str = self.file_path+'/multitask_holdouts/CCGP_scores/'+self.model_name+'/layertask_task_multi_seed{}.npy'
            self.load_multi_measure(load_str)
        elif self.mode == 'val': 
            load_str = self.file_path+'/multitask_holdouts/val_perf/'+self.model_name+'/'+self.model_name+'_val_perf_seed{}'
            self.load_multi_measure(load_str)

        elif self.mode == 'multi_comp': 
            load_str = self.file_path+'/multitask_holdouts/multi_comp_perf/'+self.model_name+'/'+self.model_name+'_multi_comp_perf_seed{}'
            self.load_multi_measure(load_str)

        elif self.mode == 'training': 
            self.load_training_data()
        else: 
            load_holdout_data()

    def get_k_shot(self, k, task=None): 
        if task is None: 
            return self.data[:, :, k]
        else: 
            return self.data[:,TASK_LIST.index(task),k]

    def get_task(self, task:str): 
        return self.data[:, TASK_LIST.index(task), :]

    def avg_seeds(self, task=None, k_shot=None): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            i = TASK_LIST.index(task) if task in TASK_LIST else slice(0, len(TASK_LIST))

            if k_shot is None: 
                k_shot=slice(0, self.num_batches)

            data = self.data[:, i, k_shot]
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            return mean, std

    def avg_tasks(self, seeds=range(len(seeds)), k_shot=None): 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if k_shot is None: 
                k_shot=slice(0, self.num_batches)


            data = self.data[seeds, :, k_shot]
            _mean = np.nanmean(data, axis=1)
            std = np.nanstd(_mean, axis=0)
            mean = np.nanmean(_mean, axis=0)

            return mean, std

    def load_holdout_data(self): 
        if 'input' in self.mode:
            self.num_batches = 1500
        else: 
            self.num_batches = 100

        data = np.full((len(seeds), len(TASK_LIST), self.num_batches), np.nan) #seeds, task, num_batches        
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

    def load_training_data(self): 
        assert len(self.holdout_file)>1
        data = np.full((5, len(TASK_LIST), 100_000), np.NaN)
        for i in self.seeds:
            seed_name = 'seed' + str(i)
            load_path = self.file_path+'/'+self.exp_type+'_holdouts/'+self.holdout_file+'/'+self.model_name+'/'+seed_name

            try:
                data_dict = pickle.load(open(load_path+'_training_'+self.perf_type, 'rb'))
            except FileNotFoundError: 
                if self.verbose:print('No folder for '+ load_path)
                
            for k, task in enumerate(TASK_LIST): 
                if task in self.exp_dict[self.holdout_file]:
                    continue
                num_examples = len(data_dict[task])
                data[i, k,:num_examples] = data_dict[task]
                
        super().__setattr__('data', data)

    def load_multi_measure(self, load_str):
        data = np.full((len(self.seeds), len(TASK_LIST), 1), np.nan)
        for i, seed in enumerate(self.seeds):
            try:
                load_str = load_str.format(str(seed))
                print(load_str)
                seed_data_arr = np.load(open(load_str, 'rb'))
                data[i, :] = seed_data_arr[:, None]
            except FileNotFoundError:
                if self.verbose: 
                    print('no data for layer {} for model {} seed {}'.format(layer, self.model_name, self.seed))
                    print(load_str)

        super().__setattr__('data', data)

    def load_holdout_CCGP(self, mode=''): 
        data = np.full((len(self.seeds), len(TASK_LIST), len(layer_list)), np.nan)        

        for i, seed in enumerate(self.seeds):
            for j, layer in enumerate(self.layer_list):
                try:
                    load_str = self.file_path+'/'+self.exp_type+'_holdouts/CCGP_scores/'+self.model_name+'/layer'+layer+'_task_holdout_seed'+str(seed)+mode+'.npy'
                    tmp_data_arr = np.load(open(task_load_str, 'rb'))
                    data[i, j, :] = tmp_data_arr
                except FileNotFoundError:
                    if self.verbose: 
                        print('no data for layer {} for model {} seed {}'.format(layer, model_name, seed))

        super().__setattr__('data', data)
