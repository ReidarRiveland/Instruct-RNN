from dataclasses import dataclass
import pickle

from attr import frozen
from task import Task
import numpy as np
from utils.utils import task_swaps_map, all_swaps

@dataclass(frozen=True)
class HoldoutDataFrame(): 
    file_path: str
    model_name: str
    perf_type: str

    mode: str = ''
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
        data = np.empty((5, len(Task.TASK_LIST), 100)) #seeds, task, num_batches        
        for i in range(5):
            seed_name = 'seed' + str(i)
            for j, task in enumerate(Task.TASK_LIST):
                task_file = task_swaps_map[task]
                load_path = self.file_path+'/'+task_file+'/'+self.model_name+'/'+self.mode+task.replace(' ', '_')+'_'+seed_name
                try:
                    data[i, j, :] = pickle.load(open(load_path+'_holdout_' + self.perf_type, 'rb'))
                except FileNotFoundError: 
                    print('No holdout data for '+ load_path)
        super().__setattr__('data', data)
        

gptNet_holdout_data  = HoldoutDataFrame('_ReLU128_4.11/swap_holdouts', 'sbertNet_tuned', 'correct')

np.mean(gptNet_holdout_data.get_k_shot(0))

holdout_data = gptNet_holdout_data.data

np.mean(holdout_data[:, :, 0])

np.mean(holdout_data[:, :, 0])

np.mean(gptNet_holdout_data.get_())

class TrainingDataFrame(): 
    def __init__(self, file_path, model_name, perf_type): 
        self.perf_type = perf_type
        self.file_path = file_path
        self.model_name = model_name


    def load_training_data(self): 
        self.training_data = np.full((2, 5, len(all_swaps), len(Task.TASK_LIST), 2000), np.NaN)
        for i in range(5):
            seed_name = 'seed' + str(i)
            for j, task_file in enumerate(all_swaps+['Multitask']):
                load_path = file_path+'/'+task_file+'/'+self.model_name+'/'+seed_name
                try:
                    correct_dict = pickle.load(open(load_path+'_training_correct', 'rb'))
                    loss_dict = pickle.load(open(load_path+'_training_loss', 'rb'))
                except FileNotFoundError: 
                    print('No folder for '+ load_path)
                    
                for k, task in enumerate(Task.TASK_LIST): 
                    try:
                        num_examples = len(correct_dict[task])
                        self.training_data[0, i, j, k,:num_examples] = correct_dict[task]
                        self.training_data[1, i, j, k, :num_examples] = loss_dict[task]
                    except: 
                        print('No training data for '+ self.model_name + ' '+seed_name+' '+task)



class PerfDataClass(): 
    def __init__(self, model_name): 
        self.model_name = model_name
        self.load_holdout_data()
        self.load_training_data()

    def get_task_data(self, task: str, perf_type: str, mode:str='normal'):
        if perf_type == 'holdout': data = self.holdout_data
        elif perf_type == 'training': data = self.holdout_data




test = PerfDataFrame('gptNet')
test.load_holdout_data('_ReLU128_4.11/swap_holdouts')

np.mean(test.holdout_data[..., 0][0][0])

task_dtype = np.dtype({'names':['mode', 'perf_type', 'seed', 'task', 'num_batches'], 'formats': ['i4', 'i4', 'i4', 'i4', 'f4']})

test_frame = np.array(test.holdout_data, task_dtype)

test_frame['mode'][0].shape

# def load_context_training_data(foldername, model_list, train_mode=''): 
#     #for each model name in the dict entry 
#     data_dict = dict.fromkeys(model_list)

#     for model_name in model_list: 
#         training_data = np.full((2, 5, len(all_swaps)+1, len(task_list), 2000), np.NaN)
#         for i in range(5):
#             seed_name = 'seed' + str(i)
#             for j, task_file in enumerate(all_swaps+['Multitask']):
#                 file_prefix = foldername+'/'+task_file+'/'+model_name+'/contexts/'+seed_name+'/'

#                 for k, task in enumerate(task_list): 

#                     try:
#                         correct_dict = pickle.load(open(file_prefix+task+'_'+train_mode+'context_correct_data20', 'rb'))
#                         loss_dict = pickle.load(open(file_prefix+task+'_'+train_mode+'context_loss_data20', 'rb'))
#                         num_examples = len(correct_dict[task])
#                         training_data[0, i, j, k,:num_examples] = correct_dict[task]
#                         training_data[1, i, j, k, :num_examples] = loss_dict[task]
#                     except FileNotFoundError: 
#                         print('No training data for '+ model_name + ' '+seed_name+' '+task)
#                         print(foldername+'/'+task_file+'/'+model_name+'/'+seed_name)
#                         continue 
#         data_dict[model_name] = training_data
#     return data_dict
