from sqlalchemy import over
import torch
from torch import Tensor, optim
import numpy as np
from yaml import warnings

from models.full_models import make_default_model
from base_trainer import BaseTrainer, masked_MSE_Loss
from dataset import TaskDataSet
from task import isCorrect
from utils.utils import get_holdout_file_name, training_lists_dict
from utils.task_info_utils import get_task_info

import pickle
import copy
from tqdm import tqdm
from attrs import define
import os
import warnings
import gc

device = torch.device(0)

EXP_FILE ='_ReLU128_4.11/aligned_holdouts'

@define
class TrainerConfig(): 
    file_path: str
    random_seed: int
    epochs: int = 35
    min_run_epochs: int = 35
    batch_len: int = 128
    num_batches: int = 500
    holdouts: list = []

    optim_alg: optim = optim.Adam
    lr: float = 0.001
    lang_lr: float = None
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.95}

    save_for_tuning_epoch: int = 30
    checker_threshold: float = 0.95
    step_last_lr: bool = True

class ModelTrainer(BaseTrainer): 
    def __init__(self, training_config:TrainerConfig=None, from_checkpoint_dict:dict=None): 
        super().__init__(training_config, from_checkpoint_dict)

    def _record_session(self, model, mode='CHECKPOINT'): 
        checkpoint_attrs = super()._record_session()
        if mode == 'CHECKPOINT':
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+'_CHECKPOINT_attrs', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix+'_CHECKPOINT')

        if mode=='FINAL': 
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+'_attrs', 'wb'))
            os.remove(self.file_path+'/'+self.model_file_path+'_CHECKPOINT_attrs')
            pickle.dump(self.loss_data, open(self.file_path+'/'+self.seed_suffix+'_training_loss', 'wb'))
            pickle.dump(self.correct_data, open(self.file_path+'/'+self.seed_suffix+'_training_correct', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix)
            os.remove(self.file_path+'/'+model.model_name+'_'+self.seed_suffix+'_CHECKPOINT.pt')

        if mode=='TESTING': 
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+'_attrs', 'wb'))
            os.remove(self.file_path+'/'+self.model_file_path+'_CHECKPOINT_attrs')
            pickle.dump(self.loss_data, open(self.file_path+'/'+self.seed_suffix+'_training_loss', 'wb'))
            pickle.dump(self.correct_data, open(self.file_path+'/'+self.seed_suffix+'_training_correct', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix)
            os.remove(self.file_path+'/'+model.model_name+'_'+self.seed_suffix+'_CHECKPOINT.pt')

    def _init_streamer(self):
        self.streamer = TaskDataSet(True, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts)

    def _init_optimizer(self, model):
        if model.is_instruct:
            if self.lang_lr is None: langLR = self.lr 
            optimizer = self.optim_alg([
                    {'params' : model.recurrent_units.parameters()},
                    {'params' : model.sensory_motor_outs.parameters()},
                    {'params' : model.langModel.parameters(), 'lr': langLR}
                ], lr=self.lr, weight_decay=self.weight_decay)
        else: 
            optimizer = self.optim_alg(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.optimizer = optimizer
        self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)

    def _save_for_tuning(self, model): 
        model_for_tuning = copy.deepcopy(model)
        cur_data_dict = {'correct_data': self.correct_data, 'loss_data': self.loss_data}
        pickle.dump(cur_data_dict, open(self.file_path+'/'+self.seed_suffix+'training_data_FOR_TUNING', 'wb'))
        model_for_tuning.save_model(self.file_path, suffix='_'+self.seed_suffix+'_FOR_TUNING')
        print('\n MODEL SAVED FOR TUNING')

    def train(self, model, is_tuning=False, is_testing=False): 
        model.to(device)
        model.train()
        self._init_streamer()

        self.model_file_path = model.model_name+'_'+self.seed_suffix
        print('\n TRAINING MODEL '+self.model_file_path + ' on holdouts ' + str(self.holdouts)+ '\n')

        if not is_tuning and model.is_instruct: 
            model.langModel.eval()

        if self.cur_epoch>0: 
            print('Resuming Training, Current lr: ' + str(self.scheduler.get_lr()))
        else:
            self._init_optimizer(model)

        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            if self.cur_epoch == self.save_for_tuning_epoch and model.is_instruct:
                self._save_for_tuning(model)

            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data
                self.optimizer.zero_grad()
                task_info = get_task_info(self.batch_len, task_type, model.is_instruct)
                out, _ = model(ins.to(device), task_info)
                loss = masked_MSE_Loss(out, tar.to(device), mask.to(device)) 
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
                self.optimizer.step()

                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                self._log_step(task_type, frac_correct, loss.item())

                if self.cur_step%50 == 0 and not is_testing:
                    self._record_session(model, mode='CHECKPOINT')
                    self._print_training_status(task_type)

                if self._check_model_training() and not is_testing:
                    self._record_session(model, mode='FINAL')
                    return True


            if self.scheduler is not None: self.scheduler.step()  
            if self.step_last_lr: self.step_scheduler.step()

        if is_testing:
            self._record_session(model, mode='TESTING')
        else:
            warnings.warn('Model has not reach specified performance threshold during training')
            return False

def check_already_trained(file_name, seed): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_training_correct', 'rb'))
        return True
    except FileNotFoundError:
        return False

def train_model_set(model_names, seeds, all_holdouts, overwrite=False, **train_config_kwargs): 
    inspection_list = []
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            for holdouts in all_holdouts: 
                file_name = EXP_FILE+'/'+get_holdout_file_name(holdouts)+'/'+model_name
                
                if check_already_trained(file_name, seed) and not overwrite:
                    print('Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
                    continue 
                
                model = make_default_model(model_name)
                trainer_config = TrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)
                trainer = ModelTrainer(trainer_config)
                is_trained = trainer.train(model)
                if not is_trained: inspection_list.append((model.model_name, seed, holdouts))
                del trainer
                del model
                gc.collect()

    return inspection_list

def tune_model_set(model_names, seeds, all_holdouts, overwrite=False, **train_config_kwargs): 
    inspection_list = []
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            for holdouts in all_holdouts: 
                file_name = EXP_FILE+'/'+get_holdout_file_name(holdouts)+'/'+model_name
                
                if check_already_trained(file_name, seed) and not overwrite:
                    print('Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
                    continue 
                
                model = make_default_model(model_name)
                model.load_model(file_name+'/'+model_name.strip('_tuned'), suffix='_seed'+str(seed)+'_FOR_TUNING')
                training_data_checkpoint = pickle.load(open(file_name+'/seed'+str(seed)+'training_data_FOR_TUNING', 'rb'))
                tuning_config = TrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)
                trainer = ModelTrainer(tuning_config, from_checkpoint_dict=training_data_checkpoint)
                is_trained = trainer.train(model, is_tuning=True)
                if not is_trained: inspection_list.append((model.model_name, seed, holdouts))
                del trainer
                del model
                gc.collect()

    return inspection_list


def test_model_set(model_names, seeds, all_holdouts, overwrite=False, **train_config_kwargs): 
    inspection_list = []
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            for holdouts in all_holdouts: 
                file_name = EXP_FILE+'/'+get_holdout_file_name(holdouts)+'/'+model_name
                
                if check_already_trained(file_name, seed) and not overwrite:
                    print('Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
                    continue 
                
                model = make_default_model(model_name)
                model.load_model(file_name+'/'+model_name, suffix='_seed'+str(seed))
                tuning_config = TrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)
                trainer = ModelTrainer(tuning_config)
                is_trained = trainer.train(model, is_testing=True)
                if not is_trained: inspection_list.append((model.model_name, seed, holdouts))
                del trainer
                del model
                gc.collect()

    return inspection_list




#save training data when checkpointing!
train_model_set(['sbertNet', 'bertNet', 'gptNet', 'simpleNet', 'simpleNetPlus'], range(5), training_lists_dict['aligned_holdouts'])            






# def tune_model(model, holdouts, epochs, holdout_file): 
#     if 'tuned' not in model.model_name: model.model_name = model.model_name+'_tuned'

#     model.load_model(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#     model.load_training_data(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#     print('model loaded:'+EXP_FILE+'/'+holdout_type+'/'+holdout_file+'\n')
#     model.langModel.train_layers=['11', '10', '9']
#     model.langModel.init_train_layers()
    
#     tried_tuning = tuning_check(model, holdouts)
#     print('tried tuning= '+str(tried_tuning))
    
#     if not tried_tuning: 
#         model.to(device)
#         data = TaskDataSet(holdouts=holdouts)

#         tune_opt_params = ALL_MODEL_PARAMS[model.model_name]['tune_opt_params']

#         opt = init_optimizer(model, tune_opt_params['init_lr'], langLR=tune_opt_params['init_lang_lr'])
#         sch = optim.lr_scheduler.ExponentialLR(opt, tune_opt_params['exp_gamma'])
#         step_params = {'milestones':[epochs-2, epochs-1], 'gamma': tune_opt_params['step_gamma']}

#         is_tuned, _ = train_model(model, data, epochs, opt, sch, step_params, tuning=True)
#         return is_tuned

#     elif model.check_model_training(0.95, 5):
#         return 'MODEL ALREADY TUNED'
    
#     else:
#         raise ValueError('examine current model, tuning tried and saved but threshold not reached')
    
# def test_model(model, holdouts_test, foldername, repeats=5, holdout_type = 'single_holdouts', save=False): 
#     holdout_file = get_holdout_file(holdouts_test)
#     model.eval()
#     for holdout in holdouts_test: 
#         for _ in range(repeats): 
#             model.load_model(foldername+'/'+holdout_type+'/'+holdout_file)
#             opt = init_optimizer(model, 0.0007)
#             step_params = {'milestones':[], 'gamma': 0}

#             data = TaskDataSet(batch_len=256, num_batches=100, task_ratio_dict={holdout:1})
#             train_model(model, data, 1, opt, None, step_params, testing=True)
        
#         correct_perf = np.mean(np.array(model._correct_data_dict[holdout]).reshape(repeats, -1), axis=0)
#         loss_perf = np.mean(np.array(model._loss_data_dict[holdout]).reshape(repeats, -1), axis=0)

#         if save: 
#             pickle.dump(correct_perf, open(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_correct', 'wb'))
#             pickle.dump(loss_perf, open(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_loss', 'wb'))
#             print(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_correct')

#     return correct_perf, loss_perf

# def train_model_set(model_configs, EXP_FILE, save_bool):
#     inspection_list = []
#     for config in model_configs:      
#         model_params_key, seed_num, holdouts = config
#         torch.manual_seed(seed_num)
#         holdout_file = get_holdout_file(holdouts)

#         print(config)
#         data = TaskDataSet(holdouts=holdouts)
#         #data.data_to_device(device)

#         model = config_model(model_params_key)
#         model.set_seed(seed_num)

#         #train 
#         if holdouts == ['Multitask']: 
#             eps = 55
#             checkpoint=10
#         else: 
#             eps = 35 
#             checkpoint=5

#         if model.model_name=='simpleNet' or model.model_name=='bowNet':
#             checkpoint=np.inf

#         opt = init_optimizer(model, 0.001)
#         sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
#         step_params = {'milestones':[eps-10, eps-2, eps-1], 'gamma': 0.5}

#         is_trained, model_for_tuning = train_model(model, data, eps, opt, sch, step_params, checkpoint_for_tuning=checkpoint)
#         is_trained=True
#         if is_trained and save_bool:
#             print('Model Trained '+str(is_trained))
#             model.save_model(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#             model.save_training_data(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#             try: 
#                 model_for_tuning.save_model(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#                 model_for_tuning.save_training_data(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#                 print('Model for tuning saved')
#             except AttributeError:
#                 print('Model for Tuning not saves, model type: ' + str(type(model_for_tuning)))
#         else: 
#             inspection_list.append(config)
        
#         print(inspection_list)

#     return inspection_list
    
# def tune_model_set(model_configs, EXP_FILE, save_bool):                
#     inspection_list = []
#     for config in model_configs: 
#         model_params_key, seed_num, holdouts = config
#         torch.manual_seed(seed_num)
#         holdout_file = get_holdout_file(holdouts)
        
#         print(config)
#         model = config_model(model_params_key)
#         model.set_seed(seed_num)

#         if holdouts == ['Multitask']: eps = 15
#         else: eps = 10
        
#         is_tuned = tune_model(model, holdouts, eps, holdout_file)
#         print('Model Tune= '+str(is_tuned))

#         if is_tuned == 'MODEL ALREADY TUNED':
#             print('model already tuned... loading next model')
#             continue

#         if is_tuned and save_bool: 
#             model.save_model(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#             model.save_training_data(EXP_FILE+'/'+holdout_type+'/'+holdout_file)
#             print('model saved!')
#         elif not is_tuned: 
#             inspection_list.append(config)
#         elif is_tuned: 
#             print('tuned and not saved')

#         print(inspection_list)

#     return inspection_list        

# def test_model_set(model_configs, EXP_FILE, save_bool):
#     for config in model_configs: 
#         print(config)
#         instruct_mode, model_params_key, seed_num, holdouts = config
#         torch.manual_seed(seed_num)
#         model = config_model(model_params_key)
#         model.set_seed(seed_num)
#         model.instruct_mode = instruct_mode
#         model.model_name 
#         model.to(device)

#         test_model(model, holdouts, repeats=5, foldername= EXP_FILE, holdout_type = holdout_type, save=save_bool)

# if __name__ == "__main__":

#     #train_mode = str(sys.argv[1])
#     train_mode = 'train'
    
#     print('Mode: ' + train_mode + '\n')

#     if train_mode == 'test': 

#         to_test = list(itertools.product(['', 'swap'], ['bertNet_tuned'], [2], [['MultiDM', 'DNMS'], ['COMP2', 'DMS']]))
#         #to_test1 = list(itertools.product(['', 'swap'], ['gptNet'], [0], training_lists_dict['swap_holdouts']))
#         #to_test1 = list(itertools.product(['', 'swap'], ['gptNet'], [0], training_lists_dict['swap_holdouts']))
#         #to_test1 = list(itertools.product(['', 'swap'], ['gptNet_tuned'], [0, 1, 2, 3, 4], training_lists_dict['swap_holdouts']))
#         #to_test = list(itertools.product(['', 'swap'], ['bertNet_tuned'], [1, 2, 3, 4], training_lists_dict['swap_holdouts']))
#         print(to_test)

#         inspection_list = test_model_set(to_test, EXP_FILE, save_bool=True)
#         print(inspection_list)

#     if train_mode == 'fine_tune': 
#         #to_tune = list(itertools.product(['gptNet'], seeds,training_lists_dict['swap_holdouts']))
#         #to_tune = [('gptNet', 0, ['MultiDM', 'DNMS'])]
#         #to_tune = list(itertools.product(['bertNet'], seeds,training_lists_dict['swap_holdouts']))
#         to_tune=[('gptNet', 0, ['Multitask']), ('gptNet', 3, ['Multitask'])]


#         print(to_tune)
#         inspection_list = tune_model_set(to_tune, EXP_FILE)
#         print(inspection_list)

#     if train_mode =='train': 
#         #to_train = [('gptNeoNet', 0, ['Multitask'])]     
#         to_train = list(itertools.product(['gptNeoNet'], [0, 1, 2, 3, 4] , [['Multitask']]))
   
#         print(to_train)
#         print(len(to_train))
#         inspection_list = train_model_set(to_train, EXP_FILE, save_bool=True)
#         print(inspection_list)


#     if train_mode == 'check_train':
#         seeds = [0, 1, 2, 3, 4]
#         to_check = list(itertools.product(['bertNet_tuned'], seeds, training_lists_dict['swap_holdouts']+[['Multitask']]))
#         to_retrain, to_tune = check_model_set(to_check)
#         print(to_retrain)
#         print(to_tune)
