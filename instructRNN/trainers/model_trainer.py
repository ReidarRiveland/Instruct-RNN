import torch
from torch import optim
import numpy as np
from yaml import warnings

import pickle
import copy
from tqdm import tqdm
from attrs import define
import os
import warnings
from os.path import exists

from instructRNN.trainers.base_trainer import *
from instructRNN.data_loaders.dataset import *
from instructRNN.tasks.task_criteria import *
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.models.full_models import make_default_model

device = torch.device(0)
print(torch.cuda.get_device_name(device), flush=True)

@define
class TrainerConfig(): 
    file_path: str
    random_seed: int
    epochs: int = 100
    min_run_epochs: int = 35
    batch_len: int = 64
    num_batches: int = 2400
    holdouts: list = []
    set_single_task: str = None
    stream_data: bool = False

    optim_alg: optim = optim.Adam
    lr: float = 0.001
    lang_lr: float = None
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_gamma: float = 0.9
    scheduler_args: dict = {}

    save_for_tuning_epoch: int = 30
    checker_threshold: float = 0.95
    step_last_lr: bool = True
    test_repeats: int = None

class ModelTrainer(BaseTrainer): 
    def __init__(self, training_config:TrainerConfig=None, from_checkpoint_dict:dict=None): 
        super().__init__(training_config, from_checkpoint_dict)

    def _record_session(self, model, mode='CHECKPOINT'): 
        checkpoint_attrs = super()._record_session()
        if os.path.exists(self.file_path):pass
        else: os.makedirs(self.file_path)
        if os.path.exists(self.file_path+'/attrs'):pass
        else: os.makedirs(self.file_path+'/attrs')

        if mode == 'CHECKPOINT':
            pickle.dump(checkpoint_attrs, open(self.file_path+'/attrs/'+self.model_file_path+'_CHECKPOINT_attrs', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix+'_CHECKPOINT')

        if mode=='FINAL': 
            pickle.dump(checkpoint_attrs.pop('loss_data'), open(self.file_path+'/'+self.seed_suffix+'_training_loss', 'wb'))
            pickle.dump(checkpoint_attrs.pop('correct_data'), open(self.file_path+'/'+self.seed_suffix+'_training_correct', 'wb'))

            pickle.dump(checkpoint_attrs, open(self.file_path+'/attrs/'+self.model_file_path+'_attrs', 'wb'))
            os.remove(self.file_path+'/attrs/'+self.model_file_path+'_CHECKPOINT_attrs')

            model.save_model(self.file_path, suffix='_'+self.seed_suffix)
            os.remove(self.file_path+'/'+self.model_file_path+'_CHECKPOINT.pt')

        if mode=='TESTING': 
            self.average_testing_data()
            tests_path = self.file_path+'/holdouts'
            if os.path.exists(tests_path):pass
            else: os.makedirs(tests_path)
            task= self.set_single_task
            pickle.dump(checkpoint_attrs, open(self.file_path+'/attrs/'+self.instruct_str+self.seed_suffix+'_'+task+'_attrs', 'wb'))
            pickle.dump(self.loss_data, open(tests_path+'/'+self.instruct_str+task +'_'+self.seed_suffix+'_loss', 'wb'))
            pickle.dump(self.correct_data, open(tests_path+'/'+self.instruct_str+task+'_'+self.seed_suffix+'_correct', 'wb'))

    def average_testing_data(self):
        correct_array = np.array(self.correct_data[self.set_single_task]).reshape(self.test_repeats, -1)
        loss_array = np.array(self.correct_data[self.set_single_task]).reshape(self.test_repeats, -1)
        self.correct_data = list(correct_array.mean(axis=0))
        self.loss_data = list(loss_array.mean(axis=0))

    def _init_streamer(self):
        self.streamer = TaskDataSet(self.file_path.partition('/')[0], 
                        self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts, 
                        self.set_single_task)

    def _init_optimizer(self, model):
        if model.info_type=='lang':
            if self.lang_lr is None: langLR = self.lr 
            else: langLR = self.lang_lr
            optimizer = self.optim_alg([
                    {'params' : model.recurrent_units.parameters()},
                    {'params' : model.sensory_motor_outs.parameters()},
                    {'params' : model.langModel.parameters(), 'lr': langLR}
                ], lr=self.lr, weight_decay=self.weight_decay)
        else: 
            optimizer = self.optim_alg(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.optimizer = optimizer
        self.scheduler = self.scheduler_class(self.optimizer, gamma=self.scheduler_gamma, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)

    def _save_for_tuning(self, model): 
        model_for_tuning = copy.deepcopy(model)
        cur_data_dict = {'correct_data': self.correct_data, 'loss_data': self.loss_data}
        pickle.dump(cur_data_dict, open(self.file_path+'/'+self.seed_suffix+'training_data_FOR_TUNING', 'wb'))
        model_for_tuning.save_model(self.file_path, suffix='_'+self.seed_suffix+'_FOR_TUNING')
        print('\n MODEL SAVED FOR TUNING')

    def _set_training_conditions(self, model, is_tuning, is_testing, instruct_mode):
        if instruct_mode is not None and is_testing: 
            self.instruct_str = instruct_mode
        elif instruct_mode is None: 
            self.instruct_str = ''
        else: 
            warnings.warn('instruct mode is not standard but doing something other than testing')
            
        self.model_file_path = model.model_name+'_'+self.seed_suffix

        if not is_tuning and model.info_type=='lang': 
            model.langModel.eval()
        
        if is_testing and hasattr(model.langModel, 'transformer'):
            model.langModel.freeze_transformer()

        if self.cur_epoch>0: 
            print('Resuming Training, Current lr: ' + str(self.scheduler.get_lr()))
        else:
            self._init_optimizer(model)
        tunable = (model.info_type == 'lang' and hasattr(model.langModel, 'transformer'))
        return tunable

    def train(self, model, is_tuning=False, is_testing=False, instruct_mode=None): 
        model.to(device)
        model.train()
        self._init_streamer()
        tunable = self._set_training_conditions(model, is_tuning, is_testing, instruct_mode)
        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            if self.cur_epoch == self.save_for_tuning_epoch and tunable:
                self._save_for_tuning(model)

            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data
                self.optimizer.zero_grad()
                task_info = get_task_info(self.batch_len, task_type, model.info_type, instruct_mode=instruct_mode)
                out, _ = model(ins.to(device), task_info)
                loss = masked_MSE_Loss(out, tar.to(device), mask.to(device)) 
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
                self.optimizer.step()

                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                self._log_step(task_type, frac_correct, loss.item())

                if self.cur_step%50 == 0:
                    self._print_training_status(task_type)
                    if not is_testing:
                        self._record_session(model, mode='CHECKPOINT')

                if self._check_model_training() and not is_testing:
                    self._record_session(model, mode='FINAL')
                    return True

            if self.scheduler is not None: self.scheduler.step()  
            if self.step_last_lr: self.step_scheduler.step()

        if not is_testing:
            warnings.warn('\n !!! Model has not reach specified performance threshold during training !!! \n')
            return False

###change to model file directory exists

def check_already_trained(file_name, seed, mode='training'): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_training_correct', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
        return True
    except FileNotFoundError:
        print('\n ' + mode+' at ' + file_name + ' at seed '+str(seed))
        return False

def check_already_tested(file_name, seed, task, mode):
    if mode is None: mode = ''
    try: 
        pickle.load(open(file_name+'/holdouts/'+ mode+task+ '_seed'+str(seed)+'_correct', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+ 'and task '+task+' aleady trained')
        return True
    except FileNotFoundError:
        return False

def train_model(exp_folder, model_name, seed, labeled_holdouts, overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)

    label, holdouts = labeled_holdouts
    file_name = exp_folder+'/'+label+'/'+model_name    
    if check_already_trained(file_name, seed) and not overwrite:
        return True
    
    model = make_default_model(model_name)
    trainer_config = TrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)
    trainer = ModelTrainer(trainer_config)
    is_trained = trainer.train(model)
    return is_trained

def tune_model(exp_folder, model_name, seed, labeled_holdouts, overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)
    
    label, holdouts = labeled_holdouts
    untuned_model_name = model_name.replace('_tuned', '')
    file_name = exp_folder+'/'+label
    
    if check_already_trained(file_name+'/'+model_name, seed, 'tuning') and not overwrite:
        return True
    
    for_tuning_model_path = file_name+'/'+untuned_model_name+'/'+\
                untuned_model_name+'_seed'+str(seed)+'_FOR_TUNING.pt'
    for_tuning_data_path = file_name+'/'+untuned_model_name+\
                '/seed'+str(seed)+'training_data_FOR_TUNING'

    print('\n Attempting to load model checkpoint for tuning')
    if not exists(for_tuning_model_path): 
        raise Exception('No model checkpoint for tuning found, train untuned version to create checkpoint \n')
    else:
        pass

    model = make_default_model(model_name)
    model.load_state_dict(torch.load(for_tuning_model_path))

    training_data_checkpoint = pickle.load(open(for_tuning_data_path, 'rb'))
    tuning_config = TrainerConfig(file_name+'/'+model_name, seed, holdouts=holdouts, 
                                    epochs=25, min_run_epochs=5, lr=2e-5, lang_lr=1e-5,
                                    save_for_tuning_epoch=np.nan, 
                                    **train_config_kwargs)

    trainer = ModelTrainer(tuning_config, from_checkpoint_dict=training_data_checkpoint)
    is_tuned = trainer.train(model, is_tuning=True)
    # if is_tuned:
    #     os.remove(for_tuning_model_path)
    #     os.remove(for_tuning_data_path)
    return is_tuned

def test_model(exp_folder, model_name, seed, labeled_holdouts, mode = None, overwrite=False, repeats=5, **train_config_kwargs): 
    torch.manual_seed(seed)
    
    label, holdouts = labeled_holdouts 
    file_name = exp_folder+'/'+label+'/'+model_name
                
    model = make_default_model(model_name)
    for task in holdouts: 
        if check_already_tested(file_name, seed, task, mode) and not overwrite:
            continue 
        else:
            print('\n testing '+model_name+' seed'+str(seed)+' on '+task)

        testing_config = TrainerConfig(file_name, seed, set_single_task=task, 
                                batch_len=256, num_batches=100, epochs=1, lr = 0.0008,
                                test_repeats = repeats, **train_config_kwargs)
        trainer = ModelTrainer(testing_config)
        for _ in range(repeats): 
            model.load_model(file_name, suffix='_seed'+str(seed))
            trainer.train(model, is_testing=True, instruct_mode=mode)
        trainer._record_session(model, mode='TESTING')


def run_pipeline(exp_folder, model_name, seed, labeled_holdouts, overwrite=False, **train_config_kwargs):
    if not '_tuned' in model_name:
        is_trained = train_model(exp_folder, model_name, seed, labeled_holdouts, overwrite=overwrite, **train_config_kwargs) 
    else: 
        is_trained = tune_model(exp_folder, model_name, seed, labeled_holdouts, overwrite=overwrite, **train_config_kwargs)
        
    if is_trained: 
        for test_mode in [None, 'swap']:
            test_model(exp_folder, model_name, seed, labeled_holdouts, mode = test_mode, overwrite=overwrite)


