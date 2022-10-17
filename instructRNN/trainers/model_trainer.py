from email.policy import strict
import torch
from torch import optim
import numpy as np
from yaml import warnings

import pickle
import copy
from tqdm import tqdm
from attrs import define, asdict
import os
import warnings
from os.path import exists
from instructRNN.analysis.model_analysis import task_eval

from instructRNN.trainers.base_trainer import *
from instructRNN.data_loaders.dataset import *
from instructRNN.tasks.task_criteria import *
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.models.full_models import make_default_model

if torch.cuda.is_available:
    device = torch.device(0)
    print(torch.cuda.get_device_name(device), flush=True)
else: 
    device = torch.device('cpu')
    
@define
class TrainerConfig(): 
    file_path: str
    random_seed: int
    epochs: int = 200
    min_run_epochs: int = 35
    batch_len: int = 64
    num_batches: int = 2400
    holdouts: list = []
    set_single_task: str = None
    stream_data: bool = True

    optim_alg: str = 'adam'
    init_lr: float = 0.001
    init_lang_lr: float = None
    weight_decay: float = 0.0

    scheduler_type: str = 'exp'
    scheduler_gamma: float = 0.95
    scheduler_args: dict = {}

    save_for_tuning_epoch: int = 30
    checker_threshold: float = 0.95
    step_last_lr: bool = True
    test_repeats: int = None


class ModelTrainer(BaseTrainer): 
    def __init__(self, training_config:TrainerConfig): 
        super().__init__(training_config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path): 
        attr_dict = pickle.load(open(checkpoint_path+'_attrs', 'rb'))
        config = TrainerConfig(**attr_dict.pop('config_dict'))
        cls = cls(config) 
        cls.checkpoint_path = checkpoint_path
        for name, value in attr_dict.items(): 
            setattr(cls, name, value)
        return cls

    def _record_session(self, model, mode='CHECKPOINT'): 
        checkpoint_attrs = super()._record_session()

        if os.path.exists(self.file_path):pass
        else: os.makedirs(self.file_path)
        
        if os.path.exists(self.file_path+'/attrs'):pass
        else: os.makedirs(self.file_path+'/attrs')

        if mode == 'CHECKPOINT':
            chk_attr_path = self.file_path+'/attrs/'+self.model_file_path+'_CHECKPOINT'
            pickle.dump(checkpoint_attrs, open(chk_attr_path+'_attrs', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix+'_CHECKPOINT')
            torch.save(self.optimizer.state_dict(), chk_attr_path+'_opt')

        if mode=='FINAL': 
            data_path = self.file_path+'/'+self.seed_suffix+'_training'
            pickle.dump(checkpoint_attrs.pop('loss_data'), open(data_path+'_loss', 'wb'))
            pickle.dump(checkpoint_attrs.pop('correct_data'), open(data_path+'_correct', 'wb'))

            attrs_path = self.file_path+'/attrs/'+self.model_file_path
            pickle.dump(checkpoint_attrs, open(attrs_path+'_attrs', 'wb'))
            os.remove(attrs_path+'_CHECKPOINT_attrs')

            model.save_model(self.file_path, suffix='_'+self.seed_suffix)
            os.remove(self.file_path+'/'+self.model_file_path+'_CHECKPOINT.pt')

        if mode=='TESTING': 
            self.average_testing_data()
            task= self.set_single_task
            tests_path = self.file_path+'/holdouts'
            test_info = self.test_name_str+task +'_'+self.seed_suffix
            if os.path.exists(tests_path):pass
            else: os.makedirs(tests_path)
            pickle.dump(self.loss_data, open(tests_path+'/'+test_info+'_loss', 'wb'))
            pickle.dump(self.correct_data, open(tests_path+'/'+test_info+'_correct', 'wb'))

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
        if self.optim_alg == 'adam': 
            optim_alg = optim.Adam

        if model.info_type=='lang':
            if self.init_lang_lr is None: langLR = self.init_lr 
            else: langLR = self.init_lang_lr
            optimizer = optim_alg([
                    {'params' : model.recurrent_units.parameters()},
                    {'params' : model.sensory_motor_outs.parameters()},
                    {'params' : model.langModel.parameters(), 'lr': langLR}
                ], lr=self.init_lr, weight_decay=self.weight_decay)
        else: 
            optimizer = optim_alg(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        self.optimizer = optimizer

    def _init_scheduler(self):
        if self.scheduler_type == 'exp': 
            scheduler_class = optim.lr_scheduler.ExponentialLR

        self.scheduler = scheduler_class(self.optimizer, gamma=self.scheduler_gamma, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)

    def _save_for_tuning(self, model): 
        model_for_tuning = copy.deepcopy(model)
        cur_data_dict = {'correct_data': self.correct_data, 'loss_data': self.loss_data}
        pickle.dump(cur_data_dict, open(self.file_path+'/'+self.seed_suffix+'training_data_FOR_TUNING', 'wb'))
        model_for_tuning.save_model(self.file_path, suffix='_'+self.seed_suffix+'_FOR_TUNING')
        print('\n MODEL SAVED FOR TUNING')

    def _set_training_conditions(self, model, is_tuning, is_testing, instruct_mode, weight_mode):

        self.model_file_path = model.model_name+'_'+self.seed_suffix
        tunable = (model.info_type == 'lang' and hasattr(model.langModel, 'transformer'))

        if not is_tuning and model.info_type=='lang': 
            model.langModel.eval()
        
        if is_testing and tunable:
            model.langModel.freeze_transformer()

        if is_testing and weight_mode=='input_only': 
            model.freeze_all_but_rnn_ins()

        if is_testing:
            if instruct_mode is None: instruct_mode = ''
            if weight_mode is None: weight_mode = ''
            self.test_name_str= instruct_mode + weight_mode
        elif not is_testing and instruct_mode is not None: 
            warnings.warn('instruct mode is not standard but doing something other than testing')
            

        return tunable

    def init_optimizer(self, model):
        self._init_optimizer(model)      
        self._init_scheduler()
        if hasattr(self, 'checkpoint_path'):
            opt_path = self.checkpoint_path + '_opt'
            self.optimizer.load_state_dict(torch.load(opt_path))  

    def train(self, model, is_tuning=False, is_testing=False, instruct_mode=None, weight_mode = None): 
        model.to(device)
        model.train()
        self._init_streamer()
        self.init_optimizer(model)
        tunable = self._set_training_conditions(model, is_tuning, is_testing, instruct_mode, weight_mode)

        for self.cur_epoch in tqdm(range(self.cur_epoch, self.epochs), desc='epochs'):
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

                if self._check_model_training() and not is_testing:
                    self._record_session(model, mode='FINAL')
                    return True

            if not is_testing:
                self._record_session(model, mode='CHECKPOINT')

            if self.scheduler is not None: self.scheduler.step()  
            if self.step_last_lr: self.step_scheduler.step()

        if not is_testing:
            warnings.warn('\n !!! Model has not reach specified performance threshold during training !!! \n')
            return False

def check_already_trained(file_name, seed, mode='training'): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_training_correct', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
        return True
    except FileNotFoundError:
        print('\n ' + mode+' at ' + file_name + ' at seed '+str(seed))
        return False

def check_already_tested(file_name, seed, task, instruct_mode, weight_mode):
    if instruct_mode is None: instruct_mode = ''
    if weight_mode is None: weight_mode = ''
    mode = instruct_mode+weight_mode
    try: 
        pickle.load(open(file_name+'/holdouts/'+ mode+task+ '_seed'+str(seed)+'_correct', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+ 'and task '+task+' aleady trained')
        return True
    except FileNotFoundError:
        return False

def load_checkpoint(model, file_name, seed): 
    checkpoint_name = model.model_name+'_seed'+str(seed)+'_CHECKPOINT'
    checkpoint_model_path = file_name+'/'+checkpoint_name+'.pt'

    print('\n Attempting to load model CHECKPOINT')
    if not exists(checkpoint_model_path): 
        raise Exception('No model checkpoint found at ' + checkpoint_model_path)
    else:
        print(checkpoint_model_path)
        model.load_state_dict(torch.load(checkpoint_model_path), strict=False)
        print('loaded model at '+ checkpoint_model_path)
    
    checkpoint_path = file_name+'/attrs/'+checkpoint_name
    trainer = ModelTrainer.from_checkpoint(checkpoint_path)
    return model, trainer

def load_tuning_checkpoint(model, trainer, file_name, seed):
    untuned_model_name = model.model_name.replace('_tuned', '')
    for_tuning_model_path = file_name+'/'+untuned_model_name+'/'+\
            untuned_model_name+'_seed'+str(seed)+'_FOR_TUNING.pt'

    print('\n Attempting to load model FOR_TUNING')
    if not exists(for_tuning_model_path): 
        raise Exception('No model FOR TUNING found, train untuned version to create checkpoint \n')
    else:
        print('loaded model at '+ for_tuning_model_path)

    model.load_state_dict(torch.load(for_tuning_model_path), strict=False)
    data_checkpoint_path = file_name+'/'+untuned_model_name+\
            '/seed'+str(seed)+'training_data_FOR_TUNING'
    data_checkpoint = pickle.load(open(data_checkpoint_path, 'rb'))
    trainer.correct_data = data_checkpoint['correct_data']
    trainer.loss_data = data_checkpoint['loss_data']

    return model, trainer

def train_model(exp_folder, model_name, seed, labeled_holdouts, use_checkpoint=False, overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)
    label, holdouts = labeled_holdouts
    file_name = exp_folder+'/'+label+'/'+model_name   

    if check_already_trained(file_name, seed) and not overwrite:
        return True
    
    model = make_default_model(model_name)

    if model_name == 'gptNet_lin':
        trainer_config = TrainerConfig(file_name, seed, holdouts=holdouts, checker_threshold=0.85, scheduler_gamma=0.95, **train_config_kwargs)
    else:
        trainer_config = TrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)

    if use_checkpoint: 
        try:
            model, trainer = load_checkpoint(model, file_name, seed)
        except: 
            print('Starting Training from untrained model')
            trainer = ModelTrainer(trainer_config)
    else: 
        trainer = ModelTrainer(trainer_config)

    is_trained = trainer.train(model)
    return is_trained

def tune_model(exp_folder, model_name, seed, labeled_holdouts, overwrite=False, use_checkpoint=False, **train_config_kwargs): 
    assert 'tuned' in model_name
    torch.manual_seed(seed)
    label, holdouts = labeled_holdouts
    file_name = exp_folder+'/'+label

    if check_already_trained(file_name+'/'+model_name, seed, 'tuning') and not overwrite:
        return True

    model = make_default_model(model_name)

    if 'gptNetXL' in model_name:
        tuning_config = TrainerConfig(file_name+'/'+model_name, seed, holdouts=holdouts, batch_len=64,
                                            epochs=60, min_run_epochs=5, init_lr=1e-5, init_lang_lr=1e-5, scheduler_gamma=0.99,
                                            save_for_tuning_epoch=np.nan, 
                                            **train_config_kwargs)

    elif 'gptNet_lin' in model_name:
        tuning_config = TrainerConfig(file_name+'/'+model_name, seed, holdouts=holdouts, batch_len=64,
                                            epochs=60, min_run_epochs=5, init_lr=1e-5, init_lang_lr=1e-5, scheduler_gamma=0.99,
                                            save_for_tuning_epoch=np.nan, checker_threshold=0.85,
                                            **train_config_kwargs)
    else: 
        tuning_config = TrainerConfig(file_name+'/'+model_name, seed, holdouts=holdouts, batch_len=64,
                                            epochs=60, min_run_epochs=5, init_lr=1e-5, init_lang_lr=1e-4, scheduler_gamma=0.99,
                                            save_for_tuning_epoch=np.nan, 
                                            **train_config_kwargs)
    trainer = ModelTrainer(tuning_config)

    if use_checkpoint: 
        try: 
            model, trainer = load_checkpoint(model, file_name+'/'+model.model_name, seed)
        except: 
            print('couldnt load checkpoint, starting from standard tuning checkpoint')
            model, trainer = load_tuning_checkpoint(model, trainer, file_name, seed)
    else: 
        model, trainer = load_tuning_checkpoint(model, trainer, file_name, seed)

    is_tuned = trainer.train(model, is_tuning=True)
    return is_tuned

def test_model(exp_folder, model_name, seed, labeled_holdouts, instruct_mode = None, weight_mode = None, overwrite=False, repeats=5, **train_config_kwargs): 
    torch.manual_seed(seed)
    
    label, holdouts = labeled_holdouts 
    file_name = exp_folder+'/'+label+'/'+model_name
                
    model = make_default_model(model_name)

    for task in holdouts: 
        if check_already_tested(file_name, seed, task, instruct_mode, weight_mode) and not overwrite:
            continue 
        else:
            print('\n testing '+model_name+' seed'+str(seed)+' on '+task)

        testing_config = TrainerConfig(file_name, seed, set_single_task=task, 
                                batch_len=256, num_batches=100, epochs=1, init_lr = 0.0008,
                                test_repeats = repeats, **train_config_kwargs)
        trainer = ModelTrainer(testing_config)
        for _ in range(repeats): 
            model.load_model(file_name, suffix='_seed'+str(seed))
            trainer.train(model, is_testing=True, instruct_mode=instruct_mode, weight_mode=weight_mode)
        trainer._record_session(model, mode='TESTING')

def run_pipeline(exp_folder, model_name, seed, labeled_holdouts, overwrite=False, ot=False, use_checkpoint=False, **train_config_kwargs):
    if not '_tuned' in model_name:
        is_trained = train_model(exp_folder, model_name, seed, labeled_holdouts, use_checkpoint = use_checkpoint, overwrite=overwrite, **train_config_kwargs) 
    else: 
        is_trained = tune_model(exp_folder, model_name, seed, labeled_holdouts, use_checkpoint = use_checkpoint, overwrite=overwrite, **train_config_kwargs)
        
    if is_trained: 
        for instruct_mode in [None, 'swap', 'combined', 'swap_combined']:
            test_model(exp_folder, model_name, seed, labeled_holdouts, instruct_mode = instruct_mode, overwrite=ot)
        