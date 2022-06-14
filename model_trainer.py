import torch
from torch import Tensor, optim
import numpy as np
from yaml import warnings

from models.full_models import make_default_model
from base_trainer import BaseTrainer, masked_MSE_Loss
from data_loaders.dataset import TaskDataSet
from tasks.task_criteria import isCorrect
from instructions.instruct_utils import get_task_info

import pickle
import copy
from tqdm import tqdm
from attrs import define
import os
import warnings
import gc

device = torch.device(0)

@define
class TrainerConfig(): 
    file_path: str
    random_seed: int
    epochs: int = 100
    min_run_epochs: int = 35
    batch_len: int = 64
    num_batches: int = 1200
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

class ModelTrainer(BaseTrainer): 
    def __init__(self, training_config:TrainerConfig=None, from_checkpoint_dict:dict=None): 
        super().__init__(training_config, from_checkpoint_dict)

    def _record_session(self, model, mode='CHECKPOINT'): 
        checkpoint_attrs = super()._record_session()
        if os.path.exists(self.file_path):
            pass
        else: os.makedirs(self.file_path)

        if mode == 'CHECKPOINT':
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+'_CHECKPOINT_attrs', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix+'_CHECKPOINT')

        if mode=='FINAL': 
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+'_attrs', 'wb'))
            os.remove(self.file_path+'/'+self.model_file_path+'_CHECKPOINT_attrs')
            pickle.dump(self.loss_data, open(self.file_path+'/'+self.seed_suffix+'_training_loss', 'wb'))
            pickle.dump(self.correct_data, open(self.file_path+'/'+self.seed_suffix+'_training_correct', 'wb'))
            model.save_model(self.file_path, suffix='_'+self.seed_suffix)
            os.remove(self.file_path+'/'+self.model_file_path+'_CHECKPOINT.pt')

        if mode=='TESTING': 
            task_file_name = self.set_single_task.replace(' ', '_')
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.seed_suffix+task_file_name+'_holdout_attrs', 'wb'))
            pickle.dump(self.loss_data, open(self.file_path+'/'+task_file_name + '_'+self.seed_suffix+'_holdout_loss', 'wb'))
            pickle.dump(self.correct_data, open(self.file_path+'/'+task_file_name+'_'+self.seed_suffix+'_holdout_correct', 'wb'))

    def _init_streamer(self):
        self.streamer = TaskDataSet(MODEL_FOLDER+'/training_data', 
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

    def train(self, model, is_tuning=False, is_testing=False): 
        model.to(device)
        model.train()
        self._init_streamer()

        self.model_file_path = model.model_name+'_'+self.seed_suffix

        if not is_tuning and model.info_type=='lang': 
            model.langModel.eval()
        
        if is_testing and model.info_type=='lang':
            model.langModel.freeze_transformer()

        if self.cur_epoch>0: 
            print('Resuming Training, Current lr: ' + str(self.scheduler.get_lr()))
        else:
            self._init_optimizer(model)

        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            if self.cur_epoch == self.save_for_tuning_epoch and model.info_type=='lang':
                self._save_for_tuning(model)

            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data
                self.optimizer.zero_grad()
                task_info = get_task_info(self.batch_len, task_type, model.info_type)
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

        if is_testing:
            self._record_session(model, mode='TESTING')
        else:
            warnings.warn('\n !!! Model has not reach specified performance threshold during training !!! \n')
            return False

def check_already_trained(file_name, seed): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_training_correct', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
        return True
    except FileNotFoundError:
        print('\n Training at ' + file_name + ' at seed '+str(seed))
        return False

def check_already_tested(file_name, seed, task):
    task_file_name = task.replace(' ', '_')
    try: 
        pickle.load(open(file_name+'/'+ task_file_name + '_seed'+str(seed)+'_holdout_correct', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+ 'and task '+task+' aleady trained')
        return True
    except FileNotFoundError:
        return False

def train_model_set(model_names, seeds, label_holdout_list, overwrite=False, **train_config_kwargs): 
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            if '_tuned' in model_name: 
                warnings.warn('\n !!!TUNED MODELS SHOULD BE FINE TUNED USING THE TUNING FUNCTION!!!')
                continue
            for label, holdouts in label_holdout_list: 
                file_name = EXP_FOLDER+'/'+label+'/'+model_name
                
                if check_already_trained(file_name, seed) and not overwrite:
                    continue 
                
                model = make_default_model(model_name)
                trainer_config = TrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)
                trainer = ModelTrainer(trainer_config)
                trainer.train(model)


def tune_model_set(model_names, seeds, label_holdout_list, overwrite=False, **train_config_kwargs): 
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            if '_tuned' not in model_name: 
                warnings.warn('\n !!!UNTUNED MODELS SHOULD BE TRAINED USING THE TRAIN FUNCTION!!!')
                continue
            for label, holdouts in label_holdout_list: 
                untuned_model_name = model_name.replace('_tuned', '')
                file_name = EXP_FOLDER+'/'+label
                
                if check_already_trained(file_name+'/'+model_name, seed) and not overwrite:
                    continue 
                
                for_tuning_model_path = file_name+'/'+untuned_model_name+'/'+\
                            untuned_model_name+'_seed'+str(seed)+'_FOR_TUNING.pt'
                for_tuning_data_path = file_name+'/'+untuned_model_name+\
                            '/seed'+str(seed)+'training_data_FOR_TUNING'
            
                model = make_default_model(model_name)
                model.load_state_dict(torch.load(for_tuning_model_path))

                training_data_checkpoint = pickle.load(open(for_tuning_data_path, 'rb'))
                tuning_config = TrainerConfig(file_name+'/'+model_name, seed, holdouts=holdouts, 
                                                epochs=25, min_run_epochs=5, lr=2e-5, lang_lr=1e-5,
                                                save_for_tuning_epoch=np.nan, 
                                                **train_config_kwargs)

                trainer = ModelTrainer(tuning_config, from_checkpoint_dict=training_data_checkpoint)
                trainer.train(model, is_tuning=True)


def test_model_set(model_names, seeds, label_holdout_list, overwrite=False, **train_config_kwargs): 
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            for label, holdouts in label_holdout_list: 
                file_name = EXP_FOLDER+'/'+label+'/'+model_name
                
                model = make_default_model(model_name)
                model.load_model(file_name, suffix='_seed'+str(seed))
                for task in holdouts: 
                    if check_already_tested(file_name, seed, task) and not overwrite:
                        continue 
                    else:
                        print('\n testing '+model_name+' seed'+str(seed)+' on '+task)
                    tuning_config = TrainerConfig(file_name, seed, set_single_task=task, 
                                            batch_len=256, num_batches=100, epochs=1, lr = 0.0007,
                                            **train_config_kwargs)
                    trainer = ModelTrainer(tuning_config)
                    trainer.train(model, is_testing=True)




if __name__ == "__main__":
    import argparse
    from tasks.tasks import SWAPS_DICT
    
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', required=True)
    parser.add_argument('exp', required=True)
    parser.add_argument('-mode', default='train')
    parser.add_argument('-models', default=None, nargs='*')
    parser.add_argument('--holdouts', type=int, default=None,  nargs='*')
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args('6.7models swap_holdouts'.split())

    os.environ['MODEL_FOLDER'] = args.folder
    MODEL_FOLDER = args.model_folder
    EXP_FOLDER =MODEL_FOLDER+'/'+args.exp_type

    if args.holdout_set is None: holdouts = list(SWAPS_DICT.items())
    else: holdouts = list(SWAPS_DICT.items())[:]

    if args.model_name is None:
        holdouts = list(SWAPS_DICT.items())
    else:
        holdouts = list(SWAPS_DICT.items())[:]




    if args.mode == 'train': 
        train_model_set(args.model_name, np.range(args.num_seeds), list(SWAPS_DICT.items()), overwrite=args.overwrite)     
    if args.mode == 'tune': 
        tune_model_set(args.model_name,  
            [0], list(SWAPS_DICT.items()), overwrite=False)     
    if args.mode == 'test': 
        test_model_set(args.model_name, 
            [0], list(SWAPS_DICT.items()), overwrite=False)     

    # MODEL_FOLDER = '6.7models'
    # EXP_FOLDER =MODEL_FOLDER+'/swap_holdouts'

    # train_model_set(['simpleNet'],  
    #     [0], list(SWAPS_DICT.items()), overwrite=True, stream_data=False)     

    # train_model_set(['sbertNet'],  
    #     [0], [['Multitask','Multitask']], overwrite=False, stream_data=False)     
    
    # tune_model_set(['sbertNet_tuned'],  
    #     [0], [['Multitask','Multitask']], overwrite=True, stream_data=False)     

    # train_model_set(['gptNetXL'],  
    #     [0], [['Multitask','Multitask']], overwrite=False, stream_data=True)     

    # train_model_set(['sifNet'],  
    #     [0], [['Multitask','Multitask']], overwrite=False, stream_data=True)     
