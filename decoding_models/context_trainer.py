
from utils.utils import isCorrect, get_holdout_file, training_lists_dict, train_instruct_dict
from models.full_models import make_default_model
from base_trainer import masked_MSE_Loss, BaseTrainer
from dataset import TaskDataSet
from task import Task

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pickle
import itertools
from tqdm import tqdm
from attrs import define

device = torch.device(0)

EXP_FILE = '13.4models/swap_holdouts'

@define 
class ContextTrainerConfig(): 
    file_path: str
    random_seed: int

    context_dim = 20

    epochs: int = 60
    min_run_epochs: int = 20
    batch_len: int = 128
    num_batches: int = 500

    optim_alg: optim = optim.Adam
    lr: float = 0.001
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.999999999}

    checker_threshold: float = 0.95
    step_last_lr: bool = True

class ContextTrainer(BaseTrainer): 
    def __init__(self, context_training_config: ContextTrainerConfig = None, from_checkpoint_dict:dict = None): 
        super().__init__(context_training_config, from_checkpoint_dict)

    def _record_session(self, contexts, task):
        checkpoint_attrs = super()._record_session()
        filename = self.file_path+task+'supervised'
        pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+'_attrs', 'wb'))
        pickle.dump(contexts, open(filename+'_context_vecs'+str(self.context_dim), 'wb'))
        pickle.dump(self.model._correct_data_dict, open(filename+'_context_correct_data'+str(self.context_dim), 'wb'))
        pickle.dump(self.model._loss_data_dict, open(filename+'_context_loss_data'+str(self.context_dim), 'wb'))

    def _init_contexts(self): 
        context = nn.Parameter(torch.randn((self.num_contexts, self.context_dim), device=device))
        return context
    
    def _init_optimizer(self, context):
        self.optimizer = self.optim_alg([context], lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = self.scheduler_class(self.optimizer, **self.schedular_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)
        
    def train(self, model, task): 
        model.load_model(self.file_path, suffix='_'+self.seed_suffix)
        model.to(device)
        model.freeze_weights()
        model.eval()
        self.streamer = TaskDataSet(self.batch_len, 
                        self.num_batches,
                        set_single_task=task)
        contexts = self._init_contexts()

        self.model_file_path = model.model_name+'_'+self.seed_suffix
        print('\n TRAINING CONTEXTS for '+self.model_file_path + ' on holdouts ' + str(self.holdouts)+ '\n')

        if self.cur_epoch>0: 
            print('Resuming Training, Current lr: ' + str(self.scheduler.get_lr()))
        else:
            self._init_optimizer(model)

        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data
                self.optimizer.zero_grad()
                out, _ = model(ins.to(device), context=contexts)
                loss = masked_MSE_Loss(out, tar.to(device), mask.to(device)) 
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
                self.optimizer.step()

                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                self._log_step(task_type, frac_correct, loss.item())

                if self.cur_step%50 == 0:
                    self._print_training_status(task_type)

                if self._check_model_training():
                    self._record_session()

            if self.scheduler is not None: self.scheduler.step()  
            if self.step_last_lr: self.step_scheduler.step()

def check_already_trained(file_name, seed): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_training_correct', 'rb'))
        return True
    except FileNotFoundError:
        return False

def train_context_set(model_names, seeds, holdouts_folder, overwrite=False, **train_config_kwargs): 
    inspection_list = []
    for seed in seeds: 
        torch.manual_seed(seed)
        for model_name in model_names: 
            for task in Task.TASK_LIST: 
                file_name = EXP_FILE+'/'+holdouts_folder+'/'+model_name+'/contexts'
                
                if check_already_trained(file_name, seed) and not overwrite:
                    print('Model at ' + file_name + ' for seed '+str(seed)+' aleady trained')
                    continue 
                
                model = make_default_model(model_name)
                trainer_config = ContextTrainerConfig(file_name, seed, holdouts=holdouts, **train_config_kwargs)
                trainer = ContextTrainer(trainer_config)
                is_trained = trainer.train(model)
                if not is_trained: inspection_list.append((model.model_name, seed))
                del model

    return inspection_list


if __name__ == "__main__":
    model_file = '_ReLU128_4.11'

    #train_mode = str(sys.argv[1])
    train_mode = 'train_contexts'
    if train_mode == 'train_contexts': 
        holdout_type = 'swap_holdouts'
        seeds = [3]
        to_train_contexts = list(itertools.product(['sbertNet_tuned'], seeds, [['Multitask']]))
        print(to_train_contexts)
        inspection_dict = get_all_contexts_set(to_train_contexts)
        print(inspection_dict)

    if train_mode == 'test_contexts': 
        holdout_type = 'swap_holdouts'
        seeds = [0]
        to_test = list(itertools.product(ALL_MODEL_PARAMS.keys(), seeds,  training_lists_dict['swap_holdouts']))

        for config in to_test: 
            print(config)
            model_params_key, seed_num, holdouts = config
            torch.manual_seed(seed_num)
            for holdout in holdouts:
                try:
                    holdout_file = get_holdout_file(holdouts)
                    pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'/contexts_test'+holdout.replace(' ', '_')+'_seed'+str(seed_num)+'_holdout_correct', 'rb'))
                    print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout)
                    continue
                except FileNotFoundError: 
                    model, _, _, _ = config_model(model_params_key)
                    model.set_seed(seed_num)
                    model.to(device)
                    test_context(model, holdouts, foldername= model_file, holdout_type = holdout_type, save=True)
                  
