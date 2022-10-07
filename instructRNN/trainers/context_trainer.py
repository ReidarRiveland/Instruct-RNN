import warnings

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pickle
from tqdm import tqdm
from attrs import define
from copy import copy
import os

from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.tasks import TASK_LIST
from instructRNN.trainers.base_trainer import *
from instructRNN.data_loaders.dataset import TaskDataSet
from instructRNN.tasks.task_criteria import isCorrect

device = torch.device(0)


@define 
class ContextTrainerConfig(): 
    file_path: str
    random_seed: int
    context_dim: int    
    num_contexts: int = 128

    epochs: int = 15
    min_run_epochs: int = 1
    batch_len: int = 256
    num_batches: int = 800
    stream_data: bool = True

    optim_alg: optim = optim.Adam
    lr: float = 0.01
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.99}

    checker_threshold: float = 0.85
    step_last_lr: bool = False

class ContextTrainer(BaseTrainer): 
    def __init__(self, context_training_config: ContextTrainerConfig = None): 
        super().__init__(context_training_config)

    def _record_session(self, contexts, task):
        if os.path.exists(self.file_path):pass
        else: os.makedirs(self.file_path)
        filename = self.file_path+'/'+self.seed_suffix+'_'+task
        pickle.dump(contexts.detach().cpu().numpy(), open(filename+'_context_vecs'+str(self.context_dim), 'wb'))

    def _log_step(self, task_type, frac_correct, loss, task_loss= None, sparsity_loss=None): 
        self.correct_data[task_type].append(frac_correct)
        self.loss_data[task_type].append(loss)
        if sparsity_loss is not None: 
            self.task_loss_data[task_type].append(task_loss)
            self.sparsity_loss_data[task_type].append(sparsity_loss)
    
    def _print_training_status(self, task_type):
        status_str = '\n Training Step: ' + str(self.cur_step)+ \
                ' ----- Task Type: '+task_type+\
                ' ----- Performance: ' + str(self.correct_data[task_type][-1])+\
                ' ----- Loss: ' + "{:.3e}".format(self.loss_data[task_type][-1])
        print(status_str, flush=True)

    def _init_contexts(self, batch_len): 
        context = nn.Parameter(torch.empty((batch_len, self.context_dim), device=device))
        nn.init.normal_(context, std=1.5)
        return context
    
    def _init_optimizer(self, context):
        self.optimizer = self.optim_alg([context], lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_args)
            self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.5)
        else:
            self.scheduler = None
            self.step_scheduler = None

    def _train(self, model, contexts): 
        model.load_model(self.file_path.replace('contexts', '')[:-1], suffix='_'+self.seed_suffix)
        model.to(device)
        model.freeze_weights()
        model.eval()

        self.model_file_path = model.model_name+'_'+self.seed_suffix
        self._init_optimizer(contexts)

        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data
                self.optimizer.zero_grad()
                in_contexts = torch.mean(contexts.repeat(self.batch_len, 1, 1), axis=1)

                out, _ = model(ins.to(device), context=in_contexts)
                loss = masked_MSE_Loss(out, tar.to(device), mask.to(device)) 
                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                self._log_step(task_type, frac_correct, loss.item())

                loss.backward()
                self.optimizer.step()

                if self.cur_step%50 == 0:
                    self._print_training_status(task_type)

                if self._check_model_training():
                    return True

            if self.scheduler is not None: self.scheduler.step()  
            if self.step_last_lr: self.step_scheduler.step()

        warnings.warn('Model has not reach specified performance threshold during training')
        return False
    
    def train(self, model, task, as_batch=True):
        self.streamer = TaskDataSet(self.file_path.partition('/')[0], 
                self.stream_data, 
                self.batch_len, 
                self.num_batches,
                set_single_task=task)

        if as_batch: 
            assert self.num_contexts == self.batch_len
            contexts = self._init_contexts(self.num_contexts)
            is_trained = self._train(model, contexts)
            if is_trained: 
                self._record_session(contexts, task)
            return is_trained

        else: 
            all_contexts = torch.empty((self.num_contexts, self.context_dim))
            for i in range(self.num_contexts): 
                is_trained = False 
                while not is_trained: 
                    print('Training '+str(i)+'th context')
                    context = self._init_contexts(20)
                    is_trained = self._train(model, context)
                all_contexts[i, :] = torch.mean(context, dim=0)
            self._record_session(all_contexts, task)                


def check_already_trained(file_name, seed, task, context_dim): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_'+task+'_context_vecs'+str(context_dim), 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+' and task '+task+' aleady trained')
        return True
    except FileNotFoundError:
        return False

def train_contexts(exp_folder, model_name,  seed, labeled_holdouts, layer, 
                    as_batch = False, tasks = TASK_LIST, overwrite=False, **train_config_kwargs): 
     
    torch.manual_seed(seed)
    labels, _ = labeled_holdouts
            
    model = make_default_model(model_name)

    if layer=='emb': 
        context_dim = model.langModel.LM_out_dim
    elif layer=='last': 
        context_dim = model.langModel.LM_intermediate_lang_dim 
    file_name = exp_folder+'/'+labels+'/'+model_name+'/contexts'

    for task in tasks: 
        if not overwrite and check_already_trained(file_name, seed, task, context_dim):
            continue 
        else:        
            print('\n TRAINING CONTEXTS at ' + file_name + ' for task '+task+ '\n')
            trainer_config = ContextTrainerConfig(file_name, seed, context_dim, **train_config_kwargs)
            trainer = ContextTrainer(trainer_config)
            trainer.train(model, task, as_batch=as_batch)

