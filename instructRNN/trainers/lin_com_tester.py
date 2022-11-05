from tabnanny import check
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
from instructRNN.tasks.tasks import TASK_LIST, construct_trials
from instructRNN.trainers.base_trainer import *
from instructRNN.data_loaders.dataset import TaskDataSet
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.analysis.model_analysis import get_instruct_reps

device = torch.device(0)

@define 
class LinCompTrainerConfig(): 
    file_path: str
    random_seed: int
    comp_vec_dim: int = 45 
    mode: str = ''
    num_contexts: int = 100

    epochs: int = 20
    min_run_epochs: int = 1
    batch_len: int = 64
    num_batches: int = 500
    stream_data: bool = True

    optim_alg: optim = optim.Adam
    lr: float = 0.005

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.99}

    checker_threshold: float = 0.95
    step_last_lr: bool = False

class LinCompTrainer(BaseTrainer): 
    def __init__(self, context_training_config: LinCompTrainerConfig = None): 
        super().__init__(context_training_config)
        self.all_contexts = torch.full((self.num_contexts, self.comp_vec_dim), np.nan)
        self.all_correct_data = []
        self.all_loss_data = []
        self.range_start = 0 

    def _record_session(self, task, is_trained_list, checkpoint=False):
        self.all_correct_data.append(self.correct_data.pop(task))
        self.all_loss_data.append(self.loss_data.pop(task))

        if os.path.exists(self.file_path):pass
        else: os.makedirs(self.file_path)

        if checkpoint: chk_str = '_chk'
        else: chk_str = ''

        filename = self.file_path+'/'+self.seed_suffix+'_'+task
        pickle.dump(is_trained_list, open(filename+self.mode+chk_str+'_is_trained', 'wb'))
        pickle.dump((self.all_correct_data, self.all_loss_data), open(filename+self.mode+chk_str+'_comp_data', 'wb'))
        pickle.dump(self.all_contexts.detach().cpu().numpy(), open(filename+self.mode+chk_str+'_comp_vecs', 'wb'))

    def _log_step(self, task_type, frac_correct, loss): 
        self.correct_data[task_type].append(frac_correct)
        self.loss_data[task_type].append(loss)
    
    def _print_training_status(self, task_type):
        status_str = '\n Training Step: ' + str(self.cur_step)+ \
                ' ----- Task Type: '+task_type+\
                ' ----- Performance: ' + str(self.correct_data[task_type][-1])+\
                ' ----- Loss: ' + "{:.3e}".format(self.loss_data[task_type][-1])
        print(status_str, flush=True)

    def _init_comp_vec(self): 
        context = nn.Parameter(torch.empty((1, self.comp_vec_dim), device=device))
        nn.init.sparse_(context, sparsity=0.1)
        return context
    
    def _init_optimizer(self, context):
        self.optimizer = self.optim_alg([context], lr=self.lr, weight_decay = 0.01)
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_args)
        else:
            self.scheduler = None
    
    def load_chk(self, file_name, seed, task): 
        try: 
            chk_contexts = pickle.load(open(file_name+'/seed'+str(seed)+'_'+task+self.mode+'_chk_comp_vecs', 'rb'))
            correct_data, loss_data = pickle.load(open(file_name+'/seed'+str(seed)+'_'+task+self.mode+'_chk_comp_data', 'rb'))
            chk_index_start = np.max(np.where(np.isnan(chk_contexts)[:, 0] == False))
            self.all_correct_data = correct_data
            self.all_loss_data = loss_data
            self.range_start = chk_index_start

            print('\n comp_vec at ' + file_name + ' for seed '+str(seed)+' and task '+task+' loading checkpoint')
        except FileNotFoundError:
            pass

    def set_rule_basis(self, model, holdouts):
        task_indices = [TASK_LIST.index(task) for task in TASK_LIST if task not in holdouts]
        if hasattr(model, 'langModel'):
            reps = get_instruct_reps(model.langModel)
            self.task_info_basis = torch.tensor(np.mean(reps, axis=1)[task_indices, :])
        else: 
            self.task_info_basis = model.rule_transform[task_indices, :]
        self.task_info_basis.to(device)

    def _train(self, model, comp_vec, mode): 
        self._init_optimizer(comp_vec)
        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            for self.cur_step in range(self.num_batches): 
                if self.mode == 'exemplar': data = self._expand_exemplar(self.exemplar_data)
                else: data = next(self.streamer.stream_batch())
                
                ins, tar, mask, tar_dir, task_type = data

                self.optimizer.zero_grad()
                contexts = torch.matmul(comp_vec, self.task_info_basis.float().to(device))
                in_contexts = contexts.repeat(self.batch_len, 1)

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

    def train(self, model, task):
        model.load_model(self.file_path.replace('lin_comp', '')[:-1], suffix='_'+self.seed_suffix)
        model.to(device)
        model.freeze_weights()
        model.eval()
        self.model_file_path = model.model_name+'_'+self.seed_suffix
        is_trained_list = []

        if mode == 'exemplar': 
            self._load_exemplar(task)
        else:
            self.streamer = TaskDataSet(self.file_path.partition('/')[0], 
                self.stream_data, 
                self.batch_len, 
                self.num_batches,
                set_single_task=task)

        for i in range(self.range_start, self.num_contexts): 
            is_trained = False 
            print('Training '+str(i)+'th context')
            comp_vec = self._init_comp_vec()
            is_trained = self._train(model, comp_vec, mode)
            is_trained_list.append(is_trained)
            self.all_contexts[i, :] = comp_vec.squeeze()
            self._record_session(task, is_trained_list, checkpoint=True)
        self._record_session(task, is_trained_list)                

def check_already_trained(file_name, seed, task, mode): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+'_'+task+mode+'_comp_vecs', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+ ' and mode ' + mode + ' and task '+task+' aleady trained')
        return True
    except FileNotFoundError:
        return False

def train_lin_comp(exp_folder, model_name,  seed, labeled_holdouts, mode = '', tasks = None, overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)
    labels, holdouts = labeled_holdouts
            
    model = make_default_model(model_name)

    file_name = exp_folder+'/'+labels+'/'+model_name+'/lin_comp'

    if tasks is None: 
        tasks = holdouts

    for task in tasks: 
        if not overwrite and check_already_trained(file_name, seed, task, mode):
            continue 
        else:        
            print('\n TRAINING LIN COMP at ' + file_name + ' for task '+task+ ' for mode ' + mode+ '\n')
            # if (task == 'DMC' or task =='DNMC') and 'swap' in labels:
            #     trainer_config = ContextTrainerConfig(file_name, seed, context_dim, batch_len=64, checker_threshold=0.8, mode=mode, **train_config_kwargs)
            # else:
            trainer_config = LinCompTrainerConfig(file_name, seed, mode=mode, **train_config_kwargs)
            trainer = LinCompTrainer(trainer_config)
            trainer.set_rule_basis(model, holdouts)
            # if not overwrite: 
            #     trainer.load_chk(file_name, seed, task)
            trainer.train(model, task)
