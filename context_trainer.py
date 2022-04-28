from matplotlib.style import context
from utils.utils import training_lists_dict
from models.full_models import make_default_model
from base_trainer import masked_MSE_Loss, BaseTrainer
from dataset import TaskDataSet
from task import Task, isCorrect
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pickle
import itertools
from tqdm import tqdm
from attrs import define
from collections import defaultdict
from copy import copy

device = torch.device(0)

EXP_FILE = '_ReLU128_4.11/swap_holdouts'

class KL_divergence():
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, hidden_rep):
        rho_hat = torch.mean(torch.sigmoid(hidden_rep), 1) # sigmoid because we need the probability distributions
        rho = torch.tensor([self.rho] * len(rho_hat)).to(device)
        return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

@define 
class ContextTrainerConfig(): 
    file_path: str
    random_seed: int

    context_dim: int = 20
    num_contexts: int = 128

    epochs: int = 60
    min_run_epochs: int = 5
    batch_len: int = 128
    num_batches: int = 500

    optim_alg: optim = optim.Adam
    lr: float = 0.1
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.95}
    sparsity_prior: float = None
    sparsity_weight: float = 1

    checker_threshold: float = 0.95
    step_last_lr: bool = False

class ContextTrainer(BaseTrainer): 
    def __init__(self, context_training_config: ContextTrainerConfig = None, from_checkpoint_dict:dict = None): 
        super().__init__(context_training_config, from_checkpoint_dict)
        if self.sparsity_prior is not None: 
            self.KL_loss = KL_divergence(self.sparsity_prior)
            self.sparsity_loss_data = defaultdict(list)
            self.task_loss_data = defaultdict(list)

    def _record_session(self, contexts, task):
        checkpoint_attrs = super()._record_session()
        filename = self.file_path+'/'+self.seed_suffix+task+'_supervised'
        pickle.dump(checkpoint_attrs, open(self.file_path+'/'+task+'_attrs', 'wb'))
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
        
        if self.sparsity_prior is not None: 
            status_str += ' ----- Task Loss: ' + "{:.3e}".format(self.task_loss_data[task_type][-1])+\
                    ' ----- Sparsity Loss: ' + "{:.3e}".format(self.sparsity_loss_data[task_type][-1])

        print(status_str)

    def _init_contexts(self, batch_len): 
        context = nn.Parameter(torch.relu(torch.randn((batch_len, self.context_dim), device=device))+1e-3)
        return context
    
    def _init_optimizer(self, context):
        self.optimizer = self.optim_alg([context], lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)
        
    def _train(self, model, contexts): 
        model.load_model(self.file_path.replace('contexts', ''), suffix='_'+self.seed_suffix)
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
                if contexts.shape[0]==1: 
                    in_contexts = contexts.repeat(self.batch_len, 1).clamp(min=0.0)
                else: 
                    in_contexts = contexts.clamp(min=0.0)

                out, _ = model(ins.to(device), context=in_contexts)
                task_loss = masked_MSE_Loss(out, tar.to(device), mask.to(device)) 
                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)

                if self.sparsity_prior is not None: 
                    sparsity_loss = self.KL_loss(contexts)*self.sparsity_weight
                    loss = sparsity_loss + task_loss
                    self._log_step(task_type, frac_correct, loss.item(), task_loss.item(), sparsity_loss.item())
                else: 
                    loss = task_loss
                    self._log_step(task_type, frac_correct, loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
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
        self.streamer = TaskDataSet(False, 
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
                    context = self._init_contexts(1)
                    is_trained = self._train(model, context)
                all_contexts[i, :] = context
            self._record_session(all_contexts, task)                


def check_already_trained(file_name, seed, task): 
    try: 
        pickle.load(open(file_name+'/seed'+str(seed)+task+'_supervised_context_vecs20', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+' and task '+task+' aleady trained')
        return True
    except FileNotFoundError:
        return False

def train_context_set(model_names, seeds, holdouts_folders, as_batch = False, tasks = Task.TASK_LIST, overwrite=False, **train_config_kwargs): 
    inspection_list = []
    for seed in seeds: 
        torch.manual_seed(seed)
        for holdouts_folder in holdouts_folders:
            for model_name in model_names: 
                file_name = EXP_FILE+'/'+holdouts_folder+'/'+model_name+'/contexts'

                model = make_default_model(model_name)
                for task in tasks: 
                    if check_already_trained(file_name, seed, task) and not overwrite:
                        continue 
                    else:        
                        print('\n TRAINING CONTEXTS at ' + file_name + '\n')
                        trainer_config = ContextTrainerConfig(file_name, seed, **train_config_kwargs)
                        trainer = ContextTrainer(trainer_config)
                        is_trained = trainer.train(model, task, as_batch=as_batch)
                        if not is_trained: inspection_list.append((model.model_name, seed))

                del model

        return inspection_list

if __name__ == "__main__":
    ##TEST BY BATCH WITH NEW INITIALIZATION AND CLIP!!!
    # train_context_set(['sbertNet_tuned'], 
    #                     [0], 
    #                     ['Go_Anti_DM'], 
    #                     tasks = Task.TASK_LIST, 
    #                     batch_len = 64, lr=0.04, min_run_epochs=2, epochs=5, step_last_lr=False)

    train_context_set(['sbertNet_tuned'], 
                        [0], 
                        ['MultiDM_DNMS'], 
                        as_batch=False,  
                        batch_len = 64, lr=0.08, min_run_epochs=3, epochs=5, step_last_lr=False)

