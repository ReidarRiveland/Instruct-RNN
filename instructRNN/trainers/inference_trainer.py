from email.policy import strict
import torch
import torch.nn as nn
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
from instructRNN.analysis.model_eval import task_eval

from instructRNN.trainers.base_trainer import *
from instructRNN.data_loaders.dataset import *
from instructRNN.tasks.task_criteria import *
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.models.full_models import make_default_model
from instructRNN.analysis.model_analysis import get_rule_reps
from instructRNN.models.sensorimotor_models import InferenceNet

if torch.cuda.is_available():
    device = torch.device(0)
    print(torch.cuda.get_device_name(device), flush=True)
else: 
    device = torch.device('cpu')
    
@define
class InferenceTrainerConfig(): 
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


class InferenceTrainer(BaseTrainer): 
    def __init__(self, training_config:InferenceTrainerConfig): 
        super().__init__(training_config)

    def _init_streamer(self):
        self.streamer = TaskDataSet(
                        self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts, 
                        self.set_single_task)

    def init_optimizer(self, model):
        if self.optim_alg == 'adam': 
            optim_alg = optim.Adam

        optimizer = optim_alg(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.optimizer = optimizer

    def _init_scheduler(self):
        if self.scheduler_type == 'exp': 
            scheduler_class = optim.lr_scheduler.ExponentialLR

        self.scheduler = scheduler_class(self.optimizer, gamma=self.scheduler_gamma, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)

    def train(self, inference_model, sm_model): 
        inference_model.train()
        inference_model.to(device)
        self._init_streamer()
        self.init_optimizer(inference_model)

        criteria = nn.MSELoss()
        rule_reps = torch.tensor(get_rule_reps(sm_model, use_rule_encoder=True))

        for self.cur_epoch in tqdm(range(self.cur_epoch, self.epochs), desc='epochs'):

            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data

                self.optimizer.zero_grad()
                out, _ = inference_model(ins.to(device))
                target = sm_model.expand_info(rule_reps[TASK_LIST.index(task_type)].repeat(self.batch_len, 1), 150, 0).to(device)
                loss = criteria(out, target) 

                loss.backward()
                self.optimizer.step()

                if self.cur_step%50 == 0:
                    print('\n')
                    print(task_type)
                    print(loss.item())

                # if self._check_model_training() and not is_testing:
                #     self._record_session(model, mode='FINAL')
                #     return True

            # if self.scheduler is not None: self.scheduler.step()  
            # if self.step_last_lr: self.step_scheduler.step()

        if not is_testing:
            warnings.warn('\n !!! Model has not reach specified performance threshold during training !!! \n')
            return False


# def train_model(exp_folder, model_name, seed, labeled_holdouts, use_checkpoint=False, overwrite=False, **train_config_kwargs): 
#     torch.manual_seed(seed)
#     label, holdouts = labeled_holdouts
#     file_name = exp_folder+'/'+label+'/'+model_name   

#     if check_already_trained(file_name, seed) and not overwrite:
#         return True
    
#     model = make_default_model(model_name)


#     if use_checkpoint: 
#         try:
#             model, trainer = load_checkpoint(model, file_name, seed)
#         except: 
#             print('Starting Training from untrained model')
#             trainer = ModelTrainer(trainer_config)
#     else: 
#         trainer = ModelTrainer(trainer_config)

#     for n, p in model.named_parameters(): 
#         if p.requires_grad: print(n)

#     is_trained = trainer.train(model)
#     return is_trained
