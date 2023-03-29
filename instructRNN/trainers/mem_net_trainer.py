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
from instructRNN.tasks.task_factory import OUTPUT_DIM, INPUT_DIM
from instructRNN.analysis.model_analysis import get_instruct_reps, get_rule_embedder_reps
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.models.script_gru import ScriptGRU

device = torch.device(0)

class MemNet(nn.Module): 
    def __init__(self, out_dim, rnn_hidden_dim):
        super(MemNet, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.out_dim = out_dim
        self.rnn_hiddenInitValue = 0.1
        self.__device__ = 'cpu'
        self.rnn = nn.GRU(OUTPUT_DIM+INPUT_DIM+256, self.rnn_hidden_dim, 1, batch_first=True)
        self.lin_out= nn.Linear(rnn_hidden_dim, self.out_dim)

    def __initHidden__(self, batch_size):
        return torch.full((1, batch_size, self.rnn_hidden_dim), 
                self.rnn_hiddenInitValue, device=torch.device(self.__device__))

    def forward(self, ins, tar, sm_hidden): 
        h0 = self.__initHidden__(ins.shape[0])
        rnn_ins = torch.cat((ins, tar, sm_hidden), axis=-1)
        rnn_hid, _ = self.rnn(rnn_ins, h0)
        out = torch.relu(self.lin_out(rnn_hid))
        return out, rnn_hid

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.__device__ = cuda_device
        

@define 
class MemNetTrainerConfig(): 
    file_path: str
    random_seed: int
    holdouts: list
    task:str = None
    mode: str = ''
    min_run_epochs:int = 1

    epochs: int = 80
    batch_len: int = 64
    num_batches: int = 800
    stream_data: bool = True

    optim_alg: str = 'adam'
    init_lr: float = 0.001

    scheduler_type: str = 'exp'
    scheduler_gamma: float = 0.99
    scheduler_args: dict = {}

    checker_threshold: float = 0.95
    step_last_lr: bool = False


class MemNetTrainer(BaseTrainer): 
    def __init__(self, training_config:MemNetTrainerConfig): 
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
        if self.task is None: 
            task_str = ''
        

        if mode == 'CHECKPOINT':
            chk_attr_path = self.file_path+'/'+self.model_file_path+task_str+'_memNet_CHECKPOINT'
            pickle.dump(checkpoint_attrs, open(chk_attr_path+'_attrs', 'wb'))
            torch.save(self.memNet.state_dict(), chk_attr_path+'.pt')

        if mode=='FINAL': 
            data_path = self.file_path+'/'+self.model_file_path+task_str+'_memNet'
            pickle.dump(checkpoint_attrs.pop('loss_data'), open(data_path+'_loss', 'wb'))
            pickle.dump(checkpoint_attrs.pop('correct_data'), open(data_path+'_correct', 'wb'))


            pickle.dump(checkpoint_attrs, open(data_path+'_attrs', 'wb'))
            os.remove(data_path+'_CHECKPOINT_attrs')

            torch.save(self.memNet.state_dict(), data_path+'.pt')
            os.remove(self.file_path+'/'+self.model_file_path+task_str+'_memNet_CHECKPOINT.pt')

    def _log_step(self, task_type, loss, frac_correct=None): 
        if frac_correct is not None: 
            self.correct_data[task_type].append(frac_correct)
        self.loss_data[task_type].append(loss)
    
    def _init_streamer(self):
        self.streamer = TaskDataSet(self.file_path.partition('/')[0], 
                        self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts,
                        set_single_task=self.task)

    def init_optimizer(self, parameters):
        if self.optim_alg == 'adam': 
            optim_alg = optim.Adam

        self.optimizer = optim_alg(parameters, lr=self.init_lr)
        self._init_scheduler()

    def _init_scheduler(self):
        if self.scheduler_type == 'exp': 
            scheduler_class = optim.lr_scheduler.ExponentialLR

        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5, eta_min=0.005, last_epoch=- 1, verbose=False)
        self.scheduler = scheduler_class(self.optimizer, gamma=self.scheduler_gamma, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)

    def train(self, model, memNet): 
        self.memNet = memNet
        model.to(device)
        model.eval()
        self.model_file_path = model.model_name+'_'+self.seed_suffix
        self.memNet.to(device)
        self.memNet = memNet
        self._init_streamer()
        self.init_optimizer(self.memNet.parameters())
        self.mse = nn.MSELoss()
        if hasattr(model, 'langModel'):
            self.rule_basis = np.mean(get_instruct_reps(model.langModel), axis=1)
        elif model.model_name == 'simpleNetPlus':
            self.rule_basis = np.mean(get_rule_embedder_reps(model), axis=1)

        for self.cur_epoch in tqdm(range(self.cur_epoch, self.epochs), desc='epochs'):
            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data
                task_info = get_task_info(self.batch_len, task_type, model.info_type)
                out, rnn_hid = model(ins.to(device), task_info)

                self.optimizer.zero_grad()
                mem_out, hid = self.memNet(ins.float().to(device), tar.float().to(device), rnn_hid)
                target_embed = torch.tensor(self.rule_basis[TASK_LIST.index(task_type),:])[None, None,  :].repeat(self.batch_len,  20, 1).float()
                loss = self.mse(mem_out[:, -20:, :], target_embed.to(device))
                loss.backward()
                self.optimizer.step()

                self._log_step(task_type, loss.item())

                if self.cur_step%50 == 0:
                    out, _ = model(ins.to(device), None, info_embedded=mem_out[:,-1,:])
                    frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                    self._log_step(task_type, loss.item(), frac_correct=frac_correct)
                    self._print_training_status(task_type)
                else: 
                    self._log_step(task_type, loss.item())

            self._record_session(model, mode='CHECKPOINT')
        
        self._record_session(model, mode='FINAL')


    def tune(self, model, memNet): 
        self.memNet = memNet
        self.init_optimizer(memNet.lin_out.parameters())
        self._init_streamer()
        model.to(device)
        self.memNet.to(device)

        for self.cur_epoch in tqdm(range(self.epochs), desc='epochs'):
            for self.cur_step in range(self.num_batches): 
                data = next(self.streamer.stream_batch())
                ins, tar, mask, tar_dir, task_type = data

                self.optimizer.zero_grad()
                out, _ = model(ins.to(device), context=in_contexts)
                loss = masked_MSE_Loss(out, tar.to(device), mask.to(device)) 
                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                self._log_step(task_type, frac_correct, loss.item())

                loss.backward()
                self.optimizer.step()

                if self.cur_step%50 == 0 or self.cur_step == 5 or self.cur_step == 10:
                    self._print_training_status(task_type)

                if self._check_model_training():
                    return True

            #if self.scheduler is not None: self.scheduler.step()  
            #if self.step_last_lr: self.step_scheduler.step()

        warnings.warn('Model has not reach specified performance threshold during training')
        return False

# def check_already_trained(file_name, seed, mode): 
#     try: 
#         pickle.load(open(file_name+'/seed'+str(seed)+'_'+task+mode+'_mem_net', 'rb'))
#         print('\n Model at ' + file_name + ' for seed '+str(seed)+ ' and mode ' + mode + ' and task '+task+' aleady trained')
#         return True
#     except FileNotFoundError:
#         return False

def train_memNet(exp_folder, model_name,  seed, labeled_holdouts, mode = '', tasks = None, overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)
    labels, holdouts = labeled_holdouts
    model = make_default_model(model_name)
    model.load_model(exp_folder+'/'+labels+'/'+model_name, suffix='_seed'+str(seed))
    file_name = exp_folder+'/'+labels+'/'+model_name+'/mem_net'
    memNet = MemNet(64, 256)

    # if not overwrite and check_already_trained(file_name, seed, mode):
    #     print('ALREADY TRAININED')
    # else:        
    
    print('\n TRAINING MEMNET at ' + file_name + ' for holdouts '+labels+ ' for mode ' + mode+ '\n')

    trainer_config = MemNetTrainerConfig(file_name, seed, holdouts=holdouts, mode=mode, **train_config_kwargs)
    trainer = MemNetTrainer(trainer_config)

    trainer.train(model, memNet)

def tune_memNet(exp_folder, model_name,  seed, labeled_holdouts, mode = '', overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)
    labels, holdouts = labeled_holdouts
    model = make_default_model(model_name)
    model.load_model(exp_folder+'/'+labels+'/'+model_name, suffix='_seed'+str(seed))
    file_name = exp_folder+'/'+labels+'/'+model_name+'/mem_net/'+model_name+'_seed'+str(seed)+'_memNet.pt'
    memNet = MemNet(64, 516)
    memNet.load_state_dict(torch.load(file_name))

    # if not overwrite and check_already_trained(file_name, seed, mode):
    #     print('ALREADY TRAININED')
    # else:        
    for task in holdouts: 
        print('\n TRAINING MEMNET at ' + file_name + ' for holdouts '+labels+ ' for mode ' + mode+ '\n')

        trainer_config = MemNetTrainerConfig(file_name, seed, init_lr=0.05, task=task, holdouts=[], mode=mode, **train_config_kwargs)
        trainer = MemNetTrainer(trainer_config)

        trainer.tune(model, memNet)