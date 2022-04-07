from pyexpat import model
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from task import Task

from collections import defaultdict

import pandas as pd
import pickle
from script_gru import ScriptGRU
from utils import get_input_rule, get_instructions


class BaseNet(nn.Module): 
    def __init__(self, config):
        super(BaseNet, self).__init__()
        for name, value in config.__dict__.items(): 
            setattr(self, name, value)

        if self.rnn_activ_func == 'relu':
            self._activ_func = torch.relu

        self.recurrent_units = ScriptGRU(self.rnn_in_dim, 
                                        self.rnn_hidden_dim, 
                                        self.rnn_layers, 
                                        activ_func = self._activ_func, 
                                        batch_first=True)

        self.sensory_motor_outs = nn.Sequential(
                                    nn.Linear(self.rnn_hidden_dim, self.sensorimotor_out_dim), 
                                    nn.Sigmoid())

        self.recurrent_units.__weights_init__()
        self.__device__ = torch.device('cpu')

        self.foldername=None

    def __initHidden__(self, batch_size):
        return torch.full((self.num_layers, batch_size, self.hid_dim), 
                self.hiddenInitValue, device=self.__device__.type)

    def forward(self, x, task_info, t=120): 
        h0 = self.__initHidden__(x.shape[0])
        task_info_block = task_info.unsqueeze(1).repeat(1, t, 1)
        rnn_ins = torch.cat((task_info_block, x.type(torch.float32)), 2)
        rnn_hid, _ = self.recurrent_units(rnn_ins, h0)
        out = self.sensory_motor_outs(rnn_hid)
        return out, rnn_hid

    def freeze_weights(self): 
        for p in self.parameters(): 
            p.requires_grad=False

    def set_file_path(self, foldername, seed_num): 
        self.seed_num = seed_num
        self.seed_num_str = 'seed'+str(seed_num)
        self.foldername = foldername
        self.data_file_path = foldername+'/'+self.model_name+'/'+self.seed_num_str
        self.model_file_path = foldername+'/'+self.model_name+'/'+self.model_name+'_'+self.seed_num_str+'.pt'

    def save_training_data(self): 
        pickle.dump(self._correct_data_dict, open(self.file_path+'_training_correct', 'wb'))
        pickle.dump(self._loss_data_dict, open(self.file_path+'_training_loss', 'wb'))

    def save_model(self): 
        torch.save(self.state_dict(), self.model_file_path)

    def load_model(self): 
        self.load_state_dict(torch.load(self.model_file_path, map_location='cpu'))

    def reset_training_data(self): 
        self._loss_data_dict = defaultdict(list)
        self._correct_data_dict = defaultdict(list)

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.__device__ = cuda_device

class SimpleNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        ortho_rules = pickle.load(open('ortho_rule_vecs', 'rb'))
        self.rule_transform = torch.Tensor(ortho_rules)

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.rule_transform = self.rule_transform.to(cuda_device)

    def get_task_info(self, batch_len, task_type): 
        return get_input_rule(batch_len, task_type, self.instruct_mode).to(self.__device__)

    def forward(self, x, task_rule):
        task_rule = torch.matmul(task_rule, self.rule_transform)
        outs, rnn_hid = super().forward(task_rule, x)
        return outs, rnn_hid

class InstructNet(BaseNet): 
    def __init__(self, config): 
        super().__init__(config)
        self.langModel = langModel
        self.model_name = self.langModel.embedder_name + 'Net' 

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.langModel.__device__ = cuda_device

    def get_task_info(self, batch_len, task_type): 
        return get_instructions(batch_len, task_type, self.instruct_mode)

    def forward(self, x, instruction = None, context = None):
        assert instruction is not None or context is not None, 'must have instruction or context input'
        if instruction is not None: info_embedded = self.langModel(instruction)
        else: info_embedded = context

        outs, rnn_hid = super().forward(x, info_embedded)
        return outs, rnn_hid
