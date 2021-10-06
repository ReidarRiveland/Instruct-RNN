from custom_GRU import CustomGRU
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import numpy as np

from collections import defaultdict

import pandas as pd
import pickle
from jit_GRU import CustomGRU
from utils import get_input_rule, get_instructions


class BaseNet(nn.Module): 
    def __init__(self, in_dim, hid_dim, num_layers, activ_func, instruct_mode):
        super(BaseNet, self).__init__()
        self.instruct_mode = instruct_mode
        self.in_dim = in_dim
        self.out_dim = 33
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.activ_func = activ_func
        self._loss_data_dict = defaultdict(list)
        self._correct_data_dict = defaultdict(list)
        self.__seed_num_str__ = None
        self.__hiddenInitValue__ = 0.1

        if self.activ_func != 'elman': 
            self.recurrent_units = CustomGRU(self.in_dim, hid_dim, self.num_layers, activ_func = activ_func, batch_first=True)
        else: 
            self.recurrent_units = nn.RNN(self.in_dim, hid_dim, self.num_layers, nonlinearity = 'relu', batch_first=True)
        self.sensory_motor_outs = nn.Sequential(nn.Linear(hid_dim, self.out_dim), nn.Sigmoid())

        self.__weights_init__()
        self.__device__ = torch.device('cpu')

    def __weights_init__(self):
        for n, p in self.named_parameters():
            if 'weight_ih' in n:
                for ih in p.chunk(3, 0):
                    torch.nn.init.normal_(ih, std = 1/np.sqrt(self.in_dim))
            elif 'weight_hh' in n:
                for hh in p.chunk(3, 0):
                    hh.data.copy_(torch.eye(self.hid_dim)*0.5)
            elif 'W_out' in n:
                torch.nn.init.normal_(p, std = 0.4/np.sqrt(self.hid_dim))

    def __initHidden__(self, batch_size):
        return torch.full((self.num_layers, batch_size, self.hid_dim), self.__hiddenInitValue__, device=self.__device__.type)

    def forward(self, task_info, x , t=120): 
        h0 = self.__initHidden__(x.shape[0])
        #task_info_block = torch.rand(task_info.shape[0], t, task_info.shape[1])*0.05
        task_info_block = task_info.unsqueeze(1).repeat(1, t, 1)

        rnn_ins = torch.cat((task_info_block, x.type(torch.float32)), 2)
        rnn_hid, _ = self.recurrent_units(rnn_ins, h0)
        out = self.sensory_motor_outs(rnn_hid)
        return out, rnn_hid

    def freeze_weights(self): 
        for p in self.parameters(): 
            p.requires_grad=False

    def get_training_df(self): 
        df_correct = pd.DataFrame(self._correct_data_dict.values()).T
        df_correct.columns = self._correct_data_dict.keys()
        df_loss = pd.DataFrame(self._loss_data_dict.values()).T
        df_loss.columns = self._loss_data_dict.keys()
        return df_correct, df_loss

    def save_training_data(self, foldername): 
        pickle.dump(self._correct_data_dict, open(foldername+'/'+self.model_name+'/'+self.__seed_num_str__+'_training_correct', 'wb'))
        pickle.dump(self._loss_data_dict, open(foldername+'/'+self.model_name+'/'+self.__seed_num_str__+'_training_loss', 'wb'))

    def save_model(self, foldername): 
        torch.save(self.state_dict(), foldername+'/'+self.model_name+'/'+self.model_name+'_'+self.__seed_num_str__+'.pt')

    def load_training_data(self, foldername): 
        self._correct_data_dict = pickle.load(open(foldername+'/'+self.model_name+'/'+self.__seed_num_str__+'_training_correct', 'rb'))
        self._loss_data_dict = pickle.load(open(foldername+'/'+self.model_name+'/'+self.__seed_num_str__+'_training_loss', 'rb'))

    def load_model(self, foldername): 
        if 'sbertNet' in self.model_name:
            sbert_state_dict = torch.load(foldername+'/'+self.model_name+'/'+self.model_name+'_'+self.__seed_num_str__+'.pt', map_location='cpu')
            for key in list(sbert_state_dict.keys()):
                sbert_state_dict[key.replace('transformer.0.auto_model.', 'transformer.')] = sbert_state_dict.pop(key)
            self.load_state_dict(sbert_state_dict)
        else: 
            self.load_state_dict(torch.load(foldername+'/'+self.model_name+'/'+self.model_name+'_'+self.__seed_num_str__+'.pt', map_location='cpu'))

    def set_seed(self, seed_num): 
        self.__seed_num_str__ = 'seed'+str(seed_num)

class SimpleNet(BaseNet):
    def __init__(self, hid_dim, num_layers, activ_func=torch.relu, instruct_mode='', use_ortho_rules=False):
        self.model_name = 'simpleNet'
        if use_ortho_rules:
            super().__init__(85, hid_dim, num_layers, activ_func, instruct_mode)
            ortho_rules = pickle.load(open('ortho_rule_vecs', 'rb'))
            self.rule_transform = torch.Tensor(ortho_rules)
        else:
            super().__init__(81, hid_dim, num_layers, activ_func, instruct_mode)
            self.rule_transform = torch.diag(torch.ones(16))
        self.to(self.__device__)

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.__device__ = cuda_device
        self.rule_transform = self.rule_transform.to(cuda_device)

    def reset_weights(self): 
        self.__weights_init__()

    def get_task_info(self, batch_len, task_type): 
        return get_input_rule(batch_len, task_type, self.instruct_mode).to(self.__device__)

    def forward(self, task_rule, x):
        task_rule = torch.matmul(task_rule, self.rule_transform)
        outs, rnn_hid = super().forward(task_rule, x)
        return outs, rnn_hid

class InstructNet(BaseNet): 
    def __init__(self, langModel, hid_dim, num_layers, activ_func = torch.relu, instruct_mode=''): 
        super().__init__(langModel.out_dim +65, hid_dim, num_layers, activ_func, instruct_mode)
        self.langModel = langModel
        self.model_name = self.langModel.embedder_name + 'Net' 

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.__device__ = cuda_device
        self.langModel.__device__ = cuda_device
        
    def reset_weights(self):
        super().__weights_init__()
        self.langModel.__init__(self.langModel.out_dim)

    def get_task_info(self, batch_len, task_type): 
        return get_instructions(batch_len, task_type, self.instruct_mode)

    def forward(self, instructions, x):
        instruct_embedded = self.langModel(instructions)
        outs, rnn_hid = super().forward(instruct_embedded, x)
        return outs, rnn_hid
    
    def load_model_weights(self, foldername): 
        self.load_state_dict(torch.load(foldername+'/'+self.model_name+'.pt'))


