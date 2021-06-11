import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import numpy as np

from collections import defaultdict

import pandas as pd
import pickle
from custom_GRU import CustomGRU
from utils import get_input_rule, get_instructions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        if self.activ_func is not 'elman': 
            self.recurrent_units = CustomGRU(self.in_dim, hid_dim, self.num_layers, activ_func = activ_func, batch_first=True)
        else: 
            self.recurrent_units = nn.RNN(self.in_dim, hid_dim, self.num_layers, nonlinearity = 'relu', batch_first=True)
        self.sensory_motor_outs = nn.Sequential(nn.Linear(hid_dim, self.out_dim), nn.Sigmoid())

        self.__weights_init__()
        
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

    def __initHidden__(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value, device=device.type)

    def forward(self, task_info, x , t=120): 
        h0 = self.__initHidden__(x.shape[0], 0.1)
        task_info_block = task_info.unsqueeze(1).repeat(1, t, 1)
        rnn_ins = torch.cat((task_info_block, x.type(torch.float32)), 2)
        rnn_hid, _ = self.recurrent_units(rnn_ins, h0)
        out = self.sensory_motor_outs(rnn_hid)
        return out, rnn_hid

    def get_training_df(self): 
        df_correct = pd.DataFrame(self._correct_data_dict.values()).T
        df_correct.columns = self._correct_data_dict.keys()
        df_loss = pd.DataFrame(self._loss_data_dict.values()).T
        df_loss.columns = self._loss_data_dict.keys()
        return df_correct, df_loss

    def save_training_data(self, holdout_task,  foldername, name): 
        df_correct, df_loss = self.get_training_df()
        holdout_task = holdout_task.replace(' ', '_')
        pickle.dump(df_correct, open(foldername+'/'+holdout_task+'/'+name+'_training_correct', 'wb'))
        pickle.dump(df_loss, open(foldername+'/'+holdout_task+'/'+name+'_training_loss', 'wb'))


class SimpleNet(BaseNet):
    def __init__(self, hid_dim, num_layers, activ_func='relu', instruct_mode=None):
        super().__init__(81, hid_dim, num_layers, activ_func, instruct_mode)
        self.model_name = 'simpleNet'

    def reset_weights(self): 
        self.__weights_init__()

    def get_task_info(self, batch_len, task_type): 
        return get_input_rule(batch_len, task_type, self.instruct_mode).to(device)

    def forward(self, task_rule, x):
        outs, rnn_hid = super().forward(task_rule, x)
        return outs, rnn_hid

class InstructNet(BaseNet): 
    def __init__(self, langModel, hid_dim, num_layers, activ_func = 'relu', instruct_mode=None): 
        super().__init__(langModel.out_dim+65, hid_dim, num_layers, activ_func, instruct_mode)
        self.langModel = langModel
        self.langModel.device = device
        self.model_name = self.langModel.embedderStr + 'Net'

    def reset_weights(self):
        super().__weights_init__()
        self.langModel.__init__(self.langModel.out_dim)

    def get_task_info(self, batch_len, task_type): 
        return get_instructions(batch_len, task_type, self.instruct_mode)

    def forward(self, instructions, x):
        instruct_embedded = self.langModel(instructions)
        outs, rnn_hid = super().forward(instruct_embedded, x)
        return outs, rnn_hid

