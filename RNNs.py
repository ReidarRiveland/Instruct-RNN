import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch.autograd import Variable

import warnings
from typing import List, Tuple
from torch import Tensor

import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
 
class simpleNet(nn.Module): 
    def __init__(self, in_dim, hid_dim, num_layers, activ_func, instruct_mode=None):
        super(simpleNet, self).__init__()
        self.tune_langModel = None
        self.instruct_mode = instruct_mode
        self.in_dim = in_dim
        self.out_dim = 33
        self.isLang = False
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.activ_func = activ_func

        if self.activ_func is not 'tanh': 
            self.recurrent_units = customGRU(self.in_dim, hid_dim, self.num_layers, activ_func = activ_func, batch_first=True)
        else: 
            self.recurrent_units = nn.GRU(self.in_dim, hid_dim, self.num_layers, batch_first=True)

        self.weights_init()
        self.W_out = nn.Linear(hid_dim, self.out_dim)

    def weights_init(self):
        for n, p in self.named_parameters():
            if 'weight_ih' in n:
                for ih in p.chunk(3, 0):
                    torch.nn.init.normal_(ih, std = 1/np.sqrt(self.in_dim))
            elif 'weight_hh' in n:
                for hh in p.chunk(3, 0):
                    hh.data.copy_(torch.eye(self.hid_dim)*0.5)
            elif 'W_out' in n:
                torch.nn.init.normal_(p, std = 0.4/np.sqrt(self.hid_dim))

    def forward(self, x, h): 
        rnn_hid, _ = self.rnn(x, h)
        motor_out = self.W_out(rnn_hid)
        out = torch.sigmoid(motor_out)
        return out, rnn_hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)


class instructNet(nn.Module): 
    def __init__(self, langMod, hid_dim, num_layers, activ_func = 'tanh', drop_p = 0.0, instruct_mode=None, tune_langModel = False, langLayerList = []): 
        super(instructNet, self).__init__()
        self.instruct_mode = instruct_mode
        self.tune_langModel = tune_langModel
        self.sensory_in_dim = 65
        self.isLang = True 
        self.hid_dim = hid_dim
        self.embedderStr = langMod.embedderStr
        self.langModel = langMod.langModel
        self.langMod = langMod
        self.num_layers = num_layers
        self.lang_embed_dim = langMod.langModel.out_dim
        self.activ_func = activ_func
        self.rnn = simpleNet(self.lang_embed_dim + self.sensory_in_dim, hid_dim, self.num_layers, self.activ_func)
        
        if tune_langModel:
            self.langModel.train()
            if len(langLayerList) == 0:  
                
                for param in self.langModel.parameters(): 
                    param.requires_grad = True
            else: 
                for n,p in self.langModel.named_parameters(): 
                    if any([layer in n for layer in langLayerList]):
                        p.requires_grad=True
                    else: 
                        p.requires_grad=False
        else: 
            for param in self.langModel.model.parameters(): 
                param.requires_grad = False
            self.langModel.eval()

    def weights_init(self): 
        self.rnn.weights_init()

    def forward(self, instruction_tensor, x, h):
        embedded_instruct = self.langModel(instruction_tensor)
        seq_blocked = embedded_instruct.unsqueeze(1).repeat(1, 120, 1)
        rnn_ins = torch.cat((seq_blocked, x.type(torch.float32)), 2)
        outs, rnn_hid = self.rnn(rnn_ins, h)
        return outs, rnn_hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)


class customGRUCell(nn.Module): 
    def __init__(self, input_size, hidden_size, activ_func): 
        super(customGRUCell, self).__init__()
        assert activ_func in ['relu', 'sigmoid', 'tanh', 'AsymSigmoid']
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.activ_func = activ_func

        torch.nn.init.normal_(self.bias_ih, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bias_hh, mean=0.0, std=1.0)

        if self.activ_func == 'relu':
            self.nonlinearity = torch.relu
        elif self.activ_func == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        elif self.activ_func == 'AsymSigmoid': 
            self.nonlinearity = lambda x: torch.sigmoid(x-torch.Tensor([10]).to(x.get_device()))
        else: 
            self.nonlinearity = torch.tanh


    def forward(self, input, hx):
        igates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih)
        hgates = (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        r_in, z_in, c_in = igates.chunk(3, 1)
        r_hid, z_hid, c_hid = hgates.chunk(3, 1)

        r_gate = torch.sigmoid(r_in+r_hid)
        z_gate = torch.sigmoid(z_in + z_hid)
        pre_c = c_in + (r_gate *c_hid)
        c = self.nonlinearity(pre_c)
        h_new = ((1-z_gate)*c) + (z_gate * hx)

        return h_new

class customGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, activ_func = torch.tanh,
                 use_bias=True, batch_first=False, dropout=0):
        super(customGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activ_func = activ_func
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = customGRUCell(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              activ_func = self.activ_func)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.dropout_layer = nn.Dropout(dropout)

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            hx = cell(input_[time], hx)
            output.append(hx)
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)

        if hx is None:
            hx = Variable(nn.init.xavier_uniform_(torch.Tensor(self.num_layers, input_.size(1), self.hidden_size)))        

        h_n = []
        layer_output = None

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = hx[layer,:,:]
            
            if layer == 0:
                layer_output, layer_h_n = customGRU._forward_rnn(
                    cell, input_, hx_layer)
            else:
                layer_output = self.dropout_layer(layer_output)
                layer_output, layer_h_n = customGRU._forward_rnn(
                    cell, layer_output,  hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)

        output = layer_output
        h_n = torch.stack(h_n, 0)
        if self.batch_first: 
            output = output.transpose(0, 1)
        return output, h_n




