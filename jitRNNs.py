import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.jit as jit
from torch.nn import Parameter
from torch.autograd import Variable

import warnings
from typing import List, Tuple
from torch import Tensor

import numpy as np

class scriptGRUCell(jit.ScriptModule): 
    def __init__(self, input_size, hidden_size, activ_func): 
        super(scriptGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        nn.init.uniform_(self.weight_ih, -1, 1)
        nn.init.uniform_(self.weight_hh, -1, 1)

        self.activ_func = activ_func

    @jit.script_method
    def forward(self, input, hx):
        igates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih)
        hgates = (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        r_in, z_in, c_in = igates.chunk(3, 1)
        r_hid, z_hid, c_hid = hgates.chunk(3, 1)

        r_gate = torch.sigmoid(r_in+r_hid)
        z_gate = torch.sigmoid(z_in + z_hid)
        c = self.activ_func(c_in + (r_gate *c_hid))
        h_new = ((1-(z_gate)*c)) + ((z_gate) * hx)


        return h_new


class scriptGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(scriptGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, hx):
        inputs = input.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            hx = self.cell(inputs[i], hx)
            outputs.append(hx)
        return torch.stack(outputs), hx



def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class scriptGRU(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ['layers', 'num_layers']

    def __init__(self, input_dim, hidden_dim, num_layers, activ_func, batch_first = True):
        super(scriptGRU, self).__init__()
        self.layers = init_stacked_lstm(num_layers, scriptGRULayer, 
                                        [scriptGRUCell, input_dim, hidden_dim, activ_func],
                                        [scriptGRUCell, hidden_dim, hidden_dim, activ_func])        

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    @jit.script_method
    def forward(self, input, layers_hx):
        output_states = jit.annotate(List[Tensor], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            hx = layers_hx[i]
            output, out_state = rnn_layer(output, hx)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1

        return output.transpose(0,1), torch.stack(output_states)

class scriptSimpleNet(jit.ScriptModule): 
    def __init__(self, in_dim, hid_dim, num_layers, activ_func, drop_p=0.0):
        super(scriptSimpleNet, self).__init__()
        self.out_dim = 33
        self.isLang = False
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.predict_task = False
        self.activ_func = activ_func
        self.rnn = scriptGRU(in_dim, hid_dim, self.num_layers, activ_func, drop_p)
        self.tune_langModel = None

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

    @jit.script_method
    def forward(self, x, h): 
        rnn_out, hid = self.rnn(x, h)
        motor_out = self.W_out(rnn_out)
        out = torch.sigmoid(motor_out)

        return out, hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)


