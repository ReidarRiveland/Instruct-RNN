import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.jit as jit

import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers

class scriptGRUCell(jit.ScriptModule): 
    def __init__(self, input_size, hidden_size, activ_func): 
        super(scriptGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.activ_func = activ_func

        torch.nn.init.normal_(self.bias_ih, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bias_hh, mean=0.0, std=1.0)

    @jit.script_method
    def forward(self, input, hx):
        igates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih)
        hgates = (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        r_in, z_in, c_in = igates.chunk(3, 1)
        r_hid, z_hid, c_hid = hgates.chunk(3, 1)

        r_gate = torch.sigmoid(r_in+r_hid)
        z_gate = torch.sigmoid(z_in + z_hid)
        pre_c = c_in + (r_gate *c_hid)
        c = self.activ_func(pre_c)
        h_new = ((1-z_gate)*c) + (z_gate * hx)

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

def init_stacked_GRU(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class ScriptGRU(jit.ScriptModule):
    __constants__ = ['num_layers']

    def __init__(self, input_dim, hidden_dim, num_layers, activ_func, batch_first = True):
        super(ScriptGRU, self).__init__()
        self.layers = init_stacked_GRU(num_layers, scriptGRULayer, 
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
        for i, rnn_layer in enumerate(self.layers):
            hx = layers_hx[i]
            output, out_state = rnn_layer(output, hx)
            output_states += [out_state]


        return output.transpose(0,1), torch.stack(output_states)
