import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
from torch.autograd import Variable
import torch.nn.functional as functional
import math
import torch.nn.functional as F
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class myGRUCell(nn.Module): 
    def __init__(self, input_size, hidden_size, activ_func): 
        super(myGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        self.activ_func = activ_func
        self.reset_parameters()
        self.alpha = (1/5)
        self.sigma = math.sqrt(2/self.alpha) * 0.05


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx):
        igates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih)
        hgates = (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        r_in, z_in, c_in = igates.chunk(3, 1)
        r_hid, z_hid, c_hid = hgates.chunk(3, 1)

        r_gate = torch.sigmoid(r_in+r_hid)
        z_gate = torch.sigmoid(z_in + z_hid)
        c = self.activ_func(c_in + (r_gate *c_hid))
        #c += (self.sigma*torch.randn_like(c))
        #h_new = ((1-z_gate)*c) + (z_gate * hx)
        h_new = ((1- (self.alpha * z_gate)) * hx) + ((self.alpha * z_gate) * c)

        return h_new
    

class myGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, activ_func = torch.tanh,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(myGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activ_func = activ_func
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = myGRUCell(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              activ_func = self.activ_func)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.reset_parameters()
        self.dropout_layer = nn.Dropout(dropout)

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

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
                layer_output, layer_h_n = myGRU._forward_rnn(
                    cell, input_, hx_layer)
            else:
                layer_output = self.dropout_layer(layer_output)
                layer_output, layer_h_n = myGRU._forward_rnn(
                    cell, layer_output,  hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)

        output = layer_output
        h_n = torch.stack(h_n, 0)
        if self.batch_first: 
            output = output.transpose(0, 1)
        return output, h_n


class mySimpleNet(nn.Module): 
    def __init__(self, in_dim, hid_dim, num_layers, activ_func,instruct_mode=None, dropout=0.0):
        super(mySimpleNet, self).__init__()
        self.out_dim = 33
        self.in_dim = in_dim
        self.isLang = False
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.predict_task = False
        self.activ_func = activ_func
        self.tune_langModel = None
        self.instruct_mode = instruct_mode
        self.rnn = myGRU(in_dim, hid_dim, num_layers = self.num_layers, 
                            activ_func = activ_func, batch_first=True, dropout=dropout)

        self.W_out = nn.Linear(hid_dim, self.out_dim)
        self.weights_init()

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
        rnn_out, hid = self.rnn(x, h)
        motor_out = self.W_out(rnn_out)
        out = torch.sigmoid(motor_out)

        return out, hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)


class myInstructNet(nn.Module): 
    def __init__(self, langMod, hid_dim, num_layers, activ_func, drop_p = 0.0, instruct_mode=None, tune_langModel = False): 
        super(myInstructNet, self).__init__()
        self.instruct_mode = instruct_mode
        self.tune_langModel = tune_langModel
        self.sensory_in_dim = 65
        self.isLang = True 
        self.hid_dim = hid_dim
        self.embedderStr = langMod.embedderStr
        self.langModel = langMod.langModel.eval()
        self.langMod = langMod
        self.num_layers = num_layers
        self.lang_embed_dim = langMod.langModel.out_dim
        self.rnn = mySimpleNet(self.sensory_in_dim+self.lang_embed_dim, hid_dim, self.num_layers, activ_func)

        if tune_langModel:
            self.langModel.train()
            for param in self.langModel.parameters(): 
                param.requires_grad = True
        else: 
            for param in self.langModel.parameters(): 
                param.requires_grad = False



    def forward(self, instruction_tensor, x, h):
        embedded_instruct = self.langModel(instruction_tensor)
        seq_blocked = embedded_instruct.unsqueeze(1).repeat(1, 120, 1)
        rnn_ins = torch.cat((seq_blocked, x.type(torch.float32)), 2)
        outs, hid = self.rnn(rnn_ins, h)
        return outs, hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)



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
        h_new = ((1-(0.2*z_gate)*c)) + ((0.2*z_gate) * hx)
        #h_new = ((1- (z_gate)) * hx) + ((z_gate) * c)


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

    def __init__(self, input_dim, hidden_dim, num_layers, activ_func, drop_p, batch_first = True):
        super(scriptGRU, self).__init__()
        self.layers = init_stacked_lstm(num_layers, scriptGRULayer, 
                                        [scriptGRUCell, input_dim, hidden_dim, activ_func],
                                        [scriptGRUCell, hidden_dim, hidden_dim, activ_func])        

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if (num_layers == 1 and drop_p != 0):
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        self.dropout_layer = nn.Dropout(drop_p)

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

    @jit.script_method
    def forward(self, x, h): 
        rnn_out, hid = self.rnn(x, h)
        motor_out = self.W_out(rnn_out)
        out = torch.sigmoid(motor_out)

        return out, hid

    def initHidden(self, batch_size, value):
        return torch.full((self.num_layers, batch_size, self.hid_dim), value)


