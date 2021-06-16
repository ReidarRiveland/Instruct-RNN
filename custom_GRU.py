import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


class CustomGRUCell(nn.Module): 
    def __init__(self, input_size, hidden_size, activ_func): 
        super(CustomGRUCell, self).__init__()
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

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, activ_func = torch.tanh,
                 use_bias=True, batch_first=False, dropout=0.0):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activ_func = activ_func
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = CustomGRUCell(input_size=layer_input_size,
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
                layer_output, layer_h_n = CustomGRU._forward_rnn(
                    cell, input_, hx_layer)
            else:
                layer_output = self.dropout_layer(layer_output)
                layer_output, layer_h_n = CustomGRU._forward_rnn(
                    cell, layer_output,  hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)

        output = layer_output
        h_n = torch.stack(h_n, 0)
        if self.batch_first: 
            output = output.transpose(0, 1)
        return output, h_n
