import torch
import torch.nn as nn
from attrs import asdict
import pickle
from script_gru import ScriptGRU

class BaseNet(nn.Module): 
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config=config
        for name, value in asdict(config, recurse=False).items(): 
            setattr(self, name, value)

        if self.rnn_activ_func == 'relu':
            self._activ_func = torch.relu

        self.recurrent_units = ScriptGRU(self._rnn_in_dim, 
                                        self.rnn_hidden_dim, 
                                        self.rnn_layers, 
                                        activ_func = self._activ_func, 
                                        batch_first=True)

        self.sensory_motor_outs = nn.Sequential(
                                    nn.Linear(self.rnn_hidden_dim, self._sensorimotor_out_dim), 
                                    nn.Sigmoid())

        self.recurrent_units.__weights_init__()
        self.__device__ = torch.device('cpu')

    def __initHidden__(self, batch_size):
        return torch.full((self.rnn_layers, batch_size, self.rnn_hidden_dim), 
                self.rnn_hiddenInitValue, device=self.__device__.type)

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

    def save_model(self, file_path, suffix=''): 
        torch.save(self.state_dict(),
            file_path+'/'+self.model_name+suffix+'.pt')

    def load_model(self, file_path, suffix=''): 
        self.load_state_dict(torch.load(
            file_path+'/'+self.model_name+suffix+'.pt', 
            map_location='cpu'))

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.__device__ = cuda_device

class RuleNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        ortho_rules = pickle.load(open('ortho_rule_vecs', 'rb'))
        self.rule_transform = torch.Tensor(ortho_rules)

    def forward(self, x, task_rule):
        task_rule = torch.matmul(task_rule, self.rule_transform)
        outs, rnn_hid = super().forward(task_rule, x)
        return outs, rnn_hid

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.rule_transform = self.rule_transform.to(cuda_device)

class InstructNet(BaseNet): 
    def __init__(self, config): 
        super().__init__(config)
        self.langModel = self.LM_class(self.LM_config)

    def forward(self, x, instruction = None, context = None):
        assert instruction is not None or context is not None, 'must have instruction or context input'
        if instruction is not None: info_embedded = self.langModel(instruction)
        else: info_embedded = context

        outs, rnn_hid = super().forward(x, info_embedded)
        return outs, rnn_hid

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.langModel.__device__ = cuda_device
