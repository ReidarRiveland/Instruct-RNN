import torch
import torch.nn as nn
from attrs import asdict
import pickle
from script_gru import ScriptGRU
from tasks import TASK_LIST

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

        self.__device__ = torch.device('cpu')

    def __initHidden__(self, batch_size):
        return torch.full((self.rnn_layers, batch_size, self.rnn_hidden_dim), 
                self.rnn_hiddenInitValue, device=torch.device(self.__device__))

    def expand_info(self, task_info, duration, onset): 
        task_info_block = torch.zeros((task_info.shape[0], 120, task_info.shape[-1]))
        task_info = task_info.unsqueeze(1).repeat(1, duration, 1)
        task_info_block[:, onset:onset+duration, :] = task_info
        return task_info_block

    def forward(self, x, task_info, info_duration=120, info_onset=0): 
        h0 = self.__initHidden__(x.shape[0])
        task_info_block = self.expand_info(task_info, info_duration, info_onset)
        rnn_ins = torch.cat((task_info_block.to(self.__device__), x.type(torch.float32)), 2)
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
        if self.add_rule_encoder: 
            self.rule_encoder = nn.Sequential(
                nn.Linear(self.rule_dim, 128), 
                nn.ReLU(), 
                nn.Linear(128, 128),
                nn.ReLU(), 
                nn.Linear(128, self.rule_dim),
                nn.ReLU()
            )
        else: 
            self.rule_encoder = nn.Identity()

    def forward(self, x, task_rule):
        task_rule = self.rule_encoder(torch.matmul(task_rule.to(self.__device__), self.rule_transform))
        outs, rnn_hid = super().forward(x, task_rule)
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
        elif context.shape[-1] == self.langModel.LM_intermediate_lang_dim:
            info_embedded = self.langModel.proj_out(context)
        else:
            info_embedded = context

        outs, rnn_hid = super().forward(x, info_embedded)
        return outs, rnn_hid

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.langModel.__device__ = cuda_device
