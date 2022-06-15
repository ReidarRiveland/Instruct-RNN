import torch
import torch.nn as nn
from attrs import asdict, define, field
import pickle

from instructRNN.models.script_gru import ScriptGRU
from instructRNN.tasks.tasks import TASK_LIST
from instructRNN.models.language_models import InstructionEmbedder, LMConfig
from instructRNN.tasks.task_factory import INPUT_DIM, OUTPUT_DIM

SENSORY_INPUT_DIM = INPUT_DIM
MOTOR_OUTPUT_DIM = OUTPUT_DIM

@define
class BaseModelConfig(): 
    model_name: str 

    rnn_hidden_dim: int = 256
    rnn_layers: int = 1
    rnn_hiddenInitValue: int = 0.1
    rnn_activ_func: str = 'relu'
    _rnn_in_dim: int = field(kw_only=True)
    _sensorimotor_out_dim: int = MOTOR_OUTPUT_DIM

@define
class RuleModelConfig(BaseModelConfig): 
    add_rule_encoder: bool = False
    rule_encoder_hidden: int = 128
    rule_dim: int = 64

    _rnn_in_dim: int = field(kw_only=True)
    @_rnn_in_dim.default
    def _set_rnn_in_dim(self):
            return self.rule_dim + SENSORY_INPUT_DIM
    info_type: str = 'rule'

@define
class InstructModelConfig(BaseModelConfig): 
    LM_class: InstructionEmbedder = field(kw_only=True)
    LM_load_str: str = field(kw_only=True)
    LM_train_layers: list = field(kw_only=True)
    LM_reducer: str = 'mean' 
    LM_out_dim: int = 64
    LM_output_nonlinearity: str ='relu'
    LM_proj_out_layers: int = 1

    _rnn_in_dim: int = field(kw_only=True)
    @_rnn_in_dim.default
    def _set_rnn_in_dim(self):
        return self.LM_out_dim + SENSORY_INPUT_DIM

    info_type: str = 'lang'

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

class RuleEncoder(nn.Module):
    def __init__(self, rule_dim, hidden_size):
        super(RuleEncoder, self).__init__()
        self.rule_dim = rule_dim
        self.hidden_size = hidden_size
        self.rule_in = nn.Sequential(
                nn.Linear(self.rule_dim, self.hidden_size), 
                nn.ReLU(), 
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU()
                )
        self.rule_out = nn.Sequential(
                nn.Linear(self.hidden_size, self.rule_dim),
                nn.ReLU()
            )

    def forward(self, rule):
        return self.rule_out(self.rule_in(rule))

class RuleNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self._set_rule_transform()
        if self.add_rule_encoder: 
            self.rule_encoder = RuleEncoder(self.rule_dim, self.rule_encoder_hidden)
        else: 
            self.rule_encoder = nn.Identity()

    def _set_rule_transform(self):
        rule_folder = 'models/ortho_rule_vecs/'
        ortho_rules = pickle.load(open(rule_folder+'ortho_rules'+str(len(TASK_LIST))+'x'+str(self.rule_dim), 'rb'))
        self.rule_transform = torch.Tensor(ortho_rules)

    def forward(self, x, task_rule):
        rule_transformed = torch.matmul(task_rule.to(self.__device__), self.rule_transform)
        task_rule = self.rule_encoder(rule_transformed)
        outs, rnn_hid = super().forward(x, task_rule)
        return outs, rnn_hid

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.rule_transform = self.rule_transform.to(cuda_device)

class InstructNet(BaseNet): 
    def __init__(self, config): 
        super().__init__(config)
        self.LM_config = LMConfig(self.LM_load_str, 
                                self.LM_train_layers, 
                                self.LM_reducer,
                                self.LM_out_dim, 
                                self.LM_output_nonlinearity,
                                self.LM_proj_out_layers)

        self.langModel = self.LM_class(self.LM_config)

    def forward(self, x, instruction = None, context = None):
        assert instruction is not None or context is not None, 'must have instruction or context input'
        
        if instruction is not None: 
            info_embedded = self.langModel(instruction)
        elif context.shape[-1] == self.langModel.LM_intermediate_lang_dim:
            info_embedded = self.langModel.proj_out(context)
        elif context.shape[-1] == self.langModel.LM_out_dim:
            info_embedded = context

        outs, rnn_hid = super().forward(x, info_embedded)
        return outs, rnn_hid

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.langModel.__device__ = cuda_device
