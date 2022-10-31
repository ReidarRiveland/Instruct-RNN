from inspect import Parameter
import torch
import torch.nn as nn
from attrs import asdict, define, field
import pathlib
import pickle
from scipy.stats import ortho_group
from collections import OrderedDict


from instructRNN.models.script_gru import ScriptGRU
from instructRNN.tasks.tasks import TASK_LIST, construct_trials
from instructRNN.models.language_models import InstructionEmbedder, LMConfig
from instructRNN.tasks.task_factory import INPUT_DIM, OUTPUT_DIM, TRIAL_LEN
from instructRNN.instructions.instruct_utils import get_input_rule, get_instructions, one_hot_input_rule

SENSORY_INPUT_DIM = INPUT_DIM
MOTOR_OUTPUT_DIM = OUTPUT_DIM
location = str(pathlib.Path(__file__).parent.absolute())


@define
class BaseModelConfig(): 
    model_name: str 

    rnn_hidden_dim: int = 256
    rnn_layers: int = 1
    rnn_hiddenInitValue: int = 0.1
    rnn_activ_func: str = 'relu'
    use_rand_rnn: bool = False
    _rnn_in_dim: int = field(kw_only=True)
    _sensorimotor_out_dim: int = MOTOR_OUTPUT_DIM

@define
class RuleModelConfig(BaseModelConfig): 
    num_tasks: int = len(TASK_LIST)
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

        if self.use_rand_rnn: 
            self.freeze_rnn_weights()

        self.__device__ = torch.device('cpu')

    def set_inactiv_units(self, units_idx): 
        self.recurrent_units._set_inactiv_mask(units_idx)

    def __initHidden__(self, batch_size):
        return torch.full((self.rnn_layers, batch_size, self.rnn_hidden_dim), 
                self.rnn_hiddenInitValue, device=torch.device(self.__device__))

    def expand_info(self, task_info, duration, onset): 
        task_info_block = torch.zeros((task_info.shape[0], TRIAL_LEN, task_info.shape[-1]))
        task_info = task_info.unsqueeze(1).repeat(1, duration, 1)
        task_info_block[:, onset:onset+duration, :] = task_info
        return task_info_block

    def forward(self, x, task_info, info_duration=TRIAL_LEN, info_onset=0): 
        h0 = self.__initHidden__(x.shape[0])
        task_info_block = self.expand_info(task_info, info_duration, info_onset)
        rnn_ins = torch.cat((task_info_block.to(self.__device__), x.type(torch.float32)), 2)
        rnn_hid, _ = self.recurrent_units(rnn_ins, h0)
        out = self.sensory_motor_outs(rnn_hid)
        return out, rnn_hid

    def freeze_weights(self): 
        for p in self.parameters(): 
            p.requires_grad=False
    
    def freeze_all_but_rnn_ins(self):
        for n, p in self.named_parameters(): 
            if 'ih' not in n: p.requires_grad=False
    
    def freeze_recurrent_weights(self): 
        for n, p in self.recurrent_units.named_parameters(): 
            if 'hh' in n: p.requires_grad=False

    def save_model(self, file_path, suffix=''): 
        torch.save(self.state_dict(),
            file_path+'/'+self.model_name+suffix+'.pt')

    def load_model(self, file_path, suffix=''): 
        self.load_state_dict(torch.load(file_path+'/'+self.model_name+suffix+'.pt', 
                map_location='cpu'), strict=False)

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.recurrent_units._mask_to(cuda_device)
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
    _tuning_epoch=None
    def __init__(self, config):
        super().__init__(config)
        self.rule_transform = nn.Parameter(self.gen_ortho_rules(), requires_grad=False)

        if self.add_rule_encoder: 
            self.rule_encoder = RuleEncoder(self.rule_dim, self.rule_encoder_hidden)
        else: 
            self.rule_encoder = nn.Identity()

    def gen_ortho_rules(self): 
        m = ortho_group.rvs(dim=self.rule_dim)
        ortho = m[:self.num_tasks,:]
        return torch.tensor(ortho)

    def forward(self, x, task_rule=None, context = None, comp_task =None):

        if comp_task is not None: 
            ref_tasks = construct_trials(comp_task, None).comp_ref_tasks
            task_infos = [one_hot_input_rule(task) for task in ref_tasks]
            comp_rule = (task_infos[0] - task_infos[1]) + task_infos[2]
            rule_transformed = torch.matmul(comp_rule.to(self.__device__), self.rule_transform.float())
            task_rule = self.rule_encoder(rule_transformed)

        elif task_rule is not None:
            rule_transformed = torch.matmul(task_rule.to(self.__device__), self.rule_transform.float())
            task_rule = self.rule_encoder(rule_transformed)
        else: 
            task_rule = context
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
    

    def forward(self, x, instruction = None, context = None, comp_task=None):
        assert instruction is not None or context is not None, 'must have instruction or context input'
        

        if instruction is not None: 
            info_embedded = self.langModel(instruction)
        elif comp_task is not None: 
            ref_tasks = construct_trials(comp_task, None).comp_ref_tasks
            task_infos = [self.langModel(get_instructions(x.shape[0], task)) for task in ref_tasks]
            info_embedded = (task_infos[0] - task_infos[1]) + task_infos[2]
        elif context.shape[-1] == self.langModel.LM_intermediate_lang_dim:
            info_embedded = self.langModel.proj_out(context)
        elif context.shape[-1] == self.langModel.LM_out_dim:
            info_embedded = context

        outs, rnn_hid = super().forward(x, info_embedded)
        return outs, rnn_hid

    def save_model(self, file_path, suffix=''):
        if 'bow' in self.model_name: 
            super().save_model(file_path, suffix)
        else: 
            reduced_state_dict = OrderedDict()
            load_layers = self.langModel.LM_train_layers[:]

            if self.model_name == 'clipNet_lin_tuned' or self.model_name == 'clipNet_tuned': 
                load_layers += ['bias']


            for n, p in self.state_dict().items(): 
                if 'transformer' not in n or any([layer in n for layer in load_layers]): 
                    reduced_state_dict[n] = p

            torch.save(reduced_state_dict,
                file_path+'/'+self.model_name+suffix+'.pt')

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.langModel.__device__ = cuda_device




