from tasks import TASK_LIST
from models.language_models import InstructionEmbedder
from attrs import define, field
from tasks.task_factory import INPUT_DIM, OUTPUT_DIM

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
class LMConfig(): 
    LM_load_str: str
    LM_train_layers: list 
    LM_reducer: str 
    LM_out_dim: int 
    LM_output_nonlinearity: str 
    LM_proj_out_layers: int

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
    is_instruct: bool = True

@define
class RuleModelConfig(BaseModelConfig): 
    add_rule_encoder: bool = False
    rule_encoder_hidden = 128
    rule_dim: int = len(TASK_LIST)
    rule_transformer_dim: int = 64

    _rnn_in_dim: int = field(kw_only=True)
    @_rnn_in_dim.default
    def _set_rnn_in_dim(self):
            return self.rule_dim + SENSORY_INPUT_DIM
    is_instruct: bool = False




