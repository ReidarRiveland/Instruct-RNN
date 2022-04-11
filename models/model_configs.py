from task import Task
from models.language_models import InstructionEmbedder
from attrs import define, field


SENSORY_INPUT_DIM = Task.INPUT_DIM
MOTOR_OUTPUT_DIM = Task.OUTPUT_DIM

@define
class BaseModelConfig(): 
    model_name: str 

    rnn_hidden_dim: int = 128
    rnn_layers: int = 1
    rnn_hiddenInitValue: int = 0.1
    rnn_activ_func: str = 'relu'
    _rnn_in_dim: int = field(kw_only=True)
    _sensorimotor_out_dim: int = MOTOR_OUTPUT_DIM

@define
class LMConfig(): 
    LM_train_layers: list 
    LM_load_str: str
    LM_reducer: str = 'mean'
    LM_out_dim: int = 20
    LM_output_nonlinearity: str = 'relu'

@define
class InstructModelConfig(BaseModelConfig): 
    LM_class: InstructionEmbedder = field(kw_only=True)
    LM_config: LMConfig = field(kw_only=True)

    _rnn_in_dim: int = field(kw_only=True)
    @_rnn_in_dim.default
    def _set_rnn_in_dim(self):
        return self.LM_config.LM_out_dim + SENSORY_INPUT_DIM
    _is_instruct: bool = True

@define
class RuleModelConfig(BaseModelConfig): 
    _rule_dim: int = 20

    _rnn_in_dim: int = field(kw_only=True)
    @_rnn_in_dim.default
    def _set_rnn_in_dim(self):
            return self._rule_dim + SENSORY_INPUT_DIM
    _is_instruct: bool = False



