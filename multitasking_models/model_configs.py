from dataclasses import dataclass, field
from task import Task

SENSORY_INPUT_DIM = Task.INPUT_DIM
MOTOR_OUTPUT_DIM = Task.OUTPUT_DIM

@dataclass
class SensorimotorModelConfig(): 
    model_name: str

    ##you'd want this to be dynamics so you can update with larger embeddings
    rnn_in_dim: int
    rnn_layers: int = 1
    sensorimotor_out_dim: int = MOTOR_OUTPUT_DIM
    rnn_hidden_dim: int = 128
    rnn_hiddenInitValue: int = 0.1
    rnn_activ_func: str = 'relu'

@dataclass
class LanguageModelConfig(): 
    LM_train_layers: list 
    LM_load_str: str
    LM_reducer: str = 'mean'
    LM_out_dim: int = 20
    LM_output_nonlinearity: str = 'relu'


@dataclass
class FullModelConfig(SensorimotorModelConfig): 
    LM: InstructionEmbedder = None
    LM_config: LanguageModelConfig = None
    rnn_in_dim = 

@dataclass
class SimpleNetConfig(SensorimotorModelConfig): 
    model_name: str = 'simpleNet'
    rule_dim: int = 20
    rnn_in_dim = int = rule_dim + SENSORY_INPUT_DIM


@dataclass
class GPTNetConfig(SensorimotorModelConfig): 
    LM: InstructionEmbedder = None
    embedder_config: EmbedderConfig = EmbedderConfig(LM_train_layers=[], LM_load_str='gpt2')


gpt_config = GPTConfig(LM_train_layers = [])

from multitasking_models.language_models import GPT, InstructionEmbedder

from multitasking_models.sensorimotor_models import SimpleNet

gpt_test = GPT(gpt_config)

gpt_test