from language_models import GPT, BERT, SBERT, GPTNeo
from sensorimotor_models import SimpleNet, InstructNet



class GPTNet(InstructNet): 
    InstructNet.__init__(gptnet_config)
