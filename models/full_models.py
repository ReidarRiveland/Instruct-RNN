from models.language_models import GPT, BERT, SBERT, GPTNeo, BoW, CLIP
from models.sensorimotor_models import RuleNet, InstructNet
from models.model_configs import RuleModelConfig, InstructModelConfig, LMConfig

_all_models = ['clipNet', 'clipNet_tuned', 
            'sbertNet_tuned', 'sbertNet', 
            'bertNet_tuned', 'bertNet', 
            'gptNet_tuned', 'gptNet', 
            'gptNeoNet', 'gptNeoNet_tuned', 
            'bowNet', 
            'simpleNet']

class SimpleNet(RuleNet):
    DEFAULT_CONFIG = RuleModelConfig(model_name = 'simpleNet')
    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class GPTNet(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'gptNet', 
                        LM_class=GPT, 
                        LM_config=LMConfig(
                                    LM_load_str = 'gpt2',
                                    LM_train_layers=[])
                                    )
    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class GPTNet_tuned(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'gptNet_tuned', 
                        LM_class=GPT, 
                        LM_config=LMConfig(
                                LM_load_str = 'gpt2',
                                LM_train_layers=['9', '10', '11', 'ln_f'])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class BERTNet(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'bertNet', 
                        LM_class=BERT, 
                        LM_config=LMConfig(
                                LM_load_str = 'bert-base-uncased',
                                LM_train_layers=[])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class BERTNet_tuned(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'bertNet_tuned', 
                        LM_class=BERT, 
                        LM_config=LMConfig(
                                LM_load_str = 'bert-base-uncased',
                                LM_train_layers=['9', '10', '11', 'pooler'])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class SBERTNet(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'sbertNet', 
                        LM_class=SBERT, 
                        LM_config=LMConfig(
                                LM_load_str = 'sbert_raw.pt',
                                LM_train_layers=[])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class SBERTNet_tuned(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'sbertNet', 
                        LM_class=SBERT, 
                        LM_config=LMConfig(
                                LM_load_str = 'sbert_raw.pt',
                                LM_train_layers=['9', '10', '11', 'pooler'])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)

class GPTNeoNet(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'gptNeoNet', 
                        LM_class=GPTNeo, 
                        LM_config=LMConfig(
                                LM_load_str = "EleutherAI/gpt-neo-1.3B",
                                LM_train_layers=[])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)
        

class GPTNeoNet_tuned(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'gptNeoNet_tuned', 
                        LM_class=GPTNeo, 
                        LM_config=LMConfig(
                                LM_load_str = "EleutherAI/gpt-neo-1.3B",
                                LM_train_layers=['21', '22', '23', 'ln_f'])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)
        

class CLIPNet(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'clipNet', 
                        LM_class=CLIP, 
                        LM_config=LMConfig(
                                LM_load_str = 'openai/clip-vit-base-patch32',
                                LM_train_layers=[])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)
        

class CLIPNet_tuned(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'clipNet_tuned', 
                        LM_class=CLIP, 
                        LM_config=LMConfig(
                                LM_load_str = 'openai/clip-vit-base-patch32',
                                LM_train_layers=['9', '10', '11', 'pooler'])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)
        
        
class BoWNet(InstructNet):
    DEFAULT_CONFIG = InstructModelConfig(model_name = 'bowNet', 
                        LM_class=BoW, 
                        LM_config=LMConfig(
                                LM_load_str = '',
                                LM_train_layers=[])
                                )

    def __init__(self, config=DEFAULT_CONFIG): 
        super().__init__(config)


def make_default_model(model_str): 
    assert model_str in _all_models, 'invalid model name'
    if model_str == 'simpleNet':
        return SimpleNet()
    if model_str == 'gptNet': 
        return GPTNet()
    if model_str == 'gptNet_tuned': 
        return GPTNet_tuned()
    if model_str == 'bertNet': 
        return BERTNet()
    if model_str == 'bertNet_tuned': 
        return BERTNet_tuned()
    if model_str == 'sbertNet': 
        return SBERTNet()
    if model_str == 'sbertNet_tuned': 
        return SBERTNet_tuned()
    if model_str == 'gptNeoNet': 
        return GPTNeoNet()
    if model_str == 'gptNeoNet_tuned': 
        return GPTNeoNet_tuned()
    if model_str == 'clipNet': 
        return CLIPNet()
    if model_str == 'clipNet_tuned': 
        return CLIPNet_tuned()
    if model_str == 'bowNet': 
        return BoWNet()
