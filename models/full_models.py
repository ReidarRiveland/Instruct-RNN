from models.language_models import GPT, BERT, SBERT, BoW, CLIP
from models.sensorimotor_models import RuleNet, InstructNet
from models.model_configs import RuleModelConfig, InstructModelConfig

_all_models = ['clipNet', 'clipNet_tuned', 
            'sbertNet_tuned', 'sbertNet', 
            'bertNet_tuned', 'bertNet', 
            'gptNet_tuned', 'gptNet', 
            'gptNetXL_tuned', 'gptNetXL', 

            'gptNeoNet', 'gptNeoNet_tuned', 
            'bowNet', 
            'simpleNet', 'simpleNetPlus']

class SimpleNet(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNet', **kw_args)
        super().__init__(config)

class SimpleNetPlus(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNetPlus', add_rule_encoder=True, **kw_args)
        super().__init__(config)

class GPTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet', 
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2', 
                                    LM_train_layers=[], 
                                    LM_proj_out_layers=3,
                                    **kw_args)
        super().__init__(config)

class GPTNet_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet_tuned', 
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2', 
                                    LM_train_layers=['9', '10', '11', 'ln_f'], 
                                    **kw_args)
        super().__init__(config)

class GPTNetXL(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNetXL', 
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2-xl', 
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class GPTNetXL_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNetXL_tuned', 
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2-xl', 
                                    LM_train_layers=['9', '10', '11', 'ln_f'], 
                                    **kw_args)
        super().__init__(config)

class BERTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bertNet', 
                                    LM_class= BERT,
                                    LM_load_str = 'bert-base-uncased',
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class BERTNet_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bertNet_tuned', 
                                    LM_class= BERT,
                                    LM_load_str = 'bert-base-uncased',
                                    LM_train_layers=['9', '10', '11', 'pooler'],
                                    **kw_args)
        super().__init__(config)

class SBERTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('sbertNet', 
                                    LM_class= SBERT,
                                    LM_load_str = 'sbert_raw.pt', 
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class SBERTNet_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('sbertNet_tuned', 
                                    LM_class= SBERT,
                                    LM_load_str = 'sbert_raw.pt', 
                                    LM_train_layers=['9', '10', '11', 'pooler'],
                                    **kw_args)
        super().__init__(config)

class CLIPNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('clipNet', 
                                    LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-base-patch32',
                                    LM_train_layers=[],
                                    **kw_args)
        super().__init__(config)
        

class CLIPNet_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('clipNet_tuned', 
                                    LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-base-patch32',
                                    LM_train_layers=['9', '10', '11', 'pooler'],
                                    **kw_args)
        super().__init__(config)
            
class BoWNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bowNet', 
                                    LM_class= BoW,
                                    LM_load_str = '',
                                    LM_train_layers=[],
                                    **kw_args)
        super().__init__(config)
            

def make_default_model(model_str): 
    assert model_str in _all_models, 'invalid model name'
    if model_str == 'simpleNet':
        return SimpleNet()
    if model_str == 'simpleNetPlus':
        return SimpleNetPlus()
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
    if model_str == 'gptNetXL':
        return GPTNetXL()
    if model_str == 'gptNetXL_tuned':
        return GPTNetXL_tuned()
    if model_str == 'clipNet': 
        return CLIPNet()
    if model_str == 'clipNet_tuned': 
        return CLIPNet_tuned()
    if model_str == 'bowNet': 
        return BoWNet()

