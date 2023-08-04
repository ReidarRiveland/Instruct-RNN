from instructRNN.models.language_models import GPT, BERT, SBERT, CLIP, BoW
from instructRNN.models.sensorimotor_models import RuleModelConfig, InstructModelConfig, RuleNet, InstructNet

all_models = ['simpleNet', 'simpleNetPlus', 'combNet',

            'gptNetXL_lin', 'gptNetXL_lin_tuned',

            'gptNetXL_L_lin',

            'gptNet_lin', 'gptNet_lin_tuned',

            'gptNet_L_lin',

            'bertNet_lin', 'bertNet_lin_tuned',

            'sbertNet_lin', 'sbertNet_lin_tuned',

            'clipNet_lin', 'clipNet_lin_tuned', 

            'bowNet_lin'
            ]

small_models = [model for model in all_models if 'XL' not in model] 
big_models = [model for model in all_models if model not in small_models]
shallow_models = [model for model in all_models if 'simple' in model or 'com' in model]                        
untuned_models = [model_name for model_name in all_models if '_tuned' not in model_name]
tuned_models = [model_name for model_name in all_models if '_tuned' in model_name]

class SimpleNet(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNet', **kw_args)
        super().__init__(config)

class CombNet(RuleNet): 
    def __init__(self, **kw_args): 
        config = RuleModelConfig('combNet', info_type='comb', **kw_args)
        super().__init__(config)

class SimpleNetPlus(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNetPlus', add_rule_encoder=True, **kw_args)
        super().__init__(config)

class GPTNet_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet_lin', 
                                LM_class=GPT, 
                                LM_load_str = 'gpt2',
                                LM_output_nonlinearity='lin',
                                LM_train_layers=[],
                                **kw_args)
        super().__init__(config)

class GPTNet_L_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet_L_lin', 
                                LM_class=GPT, 
                                LM_load_str = 'gpt2',
                                LM_output_nonlinearity='lin',
                                LM_train_layers=[],
                                LM_reducer='last',
                                **kw_args)
        super().__init__(config)

class GPTNet_lin_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet_lin_tuned', 
                                LM_class=GPT, 
                                LM_load_str = 'gpt2',
                                LM_output_nonlinearity='lin',
                                LM_train_layers=['9', '10', '11', 'ln_f'],
                                **kw_args)
        super().__init__(config)

class GPTNetXL_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNetXL_lin', 
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2-xl', 
                                    LM_output_nonlinearity='lin',
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class GPTNetXL_L_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNetXL_L_lin', 
                                LM_class=GPT, 
                                LM_load_str = 'gpt2-xl',
                                LM_output_nonlinearity='lin',
                                LM_train_layers=[],
                                LM_reducer='last',
                                **kw_args)
        super().__init__(config)

class GPTNetXL_lin_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNetXL_lin_tuned', 
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2-xl', 
                                    LM_output_nonlinearity='lin',
                                    LM_train_layers=['9', '10', '11', 'ln_f'], 
                                    **kw_args)
        super().__init__(config)

class BERTNet_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bertNet_lin', 
                                    LM_class= BERT,
                                    LM_output_nonlinearity='lin',

                                    LM_load_str = 'bert-base-uncased',
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class BERTNet_lin_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bertNet_lin_tuned', 
                                    LM_class= BERT,
                                    LM_output_nonlinearity='lin',
                                    LM_load_str = 'bert-base-uncased',
                                    LM_train_layers=['9', '10', '11', 'pooler'],
                                    **kw_args)
        super().__init__(config)



class SBERTNet_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('sbertNet_lin', 
                                    LM_class= SBERT,
                                    LM_load_str = 'sbert-base-nli-mean-tokens.pt', 
                                    LM_output_nonlinearity='lin',
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class SBERTNet_lin_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('sbertNet_lin_tuned', 
                                    LM_class= SBERT,
                                    LM_load_str = 'sbert-base-nli-mean-tokens.pt', 
                                    LM_output_nonlinearity='lin', 
                                    LM_train_layers=['9', '10', '11', 'pooler'],
                                    **kw_args)
        super().__init__(config)
            
class CLIPNet_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('clipNet_lin', 
                                    LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-base-patch32',
                                    LM_output_nonlinearity='lin',
                                    LM_train_layers=['bias'],
                                    **kw_args)
        super().__init__(config)
        

class CLIPNet_lin_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('clipNet_lin_tuned', 
                                    LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-base-patch32',
                                    LM_output_nonlinearity='lin',

                                    LM_train_layers=['bias', '9', '10', '11', 'pooler'],
                                    **kw_args)
        super().__init__(config)

class BoWNet_lin(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bowNet_lin', 
                                    LM_class= BoW,
                                    LM_output_nonlinearity='lin',
                                    LM_load_str = '',
                                    LM_train_layers=[],
                                    **kw_args)
        super().__init__(config)
            

def make_default_model(model_str): 
    if model_str == 'simpleNet':
        return SimpleNet()
    if model_str == 'simpleNetPlus':
        return SimpleNetPlus()   
    
    if model_str == 'combNet':
        return CombNet()
    
    
    if model_str == 'gptNet_lin': 
        return GPTNet_lin()
    if model_str == 'gptNet_lin_tuned': 
        return GPTNet_lin_tuned()
    
    if model_str == 'gptNet_L_lin':
        return GPTNet_L_lin()
    
    if model_str == 'gptNetXL_L_lin':
        return GPTNetXL_L_lin()


    if model_str == 'bertNet_lin': 
        return BERTNet_lin()
    if model_str == 'bertNet_lin_tuned': 
        return BERTNet_lin_tuned()

    if model_str == 'sbertNet_lin': 
        return SBERTNet_lin()
    if model_str == 'sbertNet_lin_tuned': 
        return SBERTNet_lin_tuned()
    
    
    if model_str == 'gptNetXL_lin':
        return GPTNetXL_lin()
    if model_str == 'gptNetXL_lin_tuned':
        return GPTNetXL_lin_tuned()

    if model_str == 'clipNet_lin': 
        return CLIPNet_lin()
    if model_str == 'clipNet_lin_tuned': 
        return CLIPNet_lin_tuned()

    if model_str == 'bowNet_lin': 
        return BoWNet_lin()
    else: 
        raise Exception('Model not found in make_default_model function, make sure its included there')


