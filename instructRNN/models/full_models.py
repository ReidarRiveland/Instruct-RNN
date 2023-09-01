from instructRNN.models.language_models import GPT, BERT, SBERT, CLIP, RawBERT, BoW
from instructRNN.models.sensorimotor_models import RuleModelConfig, InstructModelConfig, RuleNet, InstructNet

all_models = ['simpleNet', 'simpleNetPlus', 'combNet', 'combNetPlus', 

            'gptNetXL', 'gptNetXL_lin', 'gptNetXL_lin_tuned', 'gptNetXL_L_lin', 

            'gptNet', 'gptNet_lin', 'gptNet_lin_tuned', 'gptNet_L_lin',

            'bertNet', 'bertNet_lin', 'bertNet_lin_tuned', 'rawBertNet_lin',

            'sbertNet', 'sbertNet_lin', 'sbertNet_lin_tuned',

            'sbertNetXL', 'sbertNetXL_lin', 'sbertNetXL_lin_tuned', 'sbertNetXL_lin_new', 

            'clipNet', 'clipNet_lin', 'clipNet_lin_tuned', 'clipNet_lin_new', 'clipNet_lin_new_tuned', 
            
            'clipNetXL', 'clipNetXL_lin', 'clipNetXL_lin_tuned',


            'bowNet', 'bowNet_lin_plus'
            ]

small_models = [model for model in all_models if 'XL' not in model] 
big_models = [model for model in all_models if model not in small_models]
shallow_models = [model for model in all_models if 'simple' in model or 'com' in model]                        
untuned_models = [model_name for model_name in all_models if '_tuned' not in model_name]
tuned_models = [model_name for model_name in all_models if '_tuned' in model_name]
nonlin_models = [model_name for model_name in all_models if 'lin' not in model_name 
                                                            and 'simple' not in model_name 
                                                            and model_name != 'combNet']

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

class CombNetPlus(RuleNet): 
    def __init__(self, **kw_args): 
        config = RuleModelConfig('combNet', info_type='comb', add_rule_encoder=True, **kw_args)
        super().__init__(config)

class GPTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(
                                LM_class=GPT, 
                                LM_load_str = 'gpt2',
                                **kw_args)
        super().__init__(config)

class GPTNetXL(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(
                                    LM_class= GPT,
                                    LM_load_str = 'gpt2-xl', 
                                    **kw_args)
        super().__init__(config)

class BERTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= BERT,
                                    LM_load_str = 'bert-base-uncased',
                                    **kw_args)
        super().__init__(config)

class RawBERTNet(InstructNet): 
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= RawBERT,
                                    LM_load_str = 'bert-base-uncased',
                                    **kw_args)
        super().__init__(config)


class SBERTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= SBERT,
                                    LM_load_str = 'sbert-base-nli-mean-tokens.pt', 
                                    **kw_args)
        super().__init__(config)

class SBERTNetXL(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= SBERT,
                                    LM_load_str = 'sentence-transformers/bert-large-nli-mean-tokens', 
                                    **kw_args)
        super().__init__(config)
            
class CLIPNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-base-patch32',
                                    **kw_args)
        super().__init__(config)
        
class CLIPNetXL(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-large-patch14',
                                    **kw_args)
        super().__init__(config)
        

class BoWNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig(LM_class= BoW,
                                    LM_load_str = '',
                                    **kw_args)
        super().__init__(config)
            

def make_default_model(model_str): 
    if model_str == 'simpleNet':
        return SimpleNet()
    if model_str == 'simpleNetPlus':
        return SimpleNetPlus()   
    if model_str == 'combNet':
        return CombNet()

    if model_str == 'combNetPlus':
        return CombNetPlus()
    
    if model_str == 'gptNet': 
        return GPTNet(model_name = model_str, 
                    LM_output_nonlinearity='relu',
                    LM_train_layers = [])

    if model_str == 'gptNet_lin': 
        return GPTNet(model_name=model_str, 
                    LM_output_nonlinearity='lin',
                    LM_train_layers = [])

    if model_str == 'gptNet_lin_tuned': 
        return GPTNet(model_name=model_str, 
                    LM_output_nonlinearity='lin', 
                    LM_train_layers=['9', '10', '11', 'ln_f'],
                    )
    
    if model_str == 'gptNetXL':
        return GPTNetXL(model_name=model_str, 
                    LM_output_nonlinearity='relu',
                    LM_train_layers = [])

    if model_str == 'gptNetXL_lin':
        return GPTNetXL(model_name=model_str, 
                    LM_output_nonlinearity='lin',
                    LM_train_layers = [])

    if model_str == 'gptNetXL_lin_tuned':
        return GPTNetXL(model_name=model_str, 
                    LM_output_nonlinearity='lin',
                    LM_train_layers = ['21', '22' '23', 'ln_f'])

    if model_str == 'gptNet_L_lin':
        return GPTNet(model_name=model_str, 
                    LM_output_nonlinearity='lin',
                    LM_train_layers = [], 
                    LM_reducer='last')


    if model_str == 'gptNetXL_L_lin':
        return GPTNetXL(model_name=model_str, 
                    LM_output_nonlinearity='lin',
                    LM_reducer = 'last',
                    LM_train_layers = [])

    if model_str == 'bertNet': 
        return BERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'relu', 
                        LM_train_layers = [])

    if model_str == 'bertNet_lin': 
        return BERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = [])

    if model_str == 'bertNet_lin_tuned': 
        return BERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = ['9', '10', '11', 'pooler'])
                        
    if model_str == 'rawBertNet_lin': 
        return RawBERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = ['0'])

    if model_str == 'sbertNet': 
        return SBERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'relu', 
                        LM_train_layers = [])

    if model_str == 'sbertNet_lin': 
        return SBERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = [])

    if model_str == 'sbertNet_lin_tuned': 
        return SBERTNet(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = ['9', '10', '11', 'pooler'])  

    if model_str == 'sbertNetXL': 
        return SBERTNetXL(model_name = model_str, 
                        LM_output_nonlinearity = 'relu', 
                        LM_train_layers = [])


    if model_str == 'sbertNetXL_lin': 
        return SBERTNetXL(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = [])
    

    if model_str == 'sbertNetXL_lin_new': 
        return SBERTNetXL(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = [])



    if model_str == 'sbertNetXL_lin_tuned': 
        return SBERTNetXL(model_name = model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_train_layers = ['21', '22', '23', 'pooler'])

    if model_str == 'clipNet': 
        return CLIPNet(model_name = model_str, 
                    LM_output_nonlinearity = 'relu', 
                    LM_train_layers = [])

    if model_str == 'clipNet_lin': 
        return CLIPNet(model_name=model_str, 
                    LM_output_nonlinearity = 'lin', 
                    LM_train_layers = [])

    if model_str == 'clipNet_lin_new': 
        return CLIPNet(model_name=model_str, 
                    LM_output_nonlinearity = 'lin', 
                    LM_train_layers = [])

    if model_str == 'clipNet_lin_new_tuned': 
        return CLIPNet(model_name=model_str, 
                    LM_output_nonlinearity = 'lin', 
                    LM_train_layers = ['9', '10', '11', 'text_projection', 'pooler'])

    if model_str == 'clipNetXL': 
        return CLIPNetXL(model_name = model_str, 
                    LM_output_nonlinearity = 'relu', 
                    LM_train_layers = [])

    if model_str == 'clipNetXL_lin': 
        return CLIPNetXL(model_name=model_str, 
                    LM_output_nonlinearity = 'lin', 
                    LM_train_layers = [])


    if model_str == 'clipNetXL_lin_tuned': 
        return CLIPNetXL(model_name=model_str, 
                    LM_output_nonlinearity = 'lin', 
                    LM_train_layers = ['9', '10', '11', 'text_projection', 'pooler'])

    if model_str == 'bowNet': 
        return BoWNet(model_name=model_str, 
                        LM_output_nonlinearity = 'relu', 
                            LM_train_layers = [])

    if model_str == 'bowNet_lin': 
        return BoWNet(model_name=model_str, 
                        LM_output_nonlinearity = 'lin',                     
                        LM_train_layers = [])


    if model_str == 'bowNet_lin_plus': 
        return BoWNet(model_name=model_str, 
                        LM_output_nonlinearity = 'lin', 
                        LM_proj_out_layers=5,
                        LM_train_layers = [])


    raise Exception('Model not found in make_default_model function, make sure its included there')



