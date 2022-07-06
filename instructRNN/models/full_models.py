from instructRNN.models.language_models import *
from instructRNN.models.sensorimotor_models import *

all_models = ['simpleNet', 'simpleNetPlus',
            'comNet', 'comNetPlus',
            'gptNet', 'gptNet_tuned',
            'gptNetXL', 'gptNetXL_tuned',
            'bertNet', 'bertNet_tuned',
            'sbertNet', 'sbertNet_tuned', 
            #'sbertNet_lin', 'sbertNet_lin_tuned',
            'clipNet',
            'bowNet', 'clipNet_tuned', 
            ]
            
untuned_models = [model_name for model_name in all_models if '_tuned' not in model_name]
tuned_models = [model_name for model_name in all_models if '_tuned' in model_name]

class SimpleNet(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNet', **kw_args)
        super().__init__(config)

class SimpleNetPlus_rand(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNetPlus_rand',
                                add_rule_encoder=True,
                                use_rand_rnn=True,
                                 **kw_args)
        super().__init__(config)

class SimpleNetPlus(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('simpleNetPlus', add_rule_encoder=True, **kw_args)
        super().__init__(config)

class ComNet(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('comNet', info_type='comp', **kw_args)
        super().__init__(config)

class ComNetPlus(RuleNet):
    def __init__(self, **kw_args):
        config = RuleModelConfig('comNetPlus', add_rule_encoder=True, info_type='comp', **kw_args)
        super().__init__(config)

class GPTNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet', 
                                LM_class=GPT, 
                                LM_load_str = 'gpt2',
                                LM_train_layers=[],
                                **kw_args)
        super().__init__(config)

class GPTNet_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('gptNet_tuned', 
                                LM_class=GPT, 
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
                                    LM_load_str = 'sbert-base-nli-mean-tokens.pt', 
                                    LM_train_layers=[], 
                                    **kw_args)
        super().__init__(config)

class SBERTNet_tuned(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('sbertNet_tuned', 
                                    LM_class= SBERT,
                                    LM_load_str = 'sbert-base-nli-mean-tokens.pt', 
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


class CLIPNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('clipNet', 
                                    LM_class= CLIP,
                                    LM_load_str = 'openai/clip-vit-base-patch32',
                                    LM_train_layers=['bias'],
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
            
# class SIFNet(InstructNet):
#     def __init__(self, **kw_args):
#         config = InstructModelConfig('sifNet', 
#                                     LM_class= SIF,
#                                     LM_load_str = 'sif_model',
#                                     LM_proj_out_layers=2,
#                                     LM_train_layers=[],
#                                     **kw_args)
#         super().__init__(config)
            

class BoWNet(InstructNet):
    def __init__(self, **kw_args):
        config = InstructModelConfig('bowNet', 
                                    LM_class= BoW,
                                    LM_load_str = '',
                                    LM_train_layers=[],
                                    **kw_args)
        super().__init__(config)
            

def make_default_model(model_str): 
    if model_str == 'simpleNet':
        return SimpleNet()
    if model_str == 'simpleNetPlus':
        return SimpleNetPlus()
    if model_str == 'comNet':
        return ComNet()
    if model_str == 'comNetPlus':
        return ComNetPlus()
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
    if model_str == 'sbertNet_lin': 
        return SBERTNet_lin()
    if model_str == 'sbertNet_lin_tuned': 
        return SBERTNet_lin_tuned()
    if model_str == 'gptNetXL':
        return GPTNetXL()
    if model_str == 'gptNetXL_tuned':
        return GPTNetXL_tuned()
    if model_str == 'simpleNetPlus_rand':
        return SimpleNetPlus_rand()
    if model_str == 'clipNet': 
        return CLIPNet()
    if model_str == 'clipNet_tuned': 
        return CLIPNet_tuned()
    if model_str == 'bowNet': 
        return BoWNet()
    else: 
        raise Exception('Model not found in make_default_model function, make sure its included there')

