import torch
import torch.nn as nn
from attrs import asdict

from transformers import GPT2Model, GPT2Tokenizer, GPTNeoForCausalLM
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertModel, BertTokenizer
from transformers import GPTNeoForCausalLM

from utils.utils import sort_vocab

class InstructionEmbedder(nn.Module): 
    def __init__(self, config): 
        super(InstructionEmbedder, self).__init__()
        self.config=config
        for name, value in asdict(config).items(): 
            setattr(self, name, value)

        if self.LM_output_nonlinearity == 'relu': 
            self._output_nonlinearity = nn.ReLU()
        
        if self.LM_reducer == 'mean': 
            self._reducer = torch.mean

        self.__device__ = 'cpu'

    def __init_proj_out__(self): 
        self.proj_out = nn.Sequential(
            nn.Linear(self.LM_intermediate_lang_dim, self.LM_out_dim), 
            self._output_nonlinearity)

    def set_train_layers(self, train_layers): 
        all_train_layers = train_layers+['proj_out']
        for n,p in self.named_parameters(): 
            if any([layer in n for layer in all_train_layers]):
                p.requires_grad=True
            else: 

                p.requires_grad=False

class TransformerEmbedder(InstructionEmbedder): 
    def __init__(self, config): 
        super().__init__(config)

    def tokens_to_tensor(self, x):
        tokens = self.tokenizer(x, return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(self.__device__)
        return tokens

    def forward_transformer(self, x): 
        tokens = self.tokens_to_tensor(x)
        trans_out = self.transformer(**tokens)
        return self._reducer(trans_out.last_hidden_state, dim=1), trans_out[2]

    def forward(self, x): 
        return self.proj_out(self.forward_transformer(x)[0])

class BERT(TransformerEmbedder):
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = BertModel.from_pretrained(self.LM_load_str, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.LM_load_str)
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        ###USING EOS BUT NOT INITIALIZAED?
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

class SBERT(TransformerEmbedder): 
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

        self.transformer.load_state_dict(self._convert_state_dict_format(self.LM_load_str))

    def _convert_state_dict_format(self, state_dict_file): 
        print('converting state dict keys to BERT format')
        sbert_state_dict = torch.load(state_dict_file, map_location='cpu')
        for key in list(sbert_state_dict.keys()):
            sbert_state_dict[key.replace('0.auto_model.', '')] = sbert_state_dict.pop(key)
        return sbert_state_dict

class GPT(TransformerEmbedder): 
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = GPT2Model.from_pretrained(self.LM_load_str, output_hidden_states=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.LM_intermediate_lang_dim = self.transformer.config.n_embd
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

class CLIP(TransformerEmbedder): 
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = CLIPTextModel.from_pretrained(self.LM_load_str)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.LM_load_str)
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self._reducer = None
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

    def forward_transformer(self, x):
        tokens = self.tokens_to_tensor(x)
        trans_out = self.transformer(**tokens, output_hidden_states=True)
        return trans_out.pooler_output, trans_out.hidden_states

class GPTNeo(TransformerEmbedder): 
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = GPTNeoForCausalLM.from_pretrained(self.LM_load_str, output_hidden_states=True).transformer
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

class BoW(InstructionEmbedder): 
    VOCAB = sort_vocab()
    def __init__(self, config): 
        super().__init__(config)
        if self.LM_out_dim == None: 
            self.out_dim=len(self.VOCAB)
        self.LM_intermediate_lang_dim = len(self.VOCAB)

    def make_freq_tensor(self, instruct): 
        out_vec = torch.zeros(len(self.VOCAB))
        for word in instruct.split():
            index = self.VOCAB.index(word)
            out_vec[index] += 1
        return out_vec

    def forward(self, x): 
        freq_tensor = torch.stack(tuple(map(self.make_freq_tensor, x))).to(self.__device__)
        bow_out = self.proj_out(freq_tensor).to(self.__device__)
        return bow_out


