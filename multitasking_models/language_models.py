from typing import OrderedDict
from matplotlib.pyplot import get
from numpy.lib import utils
import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import GPTNeoModel

from utils import sort_vocab


class InstructionEmbedder(nn.Module): 
    def __init__(self, config): 
        super(InstructionEmbedder, self).__init__()
        self.config=config
        for name, value in config.__dict__.items(): 
            setattr(self, name, value)

        self.__device__ = 'cpu'
    
        if self.LM_output_nonlinearity == 'relu': 
            self._output_nonlinearity = nn.ReLU()
        
        if self.LM_reducer == 'mean': 
            self._reducer = torch.mean

    def __init_proj_out__(self): 
        self.proj_out = nn.Sequential(
            nn.Linear(self.intermediate_lang_dim, self.LM_out_dim), 
            self._output_nonlinearity)

    def __init_train_layers__(self): 
        for n,p in self.named_parameters(): 
            if any([layer in n for layer in self.LM_train_layers]):
                p.requires_grad=True
            else: 
                p.requires_grad=False

class TransformerEmbedder(InstructionEmbedder): 
    def __init__(self, config): 
        super().__init__(config)

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x, return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(self.__device__)

        trans_out = self.transformer(**tokens)
        return self.reducer(trans_out.last_hidden_state, dim=1), trans_out[2]

    def forward(self, x): 
        return self.proj_out(self.forward_transformer(x)[0])

class BERT(TransformerEmbedder):
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = BertModel.from_pretrained(self.LM_load_str, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.LM_load_str)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.__init_proj_out__()
        self.__init_train_layers__()

class SBERT(BERT): 
    def __init__(self, config): 
        super().__init__(config)
        self.transformer.load_state_dict(self._convert_state_dict_format('sbert_raw.pt'))

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
        self.intermediate_lang_dim = self.transformer.config.n_embd
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.__init_proj_out__()
        self.__init_train_layers__()

class GPTNeo(TransformerEmbedder): 
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B", output_hidden_states=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.init_train_layers()

class BoW(InstructionEmbedder): 
    VOCAB = sort_vocab()
    def __init__(self, config): 
        super().__init__(config)
        if self.LM_out_dim == None: 
            self.out_dim=len(self.VOCAB)
        
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

