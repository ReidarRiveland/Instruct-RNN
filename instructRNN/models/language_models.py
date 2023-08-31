import torch
import torch.nn as nn

from attrs import asdict
import itertools
from attrs import define
import pathlib

from transformers import GPT2Model, GPT2Tokenizer
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertModel, BertTokenizer, BertLayer, BertConfig

from instructRNN.instructions.instruct_utils import get_all_sentences, sort_vocab
from transformers import logging
#logging.set_verbosity_error()

location = str(pathlib.Path(__file__).parent.absolute())

@define
class LMConfig(): 
    LM_load_str: str
    LM_train_layers: list 
    LM_reducer: str 
    LM_out_dim: int 
    LM_output_nonlinearity: str 
    LM_proj_out_layers: int

def final_embedding(trans_last_hid):
    return trans_last_hid[:, -1, ...]

def mean_embedding(trans_last_hid): 
    return torch.mean(trans_last_hid, dim=1)

class InstructionEmbedder(nn.Module): 
    def __init__(self, config): 
        super(InstructionEmbedder, self).__init__()
        self.config=config
        for name, value in asdict(config).items(): 
            setattr(self, name, value)

        if self.LM_output_nonlinearity == 'relu': 
            self._output_nonlinearity = nn.ReLU()
        elif self.LM_output_nonlinearity == 'lin': 
            self._output_nonlinearity = nn.Identity()
        
        if self.LM_reducer == 'mean': 
            self._reducer = mean_embedding
        elif self.LM_reducer == 'last': 
            self._reducer = final_embedding

        self.__device__ = 'cpu'

    def __init_proj_out__(self): 
        if self.LM_proj_out_layers==1:
            self.proj_out = nn.Sequential(
                nn.Linear(self.LM_intermediate_lang_dim, self.LM_out_dim), 
                self._output_nonlinearity)
        else:
            layer_list = []
            layer_list.append(nn.Linear(self.LM_out_dim, 128)) 
            layer_list.append(self._output_nonlinearity)

            for _ in range(self.LM_proj_out_layers):
                layer_list.append(nn.Linear(self.LM_out_dim, 128)) 
                layer_list.append(self._output_nonlinearity)

            layer_list.append(nn.Linear(128, self.LM_out_dim)) 
            layer_list.append(self._output_nonlinearity)
            
            self.proj_out= nn.Sequential(
                nn.Linear(self.LM_intermediate_lang_dim, self.LM_out_dim), 
                self._output_nonlinearity, 
                *layer_list,
                )

    def set_train_layers(self, train_layers): 
        all_train_layers = train_layers+['proj_out']
        for n,p in self.named_parameters(): 
            if any([layer in n for layer in all_train_layers]):
                p.requires_grad=True
            else: 

                p.requires_grad=False
    
    def to(self, cuda_device):
        super().to(cuda_device)
        self.__device__ = cuda_device

class TransformerEmbedder(InstructionEmbedder): 
    def __init__(self, config): 
        super().__init__(config)

    def freeze_transformer(self):
        for p in self.transformer.parameters():
            p.requires_grad = False

    def tokens_to_tensor(self, x):
        tokens = self.tokenizer(x, return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(self.__device__)
        return tokens

    def forward_transformer(self, x): 
        tokens = self.tokens_to_tensor(x)
        trans_out = self.transformer(**tokens)
        return self._reducer(trans_out.last_hidden_state), trans_out[2]

    def forward(self, x): 
        return self.proj_out(self.forward_transformer(x)[0])


class BERT(TransformerEmbedder):
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = BertModel.from_pretrained(self.LM_load_str, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.LM_load_str)
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

class RawBertTransformer(nn.Module): 
    def __init__(self, n_layers, LM_load_str): 
        super(RawBertTransformer, self).__init__()
        self.bert_config = BertConfig()
        self.n_layers = n_layers
        self.embeddings = BertModel.from_pretrained(LM_load_str, output_hidden_states=True).embeddings
        self.layers = nn.ModuleList([BertLayer(self.bert_config)]*self.n_layers)

    def forward(self, x): 
        all_hiddens = []
        x = self.embeddings(x['input_ids'])
        all_hiddens.append(x)
        for layer in self.layers: 
            x = layer(x)[0]
            all_hiddens.append(x)

        return x, tuple(all_hiddens)

class RawBERT(TransformerEmbedder):
    def __init__(self, config): 
        super().__init__(config)
        self.transformer = RawBertTransformer(2, self.LM_load_str)
        self.tokenizer = BertTokenizer.from_pretrained(self.LM_load_str)
        self.LM_intermediate_lang_dim = self.transformer.bert_config.hidden_size
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()


    def forward_transformer(self, x): 
        tokens = self.tokens_to_tensor(x)
        trans_out = self.transformer(tokens)
        return self._reducer(trans_out[0]), trans_out[1]


class SBERT(TransformerEmbedder): 
    def __init__(self, config): 
        super().__init__(config)
        if 'large' in self.LM_load_str: bert_model = 'bert-large-uncased'
        else: bert_model = 'bert-base-uncased'

        self.transformer = BertModel.from_pretrained(bert_model, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()
        self.transformer.load_state_dict(self._convert_state_dict_format(location+'/_pretrained_state_dicts/'+self.LM_load_str))

    def _convert_state_dict_format(self, state_dict_file): 
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
        self.transformer = CLIPTextModel.from_pretrained(self.LM_load_str, output_hidden_states=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.LM_load_str)
        self.LM_intermediate_lang_dim = self.transformer.config.hidden_size
        self._reducer = mean_embedding
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

    def forward_transformer(self, x):
        tokens = self.tokens_to_tensor(x)
        trans_out = self.transformer(**tokens, output_hidden_states=True)
        return trans_out.pooler_output, trans_out.hidden_states

class BoW(InstructionEmbedder): 
    VOCAB = sort_vocab()
    def __init__(self, config): 
        super().__init__(config)
        if self.LM_out_dim == None: 
            self.out_dim=len(self.VOCAB)
        self.LM_intermediate_lang_dim = len(self.VOCAB)
        self.set_train_layers(self.LM_train_layers)
        self.__init_proj_out__()

    def _make_freq_tensor(self, instruct): 
        out_vec = torch.zeros(len(self.VOCAB))
        for word in instruct.split():
            index = self.VOCAB.index(word)
            out_vec[index] += 1
        return out_vec

    def forward(self, x): 
        freq_tensor = torch.stack(tuple(map(self._make_freq_tensor, x))).to(self.__device__)
        bow_out = self.proj_out(freq_tensor).to(self.__device__)
        return bow_out

