import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer

from utils import sort_vocab

class InstructionEmbedder(nn.Module): 
    def __init__(self, embedder_name, intermediate_lang_dim, out_dim, output_nonlinearity): 
        super(InstructionEmbedder, self).__init__()
        self.__device__ = 'cpu'
        self.embedder_name = embedder_name

        self.intermediate_lang_dim = intermediate_lang_dim
        self.out_dim = out_dim
        self.output_nonlinearity = output_nonlinearity

        try: 
            self.proj_out = nn.Sequential(nn.Linear(self.intermediate_lang_dim, self.out_dim), self.output_nonlinearity)
        except TypeError: 
            self.proj_out = nn.Identity()

class TransformerEmbedder(InstructionEmbedder): 
    SET_TRAIN_LAYER_LIST = ['proj_out', 'pooler', 'ln_f']
    def __init__(self, embedder_name, out_dim,  reducer, train_layers, output_nonlinearity): 
        super().__init__(embedder_name, 768, out_dim, output_nonlinearity )
        self.reducer = reducer
        self.train_layers = train_layers

    def set_train_layers(self): 
        if len(self.train_layers)>0: tmp_train_layers = self.train_layers+self.SET_TRAIN_LAYER_LIST
        else: tmp_train_layers = ['proj_out']
        for n,p in self.named_parameters(): 
            if any([layer in n for layer in tmp_train_layers]):
                p.requires_grad=True
            else: 
                p.requires_grad=False

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x, return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(self.__device__)

        trans_out = self.transformer(**tokens).last_hidden_state
        return self.reducer(trans_out, dim=1)

    def forward(self, x): 
        return self.output_nonlinearity(self.proj_out(self.forward_transformer(x)))

class BERT(TransformerEmbedder):
    def __init__(self, out_dim, reducer=torch.mean, train_layers = [], output_nonlinearity = nn.ReLU()): 
        super().__init__('bert', out_dim, reducer, train_layers, output_nonlinearity)
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.set_train_layers()

class GPT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=torch.mean, train_layers = [], output_nonlinearity = nn.ReLU()): 
        super().__init__('gpt', out_dim,  reducer,  train_layers, output_nonlinearity)
        self.transformer = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_train_layers()

class SBERT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=None, train_layers = [], output_nonlinearity = nn.ReLU()): 
        super().__init__('sbert', out_dim, reducer, train_layers, output_nonlinearity)
        self.transformer = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tokenizer = self.transformer.tokenize
        self.set_train_layers()

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x)
        for key, value in tokens.items():
            tokens[key] = value.to(self.__device__)
        sent_embedding = self.transformer(tokens)['sentence_embedding']
        return sent_embedding

class BoW(InstructionEmbedder): 
    VOCAB = sort_vocab()
    def __init__(self, out_dim =  None, output_nonlinearity=nn.Identity()): 
        super().__init__('bow', len(self.VOCAB), out_dim, output_nonlinearity)
        if out_dim == None: 
            self.out_dim=len(self.VOCAB)

    def make_freq_tensor(self, instruct): 
        out_vec = torch.zeros(len(self.VOCAB))
        for word in instruct.split():
            index = self.VOCAB.index(word)
            out_vec[index] += 1
        return out_vec

    def forward(self, x): 
        bow_out = torch.stack(tuple(map(self.make_freq_tensor, x))).to(self.__device__)
        return bow_out




