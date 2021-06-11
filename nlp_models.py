import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer

from utils import sort_vocab

class TransformerEmbedder(nn.Module): 
    def __init__(self, out_dim, embedder_name, reducer, train_layers, output_nonlinearity): 
        super(TransformerEmbedder, self).__init__()
        self.device = 'cpu'
        self.reducer = reducer
        self.out_dim = out_dim
        self.train_layers = train_layers
        self.output_nonlinearity = output_nonlinearity
        self.embedderStr = embedder_name
        self.proj_out = nn.Linear(768, self.out_dim), output_nonlinearity

    def set_train_layers(self, train_layers): 
        for n,p in self.named_parameters(): 
            if any([layer in n for layer in train_layers]):
                p.requires_grad=True
            else: 
                p.requires_grad=False

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x, return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(self.device)

        trans_out = self.transformer(**tokens).last_hidden_state
        return self.reducer(trans_out, dim=1)

    def forward(self, x): 
        return self.output_nonlinearity(self.proj_out(self.forward_transformer(x)))

class BERT(TransformerEmbedder):
    def __init__(self, out_dim, reducer=torch.mean, train_layers = [], output_nonlinearity = torch.relu): 
        super().__init__(out_dim, 'bert', reducer, train_layers, output_nonlinearity)
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.set_train_layers(self.train_layers)

class GPT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=torch.mean, train_layers = [], output_nonlinearity = torch.relu): 
        super().__init__(out_dim, 'gpt', reducer,  train_layers, output_nonlinearity)
        self.transformer = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_train_layers(self.train_layers)

class SBERT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=None, train_layers = [], output_nonlinearity = torch.relu): 
        super().__init__(out_dim, 'sbert', reducer, train_layers, output_nonlinearity)
        self.transformer = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tokenizer = self.transformer.tokenize
        self.set_train_layers(self.train_layers)

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x)
        for key, value in tokens.items():
            tokens[key] = value.to(self.device)
        sent_embedding = self.transformer(tokens)['sentence_embedding']
        return sent_embedding

class BoW(nn.Module): 
    def __init__(self, reduction_dim = None): 
        super(BoW, self).__init__()
        self.vocab = sort_vocab()
        self.reduction_dim = reduction_dim
        self.embedderStr = 'bow'
        self.device = 'cpu'

        if self.reduction_dim == None: 
            self.out_dim = len(self.vocab)    
        else: 
            self.out_dim = reduction_dim
            self.proj_out = nn.Sequential(nn.Linear(len(self.vocab), self.out_dim), nn.ReLU())

    def make_freq_tensor(self, instruct): 
        out_vec = torch.zeros(len(self.vocab))
        for word in instruct.split():
            index = self.vocab.index(word)
            out_vec[index] += 1
        return out_vec

    def forward(self, x): 
        bow_out = torch.stack(tuple(map(self.make_freq_tensor, x))).to(self.device)
        if self.reduction_dim is not None: 
            bow_out = self.lin(bow_out)
        return bow_out




