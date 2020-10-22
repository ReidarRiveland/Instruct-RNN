import torch
import torch.nn as nn

import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from fse.models import SIF
from fse import IndexedList

import numpy as np
import pickle
import os

from Task import Task
from LangModule import PAD_LEN, train_instruct_dict, test_instruct_dict

task_list = Task.TASK_LIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'


def wordFreq():
    freq_dict = {}
    all_words = []
    all_sentences = []
    for task_type in task_list:
        instructions = train_instruct_dict[task_type] + test_instruct_dict[task_type]
        for sentence in instructions: 
            all_sentences.append(sentence)
            for word in sentence.split(): 
                all_words.append(word)
    for word in set(all_words):
        freq_dict[word] = all_words.count(word)
    
    split_sentences = []
    for sentence in all_sentences:
        list_sent = sentence.split()
        split_sentences.append(list_sent)

    return freq_dict, sorted(list(set(all_words))), all_sentences, split_sentences

freq_dict, vocab, all_sentences, split_sentences = wordFreq()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LangTransformerTokenizer():
    def encode(self, instruct): 
        return torch.Tensor([vocab.index(word) for word in instruct.split()])

class LangTransformer(nn.Module): 
    def __init__(self, out_dim, d_reduce = 'avg', size = 'base'): 
        super(LangTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.d_reduce = d_reduce
        self.embedderStr = 'LangTransformer'
        self.out_dim = out_dim
        self.tokenizer = LangTransformerTokenizer()

        if size == 'large': 
            self.d_model, nheads, nlayers, d_ff = 1024, 16, 24, 3072
        else: 
            self.d_model, nheads, nlayers, d_ff = 768, 12, 12, 3072

        dropout = 0.1
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        encoder_layers = TransformerEncoderLayer(self.d_model, nheads, d_ff)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(len(vocab), self.d_model)

        if self.d_reduce == 'linear':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
 

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        trans_out = self.transformer_encoder(src, self.src_mask)

        if self.d_reduce == 'linear': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.d_reduce ==  'max': 
            trans_out = torch.max((trans_out), dim=1)
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg': 
            trans_out = torch.mean((trans_out), dim =1) 
            out =self.proj_out(trans_out)
        return out


class gpt2(nn.Module): 
    def __init__(self, out_dim, d_reduce='avg', size = 'base'): 
        super(gpt2, self).__init__()
        from transformers import GPT2Model, GPT2Tokenizer
        self.d_reduce = d_reduce
        self.embedderStr = 'gpt'
        self.out_dim = out_dim

        if self.d_reduce == 'linear':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        if size == 'large': 
            self.transformer = GPT2Model.from_pretrained('gpt2-medium')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        elif size == 'XL': 
            self.transformer = GPT2Model.from_pretrained('gpt2-xl')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        else: 
            self.transformer = GPT2Model.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


    def forward(self, x): 
        trans_out = self.transformer(x)[0]
        if self.d_reduce == 'linear': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.d_reduce ==  'max': 
            trans_out = torch.max((trans_out), dim=1)
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg': 
            trans_out = torch.mean((trans_out), dim =1) 
            out =self.proj_out(trans_out)
        return out

class BERT(nn.Module):
    def __init__(self, out_dim, d_reduce='avg', size = 'base'): 
        super(BERT, self).__init__()
        from transformers import BertModel, BertTokenizer

        self.d_reduce = d_reduce
        self.embedderStr = 'BERT'
        self.out_dim = out_dim

        if self.d_reduce == 'linear':
            self.proj_out = nn.Sequential(nn.Linear(PAD_LEN*768, 768), nn.ReLU(), nn.Linear(768, self.out_dim), nn.ReLU())
        else: 
            self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

        if size == 'large': 
            self.transformer = BertModel.from_pretrained('bert-large-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        else: 
            self.transformer = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def forward(self, x): 
        trans_out = self.transformer(x)[0]
        if self.d_reduce == 'linear': 
            out = self.proj_out(trans_out.flatten(1))
        elif self.d_reduce ==  'max': 
            trans_out = torch.max((trans_out), dim=1)
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg': 
            trans_out = torch.mean((trans_out), dim =1) 
            out =self.proj_out(trans_out)
        elif self.d_reduce == 'avg_conv': 
            trans_out = nn.AvgPool2d((1, ), (2,1))
        return out


class SBERT(nn.Module): 
    def __init__(self, out_dim, size = 'base'): 
        super(SBERT, self).__init__()
        from sentence_transformers import SentenceTransformer
        if size == 'large': 
            self.model = SentenceTransformer('bert-large-nli-mean-tokens')
        else: 
            self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.embedderStr = 'SBERT'
        self.tokenizer = None
        self.out_dim = out_dim
        self.lin = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())
        
    def forward(self, x): 
        sent_embedding = np.array(self.model.encode(x))
        sent_embedding = self.lin(torch.Tensor(sent_embedding).to(device))
        return sent_embedding


class InferSent(nn.Module): 
    def __init__(self, out_dim): 
        super(InferSent, self).__init__()
        self.model = self.load_model_config()
        self.embedderStr = 'InferSent'
        self.tokenizer = None 
        self.out_dim = out_dim
        self.proj_out = nn.Sequential(nn.Linear(4096, self.out_dim), nn.ReLU())

    def load_model_config(self):
        owd = os.getcwd()
        os.chdir('PreTrainedLanguageModels/SentInfer')
        from models import InferSent
        model_version = 1
        MODEL_PATH = "encoder/infersent%s.pkl" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        os.chdir(owd)
        W2V_PATH = 'PreTrainedLanguageModels/GloVe/glove.840B.300d.txt'
        model.set_w2v_path(W2V_PATH)
        model.build_vocab_k_words(K=100000)
        return model

    def forward(self, x): 
        sent_embedding = np.array(self.model.encode(x))
        sent_embedding = self.proj_out(torch.Tensor(sent_embedding).to(device))
        return sent_embedding

class SIFmodel(nn.Module): 
    def __init__(self): 
        super(SIFmodel, self).__init__()
        self.out_shape = ['batch_len', 50]
        self.in_shape = ['batch_len', 50]
        self.out_dim = 50
        self.embedderStr = 'SIF'
        self.tokenizer = None
        self.embedder = pickle.load(open('PreTrainedLanguageModels/glove-50-sif', 'rb'))

    def get_SIF_embedding(self, instruct): 
        if instruct in all_sentences: 
            index = IndexedList(split_sentences).items.index(instruct.split())
            embedded = self.embedder.sv[index]
        else: 
            tmp = (instruct.split(), 0)
            embedded = self.embedder.infer([tmp])
        return embedded.squeeze()

    def forward(self, x):
        embedded = [self.get_SIF_embedding(x[i]) for i in range(len(x))]
        return torch.Tensor(np.array(embedded, dtype = np.float32)).to(device)

class BoW(nn.Module): 
    def __init__(self): 
        super(BoW, self).__init__()
        self.embedderStr = 'BoW'
        self.tokenizer = None
        self.out_dim = len(vocab)    

    def forward(self, x): 
        batch_vectorized = []
        for instruct in x: 
            out_vec = torch.zeros(self.out_dim)
            for word in instruct.split():
                index = vocab.index(word)
                out_vec[index] += 1
            batch_vectorized.append(out_vec)
        return torch.stack(batch_vectorized).to(device)

