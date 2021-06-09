import torch
import torch.nn as nn

from Task import Task
from LangModule import train_instruct_dict, test_instruct_dict

task_list = Task.TASK_LIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sentence_transformers import SentenceTransformer
from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer

import numpy as np 
import pickle

train_instruct_dict = pickle.load(open('Instructions/train_instruct_dict2', 'rb'))
test_instruct_dict = pickle.load(open('Instructions/test_instruct_dict2', 'rb'))


swaps= [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['COMP2', 'RT Go']]
swapped_task_list = ['Anti DM', 'COMP2', 'Anti Go', 'DMC', 'DM', 'Go', 'MultiDM', 'Anti MultiDM', 'COMP1', 'RT Go', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'Anti RT Go', 'DNMC']
instruct_swap_dict = dict(zip(swapped_task_list, train_instruct_dict.values()))

def shuffle_instruction(instruct): 
    instruct = instruct.split()
    shuffled = np.random.permutation(instruct)
    instruct = ' '.join(list(shuffled))
    return instruct

def get_batch(batch_size, task_type, instruct_mode = None):
    assert instruct_mode in [None, 'instruct_swap', 'shuffled', 'comp', 'validation', 'random']

    if instruct_mode == 'instruct_swap': 
        instruct_dict = instruct_swap_dict
    elif instruct_mode == 'validation': 
        instruct_dict = test_instruct_dict
    else: 
        instruct_dict = train_instruct_dict

    instructs = np.random.choice(instruct_dict[task_type], size=batch_size)
    if instruct_mode == 'shuffled': 
        instructs = list(map(shuffle_instruction, instructs))

    return instructs

def sort_vocab(): 
    combined_instruct= {key: list(train_instruct_dict[key]) + list(test_instruct_dict[key]) for key in train_instruct_dict}
    all_sentences = sum(list(combined_instruct.values()), [])
    sorted_vocab = sorted(list(set(' '.join(all_sentences).split(' '))))
    return sorted_vocab


class TransformerEmbedder(nn.Module): 
    def __init__(self, out_dim, embedderStr, reducer): 
        super(TransformerEmbedder, self).__init__()
        self.reducer = reducer
        self.out_dim = out_dim
        self.embedderStr = embedderStr
        self.proj_out = nn.Sequential(nn.Linear(768, self.out_dim), nn.ReLU())

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x, return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(device)

        trans_out = self.model(**tokens).last_hidden_state
        return self.reducer(trans_out, dim=1)

    def forward(self, x): 
        return self.proj_out(self.forward_transformer(x))


class BERT(TransformerEmbedder):
    def __init__(self, out_dim, reducer=torch.mean): 
        super().__init__(out_dim, 'BERT', reducer)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class GPT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=torch.mean): 
        super().__init__(out_dim, 'GPT', reducer)
        self.model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

class SBERT(TransformerEmbedder): 
    def __init__(self, out_dim, d_reduce=None): 
        super().__init__(out_dim, 'SBERT', d_reduce)
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tokenizer = self.model.tokenize

    def forward_transformer(self, x): 
        tokens = self.tokenizer(x)
        for key, value in tokens.items():
            tokens[key] = value.to(device)
        sent_embedding = self.model(tokens)['sentence_embedding']
        return sent_embedding

class BoW(nn.Module): 
    def __init__(self, reduction_dim = None): 
        super(BoW, self).__init__()
        self.vocab = sorted_vocab()
        self.embedderStr = 'BoW'
        self.tokenizer = None
        self.reduction_dim = reduction_dim
        if self.reduction_dim == None: 
            self.out_dim = len(self.vocab)    
        else: 
            self.out_dim = reduction_dim
            self.lin = nn.Sequential(nn.Linear(len(self.vocab), self.out_dim), nn.ReLU())

    def count_freq(instruct): 
                    out_vec = torch.zeros(self.out_dim)
            for word in instruct.split():
                index = vocab.index(word)
                out_vec[index] += 1

    def forward(self, x): 
        batch_vectorized = []
        for instruct in x: 
            out_vec = torch.zeros(self.out_dim)
            for word in instruct.split():
                index = vocab.index(word)
                out_vec[index] += 1
            batch_vectorized.append(out_vec)
            out = torch.stack(batch_vectorized).to(device)
        if self.reduction_dim is not None: 
            out = self.lin(out)
        return out


