import torch
import torch.nn as nn

from Task import Task
from LangModule import train_instruct_dict, test_instruct_dict

task_list = Task.TASK_LIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sentence_transformers import SentenceTransformer
from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer

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
        self.embedderStr = 'BoW'
        self.tokenizer = None
        self.reduction_dim = reduction_dim
        if self.reduction_dim == None: 
            self.out_dim = len(vocab)    
        else: 
            self.out_dim = reduction_dim
            self.lin = nn.Sequential(nn.Linear(len(vocab), self.out_dim), nn.ReLU())

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

