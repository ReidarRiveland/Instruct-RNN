from matplotlib.pyplot import get
from numpy.lib import utils
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer

from utils import sort_vocab

import warnings

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

    def init_train_layers(self): 
        if len(self.train_layers)>0: 
            tmp_train_layers = self.train_layers+self.SET_TRAIN_LAYER_LIST
            if not isinstance(self.output_nonlinearity, type(nn.ReLU())): 
                warnings.warn('output nonlinearity set to something other than relu! Use caution when trying to load pretained models')
        else: 
            tmp_train_layers = ['proj_out']
            if not isinstance(self.output_nonlinearity, type(nn.Identity())): 
                warnings.warn('output nonlinearity set to something other than Identity! Use caution when trying to load pretained models')
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
        self.init_train_layers()

class GPT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=torch.mean, train_layers = [], output_nonlinearity = nn.ReLU()): 
        super().__init__('gpt', out_dim,  reducer,  train_layers, output_nonlinearity)
        self.transformer = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.init_train_layers()

class SBERT(TransformerEmbedder): 
    def __init__(self, out_dim, reducer=None, train_layers = [], output_nonlinearity = nn.ReLU()): 
        super().__init__('sbert', out_dim, reducer, train_layers, output_nonlinearity)
        self.transformer = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tokenizer = self.transformer.tokenize
        self.init_train_layers()

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





from rnn_models import InstructNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps
import numpy as np
import torch.optim as optim


model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
model.set_seed(0) 

model.load_model('_ReLU128_5.7/single_holdouts/Anti_DM')

instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
instruct_array = np.array([instruct_set for instruct_set in train_instruct_dict.values()])
instruct_reps.shape




SOS_token = 0
EOS_token = 1

class VocabIndex:
    def __init__(self):
        self.vocab = sort_vocab()
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        for word in self.vocab: 
            self.addWord(word)
    
    def tokenize_sentence(self, sent): 
        tokens = [0]
        for word in sent.split(): 
            tokens.append(self.word2index[word])
        tokens.append(1)
        return tokens
            

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

vocab = VocabIndex()


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def get_instruct_embedding_pair(task_index, instruct_index): 
    rep = instruct_reps[task_index, instruct_index, :]
    instruct = instruct_array[task_index, instruct_index]
    return instruct, rep



target_instruct, rep = get_instruct_embedding_pair(0, 0)

target_tensor = torch.LongTensor(vocab.tokenize_sentence(target_instruct))

criterion = nn.NLLLoss()

decoder = DecoderRNN(768, vocab.n_words)

decoder_hidden = torch.Tensor(rep).view(1, 1, -1)
decoder_input = torch.tensor([[SOS_token]])
target_length = target_tensor.shape[0]



decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.001)

teacher_forcing_ratio = 0.5

with torch.autograd.set_detect_anomaly(True):
    for i in range(100): 
        loss=0
        #use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = False
        decoder_optimizer.zero_grad()
        decoded_sentence = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoded_sentence.append(vocab.index2word[topi.squeeze().detach().item()])
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoded_sentence.append(vocab.index2word[decoder_input.item()])
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break
        print(decoded_sentence)

        loss.backward()

        decoder_optimizer.step()

decoder_output

with torch.autograd.set_detect_anomaly(True):

    decoder_optimizer.zero_grad()
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        decoded_sentence.append(vocab.index2word[decoder_input.item()])
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token:
            break


    loss.backward()
    decoder_optimizer.step()
