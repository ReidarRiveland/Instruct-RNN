from torch.distributions import Categorical
import numpy as np

import torch
import torch.nn as nn
import pickle

from instructRNN.instructions.instruct_utils import train_instruct_dict, count_vocab
from instructRNN.tasks.tasks import TASK_LIST
from instructRNN.models.script_gru import ScriptGRU

device = torch.device(0)

class SMDecoder(nn.Module): 
    def __init__(self, out_dim, sm_hidden_dim, drop_p):
        super(SMDecoder, self).__init__()
        self.dropper = nn.Dropout(p=drop_p)
        self.fc1 = nn.Linear(sm_hidden_dim*2, out_dim)
        self.id = nn.Identity()
        
    def forward(self, sm_hidden): 
        out_mean = self.id(torch.mean(sm_hidden, dim=1))
        out_max = self.id(torch.max(sm_hidden, dim=1).values)
        out = torch.cat((out_max, out_mean), dim=-1)
        out = self.dropper(out)
        out = torch.relu(self.fc1(out))
        return out.unsqueeze(0)

class RNNtokenizer():
    def __init__(self):
        self.sos_token_id = 0
        self.eos_token_id = 1
        self.pad_len = 45
        self.counts, self.vocab = count_vocab()
        self.word2index = {}
        self.index2word = {0: '[CLS]', 1: '[EOS]', 2: '[PAD]'}
        self.n_words = 3
        for word in self.vocab: 
            self.addWord(word)

    def __call__(self, sent_list, use_langModel = False, pad_len=45):
        return self.tokenize_sentence(sent_list, pad_len, use_langModel)

    def _tokenize_sentence(self, sent, pad_len, use_langModel): 
        tokens = [2]*pad_len
        for i, word in enumerate(sent.split()): 
            if use_langModel:
                tokens[i] = self.bert_word2index[word]
            else: 
                tokens[i] = self.word2index[word]
        tokens[i+1]=1
        return tokens

    def tokenize_sentence(self, sent_list, pad_len, use_langModel): 
        tokenized_list = []
        for sent in sent_list:
            tokens = self._tokenize_sentence(sent, pad_len, use_langModel)
            tokenized_list.append(tokens)
        return torch.LongTensor(tokenized_list)

    def _untokenize_sentence(self, tokens): 
        sent = []
        for token in tokens: 
            sent.append(self.index2word[token])
            if sent[-1] == "[EOS]" or sent[-1] == "<|endoftext|>" :
                break
        return ' '.join(sent[:-1])

    def untokenize_sentence(self, token_array): 
        return np.array(list(map(self._untokenize_sentence, token_array.T)))

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

        self.init_instructions(train_instruct_dict)
        self.contexts = None
        self.validation_ratio = 0.2

    def init_instructions(self, instruct_dict): 
        self.instruct_dict = instruct_dict
        self.instruct_array = np.array([self.instruct_dict[task] for task in TASK_LIST]).squeeze()

    def save_model(self, save_string): 
        torch.save(self.state_dict(), save_string+'.pt')

    def save_model_data(self, save_string): 
        pickle.dump(self.teacher_loss_list, open(save_string+'_teacher_loss_list', 'wb'))
        pickle.dump(self.loss_list, open(save_string+'_loss_list', 'wb'))

    def load_model(self, load_string, suffix=''): 
        self.load_state_dict(torch.load(load_string+'decoders/'+self.decoder_name+suffix+'.pt', map_location=torch.device('cpu')))

    def get_instruct_embedding_pair(self, task_index, instruct_index, training=True): 
        assert self.contexts is not None, 'must initalize decoder contexts with init_context_set'
        if training:
            context_rep_index = np.random.randint(self.contexts.shape[1]-self.contexts.shape[1]*self.validation_ratio, size=instruct_index.shape[0])
        else: 
            context_rep_index = np.random.randint(self.contexts.shape[1],size=instruct_index.shape[0])
        rep = self.contexts[task_index, context_rep_index, :]
        instruct = self.instruct_array[task_index, instruct_index]
        return list(instruct), torch.Tensor(rep)


class DecoderRNN(BaseDecoder):
    def __init__(self, hidden_size, sm_hidden_dim=256, drop_p = 0.0):
        super().__init__()
        self.decoder_name = 'rnn_decoder'
        self.hidden_size = hidden_size
        self.tokenizer = RNNtokenizer()
        self.embedding_size=64
        self.embedding = nn.Embedding(self.tokenizer.n_words, self.embedding_size)
        self.gru = ScriptGRU(self.embedding_size, self.hidden_size, 1, activ_func = torch.relu, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.tokenizer.n_words)
        self.sm_decoder = SMDecoder(self.hidden_size, sm_hidden_dim, drop_p=drop_p)
        self.softmax = nn.LogSoftmax(dim=1)

    def draw_next(self, logits, k_sample=1):
        top_k = logits.topk(k_sample)
        probs = torch.softmax(top_k.values, dim=-1)
        dist = Categorical(probs)
        next_indices = top_k.indices.gather(1, dist.sample().unsqueeze(-1))
        return next_indices

    def _base_forward(self, ins, sm_hidden):
        embedded = torch.relu(self.embedding(ins))
        init_hidden = self.sm_decoder(sm_hidden)
        rnn_out, _ = self.gru(embedded.transpose(0,1), init_hidden)
        logits = self.out(rnn_out[:, -1,:])
        return logits, rnn_out

    def forward(self, sm_hidden):
        sos_input = torch.tensor([[self.tokenizer.sos_token_id]*sm_hidden.shape[0]]).to(sm_hidden.get_device())
        decoder_input = sos_input
        for di in range(self.tokenizer.pad_len):
            logits, decoder_hidden = self._base_forward(decoder_input, sm_hidden)
            next_index = self.draw_next(logits)
            decoder_input = torch.cat((decoder_input, next_index.T))
        decoded_indices = decoder_input.squeeze().detach().cpu().numpy()
        return self.softmax(logits), decoder_hidden, decoded_indices

    def decode_sentence(self, sm_hidden): 
        _, _, decoded_indices = self.forward(sm_hidden)
        decoded_sentences = self.tokenizer.untokenize_sentence(decoded_indices[1:,...])  # detach from history as input
        return decoded_sentences

    def to(self, cuda_device): 
        super().to(cuda_device)
        self.gru._mask_to(cuda_device)

