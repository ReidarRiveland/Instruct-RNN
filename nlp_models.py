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

model.load_model('_ReLU128_5.7/single_holdouts/Multitask')

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
        tokens = []
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
        self.task_decoder = nn.Linear(hidden_size, 16)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        task_cat = self.softmax(self.task_decoder(hidden[0]))
        return output, hidden, task_cat

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def get_instruct_embedding_pair(task_index, instruct_index): 
    rep = instruct_reps[task_index, instruct_index, :]
    instruct = instruct_array[task_index, instruct_index]
    return instruct, rep


from task import Task
criterion = nn.NLLLoss()
decoder = DecoderRNN(768, vocab.n_words)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.001)
sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.5, last_epoch=-1, verbose=False)

def train_decoder(decoder, opt, sch, epochs, init_teacher_forcing_ratio, task_loss_ratio=0.5): 
    teacher_forcing_ratio = init_teacher_forcing_ratio
    loss_list = []
    teacher_loss_list = []
    for i in range(epochs): 
        print('Epoch: ' + str(i)+'\n')
        for j in range(1000): 
            decoder_loss=0
            task_loss=0
            task_index = np.random.randint(0, 16)
            instruct_index = np.random.randint(0, 15)
            target_instruct, rep = get_instruct_embedding_pair(task_index, instruct_index)            
            target_tensor = torch.LongTensor(vocab.tokenize_sentence(target_instruct))

            decoder_hidden = torch.Tensor(rep).view(1, 1, -1)
            decoder_input = torch.tensor([[SOS_token]])
            target_length = target_tensor.shape[0]

            decoder_optimizer.zero_grad()

            use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

            decoded_sentence = []
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, task_cat = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    _, topi_task = task_cat.topk(1)

                    decoded_sentence.append(vocab.index2word[topi.squeeze().detach().item()])
                    task_loss += criterion(task_cat, torch.LongTensor([task_index]))
                    decoder_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                    decoder_input = target_tensor[di]  # Teacher forcing
                loss=decoder_loss+task_loss_ratio*task_loss
                teacher_loss_list.append(loss.item()/target_length)
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, task_cat = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    _, topi_task = task_cat.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
                    decoded_sentence.append(vocab.index2word[decoder_input.item()])
                    task_loss += criterion(task_cat, torch.LongTensor([task_index]))
                    decoder_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                    if decoder_input.item() == EOS_token:
                        break
                loss=decoder_loss+task_loss_ratio*task_loss
                loss_list.append(loss.item()/target_length)

            loss.backward()
            decoder_optimizer.step()

            if j%50==0: 
                print('Teacher forceing: ' + str(use_teacher_forcing))
                print('Decoder Loss: ' + str(decoder_loss.item()/target_length))
                print('Task Loss: ' + str(task_loss.item()/target_length))

                print('target instruction: ' + target_instruct)
                print('decoded instruction: ' + ' '.join(decoded_sentence)+ '\n')
                print('Predicted Task: ' + Task.TASK_LIST[topi_task.item()] + ', Target Task: ' + Task.TASK_LIST[task_index] + '\n')


    sch.step()
    teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs

    return loss_list, teacher_loss_list

loss_list, teacher_loss_list = train_decoder(decoder, decoder_optimizer, sch, 15, 1.0)

import matplotlib.pyplot 

def decode_sentence(decoder, hidden_rep): 
    decoder_hidden = torch.Tensor(hidden_rep).view(1, 1, -1)
    decoder_input = torch.tensor([[SOS_token]])
    decoded_sentence = []
    for di in range(30):
        decoder_output, decoder_hidden, task_cat = decoder(decoder_input, decoder_hidden)
        _, topi_task = task_cat.topk(1)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        decoded_sentence.append(vocab.index2word[decoder_input.item()])
        if decoder_input.item() == EOS_token:
            break
    predicted_task = Task.TASK_LIST[topi_task.item()] 
    return predicted_task, ' '.join(decoded_sentence[:-1])

import pickle
context_reps = np.empty((16, 768))
context_reps_dict = {}
for i, task in enumerate(Task.TASK_LIST): 
    contexts = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_vecs', 'rb'))
    contexts_perf = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_correct_data', 'rb'))
    context_reps[i, :] = np.mean(contexts[contexts_perf[:, -1] > 0.95], axis=0)
    context_reps_dict[task] = contexts[contexts_perf[:, -1] > 0.95]

context_reps_dict['Go'].shape

for task_index, task in enumerate(Task.TASK_LIST): 
    #target_instruct, rep = get_instruct_embedding_pair(task_index, 1)            
    rep = context_reps[task_index, :]            
    print('Target Task: '+ task + ' Target Instruction: '+target_instruct)
    print(str(decode_sentence(decoder, rep)) + '\n')

opp_task_list = Task.TASK_LIST.copy()
opp_task_list[1], opp_task_list[2] = opp_task_list[2], opp_task_list[1]


confusion_matrix = np.zeros((16, 16))
instruct_reps[[1,2], ...] = instruct_reps[[2,1], ...] 

for task_index, task in enumerate(opp_task_list): 
    for j in range(15): 
        rep = instruct_reps[task_index, j, :]
        predicted, _ = decode_sentence(decoder, rep)
        confusion_matrix[task_index, opp_task_list.index(predicted)] += 1

import seaborn as sns
sns.heatmap(confusion_matrix, xticklabels=opp_task_list, yticklabels=opp_task_list, annot=True, cmap='Blues')
plt.show()


context_confusion_matrix = np.zeros((16, 16))
for task_index, task in enumerate(opp_task_list): 
    for j in range(context_reps_dict[task].shape[0]): 
        rep = context_reps_dict[task][j, :]
        predicted, _ = decode_sentence(decoder, rep)
        context_confusion_matrix[task_index, opp_task_list.index(predicted)] += 1

sns.heatmap(context_confusion_matrix, xticklabels=opp_task_list, yticklabels=opp_task_list, annot=True, cmap='Blues')
plt.show()


instruct_indices = np.random.randint(0, 15, size=128)
reps = instruct_reps[task_index, instruct_indices, :]
task_info = []
for i in range(128):
    _, instruct = decode_sentence(decoder, reps[i, :])
    task_info.append(instruct)

task_info

from task import construct_batch
from utils import isCorrect

num_repeats = 3

model.eval()
batch_len = 128
with torch.no_grad():
    perf_dict = dict.fromkeys(Task.TASK_LIST)
    for j, task in enumerate(Task.TASK_LIST):
        print(task)
        mean_list = [] 
        for _ in range(num_repeats): 
            ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)


            instruct_indices = np.random.randint(0, 15, size=128)
            reps = instruct_reps[j, instruct_indices, :]
            task_info = []
            for i in range(128):
                _, instruct = decode_sentence(decoder, reps[i, :])
                task_info.append(instruct)


            out, _ = model(task_info, torch.Tensor(ins).to(model.__device__))
            mean_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
        perf_dict[task] = np.mean(mean_list)

from plotting import plot_trained_performance

plot_trained_performance({'sbertNet_layer_11': perf_dict})