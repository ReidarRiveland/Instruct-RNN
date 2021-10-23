from sklearn.metrics.pairwise import paired_euclidean_distances
from nlp_models import SBERT
from rnn_models import InstructNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, reduce_rep
from plotting import plot_trained_performance, plot_rep_scatter
import numpy as np
import torch.optim as optim
from utils import sort_vocab, isCorrect
from task import Task
import seaborn as sns

from task import construct_batch

from transformers import GPT2Model, GPT2Tokenizer


import matplotlib.pyplot as plt

from numpy.lib import utils
import torch
import torch.nn as nn
import pickle

device = torch.device(0)

class Vocab:
    SOS_token = 0
    EOS_token = 1
    def __init__(self):
        self.pad_len = 30
        self.vocab = sort_vocab()
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3
        for word in self.vocab: 
            self.addWord(word)
    
    def _tokenize_sentence(self, sent): 
        tokens = [2]*self.pad_len
        for i, word in enumerate(sent.split()): 
            tokens[i] = self.word2index[word]
        tokens[i+1]=1
        return tokens

    def tokenize_sentence(self, sent_list): 
        return np.array(list(map(self._tokenize_sentence, sent_list)))

    def _untokenize_sentence(self, tokens): 
        sent = []
        for token in tokens: 
            sent.append(self.index2word[token])
            if sent[-1] == "EOS":
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
    def __init__(self, init_context_set_obj):
        super(BaseDecoder, self).__init__()
        self.teacher_loss_list = []
        self.loss_list = []
        self.vocab = Vocab()
        self.instruct_array = np.array([train_instruct_dict[task] for task in Task.TASK_LIST]).squeeze()
        self.contexts = self._init_context_set(init_context_set_obj)

    def _init_context_set(self, load_object):
        if type(load_object) is str: 
            contexts = pickle.load(open('_ReLU128_5.7/swap_holdouts/'+load_object, 'rb'))
        else: 
            contexts = load_object
        return contexts

    def save_model(self, save_string): 
        torch.save(self.state_dict(),'_ReLU128_5.7/swap_holdouts/'+save_string+'.pt')

    def load_model(self, save_string): 
        self.load_state_dict(torch.load('_ReLU128_5.7/swap_holdouts/'+save_string+'.pt'))

    def get_instruct_embedding_pair(self, task_index, instruct_index, training=True): 
        if training:
            context_rep_index = np.random.randint(self.contexts.shape[1]-30, size=instruct_index.shape[0])
        else: 
            context_rep_index = np.random.randint(self.contexts.shape[1],size=instruct_index.shape[0])
        rep = self.contexts[task_index, context_rep_index, :]
        instruct = self.instruct_array[task_index, instruct_index]
        return instruct, rep

    def plot_context_embeddings(self, tasks_to_plot=Task.TASK_LIST): 
        reps_reduced, _ = reduce_rep(self.contexts)
        plot_rep_scatter(reps_reduced, tasks_to_plot)
        return reps_reduced



class BaseDecoderRNN(BaseDecoder):
    def __init__(self, init_context_set_obj):
        super().__init__(init_context_set_obj)
        self.hidden_size = self.contexts.shape[-1]
        self.embedding = nn.Embedding(self.vocab.n_words, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab.n_words)
        self.softmax = nn.LogSoftmax(dim=1)

    def _base_forward(self, ins, hidden):
        output = self.embedding(ins).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden.view(1, 1, -1))
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def forward(self, ins, hidden):
        return self._base_forward(ins, hidden)

    def decode_sentence(self, hidden_rep): 
        decoder_hidden = torch.Tensor(hidden_rep).view(1, 1, -1)
        decoder_input = torch.tensor([[self.vocab.SOS_token]])
        decoded_sentence = []
        for di in range(30):
            decoder_output, decoder_hidden = self._base_forward(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoded_sentence.append(self.vocab.index2word[decoder_input.item()])
            if decoder_input.item() == self.vocab.EOS_token:
                break
        return ' '.join(decoded_sentence[:-1]), decoder_hidden




class BaseDecoderRNN_(BaseDecoder):
    def __init__(self, hidden_size, init_context_set_obj):
        super().__init__(init_context_set_obj)
        self.hidden_size = hidden_size
        self.context_dim = self.contexts.shape[-1]
        self.embedding = nn.Embedding(self.vocab.n_words, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab.n_words)
        self.context_encoder = nn.Linear(self.context_dim, self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def _base_forward(self, ins, context):
        embedded = self.embedding(ins)
        init_hidden = torch.relu(self.context_encoder(context))
        rnn_out, _ = self.gru(embedded, init_hidden.unsqueeze(0))
        output = self.softmax(self.out(rnn_out[-1, ...]))
        return output, rnn_out

    def forward(self, context):
        sos_input = torch.tensor([[self.vocab.SOS_token]*context.shape[0]]).to(context.get_device())
        decoder_input = sos_input
        for di in range(self.vocab.pad_len):
            decoder_output, decoder_hidden = self._base_forward(decoder_input, context)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.cat((decoder_input, topi.T))
        decoded_indices = decoder_input.squeeze().detach().cpu().numpy()
        return decoder_output, decoder_hidden, decoded_indices

    def decode_sentence(self, context): 
        _, _, decoded_indices = self.forward(context)
        decoded_sentences = self.vocab.untokenize_sentence(decoded_indices)  # detach from history as input
        return decoded_sentences

class DecoderTaskRNN(BaseDecoderRNN_):
    def __init__(self, hidden_size, init_context_set_obj):
        super().__init__(hidden_size, init_context_set_obj)
        self.task_decoder = nn.Linear(self.hidden_size, len(Task.TASK_LIST))

    def forward(self, context):
        output, hidden, decoded_indices = super().forward(context)
        task_cat = self.softmax(self.task_decoder(hidden))
        return output, hidden, task_cat, decoded_indices

    def decode_sentence(self, context): 
        _, hidden, task_cat, decoded_indices = self.forward(context)
        decoded_sentences = self.vocab.untokenize_sentence(decoded_indices)  
        task_cat = self.task_decoder(hidden[0, ...])
        _, topi_task = task_cat.topk(1)
        predicted_task = np.array(Task.TASK_LIST)[topi_task.detach().cpu().numpy()] 
        return decoded_sentences, predicted_task

    def get_confusion_matrix(self): 
        confusion_matrix = np.zeros((16, 16))
        for task_index, task in enumerate(Task.TASK_LIST): 
            for j in range(self.contexts.shape[1]): 
                rep = self.contexts[task_index, j, :]
                _, predicted = self.decode_sentence(rep)
                confusion_matrix[task_index, Task.TASK_LIST.index(predicted)] += 1
        fig, ax = plt.subplots(figsize=(6, 4))
        res = sns.heatmap(confusion_matrix, xticklabels=Task.TASK_LIST, yticklabels=Task.TASK_LIST, annot=True, cmap='Blues', ax=ax)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 6)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 6)

        plt.show()
        return confusion_matrix



def train_decoder_(decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None, task_loss_ratio=0.1): 
    teacher_forcing_ratio = init_teacher_forcing_ratio
    pad_len  = decoder.vocab.pad_len 
    loss_list = []
    teacher_loss_list = []
    task_indices = list(range(16))
    batch_size=12

    if holdout_tasks is not None: 
        for holdout_task in holdout_tasks:
            holdout_index = Task.TASK_LIST.index(holdout_task)
            task_indices.remove(holdout_index)

    for i in range(epochs): 
        print('Epoch: ' + str(i)+'\n')
        
        for j in range(500): 
            use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
            decoder_loss=0
            task_loss=0
            task_index = np.random.choice(task_indices)
            instruct_index = np.random.randint(0, 15, size=batch_size)
            target_instruct, rep = decoder.get_instruct_embedding_pair(task_index, instruct_index)            
            target_tensor = torch.LongTensor([decoder.vocab.tokenize_sentence(target_instruct)]).to(device)

            init_hidden = torch.Tensor(rep).to(device)
            decoder_input = torch.tensor([[decoder.vocab.SOS_token]*batch_size]).to(device)

            opt.zero_grad()

            if use_teacher_forcing:
                decoded_sentence = []
                # Teacher forcing: Feed the target as the next input
                for di in range(pad_len):
                    
                    decoder_output, decoder_hidden = decoder._base_forward(decoder_input, init_hidden)
                    task_cat = decoder.softmax(decoder.task_decoder(decoder_hidden))
                    topv, topi = decoder_output.topk(1)
                    _, topi_task = task_cat.topk(1)

                    #get words for last sentence in the batch
                    last_word_index = topi.squeeze().detach()[-1].item()
                    last_word = decoder.vocab.index2word[last_word_index]
                    decoded_sentence.append(last_word)

                    task_loss += criterion(task_cat[-1, ...], torch.LongTensor([task_index]*batch_size).to(device))*(1/(pad_len-di))
                    decoder_loss += criterion(decoder_output, target_tensor[0, :, di])
                    decoder_input = torch.cat((decoder_input, target_tensor[..., di]))
                      # Teacher forcing
                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_output, decoder_hidden, task_cat, decoded_indices = decoder(init_hidden)
                _, topi_task = task_cat.topk(1)

                for k in range(pad_len):
                    task_loss += criterion(task_cat[k, ...], torch.LongTensor([task_index]*batch_size).to(device))**(1/(pad_len-k))
                    decoder_loss += criterion(decoder_output, target_tensor[0, :, k])
                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                decoded_sentence = decoder.vocab.untokenize_sentence(decoded_indices)[-1]  # detach from history as input        
                decoder.loss_list.append(loss.item()/pad_len)

            loss.backward()
            opt.step()

            if j%50==0: 
                print('Teacher forceing: ' + str(use_teacher_forcing))
                print('Decoder Loss: ' + str(decoder_loss.item()/pad_len))
                print('Task Loss: ' + str(task_loss.item()/pad_len))

                print('target instruction: ' + target_instruct[-1])
                if use_teacher_forcing:
                    try:
                        eos_index = decoded_sentence.index('EOS')
                    except ValueError: 
                        eos_index = -1
                    decoded_sentence = ' '.join(decoded_sentence[:eos_index])
                
                print('decoded instruction: ' + decoded_sentence + '\n')
                print('Predicted Task: ' + Task.TASK_LIST[topi_task[0, -1].squeeze().item()] + ', Target Task: ' + Task.TASK_LIST[task_index] + '\n')


        sch.step()
        teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs

    return loss_list, teacher_loss_list

decoder= DecoderTaskRNN(128, 'Multitask/sbertNet_tuned/seed1_context_vecs20')
decoder.to(device)
# decoder_output, decoder_hidden, task_cat, decoded_indices = decoder(torch.randn(5, 20).to(device))
# decoded_sentence = decoder.vocab.untokenize_sentence(decoded_indices)[-1]  # detach from history as input        

criterion = nn.NLLLoss(reduction='mean')
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.0)
sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, verbose=False)
decoder.to(device)


train_decoder_(decoder, decoder_optimizer, sch, 50, 1.0, task_loss_ratio=1.0)

decoder.decode_sentence(torch.Tensor(decoder.contexts[1, ...]).to(device))
decoder.save_model('Multitask_seed1_20')


def train_decoder(decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None, task_loss_ratio=0.1): 
    teacher_forcing_ratio = init_teacher_forcing_ratio
    loss_list = []
    teacher_loss_list = []
    task_indices = list(range(16))
    if holdout_tasks is not None: 
        for holdout_task in holdout_tasks:
            holdout_index = Task.TASK_LIST.index(holdout_task)
            task_indices.remove(holdout_index)

    for i in range(epochs): 
        print('Epoch: ' + str(i)+'\n')
        for j in range(500): 
            use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
            decoder_loss=0
            task_loss=0
            task_index = np.random.choice(task_indices)
            instruct_index = np.random.randint(0, 15)
            target_instruct, rep = decoder.get_instruct_embedding_pair(task_index, instruct_index)            
            target_tensor = torch.LongTensor(decoder.vocab.tokenize_sentence(target_instruct)).to(device)

            decoder_hidden = torch.Tensor(rep).view(1, 1, -1).to(device)
            decoder_input = torch.tensor([[decoder.vocab.SOS_token]]).to(device)
            target_length = target_tensor.shape[0]

            opt.zero_grad()

            decoded_sentence = []
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, task_cat = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    _, topi_task = task_cat.topk(1)

                    decoded_sentence.append(decoder.vocab.index2word[topi.squeeze().detach().item()])
                    task_loss += criterion(task_cat, torch.LongTensor([task_index]).to(device))
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
                    decoded_sentence.append(decoder.vocab.index2word[decoder_input.item()])
                    task_loss += criterion(task_cat, torch.LongTensor([task_index]).to(device))
                    decoder_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                    if decoder_input.item() == decoder.vocab.EOS_token:
                        break
                loss=decoder_loss+task_loss_ratio*task_loss
                decoder.loss_list.append(loss.item()/target_length)

            loss.backward()
            opt.step()

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

def test_partner_model(partner_model, decoder, num_repeats=1): 
    partner_model.eval()
    batch_len = decoder.contexts.shape[1]
    with torch.no_grad():
        perf_array = np.empty(16)
        for j, task in enumerate(Task.TASK_LIST):
            print(task)
            mean_list = [] 
            task_info = []
            for _ in range(num_repeats): 
                ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)

                for i in range(batch_len):
                    instruct, _ = decoder.decode_sentence(decoder.contexts[j, i, :])
                    task_info.append(instruct)


                out, _ = partner_model(task_info, torch.Tensor(ins).to(partner_model.__device__))
                mean_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
            perf_array[j] = np.mean(mean_list)
    return perf_array



decoder= DecoderTaskRNN('Multitask/sbertNet_tuned/seed1_context_vecs20')

criterion = nn.NLLLoss()
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.0)
sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, verbose=False)
decoder.to(device)

train_decoder(decoder, decoder_optimizer, sch, 50, 1.0, task_loss_ratio=1.0)

decoder.save_model('Multitask/sbertNet_tuned/seed0lang_init_context_task_decoder')




decoder.to('cpu')

decoder.get_confusion_matrix()

for i in range(30): 
    rep = decoder.contexts[4, -i, :]
    print(decoder.decode_sentence(rep))


model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model1.model_name += '_tuned'
model1.set_seed(1) 
model1.load_model('_ReLU128_5.7/swap_holdouts/Multitask')
#model1.to(device)

model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model.model_name += '_tuned'
model.set_seed(0) 
model.load_model('_ReLU128_5.7/swap_holdouts/Multitask')
lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='12')






perf = test_partner_model(model1, decoder)



if '__name__' == '__main__': 
    for i in range(1):
        model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
        model.model_name += '_tuned'
        model.set_seed(i) 


        #model.load_model('_ReLU128_5.7/single_holdouts/Multitask')
        loss_list, teacher_loss_list = train_decoder(decoder, decoder_optimizer, sch, 15, 1.0)
        torch.save(decoder.state_dict(), '_ReLU128_5.7/single_holdouts/Multitask/sbertNet_tuned/seed'+str(i)+'_decoder')

