from nlp_models import SBERT
from rnn_models import InstructNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance
from plotting import plot_trained_performance
import numpy as np
import torch.optim as optim
from utils import sort_vocab
from task import Task
import seaborn as sns

import matplotlib.pyplot as plt

from numpy.lib import utils
import torch
import torch.nn as nn

device = torch.device(0)

# model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model.model_name += '_tuned'
# model.set_seed(1) 
# model.load_model('_ReLU128_5.7/single_holdouts/Multitask')

class Vocab:
    SOS_token = 0
    EOS_token = 1
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

vocab = Vocab()


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, context_size):
        super(DecoderRNN, self).__init__()
        self.vocab = Vocab()
        #self.instruct_reps = get_instruct_reps(langModel, train_instruct_dict, layer)
        #self.instruct_array = np.array([train_instruct_dict[task] for task in Task.TASK_LIST]).squeeze()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab.n_words, hidden_size)
        #self.context_encoder=nn.Linear(context_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, self.vocab.n_words)
        #self.task_decoder = nn.Linear(hidden_size, 16)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        #hidden_encoded = torch.relu(self.context_encoder(hidden))
        #output, hidden = self.gru(output, hidden_encoded.view(1,1,-1))
        output, hidden = self.gru(output, hidden.view(1, 1, -1))
        output = self.softmax(self.out(output[0]))
        #task_cat = self.softmax(self.task_decoder(hidden[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def get_confusion_matrix(self): 
        confusion_matrix = np.zeros((16, 16))
        instruct_reps = self.instruct_reps
        for task_index, task in enumerate(Task.TASK_LIST): 
            for j in range(15): 
                rep = instruct_reps[task_index, j, :]
                predicted, _ = self.decode_sentence(rep)
                confusion_matrix[task_index, Task.TASK_LIST.index(predicted)] += 1
        fig, ax = plt.subplots(figsize=(6, 4))
        res = sns.heatmap(confusion_matrix, xticklabels=Task.TASK_LIST, yticklabels=Task.TASK_LIST, annot=True, cmap='Blues', ax=ax)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 6)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 6)

        plt.show()
        return confusion_matrix

    def decode_sentence(self, hidden_rep): 
        decoder_hidden = torch.Tensor(hidden_rep).view(1, 1, -1)
        decoder_input = torch.tensor([[vocab.SOS_token]])
        decoded_sentence = []
        for di in range(30):
            decoder_output, decoder_hidden = self.forward(decoder_input, decoder_hidden)
            #_, topi_task = task_cat.topk(1)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoded_sentence.append(vocab.index2word[decoder_input.item()])
            if decoder_input.item() == vocab.EOS_token:
                break
        #predicted_task = Task.TASK_LIST[topi_task.item()] 
        return ' '.join(decoded_sentence[:-1])

    def get_instruct_embedding_pair(self, task_index, instruct_index): 
        if self.instruct_reps is None: 
            'Must init target instruct reps'

        rep = self.instruct_reps[task_index, instruct_index, :]
        instruct = self.instruct_array[task_index, instruct_index]
        return instruct, rep


criterion = nn.NLLLoss()
# decoder = DecoderRNN(20, vocab.n_words, model.langModel, 'full')
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0005, weight_decay=0.001)
# sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.9, verbose=False)
#decoder.to(device)

def get_decoder_loss(decoder, target_instructs, decoder_hiddens, teacher_forcing_ratio, print_progress=False):
    use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
    total_decoder_loss=0
    task_loss=0

    for decoder_hidden, target_instruct in zip(decoder_hiddens, target_instructs):
        use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
        decoder_input = torch.tensor([[vocab.SOS_token]]).to(device)
        target_tensor = torch.LongTensor(vocab.tokenize_sentence(target_instruct)).to(device)
        target_length = target_tensor.shape[0]
        decoded_sentence = []
        decoder_loss=0
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                #_, topi_task = task_cat.topk(1)

                decoded_sentence.append(vocab.index2word[topi.squeeze().detach().item()])
                #task_loss += criterion(task_cat, torch.LongTensor([task_index]).to(device))
                decoder_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                decoder_input = target_tensor[di]  # Teacher forcing
            loss=decoder_loss#+task_loss_ratio*task_loss
            #teacher_loss_list.append(loss.item()/target_length)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                #_, topi_task = task_cat.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoded_sentence.append(vocab.index2word[decoder_input.item()])
                #task_loss += criterion(task_cat, torch.LongTensor([task_index]).to(device))
                decoder_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == vocab.EOS_token:
                    break
            loss=decoder_loss#+task_loss_ratio*task_loss
        total_decoder_loss += loss

    if print_progress:
        print('Teacher forceing: ' + str(use_teacher_forcing))
        print('Decoder Loss: ' + str(decoder_loss.item()/target_length))
        #print('Task Loss: ' + str(task_loss.item()/target_length))

        print('target instruction: ' + target_instruct)
        print('decoded instruction: ' + ' '.join(decoded_sentence)+ '\n')
        #print('Predicted Task: ' + Task.TASK_LIST[topi_task.item()] + ', Target Task: ' + Task.TASK_LIST[task_index] + '\n')

    return total_decoder_loss/decoder_hidden.shape[0]


def train_decoder(decoder, opt, sch, epochs, init_teacher_forcing_ratio, task_loss_ratio=0.1): 
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
            target_instruct, rep = decoder.get_instruct_embedding_pair(task_index, instruct_index)            
            target_tensor = torch.LongTensor(vocab.tokenize_sentence(target_instruct)).to(device)

            decoder_hidden = torch.Tensor(rep).view(1, 1, -1).to(device)
            decoder_input = torch.tensor([[vocab.SOS_token]]).to(device)
            target_length = target_tensor.shape[0]

            opt.zero_grad()

            use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

            decoded_sentence = []
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, task_cat = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    _, topi_task = task_cat.topk(1)

                    decoded_sentence.append(vocab.index2word[topi.squeeze().detach().item()])
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
                    decoded_sentence.append(vocab.index2word[decoder_input.item()])
                    task_loss += criterion(task_cat, torch.LongTensor([task_index]).to(device))
                    decoder_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                    if decoder_input.item() == vocab.EOS_token:
                        break
                loss=decoder_loss+task_loss_ratio*task_loss
                loss_list.append(loss.item()/target_length)

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


# train_decoder(decoder, decoder_optimizer, sch, 30, 1.0)

# decoder.get_confusion_matrix()

# decoder.load_state_dict(torch.load('_ReLU128_5.7/single_holdouts/Multitask/sbertNet_tuned/seed1_decoder'))

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle

# def load_context_reps():
#     mean_context_reps = np.empty((16, 768))
#     context_reps_dict = {}
#     for i, task in enumerate(Task.TASK_LIST): 
#         contexts = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_vecs', 'rb'))
#         contexts_perf = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_correct_data', 'rb'))
#         mean_context_reps[i, :] = np.mean(contexts[contexts_perf[:, -1] > 0.95], axis=0)
#         context_reps_dict[task] = contexts[contexts_perf[:, -1] > 0.95]

#     return mean_context_reps, context_reps_dict



# model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model.model_name += '_tuned'
# model.set_seed(1) 
# model.load_model('_ReLU128_5.7/single_holdouts/Multitask')

# instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
# confusion_mat = get_decoder_confusion_matrix(decoder.to('cpu'))







# for task_index, task in enumerate(Task.opp_task_list): 
#     target_instruct, rep = get_instruct_embedding_pair(instruct_reps, task_index, 0)            
#     #rep = context_reps[task_index, :]            
#     print('Target Task: '+ task + ' Target Instruction: '+target_instruct)
#     print(str(decode_sentence(decoder, rep)) + '\n')

# context_vecs = pickle.load(open('_ReLU128_5.7/single_holdouts/Multitask/sbertNet_tuned/seed1_context_vecs', 'rb'))

# context_dict = {}
# for i, task in enumerate(Task.TASK_LIST): 
#     context_dict[task] = context_vecs[i, ...]


# context_confusion_matrix = np.zeros((16, 16))
# for task_index, task in enumerate(Task.opp_task_list): 
#     for j in range(context_dict[task].shape[0]): 
#         rep = context_dict[task][j, :]
#         predicted, _ = decode_sentence(decoder, rep)
#         context_confusion_matrix[task_index, Task.opp_task_list.index(predicted)] += 1

# fig, ax = plt.subplots(figsize=(6, 4))
# res = sns.heatmap(context_confusion_matrix, xticklabels=Task.opp_task_list, yticklabels=Task.opp_task_list, annot=True, cmap='Blues', ax=ax)
# res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 6)
# res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 6)
# plt.show()

# from task import construct_batch
# from utils import isCorrect


# model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model1.model_name += '_tuned'
# model1.set_seed(0) 
# model1.load_model('_ReLU128_5.7/single_holdouts/Multitask')


# num_repeats = 3

# model1.eval()
# batch_len = 128
# with torch.no_grad():
#     perf_dict = dict.fromkeys(Task.TASK_LIST)
#     for j, task in enumerate(Task.TASK_LIST):
#         print(task)
#         mean_list = [] 
#         for _ in range(num_repeats): 
#             ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)


#             instruct_indices = np.random.randint(0, 15, size=128)
#             # reps = instruct_reps[j, instruct_indices, :]
#             reps = instruct_reps[j, instruct_indices, :]
#             task_info = []
#             for i in range(128):
#                 _, instruct = decode_sentence(decoder, reps[i, :])
#                 task_info.append(instruct)


#             out, _ = model1(task_info, torch.Tensor(ins).to(model.__device__))
#             mean_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
#         perf_dict[task] = np.mean(mean_list)


# plot_trained_performance({'sbertNet_tuned': perf_dict})


if '__name__' == '__main__': 
    for i in range(1):
        model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
        model.model_name += '_tuned'
        model.set_seed(i) 


        #model.load_model('_ReLU128_5.7/single_holdouts/Multitask')
        loss_list, teacher_loss_list = train_decoder(decoder, decoder_optimizer, sch, 15, 1.0)
        torch.save(decoder.state_dict(), '_ReLU128_5.7/single_holdouts/Multitask/sbertNet_tuned/seed'+str(i)+'_decoder')

