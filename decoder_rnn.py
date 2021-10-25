from sklearn.metrics.pairwise import paired_euclidean_distances
from nlp_models import SBERT
from rnn_models import InstructNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, reduce_rep
from plotting import plot_trained_performance, plot_rep_scatter
import numpy as np
import torch.optim as optim
from utils import sort_vocab, isCorrect, task_swaps_map
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
        self.validation_holdouts = self.contexts.shape[1]/5

    def _init_context_set(self, load_object):
        if type(load_object) is str: 
            contexts = pickle.load(open('_ReLU128_5.7/swap_holdouts/'+load_object, 'rb'))
        else: 
            contexts = load_object
        return contexts

    def save_model(self, save_string): 
        torch.save(self.state_dict(), save_string+'.pt')

    def save_model_data(self, save_string): 
        pickle.dump(self.teacher_loss_list, open(save_string+'_teacher_loss_list', 'wb'))
        pickle.dump(self.loss_list, open(save_string+'_loss_list', 'wb'))

    def load_model(self, save_string): 
        self.load_state_dict(torch.load('_ReLU128_5.7/swap_holdouts/'+save_string+'.pt'))

    def get_instruct_embedding_pair(self, task_index, instruct_index, training=True): 
        if training:
            context_rep_index = np.random.randint(self.contexts.shape[1]-self.validation_holdouts, size=instruct_index.shape[0])
        else: 
            context_rep_index = np.random.randint(self.contexts.shape[1],size=instruct_index.shape[0])
        rep = self.contexts[task_index, context_rep_index, :]
        instruct = self.instruct_array[task_index, instruct_index]
        return instruct, rep

    def plot_context_embeddings(self, tasks_to_plot=Task.TASK_LIST): 
        reps_reduced, _ = reduce_rep(self.contexts)
        plot_rep_scatter(reps_reduced, tasks_to_plot)
        return reps_reduced

class DecoderRNN(BaseDecoder):
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
        decoded_sentences = self.vocab.untokenize_sentence(decoded_indices[1:,...])  # detach from history as input
        return decoded_sentences

class DecoderTaskRNN(DecoderRNN):
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

def test_partner_model(partner_model, decoder, num_repeats=1): 
    partner_model.eval()
    batch_len = decoder.contexts.shape[1]
    with torch.no_grad():
        perf_array = np.empty((num_repeats, 2, 16))
        for i, mode in enumerate(['context', 'instruct']): 
            for j, task in enumerate(Task.TASK_LIST):
                print(task)
                mean_list = [] 
                task_info = []
                for k in range(num_repeats): 
                    ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)

                    if mode == 'instruct': 
                        instruct = decoder.decode_sentence(torch.Tensor(decoder.contexts[j,...]).to(device))
                        task_info = list(instruct)
                        out, _ = partner_model(task_info, torch.Tensor(ins).to(partner_model.__device__))
                    elif mode == 'context':
                        task_info = decoder.contexts[j, ...]
                        out, _ = super(type(partner_model), partner_model).forward(torch.Tensor(task_info).to(partner_model.__device__), torch.Tensor(ins).to(partner_model.__device__))
                    
                    task_perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
                    perf_array[k, i, j] = task_perf

    return perf_array



def train_decoder_(decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None, task_loss_ratio=0.1): 
    teacher_forcing_ratio = init_teacher_forcing_ratio
    pad_len  = decoder.vocab.pad_len 
    loss_list = []
    teacher_loss_list = []
    task_indices = list(range(16))
    batch_size=32

    if holdout_tasks is not None and 'Multitask' not in holdout_tasks: 
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
                    if hasattr(decoder, 'task_decoder'):
                        task_cat = decoder.softmax(decoder.task_decoder(decoder_hidden))
                        _, topi_task = task_cat.topk(1)

                    topv, topi = decoder_output.topk(1)
                    #get words for last sentence in the batch
                    last_word_index = topi.squeeze().detach()[-1].item()
                    last_word = decoder.vocab.index2word[last_word_index]
                    decoded_sentence.append(last_word)

                    decoder_loss += criterion(decoder_output, target_tensor[0, :, di])
                    decoder_input = torch.cat((decoder_input, target_tensor[..., di]))

                if hasattr(decoder, 'task_decoder'):
                    task_loss = criterion(task_cat[0, ...], torch.LongTensor([task_index]*batch_size).to(device))#*(1/(pad_len-di))

                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_output, decoder_hidden, decoded_indices = decoder(init_hidden)
                

                for k in range(pad_len):
                    decoder_loss += criterion(decoder_output, target_tensor[0, :, k])

                if hasattr(decoder, 'task_decoder'):
                    _, topi_task = task_cat.topk(1)
                    task_loss = criterion(task_cat[0, ...], torch.LongTensor([task_index]*batch_size).to(device))#*(1/(pad_len-k))
                
                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                decoded_sentence = decoder.vocab.untokenize_sentence(decoded_indices)[-1]  # detach from history as input        
                decoder.loss_list.append(loss.item()/pad_len)

            loss.backward()
            opt.step()

            if j%50==0: 
                print('Teacher forceing: ' + str(use_teacher_forcing))
                print('Decoder Loss: ' + str(decoder_loss.item()/pad_len))
                #print('Task Loss: ' + str(task_loss.item()/pad_len))

                print('target instruction: ' + target_instruct[-1])
                if use_teacher_forcing:
                    try:
                        eos_index = decoded_sentence.index('EOS')
                    except ValueError: 
                        eos_index = -1
                    decoded_sentence = ' '.join(decoded_sentence[:eos_index])
                
                print('decoded instruction: ' + decoded_sentence + '\n')
                
                if hasattr(decoder, 'task_decoder'):
                    print('Predicted Task: ' + Task.TASK_LIST[topi_task[0, -1].squeeze().item()] + ', Target Task: ' + Task.TASK_LIST[task_index] + '\n')


        sch.step()
        teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs

    return loss_list, teacher_loss_list


# decoder= DecoderRNN(128, 'Anti_Go_MultiCOMP1/sbertNet_tuned/contexts/seed0_supervised_context_vecs20')

# decoder.load_model('Anti_Go_MultiCOMP1/sbertNet_tuned/_wHoldoutseed0_decoder')

# decoder.to(device)

# decoded_set = {}
# for i, task in enumerate(Task.TASK_LIST):
#     decoded_set[task] = set(decoder.decode_sentence(torch.Tensor(decoder.contexts[i, ...]).to(device)))


# correct_list = []
# incorrect_list = []
# for instruct in decoded_set['MultiCOMP1']: 
#     if instruct in train_instruct_dict['MultiCOMP1']:
#         correct_list.append(instruct)
#     else: 
#         incorrect_list.append(instruct)

# correct_list
# incorrect_list

# model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model1.model_name += '_tuned'
# model1.set_seed(1) 
# model1.load_model('_ReLU128_5.7/swap_holdouts/Multitask')
# #model1.to(device)

# model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model.model_name += '_tuned'
# model.set_seed(0) 
# model.load_model('_ReLU128_5.7/swap_holdouts/Multitask')
# lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='12')


# perf = test_partner_model(model1, decoder, num_repeats=2)

# from plotting import MODEL_STYLE_DICT, mpatches, Line2D
# perf
# perf
# perf_dict = {}
# perf_dict['contexts'] = perf[:, 0, ...]
# perf_dict['instructions'] = perf[:,1, ...]

# perf_dict['contexts'].shape

# def plot_trained_performance(all_perf_dict):
#     barWidth = 0.2
#     model_name = 'sbertNet_tuned'
#     for i, mode in enumerate(['contexts', 'instructions']):  
#         perf = all_perf_dict[mode]
#         values = list(np.mean(perf, axis=0))
#         std = np.std(perf, axis=0)
        
#         len_values = len(Task.TASK_LIST)
#         if i == 0:
#             r = np.arange(len_values)
#         else:
#             r = [x + barWidth for x in r]
#         if '_layer_11' in model_name: 
#             mark_size = 4
#         else: 
#             mark_size = 3
#         if mode == 'contexts': 
#             hatch_style = '/'
#             edge_color = 'white'
#         else: 
#             hatch_style = None
#             edge_color = None
#         plt.plot(r, [1.05]*16, marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
#         plt.bar(r, values, width =barWidth, label = model_name, hatch=hatch_style, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white')
#         #cap error bars at perfect performance 
#         error_range= (std, np.where(values+std>1, (values+std)-1, std))
#         print(error_range)
#         markers, caps, bars = plt.errorbar(r, values, yerr = error_range, elinewidth = 0.5, capsize=1.0, linestyle="", alpha=0.8, color = 'black', markersize=1)

#     plt.ylim(0, 1.15)
#     plt.title('Trained Performance')
#     plt.xlabel('Task Type', fontweight='bold')
#     plt.ylabel('Percentage Correct')
#     r = np.arange(len_values)
#     plt.xticks([r + barWidth+0.25 for r in range(len_values)], Task.TASK_LIST, fontsize='xx-small', fontweight='bold')
#     plt.tight_layout()
#     Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
#                 markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
#     Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
#     Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))
#     #plt.legend()
#     plt.show()

# plot_trained_performance(perf_dict)

if __name__ == "__main__": 
    import itertools
    from utils import training_lists_dict, all_models
    seeds = [0, 1, 2, 3, 4]
    model_file = '_ReLU128_5.7/swap_holdouts/'
    to_train = list(itertools.product(seeds, all_models, ['Multitask']+training_lists_dict['swaps']))
    for config in to_train: 
        seed, model_name, tasks = config 
        for holdout_train in [False, True]: 
            if holdout_train: 
                holdout_str = '_wHoldout'
                holdouts=tasks
            else: 
                holdout_str = ''
                holdouts = []


            print(seed, tasks, holdout_str, holdouts)

            task_file = task_swaps_map[tasks[0]]
            filename = model_file + task_file+'/'+ model_name 

            try: 
                pickle.load(open(filename+'/decoders/seed'+str(seed)+'_decoder'+holdout_str+'_loss_list', 'rb'))
                print(filename+'/decoders/seed'+str(seed)+'_decoder'+holdout_str+'_loss_list already trained')
            except FileNotFoundError:
                decoder= DecoderRNN(128, filename+ '/contexts/seed' +str(seed)+'_context_vecs20')
                decoder.to(device)

                criterion = nn.NLLLoss(reduction='mean')
                decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.0)
                sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, verbose=False)
                decoder.to(device)

                train_decoder_(decoder, decoder_optimizer, sch, 50, 1.0, holdout_tasks=holdouts, task_loss_ratio=0.0)
                decoder.save_model(filename+'/decoders/seed'+str(seed)+'_decoder'+holdout_str)
                decoder.save_model_data(filename+'/decoders/seed'+str(seed)+'_decoder'+holdout_str)





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

