import itertools
from seaborn.palettes import color_palette
from sklearn.metrics.pairwise import paired_euclidean_distances
from torch.nn.modules.rnn import GRU
from nlp_models import SBERT
from rnn_models import InstructNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, reduce_rep
from plotting import plot_trained_performance, plot_rep_scatter
import numpy as np
import torch.optim as optim
from utils import sort_vocab, isCorrect, task_swaps_map, inv_train_instruct_dict
from task import Task
import seaborn as sns
from collections import defaultdict


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
        self.index2word = {0: '[CLS]', 1: 'EOS', 2: '[PAD]'}
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
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.load_foldername = '_ReLU128_4.11/swap_holdouts'
        self.teacher_loss_list = []
        self.loss_list = []
        self.vocab = Vocab()
        self.instruct_array = np.array([train_instruct_dict[task] for task in Task.TASK_LIST]).squeeze()
        self.contexts = None
        self.validation_ratio = 0.2

    def init_context_set(self, task_file, model_name, seed_str, supervised_str=''):
        all_contexts = np.empty((16, 256, 20))
        for i, task in enumerate(Task.TASK_LIST):
            filename = self.load_foldername+'/'+task_file+'/'+model_name+'/contexts/'+seed_str+task+supervised_str+'_context_vecs20'
            task_contexts = pickle.load(open(filename, 'rb'))
            all_contexts[i, ...]=task_contexts
        self.contexts = all_contexts
        return all_contexts

    def save_model(self, save_string): 
        torch.save(self.state_dict(), save_string+'.pt')

    def save_model_data(self, save_string): 
        pickle.dump(self.teacher_loss_list, open(save_string+'_teacher_loss_list', 'wb'))
        pickle.dump(self.loss_list, open(save_string+'_loss_list', 'wb'))

    def load_model(self, save_string): 
        self.load_state_dict(torch.load(self.load_foldername+'/'+save_string+'.pt'))

    def get_instruct_embedding_pair(self, task_index, instruct_index, training=True): 
        assert self.contexts is not None, 'must initalize decoder contexts with init_context_set'
        if training:
            context_rep_index = np.random.randint(self.contexts.shape[1]-self.contexts.shape[1]*self.validation_ratio, size=instruct_index.shape[0])
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
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.context_dim = 20
        self.embedding = nn.Embedding(self.vocab.n_words, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
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

    def get_decoded_set(self): 
        decoded_set = {}
        confusion_mat = np.zeros((16, 17))
        for i, task in enumerate(Task.TASK_LIST):
            tasks_decoded = defaultdict(list)

            decoded_sentences = self.decode_sentence(torch.Tensor(self.contexts[i, ...]).to(device))

            for instruct in decoded_sentences:
                try: 
                    decoded_task = inv_train_instruct_dict[instruct]
                    tasks_decoded[decoded_task].append(instruct)
                    confusion_mat[i, Task.TASK_LIST.index(decoded_task)] += 1
                except KeyError:
                    tasks_decoded['other'].append(instruct)
                    confusion_mat[i, -1] += 1

            decoded_set[task] = tasks_decoded

        return decoded_set, confusion_mat

    def plot_confuse_mat(self): 
        _, confusion_mat = self.get_decoded_set()
        res=sns.heatmap(confusion_mat, xticklabels=Task.TASK_LIST+['other'], yticklabels=Task.TASK_LIST, annot=True, cmap='Blues', fmt='g', cbar=False)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
        plt.show()

class TranEmbedder(nn.Module):
    def __init__(self, embedding_size, langModel, vocab, layer):
        super(TranEmbedder, self).__init__()
        self.embedding_size = embedding_size
        self.langModel = langModel
        self.vocab = vocab
        self.trans_layer = layer
        self.W_embedder = nn.Linear(768, embedding_size)
    
    def forward(self, ins): 
        np_ins = ins.detach().cpu().numpy()
        decoded_sentences=self.vocab.untokenize_sentence(np_ins[1:,...]) 
        tokens = self.langModel.tokenizer(list(decoded_sentences), return_tensors='pt', padding=True)
        for key, value in tokens.items():
            tokens[key] = value.to(0)
        trans_out = self.langModel.transformer(**tokens)
        out = torch.relu(self.W_embedder(trans_out[2][self.layer][:, 0:-1, :]))
        return torch.swapaxes(out, 0, 1)


class TranDecoderRNN(DecoderRNN):
    def __init__(self, embedding_size, hidden_size, langModel, train_layers=None, tran_embedder_layer=11):
        super().__init__(embedding_size, hidden_size)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.langModel = langModel
        self.train_layers=train_layers
        self.tran_embedder_layer = tran_embedder_layer
        tranEmbedder = TranEmbedder(embedding_size, langModel, self.vocab, tran_embedder_layer)
        self.embedding = tranEmbedder
        langModel.train_layers = train_layers
        langModel.init_train_layers()
        for p in langModel.proj_out.parameters():
            p.requires_grad=False

def test_partner_model(partner_model, decoder, num_repeats=1, tasks=Task.TASK_LIST): 
    partner_model.eval()
    batch_len = decoder.contexts.shape[1]
    decoded_instructs = {}
    with torch.no_grad():
        perf_array = np.empty((num_repeats, 2, len(tasks)))
        for i, mode in enumerate(['context', 'instruct']): 
            for j, task in enumerate(tasks):
                print(task)
                task_info = []
                for k in range(num_repeats): 
                    ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)

                    if mode == 'instruct': 
                        instruct = decoder.decode_sentence(torch.Tensor(decoder.contexts[j,...]).to(device))
                        task_info = list(instruct)
                        decoded_instructs[task] = task_info
                        out, _ = partner_model(task_info, torch.Tensor(ins).to(partner_model.__device__))
                    elif mode == 'context':
                        task_info = decoder.contexts[j, ...]
                        out, _ = super(type(partner_model), partner_model).forward(torch.Tensor(task_info).to(partner_model.__device__), torch.Tensor(ins).to(partner_model.__device__))
                    
                    task_perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
                    perf_array[k, i, j] = task_perf

    return perf_array, decoded_instructs

def train_decoder_(decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None, task_loss_ratio=0.1): 
    criterion = nn.NLLLoss(reduction='mean')
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
                    topv, topi = decoder_output.topk(1)
                    #get words for last sentence in the batch
                    last_word_index = topi.squeeze().detach()[-1].item()
                    last_word = decoder.vocab.index2word[last_word_index]
                    decoded_sentence.append(last_word)

                    decoder_loss += criterion(decoder_output, target_tensor[0, :, di])
                    decoder_input = torch.cat((decoder_input, target_tensor[..., di]))

                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_output, decoder_hidden, decoded_indices = decoder(init_hidden)
                

                for k in range(pad_len):
                    decoder_loss += criterion(decoder_output, target_tensor[0, :, k])

                
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
                


        sch.step()
        teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs

    return loss_list, teacher_loss_list


from plotting import MODEL_STYLE_DICT, mpatches, Line2D

def plot_partner_performance(all_perf_dict):
    barWidth = 0.2
    model_name = 'sbertNet_tuned'
    for i, mode in enumerate(['instructions', 'contexts']):  
        perf = all_perf_dict[mode]
        values = list(np.mean(perf, axis=0))
        std = np.std(perf, axis=0)
        
        len_values = len(Task.TASK_LIST)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        mark_size = 3
        if mode == 'contexts': 
            hatch_style = '/'
            edge_color = 'white'
        else: 
            hatch_style = None
            edge_color = None
        plt.plot(r, [1.05]*16, linestyle="", alpha=0.8, color = ['blue', 'red'][i], markersize=mark_size)
        plt.bar(r, values, width =barWidth, label = model_name, color = ['blue', 'red'][i], edgecolor = 'white')
        #cap error bars at perfect performance 
        error_range= (std, np.where(values+std>1, (values+std)-1, std))
        print(error_range)
        markers, caps, bars = plt.errorbar(r, values, yerr = error_range, elinewidth = 0.5, capsize=1.0, linestyle="", alpha=0.8, color = 'black', markersize=1)

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], Task.TASK_LIST, fontsize='xx-small', fontweight='bold')
    plt.tight_layout()
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))
    #plt.legend()
    plt.show()


import itertools
from utils import training_lists_dict, all_models
seeds = [0, 1]
model_file = '_ReLU128_4.11/swap_holdouts/'
#to_train = list(itertools.product(seeds, ['sbertNet_tuned'], training_lists_dict['swap_holdouts']))
to_train = list(itertools.product(seeds, ['sbertNet_tuned'], [['Multitask']]))
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

        # try: 
        #     pickle.load(open(filename+'/decoders/seed'+str(seed)+'_decoder'+holdout_str+'_loss_list', 'rb'))
        #     print(filename+'/decoders/seed'+str(seed)+'_decoder'+holdout_str+'_loss_list already trained')
        #except FileNotFoundError:

        foldername = '_ReLU128_4.11/swap_holdouts/Multitask'

        model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
        model1.model_name += '_tuned'
        model1.set_seed(seed) 

        model1.load_model('_ReLU128_4.11/swap_holdouts/Multitask')

        model1.to(device)
        model1.eval()

        decoder= TranDecoderRNN(256, model1.langModel, train_layers=['11'])
        #decoder= DecoderRNN(128)
        decoder.init_context_set(task_file, model_name, 'seed'+str(seed))
        decoder.to(device)

        criterion = nn.NLLLoss(reduction='mean')

        params = [{'params' : decoder.gru.parameters()},
            {'params' : decoder.out.parameters()},
            {'params' : decoder.context_encoder.parameters()}, 
            ]

        for n,p in decoder.named_parameters():
            if p.requires_grad: print(n)

        decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-5, weight_decay=0.0)
        # decoder_optimizer = optim.Adam([
        #         {'params' : decoder.embedding.parameters(), 'lr': 1e-5}
        #     ], lr=5*1e-4)
        sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, verbose=False)
        decoder.to(device)

        train_decoder_(decoder, decoder_optimizer, sch, 80, 0.8, holdout_tasks=holdouts, task_loss_ratio=0.0)
        decoder.save_model(filename+'/decoders/seed'+str(seed)+'tran_decoder'+holdout_str)
        decoder.save_model_data(filename+'/decoders/seed'+str(seed)+'tran_decoder'+holdout_str)





foldername = '_ReLU128_4.11/swap_holdouts/Multitask'

model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model1.model_name += '_tuned'
model1.set_seed(0) 
model1.to(device)

decoder= TranDecoderRNN(128, model1.langModel)
decoder.init_context_set('Multitask', 'sbertNet_tuned', 'seed'+str(0), supervised_str='_supervised')

decoder.load_model('Multitask/sbertNet_tuned/decoders/seed0tran_decoder')
decoder.to(device)
decoded_set = decoder.get_decoded_set()
decoded_set['Go']
# model1.load_model('_ReLU128_4.11/swap_holdouts/Multitask')

# decoder.to(device)
# all_perf, decoded_instructs = test_partner_model(model1, decoder, num_repeats=5)

# plot_partner_performance({'instructions': all_perf[:, 1, :], 'contexts': all_perf[:, 0, :]})




