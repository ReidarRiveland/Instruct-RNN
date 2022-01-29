from context_trainer import ContextTrainer
from torch.distributions.utils import logits_to_probs
from torch.nn.modules.pooling import MaxPool1d
from transformers.utils.dummy_pt_objects import Conv1D
from model_trainer import config_model
from utils import train_instruct_dict
from model_analysis import reduce_rep
from plotting import plot_rep_scatter
from utils import sort_vocab, isCorrect, inv_train_instruct_dict
from task import Task, construct_batch
from jit_GRU import CustomGRU
from context_trainer import ContextTrainer
from data import TaskDataSet
import torch.optim as optim



from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from torch.distributions import Categorical

import itertools
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from plotting import MODEL_STYLE_DICT, mpatches, Line2D
import seaborn as sns

import torch
import torch.nn as nn
import pickle

device = torch.device(0)

class PenalizedSoftmax():
    def __init__(self, temp=1, theta=1.2):
        self.theta=theta
        self.temp=temp

    def __call__(self, top_k, last_decoded_indices): 
        logits = top_k.values
        batch_size = logits.shape[0]
        k = logits.shape[1]
        penalized_temps = torch.where(top_k.indices == last_decoded_indices.repeat(k, 1).T, self.temp*self.theta*torch.ones(batch_size, k, device=device), self.temp*torch.ones(batch_size, k, device=device))
        exps = logits * penalized_temps
        exps -= torch.max(exps, -1).values.repeat(k, 1).T
        normalizer = torch.sum(torch.exp(exps), dim=1)
        p = torch.exp((exps))/normalizer.repeat(k, 1).T
        return p 

# class SMDecoder(nn.Module): 
#     def __init__(self):
#         super(SMDecoder, self).__init__()
#         self.conv1 = nn.Conv1d(128, 12, 9, 1, padding=4)
#         self.maxer = nn.MaxPool1d(2, 2)
#         self.dropper = nn.Dropout(p=0.4)
#         self.fc1 = nn.Linear(12*60, 32)
#         self.fc2 =nn.Linear(32, 128)

    
#     def forward(self, sm_hidden): 
#         batch_len = sm_hidden.shape[0]
#         out = torch.relu(self.conv1(sm_hidden.transpose(1,2)))
#         out = self.dropper(self.maxer(out).view(batch_len, -1))
#         out = torch.relu(self.fc1(out))
#         out = torch.relu(self.fc2(out))
#         return out.unsqueeze(0)


class SMDecoder(nn.Module): 
    def __init__(self, out_dim):
        super(SMDecoder, self).__init__()
        # self.conv1 = nn.Conv1d(128, 12, 9, 1, padding=4)
        # self.maxer = nn.MaxPool1d(2, 2)
        self.dropper = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*2, out_dim)
        #self.fc2 =nn.Linear(128, 128)
        self.id = nn.Identity()
    
    
    def forward(self, sm_hidden): 
        out_mean = self.id(torch.mean(sm_hidden, dim=1))
        out_max = self.id(torch.max(sm_hidden, dim=1).values)
        out = torch.cat((out_max, out_mean), dim=-1)
        out = self.dropper(out)
        out = torch.relu(self.fc1(out))
        #out = torch.relu(self.fc2(out))
        return out.unsqueeze(0)


# class SM_RNNDecoder(nn.Module): 
#     def __init__(self):
#         super(SMDecoder, self).__init__()
#         # self.conv1 = nn.Conv1d(128, 12, 9, 1, padding=4)
#         # self.maxer = nn.MaxPool1d(2, 2)
#         #self.dropper = nn.Dropout(p=0.1)
#         self.rnn = 
#         self.fc2 =nn.Linear(128, 128)
#         self.id = nn.Identity()
    
#     def forward(self, sm_hidden): 
#         out_mean = self.id(torch.mean(sm_hidden, dim=1))
#         out_max = self.id(torch.max(sm_hidden, dim=1).values)
#         out = torch.cat((out_max, out_mean), dim=-1)
#         #out = self.dropper(out)
#         out = torch.relu(self.fc1(out))
#         out = torch.relu(self.fc2(out))
#         return out.unsqueeze(0)



class RNNtokenizer():
    sos_token = 0
    eos_token = 1
    def __init__(self):
        self.pad_len = 30
        self.vocab = sort_vocab()
        self.word2index = {}
        self.index2word = {0: '[CLS]', 1: 'EOS', 2: '[PAD]'}
        self.n_words = 3
        for word in self.vocab: 
            self.addWord(word)
    
    def __call__(self, sent_list, pad_len=30):
        return self.tokenize_sentence(sent_list, pad_len)

    def _tokenize_sentence(self, sent, pad_len): 
        tokens = [2]*pad_len
        for i, word in enumerate(sent.split()): 
            tokens[i] = self.word2index[word]
        tokens[i+1]=1
        return tokens

    def tokenize_sentence(self, sent_list, pad_len): 
        tokenized_list = []
        for sent in sent_list:
            tokens = self._tokenize_sentence(sent, pad_len)
            tokenized_list.append(tokens)
        return torch.LongTensor(tokenized_list)

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
        self.init_instructions(train_instruct_dict)
        self.teacher_loss_list = []
        self.loss_list = []
        self.instruct_array = np.array([self.instruct_dict[task] for task in Task.TASK_LIST]).squeeze()
        self.contexts = None
        self.validation_ratio = 0.2

    def init_instructions(self, instruct_dict): 
        self.instruct_dict = instruct_dict
        self.instruct_array = np.array([self.instruct_dict[task] for task in Task.TASK_LIST]).squeeze()

    def add_punctuation(self): 
        instruct_dict = {}
        for task, instructions in train_instruct_dict.items(): 
            new_instructions = []
            for instruct in instructions: 
                instruct = instruct.replace(' otherwise', ', otherwise')
                instruct+='.'
                new_instructions.append(instruct)
            instruct_dict[task]=new_instructions
        return instruct_dict

    def save_model(self, save_string): 
        torch.save(self.state_dict(), save_string+'.pt')

    def save_model_data(self, save_string): 
        pickle.dump(self.teacher_loss_list, open(save_string+'_teacher_loss_list', 'wb'))
        pickle.dump(self.loss_list, open(save_string+'_loss_list', 'wb'))

    def load_model(self, load_string): 
        self.load_state_dict(torch.load(load_string+'.pt'))

    def get_instruct_embedding_pair(self, task_index, instruct_index, training=True): 
        assert self.contexts is not None, 'must initalize decoder contexts with init_context_set'
        if training:
            context_rep_index = np.random.randint(self.contexts.shape[1]-self.contexts.shape[1]*self.validation_ratio, size=instruct_index.shape[0])
        else: 
            context_rep_index = np.random.randint(self.contexts.shape[1],size=instruct_index.shape[0])
        rep = self.contexts[task_index, context_rep_index, :]
        instruct = self.instruct_array[task_index, instruct_index]
        return list(instruct), torch.Tensor(rep)

    def plot_context_embeddings(self, tasks_to_plot=Task.TASK_LIST): 
        reps_reduced, _ = reduce_rep(self.contexts)
        plot_rep_scatter(reps_reduced, tasks_to_plot)
        return reps_reduced

class DecoderRNN(BaseDecoder):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = 20
        self.tokenizer = RNNtokenizer()

        self.embedding = nn.Embedding(self.tokenizer.n_words, self.hidden_size)
        #self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.gru = CustomGRU(self.hidden_size, self.hidden_size, 1, activ_func = torch.relu, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.tokenizer.n_words)
        self.sm_decoder = SMDecoder(self.hidden_size)
        self.__weights_init__()
        self.softmax = nn.LogSoftmax(dim=1)

    def __weights_init__(self):
        for n, p in self.named_parameters():
            if 'weight_ih' in n:
                for ih in p.chunk(3, 0):
                    torch.nn.init.normal_(ih, std = 1/np.sqrt(self.hidden_size))
            elif 'weight_hh' in n:
                for hh in p.chunk(3, 0):
                    hh.data.copy_(torch.eye(self.hidden_size)*0.5)
            elif 'W_out' in n:
                torch.nn.init.normal_(p, std = 0.4/np.sqrt(self.hidden_size))

    def draw_next(self, logits, k_sample=3):
        top_k = logits.topk(k_sample)
        probs = torch.softmax(top_k.values, dim=-1)
        dist = Categorical(probs)
        next_indices = top_k.indices.gather(1, dist.sample().unsqueeze(-1))
        return next_indices

    def _base_forward(self, ins, sm_hidden):
        embedded = self.embedding(ins)
        init_hidden = self.sm_decoder(sm_hidden)
        rnn_out, _ = self.gru(embedded.transpose(0,1), init_hidden)
        logits = self.out(rnn_out[:, -1,:])
        return logits, rnn_out

    def forward(self, sm_hidden):
        sos_input = torch.tensor([[self.tokenizer.sos_token]*sm_hidden.shape[0]]).to(sm_hidden.get_device())
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

class gptDecoder(BaseDecoder): 
    def __init__(self, conv_out_channels=64, kernel_size=8, stride=3):
        super().__init__()
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.gelu = nn.GELU()
        self.init_instructions(self.add_punctuation())
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.conv = nn.Conv1d(in_channels=128, out_channels = 64, kernel_size=self.kernel_size, stride=self.stride) 
        self.context_encoder = nn.Linear(64, 768)

        self.psoftmax = PenalizedSoftmax()

        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', use_cache=True)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.tokenizer.model_max_length=30
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def sm_encode(self, sm_hidden): 
        out = torch.max(self.conv(sm_hidden.transpose(1,2)), dim=-1).values
        out = self.gelu(self.context_encoder(out))
        return out.unsqueeze(1)


    def tokenize_instruction(self, instructions): 
        instructions = [instruct+' '+self.tokenizer.eos_token for instruct in instructions]
        tokenized = self.tokenizer(instructions, return_tensors='pt', padding=True)
        attention_mask=torch.cat((torch.ones(tokenized['attention_mask'].shape[0], 1), tokenized['attention_mask']), dim=1)
        return tokenized['input_ids'], attention_mask

    def draw_next(self, last_logits, decoded_indices, k_sample=5):
        if decoded_indices.shape[0] != 0:
            last_decoded_indices = decoded_indices[:, -1]
        else: 
            last_decoded_indices = torch.Tensor(last_logits.shape[0]*[torch.inf]).to(device)
        top_k = last_logits.topk(k_sample)
        probs = self.psoftmax(top_k, last_decoded_indices)
        dist = Categorical(probs)
        next_indices = top_k.indices.gather(1, dist.sample().unsqueeze(-1))
        return next_indices


    def _base_forward(self, sm_hidden=None, input_ids=None, attention_mask=None, past_keys=None): 
        if past_keys is None: 
            embedded_inputs = self.sm_encode(sm_hidden)
        else: 
            embedded_inputs = torch.Tensor([]).to(device)

        if input_ids is not None: 
            embedded_inputs = torch.cat((embedded_inputs, self.gpt.transformer.wte(input_ids)), dim=1)
        
        return self.gpt(inputs_embeds=embedded_inputs, attention_mask=attention_mask, past_key_values=past_keys)

    def forward(self, sm_hidden):
        past_keys = None
        input_ids = None
        decoded_indices = torch.Tensor([]).to(device)
        scores = torch.Tensor([]).to(device)

        for di in range(self.tokenizer.model_max_length):
            outputs = self._base_forward(sm_hidden, input_ids = input_ids, past_keys=past_keys)
            past_keys = outputs.past_key_values
            logits = outputs.logits
            #no repeat

            cur_scores = self.softmax(logits)
            last_logits = logits[:, -1, :]
            input_ids = self.draw_next(last_logits, decoded_indices)

            decoded_indices = torch.cat((decoded_indices, input_ids), dim=1)
            scores = torch.cat((scores, cur_scores), dim=1)

        return scores.squeeze(), decoded_indices

    def decode_sentence(self, sm_hidden): 
        _, decoded_indices = self.forward(sm_hidden)
        decoded_sentences = self.tokenizer.batch_decode(decoded_indices.int(), skip_special_tokens=True)  # detach from history as input
        return decoded_sentences

class EncoderDecoder(nn.Module): 
    def __init__(self, sm_model, decoder): 
        super(EncoderDecoder, self).__init__()
        self.sm_model = sm_model
        self.decoder = decoder
        self.contexts = None
        self.load_foldername = '_ReLU128_4.11/swap_holdouts'

    def load_model_componenets(self, task_file, seed, load_holdout_decoder=True):
        self.sm_model.set_seed(seed)
        self.sm_model.load(self.load_foldername+'/'+task_file)
        
        if load_holdout_decoder:
            holdout_str = '_wHoldout'
        else: 
            holdout_str = ''
        self.decoder.load(self.load_foldername+'/'+task_file+'/'+self.sm_model.model_name+'/decoders/seed'+str(seed)+'_rnn_decoder'+holdout_str)

        self.init_context_set(task_file, seed)

    def init_context_set(self, task_file, seed):
        all_contexts = np.empty((16, 128, 20))
        for i, task in enumerate(Task.TASK_LIST):
            try: 
                filename = self.load_foldername+'/'+task_file+'/'+self.sm_model.model_name+'/contexts/seed'+str(seed)+task+'_supervised_context_vecs20'
                task_contexts = pickle.load(open(filename, 'rb'))
                all_contexts[i, ...]=task_contexts[:128, :]
            except FileNotFoundError: 
                print('no contexts for '+task+' for model file '+task_file)

        self.contexts = all_contexts
        return all_contexts

    def decode_set(self, num_trials, from_contexts=False, tasks=Task.TASK_LIST, t=120): 
        decoded_set = {}
        confusion_mat = np.zeros((16, 17))

        for i, task in enumerate(tasks): 
            tasks_decoded = defaultdict(list)

            ins, _, _, _, _ = construct_batch(task, num_trials)

            if from_contexts: 
                task_index = Task.TASK_LIST.index(task)
                task_info = torch.Tensor(self.contexts[task_index, ...]).to(self.sm_model.__device__)
                _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), context=task_info)
            else: 
                task_info = self.sm_model.get_task_info(num_trials, task)
                _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), task_info)
            
            decoded_sentences = self.decoder.decode_sentence(sm_hidden[:,0:t, :]) 

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
    
    def plot_confuse_mat(self, num_trials, from_contexts=False, confusion_mat=None): 
        if confusion_mat is None:
            _, confusion_mat = self.decode_set(num_trials, from_contexts=from_contexts)
        res=sns.heatmap(confusion_mat, xticklabels=Task.TASK_LIST+['other'], yticklabels=Task.TASK_LIST, annot=True, cmap='Blues', fmt='g', cbar=False)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
        plt.show()

    def test_partner_model(self, partner_model, num_repeats=3, tasks=Task.TASK_LIST, decoded_dict=None): 
        partner_model.eval()
        if decoded_dict is None: 
            decoded_dict, _ = self.decode_set(128, from_contexts=True)
        batch_len = sum([len(item) for item in list(decoded_dict.values())[0].values()])

        decoded_instructs = {}
        perf_dict = {}
        with torch.no_grad():
            for i, mode in enumerate(['context', 'instructions']): 
                perf_array = np.empty((len(tasks), num_repeats))

                for j, task in enumerate(tasks):
                    print(task)
                    task_info = []
                    for k in range(num_repeats): 
                        ins, targets, _, target_dirs, _ = construct_batch(task, 128)

                        if mode == 'instructions': 
                            task_info = list(itertools.chain.from_iterable([value for value in decoded_dict[task].values()]))
                            decoded_instructs[task] = task_info
                            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), task_info)
                        elif mode == 'context':
                            task_info = self.contexts[j, ...]
                            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), context=torch.Tensor(task_info).to(partner_model.__device__))
                        
                        task_perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
                        perf_array[j, k] = task_perf
                perf_dict[mode] = perf_array
        return perf_dict, decoded_instructs

    def _partner_model_over_training(self, partner_model, task, num_repeats=1, task_file='Multitask'): 
        context_trainer = ContextTrainer(self.sm_model, self.decoder.context_dim, task_file)
        context_trainer.supervised_str=='supervised'
        context = nn.Parameter(torch.randn((128, self.decoder.context_dim), device=device))

        opt= optim.Adam([context], lr=5*1e-1, weight_decay=0.0)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.99)
        streamer = TaskDataSet(batch_len = 128, num_batches = 100, task_ratio_dict={task:1})
        contexts, is_trained = context_trainer.train_context(streamer, 1, opt, sch, context, decoder=self.decoder)

        task_perf_list = []
        ins, targets, _, target_dirs, _ = construct_batch(task, 128)
        for i in range(2*100): 
            if i%50==0: 
                print(i)
            task_info = context_trainer._decode_during_training_[i]
            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), task_info)
            task_perf_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
        
        return task_perf_list, context_trainer._decode_during_training_, self.sm_model._correct_data_dict

    def _partner_model_over_t(self, partner_model, task, from_contexts=True, t_set = [120]): 
        ins, targets, _, target_dirs, _ = construct_batch(task, 128)

        if from_contexts: 
            task_index = Task.TASK_LIST.index(task)
            task_info = torch.Tensor(self.contexts[task_index, :128, :]).to(self.sm_model.__device__)
            _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), context=task_info)
        else: 
            task_info = self.sm_model.get_task_info(128, task)
            _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), task_info)

        instruct_list = []
        task_perf_list = []
        for t in t_set: 
            instruct = list(self.decoder.decode_sentence((sm_hidden[:, 0:t,:])))
            instruct_list.append(instruct)
            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), instruct)
            task_perf_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
        
        return task_perf_list, instruct_list

    def plot_partner_performance(self, all_perf_dict):
        barWidth = 0.2
        model_name = 'sbertNet_tuned'
        for i, mode in enumerate(['instructions', 'context']):  

            perf = all_perf_dict[mode]
            values = list(np.mean(perf, axis=1))
            std = np.std(perf, axis=1)
            
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
    

# import smart_open
# smart_open.open = smart_open.smart_open
# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
# weights = torch.FloatTensor(model.vectors)


# from utils import task_swaps_map
# from model_trainer import config_model, training_lists_dict


# swap_tasks = training_lists_dict['swap_holdouts']

# confusion_mat = np.zeros((16, 17))
# decoded_dict = {}

# from_context = True
# seed=0

# sm_model = config_model('sbertNet_tuned')
# sm_model.set_seed(seed)
# decoder_rnn=DecoderRNN(128)
# sm_model.to(device)
# decoder_rnn.to(device)
# # for tasks in swap_tasks:
# #     task_file = task_swaps_map[tasks[0]]
# #     load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
# #     sm_model.load_model(load_str)
# #     decoder_rnn.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder')
# #     encoder_decoder = EncoderDecoder(sm_model, decoder_rnn)
# #     if from_context:
# #         encoder_decoder.init_context_set(task_file, seed)
# #     tmp_decoded, tmp_confuse_mat = encoder_decoder.decode_set(128, from_contexts=from_context, tasks=tasks)
# #     for i, task in enumerate(tasks):
# #         decoded_dict[task] =tmp_decoded[task]
# #         confusion_mat[Task.TASK_LIST.index(task),:] = tmp_confuse_mat[i,:] 



# task_file='Multitask'
# seed=0
# sm_model = config_model('sbertNet_tuned')
# sm_model.set_seed(seed)



# rnn_decoder=DecoderRNN(64)
# sm_model.to(device)
# rnn_decoder.to(device)
# sm_model.eval()
# rnn_decoder.eval()

# load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
# sm_model.load_model(load_str)

# rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_lin_wHoldout')
# encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)

# encoder_decoder.init_context_set(task_file, seed)


# encoder_decoder.plot_confuse_mat(128, from_contexts=False)



# decoded, confuse_mat = encoder_decoder.decode_set(128, from_contexts=False)

# task_info = list(itertools.chain.from_iterable([value for value in decoded['Anti Go'].values()]))
# Counter(task_info)


# from collections import Counter
# Counter(decoded['Anti DM']['other'])



# num_gen = 0
# for task in Task.TASK_LIST: 
#     num_gen += len(set(decoded[task][task]))
# task = 'Anti Go'
# set(decoded[task][task])

# num_gen/16


# from rnn_models import InstructNet
# from nlp_models import SBERT

# model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model1.model_name += '_tuned'
# model1.set_seed(0) 
# model1.load_model('_ReLU128_4.11/swap_holdouts/Multitask')
# model1.to(device)


# perf, _ = encoder_decoder.test_partner_model(model1, num_repeats=5)

# np.mean(perf['instructions'])

# encoder_decoder.plot_partner_performance(perf)

# perf

# fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(8, 8))
# for j, task in enumerate(Task.TASK_LIST):
#     t_set=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#     task_perf, instructs = encoder_decoder._partner_model_over_t(model1, task, t_set=t_set)
#     print(task_perf)
#     ax = axn.flat[j]
#     ax.set_ylim(-0.05, 1.05)
#     ax.set_xlim(-5, 125)
#     ax.set_title(task, size=6, pad=1)
#     ax.xaxis.set_tick_params(labelsize=5)
#     ax.set_xticks(t_set)
#     ax.yaxis.set_tick_params(labelsize=10)
#     ax.plot(t_set, task_perf)

# ax.get_xticks()
# ax.get_xlim()

# plt.show()

# t_set=[1, 20, 60, 80, 100, 120]



# task_perf
# sm_perf['Go']
# plt.plot(task_perf)
# plt.plot(sm_perf['Go'])
# plt.show()




# task_file='Go_Anti_DM'
# seed=0
# sm_model = config_model('sbertNet_tuned')
# sm_model.set_seed(seed)




# gpt_decoder=gptDecoder(128)
# sm_model.to(device)
# gpt_decoder.to(device)
# sm_model.eval()
# gpt_decoder.eval()

# load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
# sm_model.load_model(load_str)
# gpt_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_gpt_decoder_conv64_wHoldout')
# encoder_decoder = EncoderDecoder(sm_model, gpt_decoder)

# encoder_decoder.init_context_set(task_file, seed)

# encoder_decoder.contexts.shape

#encoder_decoder.plot_confuse_mat(128, from_contexts=True)


# decoded, confuse_mat = encoder_decoder.decode_set(128, from_contexts=True)

# len(decoded['Go']['other'])

# from collections import Counter
# Counter(decoded['DM']['other'])

# perf, _ = encoder_decoder.test_partner_model(model1, num_repeats=5)

# encoder_decoder.plot_partner_performance(perf)

# np.mean(perf['instructions'])

# dict(zip(Task.TASK_LIST, np.mean(perf['instructions'], axis=1)))



# from collections import Counter
# Counter(decoded['DM']['other'])



# encoder_decoder.decoder.psoftmax=PenalizedSoftmax(theta=1.2)


# tokenized = gpt_decoder.tokenizer('To be or not to be? That is', return_tensors='pt')
# gpt = gpt_decoder.gpt
# gpt.to('cpu')
# out_ids = gpt.generate(**tokenized)
# gpt_decoder.tokenizer.batch_decode(out_ids)






# np.min(np.mean(perf['instructions'], axis=1))

# encoder_decoder.plot_partner_performance(perf)

# decoder_rnn = DecoderRNN(128, conv_out_channels=64).to(device)
# decoder_rnn.load_state_dict(torch.load('decoderRNN_ANTIDM.pt'))


# encoder = EncoderDecoder(model1, decoder_rnn)

# for task in 
# encoder.init_context_set('Multitask', 0)

# decoded_dict, _ = encoder.decode_set(64)

# decoded_dict['Anti DM']

# # import itertools

# # list(itertools.chain.from_iterable([value for value in decoded_dict['Anti DM'].values()]))

# #encoder.plot_confuse_mat(128, from_contexts=True)






# import seaborn as sns
# import matplotlib.pyplot as plt

# def plot_heatmap(tens, index=0): 
#     sns.heatmap(tens[index, ...].T.detach().numpy())
#     plt.show()

# conv1 = nn.Conv1d(128, 1, 5, 1, padding=2)


# conv2 = nn.Conv1d(64, 32, 5, 1, padding=2)
# conv3 = nn.Conv1d(32, 12, 11, 3, padding=5)


# conv1 = nn.Conv1d(128, 1, 5, 1, padding=2)
# model = nn.Sequential(conv1, nn.ReLU())


# task = 'DM'
# ins, _, _, _, _ = construct_batch(task, 64)
# instructions = sm_model.get_task_info(64, task)

# _, sm_hidden = sm_model(torch.Tensor(ins), instructions)

# sm_decoder = SMDecoder()

# sm_decoder(sm_hidden).

# out = model(sm_hidden.transpose(1,2))

# out.shape
# torch.max(out, dim=1).values

# sns.heatmap(sm_hidden[0, ...].detach().numpy())
# plt.show()
# plot_heatmap(out, index=5)