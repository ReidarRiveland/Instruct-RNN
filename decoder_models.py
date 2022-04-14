from lib2to3.pgen2 import token

from matplotlib import use
from context_trainer import ContextTrainer
from torch.distributions.utils import logits_to_probs
from torch.nn.modules.pooling import MaxPool1d
from transformers.utils.dummy_pt_objects import Conv1D
from model_trainer import config_model
from utils.utils import train_instruct_dict
from model_analysis import reduce_rep
from plotting import plot_rep_scatter
from utils.utils import sort_vocab, isCorrect, inv_train_instruct_dict, count_vocab, training_lists_dict, get_holdout_file
from task import Task, construct_batch
from models.script_gru import CustomGRU
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

class SMDecoder(nn.Module): 
    def __init__(self, out_dim, drop_p):
        super(SMDecoder, self).__init__()
        self.dropper = nn.Dropout(p=drop_p)
        self.fc1 = nn.Linear(128*2, out_dim)
        #self.fc2 =nn.Linear(128, out_dim)
        self.id = nn.Identity()
    
    
    def forward(self, sm_hidden): 
        out_mean = self.id(torch.mean(sm_hidden, dim=1))
        out_max = self.id(torch.max(sm_hidden, dim=1).values)
        out = torch.cat((out_max, out_mean), dim=-1)
        out = self.dropper(out)
        out = torch.relu(self.fc1(out))
        #out = torch.relu(self.fc2(out))
        return out.unsqueeze(0)


class RNNtokenizer():
    def __init__(self, langModel):
        self.langModel = langModel
        self.sos_token_id = 0
        self.eos_token_id = 1
        if self.langModel is not None: 
            langModel.tokenizer.eos_token = '[EOS]'
            self.sos_token_id = self.langModel.tokenizer.cls_token_id
            self.eos_token_id = self.langModel.tokenizer.eos_token_id
        self.langModel = langModel
        self.pad_len = 30
        self.counts, self.vocab = count_vocab()
        self.word2index = {}
        self.index2word = {0: '[CLS]', 1: '[EOS]', 2: '[PAD]'}
        self.n_words = 3
        for word in self.vocab: 
            self.addWord(word)

        if self.langModel is not None:
            self.order_bert_tokens = [token[0] for token in self.langModel.tokenizer(list(self.index2word.values()), add_special_tokens=False).input_ids]
            self.bert_word2index = dict(zip(self.word2index.keys(),self.order_bert_tokens))

    def get_smoothed_freq(self, a=100): 
        freq_tensor = torch.empty(len(self.index2word.items()))
        for index, word in self.index2word.items(): 
            freq_tensor[index] = self.counts[word]
        freq_tensor[0:3] = torch.Tensor([torch.ceil(torch.mean(freq_tensor)).item()]*3)
        freq_weight = (a/(a+freq_tensor))
        return freq_weight

    def __call__(self, sent_list, use_langModel = False, pad_len=30):
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
            if sent[-1] == "[EOS]":
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
        self.load_state_dict(torch.load(load_string+'.pt', map_location=torch.device('cpu')))

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
    def __init__(self, hidden_size, drop_p = 0.1, langModel=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = 20
        self.langModel = langModel

        self.tokenizer = RNNtokenizer(langModel)
        if langModel is None: 
            self.embedding_size=64
            self.embedding = nn.Embedding(self.tokenizer.n_words, self.embedding_size)
        else: 
            self.embedding = nn.Sequential(langModel.transformer.embeddings.word_embeddings, 
                                            langModel.transformer.embeddings.LayerNorm, 
                                            langModel.transformer.embeddings.dropout)
            self.embedding_size=768

        #self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.gru = CustomGRU(self.embedding_size, self.hidden_size, 1, activ_func = torch.relu, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.tokenizer.n_words)
        self.sm_decoder = SMDecoder(self.hidden_size, drop_p=drop_p)
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
                #need an underscore
                filename = self.load_foldername+'/'+task_file+'/'+self.sm_model.model_name+'/contexts/seed'+str(seed)+task+'supervised_context_vecs20'
                task_contexts = pickle.load(open(filename, 'rb'))
                all_contexts[i, ...]=task_contexts[:128, :]
            except FileNotFoundError: 
                print('no contexts for '+task+' for model file '+task_file)

        self.contexts = all_contexts

    def decode_set(self, num_trials, num_repeats = 1, from_contexts=False, tasks=Task.TASK_LIST, t=120): 
        decoded_set = {}
        confusion_mat = np.zeros((16, 17))
        for _ in range(num_repeats): 
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
    
    def plot_confuse_mat(self, num_trials, num_repeats, from_contexts=False, confusion_mat=None, fmt='g'): 
        if confusion_mat is None:
            _, confusion_mat = self.decode_set(num_trials, num_repeats = num_repeats, from_contexts=from_contexts)
        res=sns.heatmap(confusion_mat, linewidths=0.5, linecolor='black', mask=confusion_mat == 0, xticklabels=Task.TASK_LIST+['other'], yticklabels=Task.TASK_LIST, annot=True, cmap='Blues', fmt=fmt, cbar=False)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
        plt.show()

    def test_partner_model(self, partner_model, num_repeats=3, tasks=Task.TASK_LIST, decoded_dict=None, use_others=False): 
        partner_model.eval()
        if decoded_dict is None: 
            decoded_dict, _ = self.decode_set(128, from_contexts=True)
        batch_len = sum([len(item) for item in list(decoded_dict.values())[0].values()])

        perf_dict = {}
        with torch.no_grad():
            for i, mode in enumerate(['context', 'instructions', 'others']): 
                perf_array = np.empty((len(tasks), num_repeats))
                perf_array[:] = np.nan
                for k in range(num_repeats): 
                    for j, task in enumerate(tasks):
                        print(task)
                        task_info = []
                        ins, targets, _, target_dirs, _ = construct_batch(task, 128)
                        if mode == 'others': 
                            try:
                                task_info = list(np.random.choice(decoded_dict[task]['other'], 128))
                                out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), task_info)
                            except ValueError:
                                continue
                        elif mode == 'instructions':
                            task_info = list(itertools.chain.from_iterable([value for value in decoded_dict[task].values()]))
                            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), task_info)
                        elif mode == 'context':
                            task_index = Task.TASK_LIST.index(task)
                            task_info = self.contexts[task_index, ...]
                            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), context=torch.Tensor(task_info).to(partner_model.__device__))
                        
                        task_perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
                        perf_array[j, k] = task_perf
                perf_dict[mode] = perf_array
        return perf_dict, decoded_dict

    def _decoded_over_training(self, task, num_repeats=1, task_file='Multitask', lr=1e-1): 
        context_trainer = ContextTrainer(self.sm_model, self.decoder.context_dim, task_file)
        context_trainer.supervised_str=='supervised'
        context = nn.Parameter(torch.randn((256, self.decoder.context_dim), device=device))

        opt= optim.Adam([context], lr=lr, weight_decay=0.0)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.99)
        streamer = TaskDataSet(batch_len = 256, num_batches = 100, task_ratio_dict={task:1})
        is_trained = context_trainer.train_context(streamer, 1, opt, sch, context, decoder=self.decoder)

        return context_trainer._decode_during_training_, self.sm_model._correct_data_dict

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
    

def get_all_partner_model_perf(num_repeats=5): 
    task_file='Multitask'    
    sm_model = config_model('sbertNet_tuned')
    partner_model = config_model('sbertNet_tuned')

    rnn_decoder = DecoderRNN(64, drop_p=0.1)
    sm_model.to(device)
    rnn_decoder.to(device)
    sm_model.eval()
    all_seeds = [0, 1, 2, 3, 4]

    load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
    
    for seed in [0]: 
        all_perf_dict = {}
        all_perf_dict['instructions'] = np.empty((4, 16, num_repeats))
        all_perf_dict['context'] = np.empty((4, 16, num_repeats))
        all_perf_dict['others'] = np.empty((4, 16, num_repeats))
        print('\n seed '+str(seed)+'\n')
        sm_model.set_seed(seed)
        sm_model.load_model(load_str)
        rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_lin_wHoldout')
        encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)
        encoder_decoder.init_context_set(task_file, seed)
        decoded, confuse_mat = encoder_decoder.decode_set(128, from_contexts=True)
        partner_seeds = all_seeds.copy()
        partner_seeds.remove(seed)
        for i, partner_seed in enumerate([1]): 
            print(partner_seeds)
            print('\n partner seed '+str(partner_seed)+'\n')
            partner_model.set_seed(partner_seed)
            partner_model.load_model(load_str)
            perf, _ = encoder_decoder.test_partner_model(partner_model, num_repeats=num_repeats)
            for mode in ['context', 'instructions', 'others']:
                all_perf_dict[mode][i,...] = perf[mode]
        # pickle.dump(all_perf_dict, open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_all_perf_dict', 'wb'))
        # pickle.dump(confuse_mat, open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_confuse_mat', 'wb'))
        # pickle.dump(decoded, open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_decoded_set', 'wb'))

    return all_perf_dict


def load_decoder_outcomes(num_repeats=5): 
    all_perf_dict = {}
    all_perf_dict['instructions'] = np.empty((5, 4, 16, num_repeats))
    all_perf_dict['context'] = np.empty((5, 4, 16, num_repeats))
    all_perf_dict['others'] = np.empty((5, 4, 16, num_repeats))
    all_decoded_sets = []
    all_confuse_mat = np.empty((5, 16, 17))
    task_file='Multitask'    

    load_str = '_ReLU128_4.11/swap_holdouts/'+task_file


    for seed in range(5): 
        perf_dict=pickle.load(open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_all_perf_dict', 'rb'))
        confuse_mat = pickle.load(open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_confuse_mat', 'rb'))
        decoded_set = pickle.load(open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_decoded_set', 'rb'))

        for mode in ['context', 'instructions', 'others']:
            all_perf_dict[mode][seed,...] = perf_dict[mode]        
        all_confuse_mat[seed, ...] = confuse_mat
        all_decoded_sets.append(decoded_set)

    return all_perf_dict, all_confuse_mat, all_decoded_sets


all_perf_dict, all_confuse_mat, all_decoded_sets = load_decoder_outcomes()                                

np.mean(all_perf_dict['context'][0])


all_other_dict = defaultdict(list)
for task in Task.TASK_LIST: 
    for i in range(5): 
        for item in all_decoded_sets[i][task]['other']:
            all_other_dict[task].append(item)

from collections import Counter, OrderedDict

total_other = 0
unique_other = 0

len(set(all_other_dict['Go']))

for task in Task.TASK_LIST: 
    tot=len(all_other_dict[task])
    unique = len(set(all_other_dict[task]))
    print(task + ' ' + str(tot)+ ' ' +str(unique))
    total_other += len(all_other_dict[task])
    unique_other += len(set(all_other_dict[task]))

(125*5*16)

total_other/(125*5*16)
unique_other/16

sm_model = config_model('sbertNet_tuned')
rnn_decoder = DecoderRNN(64, drop_p=0.1)
encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


15*16

get_n_params(rnn_decoder)

count_vocab()



encoder_decoder.plot_confuse_mat(0, 0, confusion_mat=np.round(np.sum(all_confuse_mat, axis=0)/(5*128), 2), fmt='.0%')

np.mean(all_perf_dict['context'])

np.mean(all_perf_dict['context'][0,1, ...])

to_plot_dict = {}
for mode in ['instructions', 'context']:
    to_plot_dict[mode]=np.mean(all_perf_dict[mode], axis=(1, -1)).T


encoder_decoder.plot_partner_performance(to_plot_dict)




task_file='Multitask'
seed=0

sm_model = config_model('sbertNet_tuned')
sm_model.set_seed(seed)

rnn_decoder = DecoderRNN(64, drop_p=0.1)

sm_model.to(device)
rnn_decoder.to(device)
sm_model.eval()
#rnn_decoder.eval()

load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
sm_model.load_model(load_str)

rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_lin_wHoldout')
encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)

encoder_decoder.init_context_set(task_file, seed)




from multitasking_models.sensorimotor_models import InstructNet
from multitasking_models.language_models import SBERT
num_repeats=3
all_perf_dict = {}
all_perf_dict['instructions'] = np.empty((4, 16, num_repeats))
all_perf_dict['context'] = np.empty((4, 16, num_repeats))
all_perf_dict['others'] = np.empty((4, 16, num_repeats))

model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model1.model_name += '_tuned'
partner_seeds = [3]
for i, partner_seed in enumerate(partner_seeds): 
    model1.set_seed(partner_seed)
    model1.to(device)
    model1.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

    perf, _ = encoder_decoder.test_partner_model(model1, num_repeats=num_repeats)
    for mode in ['context', 'instructions', 'others']:
        all_perf_dict[mode][i,...] = perf[mode]





np.mean(perf['context'])











def get_all_holdout_partners():
    num_repeats = 10
    all_perf_dict = {}
    all_perf_dict['instructions'] = np.empty((5, 16, 5, num_repeats))
    all_perf_dict['context'] = np.empty((5, 16, 5, num_repeats))
    all_perf_dict['others'] = np.empty((5, 16, 5, num_repeats))

    for seed in range(5):
        sm_model.set_seed(seed)
        sm_model.load_model(load_str)
        rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_lin_wHoldout')
        encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)
        encoder_decoder.init_context_set(task_file, seed)
        for i in range(5): 
            model1.set_seed(i) 
            for tasks in training_lists_dict['swap_holdouts']: 
                holdout_file = get_holdout_file(tasks)
                model1.load_model('_ReLU128_4.11/swap_holdouts/'+holdout_file)
                perf, _ = encoder_decoder.test_partner_model(model1, num_repeats=num_repeats, tasks=tasks)
                for mode in ['context', 'instructions', 'others']:
                    all_perf_dict[mode][seed, [Task.TASK_LIST.index(tasks[0]), Task.TASK_LIST.index(tasks[1])], i, :] = perf[mode]

    return all_perf_dict

#pickle.dump(all_perf_dict, open(load_str+'/sbertNet_tuned/decoders/+all_holdout_perf', 'wb'))


all_perf_dict = pickle.load(open(load_str+'/sbertNet_tuned/decoders/+all_holdout_perf', 'rb'))


from plotting import plot_trained_performance
np.mean(all_perf_dict['instructions'], axis=(0, -1))


to_plot_dict = {}
for mode in ['instructions', 'context']:
    to_plot_dict[mode]=np.mean(all_perf_dict[mode], axis=(2, 3)).T


encoder_decoder.plot_partner_performance(to_plot_dict)