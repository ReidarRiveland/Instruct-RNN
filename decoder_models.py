from torch.nn.modules.pooling import MaxPool1d
from transformers.utils.dummy_pt_objects import Conv1D
from model_trainer import config_model
from utils import train_instruct_dict
from model_analysis import reduce_rep
from plotting import plot_rep_scatter
from utils import sort_vocab, isCorrect, inv_train_instruct_dict
from task import Task, construct_batch


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
    def __init__(self):
        super(SMDecoder, self).__init__()
        self.conv1 = nn.Conv1d(128, 1, 5, 1, padding=2)
        self.fc = nn.Linear(120, 128)
    
    def forward(self, sm_hidden): 
        batch_len = sm_hidden.shape[0]
        conv_out = torch.relu(self.conv1(sm_hidden))
        fc_out = torch.relu(self.fc(conv_out.view(batch_len, -1)))
        return fc_out

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
    def __init__(self, hidden_size, conv_out_channels=32, kernel_size=10, stride=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = 20
        self.tokenizer = RNNtokenizer()

        self.embedding = nn.Embedding(self.tokenizer.n_words, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.tokenizer.n_words)
        self.sm_decoder = SMDecoder()
        # self.conv_out_channels=conv_out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.conv = nn.Conv1d(in_channels=128, out_channels = self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride) 
        # self.context_encoder = nn.Linear(self.conv_out_channels, self.hidden_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def sm_encode(self, sm_hidden): 
        conv_out = self.conv(sm_hidden.transpose(1, 2))
        conv_out = torch.relu(conv_out)
        max_pooled = torch.max(conv_out, dim=-1).values
        encoded = torch.relu(self.context_encoder(max_pooled)).unsqueeze(0)
        return encoded

    def _base_forward(self, ins, sm_hidden):
        embedded = self.embedding(ins)
        init_hidden = self.sm_encode(sm_hidden)
        rnn_out, _ = self.gru(embedded, init_hidden)
        output = self.softmax(self.out(rnn_out[-1, ...]))
        return output, rnn_out

    def forward(self, sm_hidden):
        sos_input = torch.tensor([[self.tokenizer.sos_token]*sm_hidden.shape[0]]).to(sm_hidden.get_device())
        decoder_input = sos_input
        for di in range(self.tokenizer.pad_len):
            decoder_output, decoder_hidden = self._base_forward(decoder_input, sm_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.cat((decoder_input, topi.T))
        decoded_indices = decoder_input.squeeze().detach().cpu().numpy()
        return decoder_output, decoder_hidden, decoded_indices

    def decode_sentence(self, sm_hidden): 
        _, _, decoded_indices = self.forward(sm_hidden)
        decoded_sentences = self.tokenizer.untokenize_sentence(decoded_indices[1:,...])  # detach from history as input
        return decoded_sentences

class gptDecoder(BaseDecoder): 
    def __init__(self, conv_out_channels=64, kernel_size=10, stride=3):
        super().__init__()
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.gelu = nn.GELU()
        self.init_instructions(self.add_punctuation())
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.context_encoder = nn.Linear(self.conv_out_channels, 768)
        self.conv = nn.Conv1d(in_channels=128, out_channels = self.conv_out_channels, kernel_size=self.kernel_size, stride=self.stride) 
        self.psoftmax = PenalizedSoftmax()

        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', use_cache=True)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.tokenizer.model_max_length=30
        self.softmax = torch.nn.LogSoftmax(dim=-1)

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

    def sm_encode(self, sm_hidden): 
        conv_out = self.conv(sm_hidden.transpose(1, 2))
        max_pooled = torch.max(conv_out, dim=-1).values
        encoded = self.gelu(self.context_encoder(max_pooled)).unsqueeze(1)
        return encoded

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
        decoded_sentences = self.tokenizer.batch_decode(decoded_indices[1:,...].int(), skip_special_tokens=True)  # detach from history as input
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

    def decode_set(self, num_trials, from_contexts=False, tasks=Task.TASK_LIST): 
        decoded_set = {}
        confusion_mat = np.zeros((16, 17))

        for i, task in enumerate(tasks): 
            tasks_decoded = defaultdict(list)

            ins, _, _, _, _ = construct_batch(task, num_trials)

            if from_contexts: 
                task_index = Task.TASK_LIST.index(task)
                task_info = torch.Tensor(self.contexts[task_index, :num_trials, :]).to(self.sm_model.__device__)
                _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), context=task_info)
            else: 
                task_info = self.sm_model.get_task_info(num_trials, task)
                _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), task_info)
            
            decoded_sentences = self.decoder.decode_sentence(sm_hidden) 

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
                        ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)

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



task_file='Go_Anti_DM'
seed=0
sm_model = config_model('sbertNet_tuned')
sm_model.set_seed(seed)

sm_decoder = SMDecoder()


rnn_decoder=DecoderRNN(128, conv_out_channels=32, kernel_size=10)
sm_model.to(device)
rnn_decoder.to(device)

load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
sm_model.load_model(load_str)
rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_conv32_wHoldout')
encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)
encoder_decoder.init_context_set(task_file, seed)

encoder_decoder.decoder.psoftmax=PenalizedSoftmax(theta=1.2)


tokenized = gpt_decoder.tokenizer('Hello, my dog is not in', return_tensors='pt')
gpt = gpt_decoder.gpt
gpt.to('cpu')
out_ids = gpt.generate(**tokenized)
gpt_decoder.tokenizer.batch_decode(out_ids)


encoder_decoder.plot_confuse_mat(128, from_contexts=False)
decoded, confuse_mat = encoder_decoder.decode_set(128, from_contexts=True)

decoded['Anti DM']



# from collections import Counter


# decoded['DM']['other']


# from rnn_models import InstructNet
# from nlp_models import SBERT


# model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model1.model_name += '_tuned'
# model1.set_seed(1) 
# model1.load_model('_ReLU128_4.11/swap_holdouts/Multitask')
# model1.to(device)

# perf, _ = encoder_decoder.test_partner_model(model1)

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







# model2 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model2.model_name += '_tuned'
# model2.set_seed(1) 
# model2.load_model('_ReLU128_4.11/swap_holdouts/Multitask')

# perf, decoded = encoder.test_partner_model(model2, num_repeats=1)
# perf
# # np.mean(perf, axis=0)

# # from plotting import MODEL_STYLE_DICT, mpatches, Line2D

# # perf

# # perf
# # encoder.plot_partner_performance(perf)


import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(tens, index=0): 
    sns.heatmap(tens[index, ...].T.detach().numpy())
    plt.show()

conv1 = nn.Conv1d(128, 1, 5, 1, padding=2)


conv2 = nn.Conv1d(64, 32, 5, 1, padding=2)
conv3 = nn.Conv1d(32, 12, 11, 3, padding=5)


conv1 = nn.Conv1d(128, 1, 5, 1, padding=2)
model = nn.Sequential(conv1, nn.ReLU())


task = 'DM'
ins, _, _, _, _ = construct_batch(task, 64)
instructions = sm_model.get_task_info(64, task)

_, sm_hidden = sm_model(torch.Tensor(ins), instructions)

out = model(sm_hidden.transpose(1,2))

out.shape
torch.max(out, dim=1).values

sns.heatmap(sm_hidden[0, ...].detach().numpy())
plt.show()
plot_heatmap(out, index=5)