import transformers
from utils import train_instruct_dict
from model_analysis import reduce_rep
from plotting import plot_rep_scatter
import numpy as np
from utils import sort_vocab, isCorrect, inv_train_instruct_dict
from task import Task
import seaborn as sns
from collections import defaultdict
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from task import construct_batch

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pickle

device = torch.device(0)

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
        return torch.Tensor(tokens).unsqueeze(0)

    def tokenize_sentence(self, sent_list, pad_len): 
        tokenized_tensor = torch.Tensor([])
        for sent in sent_list:
            tokens = self._tokenize_sentence(sent, pad_len)
            torch.cat((tokenized_tensor, tokens), dim=0)
        return tokenized_tensor

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
        self.load_foldername = '_ReLU128_4.11/swap_holdouts'
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

    def init_context_set(self, task_file, model_name, seed_str, supervised_str=''):
        all_contexts = np.empty((16, 128, self.context_dim))
        for i, task in enumerate(Task.TASK_LIST):
            filename = self.load_foldername+'/'+task_file+'/'+model_name+'/contexts/'+seed_str+task+supervised_str+'_context_vecs20'
            task_contexts = pickle.load(open(filename, 'rb'))
            all_contexts[i, ...]=task_contexts[:128, :]
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
        return list(instruct), torch.Tensor(rep)

    def plot_context_embeddings(self, tasks_to_plot=Task.TASK_LIST): 
        reps_reduced, _ = reduce_rep(self.contexts)
        plot_rep_scatter(reps_reduced, tasks_to_plot)
        return reps_reduced

class DecoderRNN(BaseDecoder):
    def __init__(self, context_dim, embedding_size, hidden_size):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.context_dim = 20
        self.tokenizer = RNNtokenizer()

        self.embedding = nn.Embedding(self.tokenizer.n_words, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.tokenizer.n_words)
        self.context_encoder = nn.Linear(self.context_dim, self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
 
    def _base_forward(self, ins, context):
        embedded = self.embedding(ins)
        init_hidden = torch.relu(self.context_encoder(context))
        rnn_out, _ = self.gru(embedded, init_hidden.unsqueeze(0))
        output = self.softmax(self.out(rnn_out[-1, ...]))
        return output, rnn_out

    def forward(self, context):
        sos_input = torch.tensor([[self.tokenizer.sos_token]*context.shape[0]]).to(context.get_device())
        decoder_input = sos_input
        for di in range(self.tokenizer.pad_len):
            decoder_output, decoder_hidden = self._base_forward(decoder_input, context)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.cat((decoder_input, topi.T))
        decoded_indices = decoder_input.squeeze().detach().cpu().numpy()
        return decoder_output, decoder_hidden, decoded_indices

    def decode_sentence(self, context): 
        _, _, decoded_indices = self.forward(context)
        decoded_sentences = self.tokenizer.untokenize_sentence(decoded_indices[1:,...])  # detach from history as input
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

class gptDecoder(BaseDecoder): 
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.init_instructions(self.add_punctuation())
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.context_encoder = nn.Linear(self.context_dim, 768)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_cache=True)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.tokenizer.model_max_length=30
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def tokenize_instruction(self, instructions): 
        instructions = [instruct+' '+self.tokenizer.eos_token for instruct in instructions]
        tokenized = self.tokenizer(instructions, return_tensors='pt', padding=True)
        attention_mask=torch.cat((torch.ones(tokenized['attention_mask'].shape[0], 1), tokenized['attention_mask']), dim=1)
        return tokenized['input_ids'], attention_mask

    def _base_forward(self, context=None, input_ids=None, attention_mask=None, past_keys=None): 
        if past_keys is None: 
            embedded_inputs = torch.sigmoid(self.context_encoder(context)).unsqueeze(1)
        else: 
            embedded_inputs = torch.Tensor([])

        if input_ids is not None: 
            embedded_inputs = torch.cat((embedded_inputs, self.gpt.transformer.wte(input_ids)), dim=1)
        
        return self.gpt(inputs_embeds=embedded_inputs, attention_mask=attention_mask, past_key_values=past_keys)

    def forward(self, context):
        past_keys = None
        input_ids = None
        decoded_indices = torch.Tensor([])
        scores = torch.Tensor([])

        for di in range(self.tokenizer.model_max_length):
            outputs = self._base_forward(context, input_ids = input_ids, past_keys=past_keys)
            past_keys = outputs.past_key_values
            logits = outputs.logits
            cur_scores = self.softmax(logits)
            input_ids = torch.argmax(cur_scores[:, -1, :], -1).unsqueeze(1)
            decoded_indices = torch.cat((decoded_indices, input_ids), dim=1)
            scores = torch.cat((scores, cur_scores), dim=1)

        return scores, decoded_indices

    def decode_sentence(self, context): 
        _, decoded_indices = self.forward(context)
        decoded_sentences = self.tokenizer.batch_decode(decoded_indices[1:,...])  # detach from history as input
        return decoded_sentences

#add periods to the end of all setnences

# rnn_decoder = DecoderRNN(20, 128)
# rnn_decoder.init_context_set('Multitask', 'sbertNet_tuned', 'seed0')

# type(rnn_decoder) is DecoderRNN

# go_instructs = train_instruct_dict['Go']

# kargs = {'pad_len': 30}

# rnn_decoder.tokenizer(go_instructs)





def train_decoder_(decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None): 
    criterion = nn.NLLLoss(reduction='none', ignore_index=decoder.tokenizer.pad_token_id)
    teacher_forcing_ratio = init_teacher_forcing_ratio
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
            task_index = np.random.choice(task_indices)
            instruct_index = np.random.randint(0, 15, size=batch_size)
            target_instruct, rep = decoder.get_instruct_embedding_pair(task_index, instruct_index)  
            pad_len = max([len(instruct.split(' ')) for instruct in target_instruct])+3
            # if type(rnn_decoder) is DecoderRNN: 
            # else: token_kargs = 
            tokenized_targets = decoder.tokenizer(target_instruct, padding= 'max_length', return_tensors='pt')
            target_ids = tokenized_targets.input_ids

            opt.zero_grad()

            if use_teacher_forcing:
                decoded_indices = torch.Tensor([])
                decoder_loss = 0
                past_keys = None
                # Teacher forcing: Feed the target as the next input
                for di in range(pad_len-1):
                    mask = tokenized_targets.attention_mask[:, :di+1]
                    outputs = decoder._base_forward(rep, input_ids=target_ids[:, di].unsqueeze(1), past_keys=past_keys)
                    #get words for last sentence in the batch
                    logits = outputs.logits
                    past_keys = outputs.past_key_values
                    scores = decoder.softmax(logits)
                    input_ids = torch.argmax(scores[:, -1, :], -1).unsqueeze(1)
                    decoded_indices = torch.cat((decoded_indices, input_ids), dim=1)

                    decoder_loss += criterion(scores[:, -1, :], target_ids[:, di])

                loss=decoder_loss/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                scores, decoded_indices = decoder(rep)
                scores = scores.squeeze().transpose(1, 2)
                #softmax_scores = decoder.softmax(outputs.logits)
                seq_loss = criterion(scores, target_ids)
                loss = torch.mean(seq_loss)
                #loss = torch.Tensor([criterion(softmax_scores[:, i, :], target_ids[:, i]) for i in range(softmax_scores.shape[1])], requires_grad=True)
                decoder.loss_list.append(loss.item()/pad_len)

            loss.backward()
            opt.step()

            if j%50==0: 
                decoded_sentence = decoder.tokenizer.batch_decode(decoded_indices)[-1]
                print('Decoder Loss: ' + str(loss.item()/pad_len))
                #print('Task Loss: ' + str(task_loss.item()/pad_len))

                print('target instruction: ' + target_instruct[-1])
                if use_teacher_forcing:
                    try:
                        eos_index = decoded_sentence.index(decoder.tokenizer.eos_token)
                    except ValueError: 
                        eos_index = -1
                    decoded_sentence = decoded_sentence[:eos_index]
                
                print('decoded instruction: ' + decoded_sentence + '\n')
                
        sch.step()
        teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs

    return loss_list, teacher_loss_list

import torch.optim as optim
gpt_decoder = gptDecoder(20)
gpt_decoder.init_context_set('Multitask', 'sbertNet_tuned', 'seed0')
decoder_optimizer = optim.Adam(gpt_decoder.parameters(), lr=1e-6, weight_decay=0.0)
sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.95, verbose=False)

gpt_decoder.tokenizer.pad_token_id

train_decoder_(gpt_decoder, decoder_optimizer, sch, 50, 0.0, holdout_tasks=['Anti DM'])

Task.TASK_LIST.index('Anti DM')

anti_dm_tensor=torch.Tensor(gpt_decoder.contexts[5])
anti_dm_out = gpt_decoder.forward(anti_dm_tensor)
anti_dm_out[1]
gpt_decoder.decode_sentence(anti_dm_out[1])

gpt_decoder.tokenizer.batch_decode(anti_dm_out[1])

gpt_decoder.init_context_set('Multitask', 'sbertNet_tuned', 'seed0')
go_contexts = gpt_decoder.contexts[0, ...]

decoded_contexts = gpt_decoder(torch.Tensor(go_contexts))

gpt_decoder.tokenizer.batch_decode(decoded_contexts[0])


import torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', use_cache=True)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def tokenize(instructions):
    instructions = [instruct+' '+tokenizer.eos_token for instruct in instructions]
    return tokenizer(instructions, return_tensors='pt', padding=True)

tokenized=tokenize(instructs)

#why am I getting different outputs here? Is it the padding?

prompt = ["In the middle of the night I", "When I was a young man in"]
past_keys = None
tokenized = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = tokenized.input_ids
decoded_inputs=inputs

for i in range(20): 
    outputs = model(input_ids=inputs, past_key_values=past_keys)
    loss = outputs.loss
    logits = outputs.logits
    past_keys = outputs.past_key_values
    scores = torch.softmax(logits, dim=-1)
    inputs = torch.argmax(scores[:, -1, :], -1).unsqueeze(1)
    decoded_inputs = torch.cat((decoded_inputs, inputs), dim=1)

tokenizer.batch_decode(decoded_inputs)

torch.cat((torch.Tensor([]), decoded_inputs))


# tokenizer.batch_decode(torch.max(scores, 2).indices[:, -1])


# scores = torch.softmax(logits, dim=-1)[-1]
# inputs = torch.argmax(scores).view(1, -1)
# inputs.shape


# .item()



# decoded_sentence

tokenizer.batch_decode(tokenized['input_ids'], skip_special_tokens=True)







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






