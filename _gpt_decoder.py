import transformers
from utils import train_instruct_dict
from model_analysis import reduce_rep
from plotting import plot_rep_scatter
import numpy as np
from utils import sort_vocab, isCorrect, inv_train_instruct_dict
from task import Task
import seaborn as sns
from collections import defaultdict
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from torch.distributions import Categorical
from decoder_models import BaseDecoder

from task import construct_batch

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pickle

device = torch.device(0)

class PenalizedSoftmax():
    def __init__(self, temp=1, theta=1.6):
        self.theta=theta
        self.temp=temp

    def __call__(self, logits, last_decoded_indices): 
        batch_size = logits.shape[0]
        k = logits.shape[1]
        penalized_temps = torch.where(logits == last_decoded_indices.repeat(k, 1).T, self.temp*self.theta*torch.ones(batch_size, k, device=device), self.temp*torch.ones(batch_size, k, device=device))
        exps = logits * penalized_temps
        exps -= torch.max(exps, -1).values.repeat(k, 1).T
        normalizer = torch.sum(torch.exp(exps), dim=1)
        p = torch.exp((exps))/normalizer.repeat(k, 1).T
        return p 

# decoded_indices = torch.randint(50257, (1, 64)).squeeze()

# logits = torch.randn((64, 1, 50257))

# top = logits.topk(5)

# theta=0.8
# temp = 1

# decoded_indices
# decoded_indices.repeat(5, 1).T

# top_indices = top.indices.squeeze()
# penalized_temps = torch.where(top_indices == decoded_indices.repeat(5, 1).T, temp*theta*torch.ones(64, 5), temp*torch.ones(64, 5))

# top_values = top.values.squeeze()

# top_values.shape

# normalizer = torch.sum(torch.exp(top_values * penalized_temps), dim=1)

# torch.exp(top_values)/286.1

# torch.exp((top_values*penalized_temps))/normalizer.repeat(5, 1).T


# psoftmax = PenalizedSoftmax()

# top.values.squeeze().shape
# decoded_indices.shape
# probs = psoftmax(top.values.squeeze(), torch.Tensor([]))

# torch.Tensor([]).shape[0] == 0

# dist = Categorical(probs)
# sam = dist.sample()
# top_indices.squeeze().shape
# sam.shape
# top_indices.squeeze()[0, sam[0]]

# top.indices.squeeze().gather(1, sam.unsqueeze(-1))

# torch.Tensor(logits.shape[0]*[-torch.inf]).shape

# def draw_next(last_logits, decoded_indices, k_sample=5):
#     if decoded_indices.shape[0] != 0:
#         last_decoded_indices = decoded_indices[-1, :]
#     else: 
#         last_decoded_indices = torch.Tensor(last_logits.shape[0]*[-torch.inf])
#     top_k = last_logits.topk(k_sample)
#     probs = psoftmax(top_k.values(), last_decoded_indices)
#     dist = Categorical(probs)
#     next_indices = top_k.indices.gather(1, dist.sample().unsqueeze(-1))
#     return next_indices




class gptDecoder(BaseDecoder): 
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.init_instructions(self.add_punctuation())
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.context_encoder = nn.Linear(128, 768)
        self.conv = nn.Conv1d(in_channels=128, out_channels = 128, kernel_size=8, stride=3) 
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
        probs = self.psoftmax(top_k.values, last_decoded_indices)
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
        decoded_sentences = self.tokenizer.batch_decode(decoded_indices[1:,...], skip_special_tokens=True)  # detach from history as input
        return decoded_sentences


def train_decoder(sm_model, decoder, data_streamer, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None): 
    criterion = nn.NLLLoss(reduction='none')
    teacher_forcing_ratio = init_teacher_forcing_ratio
    loss_list = []
    teacher_loss_list = []
    batch_size=64

    for i in range(epochs): 
        print('Epoch: ' + str(i)+'\n')
        
        for j, data in enumerate(data_streamer.stream_batch()): 
            use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
            
            ins, tar, _, tar_dir, task_type = data

            target_instruct = sm_model.get_task_info(batch_size, task_type)

            pad_len = max([len(instruct.split(' ')) for instruct in target_instruct])+3
            # if type(rnn_decoder) is DecoderRNN: 
            # else: token_kargs = 
            tokenized_targets = decoder.tokenizer(target_instruct, padding= 'max_length', return_tensors='pt')
            target_ids = tokenized_targets.input_ids

            _, sm_hidden = sm_model.forward(target_instruct, ins.to(device))

            opt.zero_grad()

            if use_teacher_forcing:
                decoded_indices = torch.Tensor([]).to(device)
                decoder_loss = 0
                past_keys = None
                # Teacher forcing: Feed the target as the next input
                for di in range(pad_len-1):
                    mask = tokenized_targets.attention_mask[:, :di+1]
                    outputs = decoder._base_forward(sm_hidden.to(device), input_ids=target_ids[:, di].unsqueeze(1).to(device), past_keys=past_keys)
                    #get words for last sentence in the batch
                    logits = outputs.logits
                    past_keys = outputs.past_key_values
                    scores = decoder.softmax(logits)
                    last_logits = logits[:, -1, :]
                    input_ids = decoder.draw_next(last_logits, decoded_indices)
                    decoded_indices = torch.cat((decoded_indices, input_ids), dim=1)

                    decoder_loss += criterion(scores[:, -1, :], target_ids[:, di].to(device))

                loss=torch.mean(decoder_loss)/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                scores, decoded_indices = decoder(sm_hidden.to(device))
                seq_loss = criterion(scores.transpose(1, 2), target_ids.to(device))
                loss = torch.mean(seq_loss)/pad_len
                decoder.loss_list.append(loss.item()/pad_len)

            loss.backward()
            opt.step()

            if j%50==0: 
                decoded_sentence = decoder.tokenizer.batch_decode(decoded_indices.int())[-1]
                print('Decoder Loss: ' + str(loss.item()/pad_len))
                #print('Task Loss: ' + str(task_loss.item()/pad_len))
                print('Teacher Forcing:' + str(use_teacher_forcing))

                print('target instruction: ' + target_instruct[-1])
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
gpt_decoder = gptDecoder()
