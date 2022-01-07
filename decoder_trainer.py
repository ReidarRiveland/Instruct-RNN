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
from decoder_models import DecoderRNN


from task import construct_batch

from transformers import GPT2Model, GPT2Tokenizer


import matplotlib.pyplot as plt

from numpy.lib import utils
import torch
import torch.nn as nn
import pickle

device = torch.device(0)



def train_decoder_(decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None, task_loss_ratio=0.1): 
    criterion = nn.NLLLoss(reduction='mean')
    teacher_forcing_ratio = init_teacher_forcing_ratio
    pad_len  = decoder.tokenizer.pad_len 
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
            target_tensor = decoder.tokenizer(target_instruct).to(device)

            init_hidden = torch.Tensor(rep).to(device)
            decoder_input = torch.tensor([[decoder.tokenizer.sos_token]*batch_size]).to(device)

            opt.zero_grad()

            if use_teacher_forcing:
                decoded_sentence = []
                # Teacher forcing: Feed the target as the next input
                for di in range(pad_len):
                    decoder_output, decoder_hidden = decoder._base_forward(decoder_input, init_hidden)
                    topv, topi = decoder_output.topk(1)
                    #get words for last sentence in the batch
                    last_word_index = topi.squeeze().detach()[-1].item()
                    last_word = decoder.tokenizer.index2word[last_word_index]
                    decoded_sentence.append(last_word)

                    decoder_loss += criterion(decoder_output, target_tensor[:, di])
                    decoder_input = torch.cat((decoder_input, target_tensor[:, di].unsqueeze(0)), dim=0)

                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_output, decoder_hidden, decoded_indices = decoder(init_hidden)
                

                for k in range(pad_len):
                    decoder_loss += criterion(decoder_output, target_tensor[:, k])

                
                loss=(decoder_loss+task_loss_ratio*task_loss)/pad_len
                decoded_sentence = decoder.tokenizer.untokenize_sentence(decoded_indices)[-1]  # detach from history as input        
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

        decoder= DecoderRNN(20, 128, 128)
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



# foldername = '_ReLU128_4.11/swap_holdouts/Multitask'

# model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model1.model_name += '_tuned'
# model1.set_seed(0) 
# model1.to(device)

# decoder= TranDecoderRNN(128, model1.langModel)
# decoder.init_context_set('Multitask', 'sbertNet_tuned', 'seed'+str(0), supervised_str='_supervised')

# decoder.load_model('Multitask/sbertNet_tuned/decoders/seed0tran_decoder')
# decoder.to(device)
# decoded_set = decoder.get_decoded_set()
# decoded_set['Go']
# model1.load_model('_ReLU128_4.11/swap_holdouts/Multitask')

# decoder.to(device)
# all_perf, decoded_instructs = test_partner_model(model1, decoder, num_repeats=5)

# plot_partner_performance({'instructions': all_perf[:, 1, :], 'contexts': all_perf[:, 0, :]})

