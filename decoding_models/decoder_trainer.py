import itertools
from model_trainer import config_model
import numpy as np
import torch.optim as optim
from utils.utils import task_swaps_map, training_lists_dict
from task import Task
from dataset import TaskDataSet
from decoding_models.decoder_models import DecoderRNN, gptDecoder
import pickle

import torch
import torch.nn as nn

torch.cuda.is_available()

device = torch.device(0)

def train_rnn_decoder(sm_model, decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=[]): 
    data_streamer = TaskDataSet(batch_len = 64, num_batches = 1000, holdouts=holdout_tasks)
    #weights = decoder.tokenizer.get_smoothed_freq().to(device)
    weights = None
    criterion = nn.NLLLoss(weight = weights, reduction='mean')
    teacher_forcing_ratio = init_teacher_forcing_ratio
    pad_len  = decoder.tokenizer.pad_len 
    loss_list = []
    teacher_loss_list = []
    task_indices = list(range(16))
    batch_size=64

    if holdout_tasks is not None and 'Multitask' not in holdout_tasks: 
        for holdout_task in holdout_tasks:
            holdout_index = Task.TASK_LIST.index(holdout_task)
            task_indices.remove(holdout_index)

    for i in range(epochs): 
        print('Epoch: ' + str(i)+'\n')
        
        for j, data in enumerate(data_streamer.stream_batch()): 
            ins, _, _, _, task_type = data

            use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

            decoder_loss=0

            target_instruct = sm_model.get_task_info(batch_size, task_type)

            target_tensor = decoder.tokenizer(target_instruct).to(device)
            _, sm_hidden = sm_model.forward(ins.to(device), target_instruct)

            opt.zero_grad()

            if use_teacher_forcing:
                
                decoder_input = torch.tensor([[decoder.tokenizer.sos_token_id]*batch_size]).to(device)
                if decoder.langModel is not None: 
                    input_token_tensor = decoder.tokenizer(target_instruct, use_langModel=True).to(device)
                else: 
                    input_token_tensor  =  target_tensor
                    
                decoded_sentence = []
                # Teacher forcing: Feed the target as the next input
                for di in range(pad_len):
                    logits, _ = decoder._base_forward(decoder_input, sm_hidden)
                    decoder_output = decoder.softmax(logits)
                    next_index = decoder.draw_next(logits)
                    #get words for last sentence in the batch
                    last_word_index = next_index.detach()[-1].item()
                    last_word = decoder.tokenizer.index2word[last_word_index]
                    decoded_sentence.append(last_word)

                    decoder_loss += criterion(decoder_output, target_tensor[:, di])
                    decoder_input = torch.cat((decoder_input, input_token_tensor[:, di].unsqueeze(0)), dim=0)

                loss=(decoder_loss)/pad_len
                teacher_loss_list.append(loss.item()/pad_len)
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_output, decoder_hidden, decoded_indices = decoder(sm_hidden)
                

                for k in range(pad_len):
                    decoder_loss += criterion(decoder_output, target_tensor[:, k])

                
                loss=(decoder_loss)/pad_len
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
                        eos_index = decoded_sentence.index('[EOS]')
                    except ValueError: 
                        eos_index = -1
                    decoded_sentence = ' '.join(decoded_sentence[:eos_index])
                
                print('decoded instruction: ' + decoded_sentence + '\n')
                
        sch.step()
        teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs
        print('Teacher Force Ratio: ' + str(teacher_forcing_ratio))

    return loss_list, teacher_loss_list


def train_decoder_set(config, decoder_type='rnn'): 
# decoder_type = 'rnn'
# #config=(0, 'sbertNet_tuned', training_lists_dict['swap_holdouts'][0])
# config=(0, 'sbertNet_tuned', ['Multitask'])
# #config=(0, 'sbertNet_tuned', ['Go', 'Anti DM'])

    model_file = '_ReLU128_4.11/swap_holdouts/'
    seed, model_name, tasks = config 
    for holdout_train in [True]:
        # if tasks == ['Multitask'] and holdout_train == True:
        #     continue

        if holdout_train: 
            holdout_str = '_wHoldout'
            holdouts=tasks
        else: 
            holdout_str = ''
            holdouts = []


        print(seed, tasks, holdout_str, holdouts)

        task_file = task_swaps_map[tasks[0]]
        filename = model_file + task_file+'/'+ model_name 

        sm_model = config_model(model_name)
        sm_model.set_seed(seed) 
        sm_model.load_model(model_file + task_file)
        sm_model.to(device)

        decoder= DecoderRNN(64, drop_p = 0.1, langModel=None)
        decoder_optimizer = optim.Adam([
            #{'params' : decoder.embedding.parameters()},
            {'params' : decoder.gru.parameters()},
            {'params' : decoder.out.parameters()},
            {'params' : decoder.embedding.parameters()},
            {'params' : decoder.sm_decoder.parameters(), 'lr': 1e-4}
        ], lr=1e-4, weight_decay=0.0)



        sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.99, verbose=False)
        for n, p in decoder.named_parameters(): 
            if p.requires_grad: print(n)


        trainer(sm_model, decoder, decoder_optimizer, sch, 80, 0.5, holdout_tasks=holdouts)
        decoder.save_model(filename+'/decoders/seed'+str(seed)+'_'+decoder_type+'_decoder_lin'+holdout_str)
        decoder.save_model_data(filename+'/decoders/seed'+str(seed)+'_'+decoder_type+'_decoder_lin'+holdout_str)



seeds = [1, 2,3,4]
model_file = '_ReLU128_4.11/swap_holdouts/'
to_train = list(itertools.product(seeds, ['sbertNet_tuned'], [['Multitask']]))
for config in to_train: 
    train_decoder_set(config, decoder_type='rnn')


