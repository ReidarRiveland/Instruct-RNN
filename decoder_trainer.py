import itertools
from model_trainer import config_model
import numpy as np
import torch.optim as optim
from utils import task_swaps_map, training_lists_dict
from task import Task
from data import TaskDataSet
from decoder_models import DecoderRNN, gptDecoder
import pickle

import torch
import torch.nn as nn

device = torch.device(0)

def train_rnn_decoder(sm_model, decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=[]): 
    data_streamer = TaskDataSet(batch_len = 64, num_batches = 1000, holdouts=holdout_tasks)
    criterion = nn.NLLLoss(reduction='mean')
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

            decoder_input = torch.tensor([[decoder.tokenizer.sos_token]*batch_size]).to(device)

            _, sm_hidden = sm_model.forward(ins.to(device), target_instruct)

            opt.zero_grad()

            if use_teacher_forcing:
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
                    decoder_input = torch.cat((decoder_input, target_tensor[:, di].unsqueeze(0)), dim=0)

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
                        eos_index = decoded_sentence.index('EOS')
                    except ValueError: 
                        eos_index = -1
                    decoded_sentence = ' '.join(decoded_sentence[:eos_index])
                
                print('decoded instruction: ' + decoded_sentence + '\n')
                
        sch.step()
        teacher_forcing_ratio -= init_teacher_forcing_ratio/epochs

    return loss_list, teacher_loss_list



def train_gpt_decoder(sm_model, decoder, opt, sch, epochs, init_teacher_forcing_ratio, holdout_tasks=None): 
    data_streamer = TaskDataSet(batch_len = 64, num_batches = 800, holdouts=holdout_tasks)

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

            _, sm_hidden = sm_model.forward(ins.to(device), target_instruct)

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



# from rnn_models import InstructNet
# from nlp_models import SBERT
# from data import TaskDataSet
# from decoder_models import DecoderRNN
# from model_trainer import config_model
# from utils import training_lists_dict

# model1 = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model1.model_name += '_tuned'
# model1.set_seed(0) 
# model1.load_model('_ReLU128_4.11/swap_holdouts/Multitask')
# model1.to(device)


# decoder_rnn = DecoderRNN(128, conv_out_channels=64).to(device)

# decoder_optimizer = optim.Adam(decoder_rnn.parameters(), lr=1e-4, weight_decay=0.0)
# sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.99, verbose=False)

# train_decoder(model1, decoder_rnn, decoder_optimizer, sch, 100, 0.8, holdout_tasks=['Anti DM'])


#def train_decoder_set(config, decoder_type='rnn'): 

# seeds = [1, 2, 3, 4]
# model_file = '_ReLU128_4.11/swap_holdouts/'
# to_train = list(itertools.product(seeds, ['sbertNet_tuned'], [['Multitask']]))
# config = to_train[0]
# decoder_type = 'rnn'

training_lists_dict['swap_holdouts']
#def train_decoder_set(config, decoder_type='rnn'): 
decoder_type = 'rnn'
#config=(0, 'sbertNet_tuned', training_lists_dict['swap_holdouts'][0])
config=(0, 'sbertNet_tuned', ['Go', 'Anti DM'])
#config=(0, 'sbertNet_tuned', ['Multitask'])

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
    # try: 
    #     pickle.load(open(filename+'/decoders/seed'+str(seed)+'_rnn_decoder'+holdout_str+'_loss_list', 'rb'))
    #     print(filename+'/decoders/seed'+str(seed)+'_rnn_decoder'+holdout_str+'   already trained')
    # except FileNotFoundError:

    sm_model = config_model(model_name)
    sm_model.set_seed(seed) 
    sm_model.load_model(model_file + task_file)
    sm_model.to(device)

    if decoder_type == 'rnn': 
        decoder= DecoderRNN(64)
        trainer= train_rnn_decoder
        decoder_optimizer = optim.Adam([
            {'params' : decoder.embedding.parameters()},
            {'params' : decoder.gru.parameters()},
            {'params' : decoder.out.parameters()},
            {'params' : decoder.sm_decoder.parameters(), 'lr': 1e-4}
        ], lr=1e-4, weight_decay=0.00001)
    else: 
        decoder= gptDecoder()
        trainer= train_gpt_decoder
        decoder_optimizer = optim.Adam([
            {'params' : decoder.gpt.parameters()},
            {'params' : decoder.sm_decoder.parameters(), 'lr': 1e-4}
        ], lr=1e-6, weight_decay=0.0)
    decoder.train()
    decoder.to(device)
    #decoder_rnn.load_model(filename+'/decoders/seed'+str(seed)+'_rnn_decoder'+holdout_str)



    sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.97, verbose=False)
    for n, p in decoder.named_parameters(): 
        if p.requires_grad: print(n)


    trainer(sm_model, decoder, decoder_optimizer, sch, 80, 0.5, holdout_tasks=holdouts)
    decoder.save_model(filename+'/decoders/seed'+str(seed)+'_'+decoder_type+'_decoder_lin'+holdout_str)
    decoder.save_model_data(filename+'/decoders/seed'+str(seed)+'_'+decoder_type+'_decoder_lin'+holdout_str)



seeds = [0]
model_file = '_ReLU128_4.11/swap_holdouts/'
to_train = list(itertools.product(seeds, ['sbertNet_tuned'], [['Multitask']]))
for config in to_train: 
    train_decoder_set(config, decoder_type='gpt')


