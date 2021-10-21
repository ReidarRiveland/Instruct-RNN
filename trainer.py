from matplotlib.pyplot import axis, stem
import numpy as np
from numpy.random import randn

import torch
import torch.nn as nn
import torch.optim as optim

from task import Task
from data import TaskDataSet
from utils import isCorrect, train_instruct_dict, training_lists_dict
from model_analysis import task_eval, get_instruct_reps

from rnn_models import SimpleNet, InstructNet
from nlp_models import BERT, SBERT, GPT, BoW
import torch.nn as nn

import itertools
import pickle
import sys

device = torch.device(0)

device

torch.cuda.is_available()
torch.cuda.get_device_name(device)



def masked_MSE_Loss(nn_out, nn_target, mask):
    """MSE loss (averaged over features then time) function with special weighting mask that prioritizes loss in response epoch 
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_num, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_num, seq_len, features)
        mask (Tensor): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: weighted loss of neural network response; shape: (1x1)
    """

    mask_applied = torch.mul(torch.pow((nn_out - nn_target), 2), mask)
    avg_applied = torch.mean(torch.mean(mask_applied, 2), 1)
    return torch.mean(avg_applied)


def init_optimizer(model, lr, milestones, weight_decay=0.0, langLR=None):
    try:
        if langLR is None: langLR = lr 
        optimizer = optim.Adam([
                {'params' : model.recurrent_units.parameters()},
                {'params' : model.sensory_motor_outs.parameters()},
                {'params' : model.langModel.parameters(), 'lr': langLR}
            ], lr=lr, weight_decay=weight_decay)
    except AttributeError: 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    
def train_model(model, streamer, epochs, optimizer, scheduler, print_eval=False, testing=False): 
    model.to(device)
    model.train()
    if testing: 
        try:
            model.langModel.eval()
        except: 
            pass

    batch_len = streamer.batch_len 
    for i in range(epochs):
        print('epoch', i)
        streamer.shuffle_stream_order()
        for j, data in enumerate(streamer.stream_batch()): 
            
            ins, tar, mask, tar_dir, task_type = data
            
            optimizer.zero_grad()

            task_info = model.get_task_info(batch_len, task_type)
            out, _ = model(task_info, ins)

            loss = masked_MSE_Loss(out, tar, mask) 
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
            optimizer.step()

            #make this a float.16
            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
            model._loss_data_dict[task_type].append(loss.item())
            model._correct_data_dict[task_type].append(frac_correct)
            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                print('Frac Correct ' + str(frac_correct) + '\n')
            if j%100 == 0: 
                if print_eval:
                    for holdout in streamer.holdouts: 
                        frac_correct = task_eval(model, holdout, 128)
                        print(holdout + ' holdout performance: '+str(frac_correct) + '\n')


        if scheduler is not None: 
            scheduler.step()    

def test_model(model, holdouts_test, repeats=5, foldername = '_ReLU128_5.7', holdout_type = 'single_holdouts', save=False): 
    if len(holdouts_test) > 1: holdout_file = '_'.join(holdouts_test)
    else: holdout_file = holdouts_test[0]
    holdout_file = holdout_file.replace(' ', '_')
    for holdout in holdouts_test: 
        for _ in range(repeats): 
            model.load_model(foldername+'/'+holdout_type+'/'+holdout_file)
            opt, _ = init_optimizer(model, 0.0007, [])

            data = TaskDataSet(data_folder = foldername+'/training_data', batch_len=256, num_batches=100, task_ratio_dict={holdout:1})
            data.data_to_device(device)
            train_model(model, data, 1, opt, None, testing=True)
        
        correct_perf = np.mean(np.array(model._correct_data_dict[holdout]).reshape(repeats, -1), axis=0)
        loss_perf = np.mean(np.array(model._loss_data_dict[holdout]).reshape(repeats, -1), axis=0)

        if save: 
            pickle.dump(correct_perf, open(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_correct', 'wb'))
            pickle.dump(loss_perf, open(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_loss', 'wb'))
            print(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_correct')

    return correct_perf, loss_perf


def _train_context_(model, data_streamer, epochs, opt, sch, context): 
    model.freeze_weights()
    model.eval()
    for i in range(epochs): 
        print('epoch', i)
        data_streamer.shuffle_stream_order()
        for j, data in enumerate(data_streamer.stream_batch()): 

            ins, tar, mask, tar_dir, task_type = data

            opt.zero_grad()
            batch_context = context
            #batch_context = context.repeat(ins.shape[0], 1).to(device)
            task_info = model.get_task_info(ins.shape[0], task_type)
            target_out, _ = model(task_info, ins)
            proj = model.langModel.proj_out(batch_context)            
            out, _ = super(type(model), model).forward(proj, ins)
            loss = masked_MSE_Loss(out, target_out, mask) 
            loss.backward()

            opt.step()

            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
            model._loss_data_dict[task_type].append(loss.item())
            model._correct_data_dict[task_type].append(frac_correct)
            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                print('Frac Correct ' + str(frac_correct) + '\n')
        sch.step()

    return context.squeeze().detach().cpu().numpy()


def get_model_contexts(model, epochs, filename='_ReLU128_5.7', init_avg=False):
    from decoder_rnn import DecoderRNN, get_decoder_loss
    decoder_loss_ratio = 0.8
    decoder_rnn = DecoderRNN(768, 768).to(device)
    context_dim = 768
    batch_len = 128
    all_contexts = np.random.normal((16, batch_len, context_dim))
    task_file = 'Anti_Go_MultiCOMP1'
    model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
    #context_dim = model.langModel.intermediate_lang_dim
    
    all_contexts_tensor =torch.randn((16, batch_len, context_dim), requires_grad=True, device=device)
    opt= optim.Adam([{'params': decoder_rnn.parameters()},
                    {'params': all_contexts_tensor}],
                    lr=0.0005, weight_decay=0.0)
                    
    sch = optim.lr_scheduler.ExponentialLR(opt, 0.90)


    streamer = TaskDataSet(filename+'/training_data', batch_len = batch_len, num_batches = 500, holdouts=['Anti Go', 'MultiCOMP1'])
    streamer.data_to_device(device)

    for i in range(epochs): 
        print('epoch', i)
        streamer.shuffle_stream_order()
        for j, data in enumerate(streamer.stream_batch()): 
            ins, tar, mask, tar_dir, task_type = data
            task_index = Task.TASK_LIST.index(task_type)
            task_info = model.get_task_info(ins.shape[0], task_type)
            target_out, _ = model(task_info, ins)
            contexts = all_contexts_tensor[task_index, ...]
            decoder_loss = get_decoder_loss(decoder_rnn, task_info, contexts, (500-j)/500, print_progress=j%50==0)
            proj_contexts = model.langModel.proj_out(contexts)
            out, _ = super(type(model), model).forward(proj_contexts, ins)
            task_loss = masked_MSE_Loss(out, target_out, mask) 
            loss = decoder_loss*decoder_loss_ratio+task_loss*(1-decoder_loss_ratio)
            loss.backward()
            opt.step

            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)

            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(task_loss.item()))
                print('Frac Correct ' + str(frac_correct) + '\n')

            model._correct_data_dict[task_type] = np.array(model._correct_data_dict[task_type])
            model._loss_data_dict[task_type] = np.array(model._correct_data_dict[task_type])

        sch.step()
    pickle.dump(all_contexts, open(filename+'/swap_holdouts/'+task_file + model.model_name+'/'+model.__seed_num_str__+'combo_context_vecs20', 'wb'))
    pickle.dump(model._correct_data_dict, open(filename+'/swap_holdouts/'+task_file + model.model_name+'/'+model.__seed_num_str__+'combo_context_holdout_correct_data20', 'wb'))
    pickle.dump(model._loss_data_dict, open(filename+'/swap_holdouts/'+task_file + model.model_name+'/'+model.__seed_num_str__+'combo_context_holdout_loss_data20', 'wb'))


def _get_model_contexts(model, num_contexts, task_file, filename='_ReLU128_5.7'):
    all_contexts = np.empty((16, num_contexts, model.langModel.intermediate_lang_dim))
    model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
    for i, task in enumerate(Task.TASK_LIST[::-1]):     
        #contexts = np.empty((num_contexts, model.langModel.intermediate_lang_dim))
        streamer = TaskDataSet(filename+'/training_data', num_batches = 500, task_ratio_dict={task:1})
        streamer.data_to_device(device)
        context = nn.Parameter(torch.randn((num_contexts, model.langModel.intermediate_lang_dim), device=device))

        #for j in range(num_contexts): 
        opt= optim.Adam([context], lr=1e-3, weight_decay=0.0)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
        contexts =_train_context_(model, streamer, 30, opt, sch, context)
        all_contexts[i, ...] = contexts


    pickle.dump(all_contexts, open(filename+'/swap_holdouts/'+task_file + '/'+model.model_name+'/'+model.__seed_num_str__+'_context_vecs768', 'wb'))
    pickle.dump(model._correct_data_dict, open(filename+'/swap_holdouts/'+task_file+'/' + model.model_name+'/'+model.__seed_num_str__+'_context_holdout_correct_data768', 'wb'))
    pickle.dump(model._loss_data_dict, open(filename+'/swap_holdouts/'+task_file + '/'+ model.model_name+'/'+model.__seed_num_str__+'_context_holdout_loss_data768', 'wb'))



def config_model_training(key): 
    params = ALL_MODEL_PARAMS[key]

    try:
        model = params['model'](params['langModel'](**params['langModel_params']), 128, 1, torch.relu)
    except:
        model = params['model'](128, 1, torch.relu, use_ortho_rules=True)

    model.model_name = params['model_name']

    opt, sch = init_optimizer(model, **params['opt_params'])
    epochs = params['epochs']

    return model, opt, sch, epochs


ALL_MODEL_PARAMS = {
    'sbertNet_tuned': {'model': InstructNet, 
                    'langModel': SBERT,
                    'model_name': 'sbertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                    'epochs': 40
                },

    'sbertNet': {'model': InstructNet, 
                'langModel': SBERT,
                'model_name': 'sbertNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 55
                },
    
    'bertNet_tuned': {'model': InstructNet, 
                    'langModel': BERT,
                    'model_name': 'bertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25], 'langLR': 1e-4},
                    'epochs': 40
                },

    'bertNet': {'model': InstructNet, 
                'langModel': BERT,
                'model_name': 'bertNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 55
                },

    'gptNet_tuned': {'model': InstructNet, 
                    'langModel': GPT,
                    'model_name': 'gptNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25], 'langLR': 1e-5},
                    'epochs': 40
                },

    'gptNet': {'model': InstructNet, 
                'langModel': GPT,
                'model_name': 'gptNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 20, 25]}, 
                'epochs': 55
                },
    
    'bowNet_flat': {'model': InstructNet, 
                'langModel': BoW,
                'model_name': 'bowNet_flat',
                'langModel_params': {'out_dim': None}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 40
                },

                    
    'bowNet': {'model': InstructNet, 
                'langModel': BoW,
                'model_name': 'bowNet',
                'langModel_params': {'out_dim': 20, 'output_nonlinearity': nn.ReLU()}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 55
                },


    'simpleNet': {'model': SimpleNet, 
                'model_name': 'simpleNet',
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 55
                }

}


if __name__ == "__main__":
    model_file = '_ReLU128_5.7'

    #train_mode = str(sys.argv[1])
    train_mode = 'train_contexts'
    
    print('Mode: ' + train_mode + '\n')

    if train_mode == 'train_contexts': 
        seeds = [0, 1]
        to_train = list(itertools.product(seeds, ['sbertNet_tuned'], ['']))

        for config in to_train: 
            print(config)
            seed_num, model_params_key, task_file = config 
            model, _, _, _ = config_model_training(model_params_key)

            model.set_seed(seed_num)
            model.to(device)

            _get_model_contexts(model, 128, 'Multitask')



    if train_mode == 'test': 
        holdout_type = 'swap_holdouts'
        instruct_mode = ''
        seeds = [0, 1, 2, 3, 4]
        to_test = list(itertools.product(['bow20Net'], seeds, training_lists_dict['swap_holdouts']))

        for config in to_test: 
            print(config)
            model_params_key, seed_num, holdouts = config
            for holdout in holdouts:
                try:
                    if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
                    else: holdout_file = holdouts[0]
                    holdout_file = holdout_file.replace(' ', '_')
                    pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'/'+instruct_mode+holdout.replace(' ', '_')+'_seed'+str(seed_num)+'_holdout_correct', 'rb'))
                    print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout)
                    continue
                except FileNotFoundError: 
                    model, _, _, _ = config_model_training(model_params_key)
                    model.set_seed(seed_num)
                    model.instruct_mode = instruct_mode
                    model.model_name 
                    model.to(device)
                    test_model(model, holdouts, foldername= model_file, holdout_type = holdout_type, save=True)
                    
    if train_mode == 'fine_tune': 
        holdout_type = 'swap_holdouts'
        seeds = [0, 1, 2, 3, 4]
        to_tune = list(itertools.product(['bertNet', 'gptNet'], seeds, [['Multitask']]))
        for config in to_tune: 
            model_params_key, seed_num, holdouts = config
            if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
            else: holdout_file = holdouts[0]
            holdout_file = holdout_file.replace(' ', '_')
            print(holdout_file)

            # try:
            #     pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'_tuned'+'/seed'+str(seed_num)+'_training_correct', 'rb'))
            #     print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)
            #     continue
            # except FileNotFoundError: 
            print(config)
            model, _, _, _ = config_model_training(model_params_key)
            opt, sch = init_optimizer(model, 5*1e-4, [-1], langLR= 5*1e-5)
            model.set_seed(seed_num)
            model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.langModel.train_layers=['11', '10', '9']
            model.langModel.init_train_layers()
            model.model_name = model.model_name+'_tuned'
            if holdouts == ['Multitask']: data = TaskDataSet(data_folder= model_file+'/training_data')
            else: data = TaskDataSet(data_folder= model_file+'/training_data', holdouts=holdouts)
            data.data_to_device(device)
            model.to(device)
            train_model(model, data, 5, opt, sch)

            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
            print('saved: '+ model_file+'/'+holdout_type+'/'+holdout_file)

    if train_mode == 'train': 
        holdout_type = 'swap_holdouts'

        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ['sbertNet', 'gptNet',  'simpleNet', 'bow20Net', 'bertNet'], [['Multitask']]))
        print(to_train)

        last_holdouts = None
        data = None
        for cur_train in to_train:      
            #get the seed, holdout task, and model to train 
            seed_num, model_params_key, holdouts = cur_train

            #checkpoint the model training 

            #format save file name 
            if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
            else: holdout_file = holdouts[0]
            holdout_file = holdout_file.replace(' ', '_')

            #build model from params 

            # try: 
            #     pickle.load(open(model_file+'/'+holdout_type+'/'+holdout_file+'/'+model_params_key+'/seed'+str(seed_num)+'_training_loss', 'rb'))
            #     print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)

            #     last_holdouts = holdouts
            #     continue
            #except FileNotFoundError:
            print(cur_train)
                    #if its a new training task, make the new data 
            if holdouts == last_holdouts and data is not None: 
                pass 
            else: 
                if holdouts == ['Multitask']: data = TaskDataSet(data_folder= model_file+'/training_data')
                else: data = TaskDataSet(data_folder= model_file+'/training_data', holdouts=holdouts)
                data.data_to_device(device)

            model, _, _, epoch = config_model_training(model_params_key)
            model.set_seed(seed_num)
            # model.load_model(model_file+'/'+holdout_type+'/'+'Multitask')
            # model.load_training_data(model_file+'/'+holdout_type+'/'+'Multitask')
            for n,p in model.named_parameters():
                if p.requires_grad: print(n)
            model.to(device)
            print(model.model_name)
            #opt, _ = init_optimizer(model, 5e-05, [])

            opt, _ = init_optimizer(model, 0.001, [])
            sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
            #train 
            train_model(model, data, epoch, opt, sch)
            #save
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)

            #to check if you should make new data 
            last_holdouts = holdouts
