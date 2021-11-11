from matplotlib.pyplot import axis, stem
import numpy as np
from numpy.random import randn

import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim
from torch.serialization import save

from task import Task
from data import TaskDataSet
from utils import get_holdout_file, isCorrect, train_instruct_dict, training_lists_dict, get_holdout_file, all_swaps, tuning_check
from model_analysis import task_eval, get_instruct_reps, get_model_performance

from rnn_models import SimpleNet, InstructNet
from nlp_models import BERT, SBERT, GPT, BoW
import torch.nn as nn

import itertools
import pickle
import sys
import copy

device = torch.device(0)


torch.cuda.is_available()
torch.cuda.get_device_name(device)


model_file = '_ReLU128_4.11'
holdout_type = 'swap_holdouts'



ALL_MODEL_PARAMS = {
    'sbertNet_tuned': {'model': InstructNet, 
                    'langModel': SBERT,
                    'model_name': 'sbertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                },

    'sbertNet': {'model': InstructNet, 
                'langModel': SBERT,
                'model_name': 'sbertNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                },
    
    'bertNet_tuned': {'model': InstructNet, 
                    'langModel': BERT,
                    'model_name': 'bertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                },

    'bertNet': {'model': InstructNet, 
                'langModel': BERT,
                'model_name': 'bertNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                },

    'gptNet_tuned': {'model': InstructNet, 
                    'langModel': GPT,
                    'model_name': 'gptNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                },

    'gptNet': {'model': InstructNet, 
                'langModel': GPT,
                'model_name': 'gptNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                },
    
    'bowNet_flat': {'model': InstructNet, 
                'langModel': BoW,
                'model_name': 'bowNet_flat',
                'langModel_params': {'out_dim': None}, 
                },

                    
    'bowNet': {'model': InstructNet, 
                'langModel': BoW,
                'model_name': 'bowNet',
                'langModel_params': {'out_dim': 20, 'output_nonlinearity': nn.ReLU()}, 
                },


    'simpleNet': {'model': SimpleNet, 
                'model_name': 'simpleNet',
                }

}

def init_optimizer(model, lr, weight_decay=0.0, langLR=None):
    try:
        if langLR is None: langLR = lr 
        optimizer = optim.Adam([
                {'params' : model.recurrent_units.parameters()},
                {'params' : model.sensory_motor_outs.parameters()},
                {'params' : model.langModel.parameters(), 'lr': langLR}
            ], lr=lr, weight_decay=weight_decay)
    except AttributeError: 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer

def config_model(key): 
    params = ALL_MODEL_PARAMS[key]

    try:
        model = params['model'](params['langModel'](**params['langModel_params']), 128, 1, torch.relu)
    except:
        model = params['model'](128, 1, torch.relu, use_ortho_rules=True)

    model.model_name = params['model_name']

    return model


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

    
def train_model(model, streamer, epochs, optimizer, scheduler, holdout_file=None, testing=False, tuning =False, checkpoint_for_tuning=np.inf): 
    model.to(device)
    model.train()

    if not tuning: 
        try:
            model.langModel.eval()
        except: 
            pass

    step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-10, epochs-3, epochs-2], gamma=0.5)

    batch_len = streamer.batch_len 
    for i in range(epochs):

        if i == epochs-checkpoint_for_tuning and holdout_file is not None: 
            model_for_tuning = copy.deepcopy(model)
            model_for_tuning.model_name += '_tuned'
            model_for_tuning.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model_for_tuning.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
            print('model checkpointed')

        print('epoch', i)
        streamer.shuffle_stream_order()
        for j, data in enumerate(streamer.stream_batch()): 
            ins, tar, mask, tar_dir, task_type = data
            

            optimizer.zero_grad()

            task_info = model.get_task_info(batch_len, task_type)
            out, _ = model(task_info, ins.to(device))

            loss = masked_MSE_Loss(out,tar.to(device), mask.to(device)) 
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
                print(model.check_model_training(0.95))


        if scheduler is not None: 
            scheduler.step()    
        if not testing: 
            step_scheduler.step()

    return model.check_model_training(0.95)

def tune_model(model, holdouts, epochs, holdout_file): 

    if 'tuned' not in model.model_name: model.model_name = model.model_name+'_tuned'
    model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
    model.load_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
    print('model loaded:'+model_file+'/'+holdout_type+'/'+holdout_file+'\n')
    tuning_check(model)

    model.langModel.train_layers=['11', '10', '9']
    model.langModel.init_train_layers()
    
    model.to(device)

    data = TaskDataSet(holdouts=holdouts)
    #data.data_to_device(device)
    opt = init_optimizer(model, 5*1e-4, langLR= 1e-4)
    sch = optim.lr_scheduler.ExponentialLR(opt, 0.9)
    
    is_tuned = train_model(model, data, epochs, opt, sch, holdout_file=holdout_file, tuning=True)
    return is_tuned

def test_model(model, holdouts_test, foldername, repeats=5, holdout_type = 'single_holdouts', save=False): 
    holdout_file = get_holdout_file(holdouts_test)
    model.eval()
    for holdout in holdouts_test: 
        for _ in range(repeats): 
            model.load_model(foldername+'/'+holdout_type+'/'+holdout_file)
            opt, _ = init_optimizer(model, 0.0007)

            data = TaskDataSet(batch_len=256, num_batches=100, task_ratio_dict={holdout:1})
            data.data_to_device(device)
            train_model(model, data, 1, opt, None, testing=True)
        
        correct_perf = np.mean(np.array(model._correct_data_dict[holdout]).reshape(repeats, -1), axis=0)
        loss_perf = np.mean(np.array(model._loss_data_dict[holdout]).reshape(repeats, -1), axis=0)

        if save: 
            pickle.dump(correct_perf, open(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_correct', 'wb'))
            pickle.dump(loss_perf, open(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_loss', 'wb'))
            print(foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/'+model.instruct_mode+holdout.replace(' ', '_')+'_'+model.__seed_num_str__+'_holdout_correct')

    return correct_perf, loss_perf

def train_model_set(model_configs, model_file, save_bool):
    inspection_list = []
    for config in model_configs:      
        seed_num, model_params_key, holdouts = config
        torch.manual_seed(seed_num)
        holdout_file = get_holdout_file(holdouts)


        print(config)
        data = TaskDataSet(holdouts=holdouts)
        data.data_to_device(device)

        model = config_model(model_params_key)
        model.set_seed(seed_num)

        opt = init_optimizer(model, 0.001)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)

        #train 
        if holdouts == ['Multitask']: 
            eps = 55
            checkpoint=10
        else: 
            eps = 35 
            checkpoint=5

        is_training = train_model(model, data, eps, opt, sch, checkpoint_for_tuning=checkpoint)

        if is_training and save_bool:
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
        else: 
            inspection_list.append(config)
        
        print(inspection_list)

    return inspection_list
    
def tune_model_set(model_configs, model_file, save_bool):                
    inspection_list = []
    for config in model_configs: 
        seed_num, model_params_key, holdouts = config
        torch.manual_seed(seed_num)
        holdout_file = get_holdout_file(holdouts)
        
        print(config)
        model = config_model(model_params_key)
        model.set_seed(seed_num)

        if holdouts == ['Multitask']: eps = 10
        else: eps = 5
        
        tried_tuning = False
        is_tuned = False
        try: 
            is_tuned = tune_model(model, holdouts, eps, holdout_file)
            tried_tuning = True
        except ValueError: 
            print(model.model_name+model.__seed_num_str__+' already tuned for holdouts '+holdout_file+'\n')

        if is_tuned and save_bool: 
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
        elif tried_tuning: 
            inspection_list.append(config)
        else: 
            print('loading next model')

        print(inspection_list)
    return inspection_list        

def test_model_set(model_configs, model_file, save_bool):
    instruct_mode = ''
    for instruct_mode in ['', 'swap']:
        for config in model_configs: 
            print(config)
            model_params_key, seed_num, holdouts = config
            torch.manual_seed(seed_num)
            for holdout in holdouts:
                model = config_model(model_params_key)
                model.set_seed(seed_num)
                model.instruct_mode = instruct_mode
                model.model_name 
                model.to(device)
                test_model(model, holdouts, foldername= model_file, holdout_type = holdout_type, save=save_bool)
               
def check_model_set(to_check):
    perf_dict = {}
    to_retrain = []
    with torch.no_grad:
        for config in to_check: 
            model_params_key, seed_num, holdouts = config
            torch.manual_seed(seed_num)
            holdout_file = get_holdout_file(holdouts) 
            print(config)
            model = config_model(model_params_key)
            model.eval()
            model.to(device)
            model.set_seed(seed_num)
            model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
            perf = get_model_performance(model, 10).round(2)
            try:
                holdout_indices = [Task.TASK_LIST.index(holdouts[0]), Task.TASK_LIST.index(holdouts[1])]
            except ValueError: 
                holdout_indices=[]
            check_array=np.delete(perf, holdout_indices)
            passed_test = all(check_array>=0.95)
            print(perf)
            print(passed_test)
            perf_dict[str(model_params_key)+ str(seed_num)+str(holdouts)] = perf
            if not passed_test: 
                to_retrain.append((model_params_key, seed_num, holdouts))
            print(to_retrain)
    return to_retrain, perf_dict


if __name__ == "__main__":

    #train_mode = str(sys.argv[1])
    train_mode = 'fine_tune'
    
    print('Mode: ' + train_mode + '\n')

    if train_mode == 'test': 
        seeds = [0, 1, 2, 3, 4]
        to_test = list(itertools.product(['simpleNet'], seeds, [['Go', 'Anti DM'], ['Anti RT Go', 'DMC']]))
        inspection_list = test_model_set(to_test, model_file)
        print(inspection_list)

    if train_mode == 'fine_tune': 
        seeds = [0]
        to_tune = list(itertools.product([0, 1], ['sbertNet', 'bertNet', 'gptNet'],  training_lists_dict['swap_holdouts']))
        print(to_tune)
        inspection_list = tune_model_set(to_tune, model_file, save_bool=True)
        print(inspection_list)

    if train_mode =='train': 
        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ['sbertNet', 'bertNet', 'gptNet'],  training_lists_dict['swap_holdouts']))
        print(to_train)

        inspection_list = train_model_set(to_train)
        print(inspection_list)


    if train_mode == 'check_train':
        seeds = [0, 1, 2, 3, 4]
        to_check = list(itertools.product(['gptNet', 'gptNet_tuned'], seeds, [['Multitask']]))
        to_retrain, perf_dict = check_model_set(to_check)
