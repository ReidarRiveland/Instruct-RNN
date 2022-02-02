from matplotlib.pyplot import axis, stem
import numpy as np
from numpy.core.fromnumeric import repeat
from numpy.random import randn

import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim
from torch.random import seed
from torch.serialization import save

from task import Task
from data import TaskDataSet
from utils import get_holdout_file, isCorrect, train_instruct_dict, training_lists_dict, get_holdout_file, all_swaps, tuning_check, all_models
from model_analysis import task_eval, get_instruct_reps, get_model_performance

from rnn_models import SimpleNet, InstructNet
from nlp_models import BERT, SBERT, GPT, BoW
import torch.nn as nn

import itertools
import pickle
import sys
import copy

device = torch.device(0)




model_file = '_ReLU128_4.11'
holdout_type = 'swap_holdouts'



ALL_MODEL_PARAMS = {
    'sbertNet_tuned': {'model': InstructNet, 
                    'langModel': SBERT,
                    'model_name': 'sbertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'tune_opt_params': {'init_lr':1e-4, 'init_lang_lr': 5*1e-5, 'exp_gamma':0.9, 'step_gamma': 0.5}
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
                    'tune_opt_params': {'init_lr':1e-4, 'init_lang_lr': 5*1e-5, 'exp_gamma':0.9, 'step_gamma': 0.5}
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
                    'tune_opt_params': {'init_lr':1e-4, 'init_lang_lr': 1e-4, 'exp_gamma':0.95, 'step_gamma': 0.2}
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


def train_model(model, streamer, epochs, optimizer, scheduler, step_params, testing=False, tuning =False, checkpoint_for_tuning=np.inf): 
    model.to(device)
    model.train()

    step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,**step_params)

    if not tuning: 
        try:
            model.langModel.eval()
        except: 
            pass

    model_for_tuning = None
    batch_len = streamer.batch_len 

    for i in range(epochs):
        if i == epochs-checkpoint_for_tuning: 
            model_for_tuning = copy.deepcopy(model)
            model_for_tuning.model_name += '_tuned'
            print('model checkpointed')

        print('epoch', i)
        if not testing:
            print('Current lr:' +str([round(lr, 7) for lr in scheduler.get_last_lr()]))

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

            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
            model._loss_data_dict[task_type].append(loss.item())
            model._correct_data_dict[task_type].append(frac_correct)
            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                print('Frac Correct ' + str(frac_correct) + '\n')
                print(model.check_model_training(0.95, 1))

            if tuning and i>= epochs-5 and model.check_model_training(0.95, 5): 
                return model.check_model_training(0.95, 5), None

        if scheduler is not None: 
            scheduler.step()    
        if not testing: 
            step_scheduler.step()

    return model.check_model_training(0.95, 5), model_for_tuning

def tune_model(model, holdouts, epochs, holdout_file): 
    if 'tuned' not in model.model_name: model.model_name = model.model_name+'_tuned'

    model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
    model.load_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
    print('model loaded:'+model_file+'/'+holdout_type+'/'+holdout_file+'\n')
    model.langModel.train_layers=['11', '10', '9']
    model.langModel.init_train_layers()
    
    tried_tuning = tuning_check(model, holdouts)
    print('tried tuning= '+str(tried_tuning))
    
    if not tried_tuning: 
        model.to(device)
        data = TaskDataSet(holdouts=holdouts)

        tune_opt_params = ALL_MODEL_PARAMS[model.model_name]['tune_opt_params']

        opt = init_optimizer(model, tune_opt_params['init_lr'], langLR=tune_opt_params['init_lang_lr'])
        sch = optim.lr_scheduler.ExponentialLR(opt, tune_opt_params['exp_gamma'])
        step_params = {'milestones':[epochs-2, epochs-1], 'gamma': tune_opt_params['step_gamma']}

        is_tuned, _ = train_model(model, data, epochs, opt, sch, step_params, tuning=True)
        return is_tuned

    elif model.check_model_training(0.95, 5):
        return 'MODEL ALREADY TUNED'
    
    else:
        raise ValueError('examine current model, tuning tried and saved but threshold not reached')
    
def test_model(model, holdouts_test, foldername, repeats=5, holdout_type = 'single_holdouts', save=False): 
    holdout_file = get_holdout_file(holdouts_test)
    model.eval()
    for holdout in holdouts_test: 
        for _ in range(repeats): 
            model.load_model(foldername+'/'+holdout_type+'/'+holdout_file)
            opt = init_optimizer(model, 0.0007)
            step_params = {'milestones':[], 'gamma': 0}

            data = TaskDataSet(batch_len=256, num_batches=100, task_ratio_dict={holdout:1})
            train_model(model, data, 1, opt, None, step_params, testing=True)
        
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
        model_params_key, seed_num, holdouts = config
        #torch.manual_seed(seed_num)
        holdout_file = get_holdout_file(holdouts)

        print(config)
        data = TaskDataSet(holdouts=holdouts)
        #data.data_to_device(device)

        model = config_model(model_params_key)
        model.set_seed(seed_num)

        #train 
        if holdouts == ['Multitask']: 
            eps = 55
            checkpoint=10
        else: 
            eps = 35 
            checkpoint=5

        if model.model_name=='simpleNet' or model.model_name=='bowNet':
            checkpoint=np.inf

    
        opt = init_optimizer(model, 0.001)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
        step_params = {'milestones':[eps-10, eps-2, eps-1], 'gamma': 0.5}

        is_trained, model_for_tuning = train_model(model, data, eps, opt, sch, step_params, checkpoint_for_tuning=checkpoint)
        is_trained=True
        if is_trained and save_bool:
            print('Model Trained '+str(is_trained))
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
            try: 
                model_for_tuning.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
                model_for_tuning.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
                print('Model for tuning saved')
            except AttributeError:
                print('Model for Tuning not saves, model type: ' + str(type(model_for_tuning)))
        else: 
            inspection_list.append(config)
        
        print(inspection_list)

    return inspection_list
    
def tune_model_set(model_configs, model_file, save_bool):                
    inspection_list = []
    for config in model_configs: 
        model_params_key, seed_num, holdouts = config
        torch.manual_seed(seed_num)
        holdout_file = get_holdout_file(holdouts)
        
        print(config)
        model = config_model(model_params_key)
        model.set_seed(seed_num)

        if holdouts == ['Multitask']: eps = 15
        else: eps = 10
        
        
        is_tuned = tune_model(model, holdouts, eps, holdout_file)
        print('Model Tune= '+str(is_tuned))

        if is_tuned == 'MODEL ALREADY TUNED':
            print('model already tuned... loading next model')
            continue

        if is_tuned and save_bool: 
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
            print('model saved!')
        elif not is_tuned: 
            inspection_list.append(config)
        elif is_tuned: 
            print('tuned and not saved')

        print(inspection_list)

    return inspection_list        

def test_model_set(model_configs, model_file, save_bool):
    for config in model_configs: 
        print(config)
        instruct_mode, model_params_key, seed_num, holdouts = config
        torch.manual_seed(seed_num)
        model = config_model(model_params_key)
        model.set_seed(seed_num)
        model.instruct_mode = instruct_mode
        model.model_name 
        model.to(device)

        test_model(model, holdouts, repeats=5, foldername= model_file, holdout_type = holdout_type, save=save_bool)

def check_model_set(to_check):
    to_retrain = []
    to_tune = []
    for config in to_check: 
        model_params_key, seed_num, holdouts = config
        holdout_file = get_holdout_file(holdouts) 
        print(config)
        foldername = '_ReLU128_4.11/swap_holdouts/'+holdout_file
        try: 
            correct_data_dict = pickle.load(open(foldername+'/'+model_params_key\
                            +'/seed'+str(seed_num)+'_training_correct', 'rb'))
        except: 
            print('No training data for '+str(config))
        example_data = list(correct_data_dict.values())[0]
        if holdouts == ['Multitask']: needs_tuning = len(example_data) < 1500
        else: needs_tuning = len(example_data) < 1100
        latest_perf = np.array([task_perf[-5:] for task_perf in correct_data_dict.values()])
        passed_test = np.all(latest_perf>0.95)
                
        if passed_test and not needs_tuning: 
            print(str(config)+' passed test')
        elif needs_tuning: 
            print('needs tuning!')
            to_tune.append(config)
        else: 
            to_retrain.append(config)
    return to_retrain, to_tune


if __name__ == "__main__":

    #train_mode = str(sys.argv[1])
    train_mode = 'test'
    
    print('Mode: ' + train_mode + '\n')

    if train_mode == 'test': 

        to_test = list(itertools.product(['', 'swap'], ['bertNet_tuned'], [2], [['MultiDM', 'DNMS'], ['COMP2', 'DMS']]))
        #to_test1 = list(itertools.product(['', 'swap'], ['gptNet'], [0], training_lists_dict['swap_holdouts']))
        #to_test1 = list(itertools.product(['', 'swap'], ['gptNet'], [0], training_lists_dict['swap_holdouts']))
        #to_test1 = list(itertools.product(['', 'swap'], ['gptNet_tuned'], [0, 1, 2, 3, 4], training_lists_dict['swap_holdouts']))
        #to_test = list(itertools.product(['', 'swap'], ['bertNet_tuned'], [1, 2, 3, 4], training_lists_dict['swap_holdouts']))
        print(to_test)

        inspection_list = test_model_set(to_test, model_file, save_bool=True)
        print(inspection_list)

    if train_mode == 'fine_tune': 
        #to_tune = list(itertools.product(['gptNet'], seeds,training_lists_dict['swap_holdouts']))
        #to_tune = [('gptNet', 0, ['MultiDM', 'DNMS'])]
        #to_tune = list(itertools.product(['bertNet'], seeds,training_lists_dict['swap_holdouts']))
        to_tune=[('gptNet', 0, ['Multitask']), ('gptNet', 3, ['Multitask'])]


        print(to_tune)
        inspection_list = tune_model_set(to_tune, model_file)
        print(inspection_list)

    if train_mode =='train': 
        to_train = [('gptNet', 0, ['Multitask'])]        
        print(to_train)
        print(len(to_train))
        inspection_list = train_model_set(to_train, model_file, save_bool=True)
        print(inspection_list)


    if train_mode == 'check_train':
        seeds = [0, 1, 2, 3, 4]
        to_check = list(itertools.product(['bertNet_tuned'], seeds, training_lists_dict['swap_holdouts']+[['Multitask']]))
        to_retrain, to_tune = check_model_set(to_check)
        print(to_retrain)
        print(to_tune)
