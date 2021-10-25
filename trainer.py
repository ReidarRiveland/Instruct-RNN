from matplotlib.pyplot import axis, stem
import numpy as np
from numpy.random import randn

import torch
import torch.nn as nn
import torch.optim as optim

from task import Task
from data import TaskDataSet
from utils import get_holdout_file, isCorrect, train_instruct_dict, training_lists_dict, get_holdout_file, all_swaps
from model_analysis import task_eval, get_instruct_reps

from rnn_models import SimpleNet, InstructNet
from nlp_models import BERT, SBERT, GPT, BoW
import torch.nn as nn

import itertools
import pickle
import sys
import copy

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
    
    
def train_model(model, streamer, epochs, optimizer, scheduler, print_eval=False, testing=False, checkpoint_for_tuning=False): 
    assert checkpoint_for_tuning != 'tuned' in model.model_name, 'checkpointed train for model for tuning'
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
        
        if checkpoint_for_tuning and i == epochs-5: 
            model_for_tuning = copy.deepcopy(model)
            model_for_tuning.model_name += '_tuned'
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)

def tune_model(model, holdouts): 
    data = TaskDataSet(data_folder= model_file+'/training_data', holdouts=holdouts)
    data.data_to_device(device)
    model.set_seed(seed_num)
    opt, sch = init_optimizer(model, 5*1e-4, [-1], langLR= 5*1e-5)
    
    model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
    model.langModel.train_layers=['11', '10', '9']
    model.langModel.init_train_layers()
    if 'tuned' not in model.model_name: model.model_name = model.model_name+'_tuned'

    model.to(device)
    train_model(model, data, 10, opt, sch)

    model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
    model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
    print('saved: '+ model_file+'/'+holdout_type+'/'+holdout_file)


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


def _train_context_(model, data_streamer, epochs, opt, sch, context, self_supervised): 
    model.freeze_weights()
    model.eval()
    for i in range(epochs): 
        print('epoch', i)
        data_streamer.shuffle_stream_order()
        for j, data in enumerate(data_streamer.stream_batch()): 

            ins, tar, mask, tar_dir, task_type = data

            opt.zero_grad()
    
            #batch_context = context.repeat(ins.shape[0], 1).to(device)
            if self_supervised:
                task_info = model.get_task_info(ins.shape[0], task_type)
                target, _ = model(task_info, ins)
            else: 
                target = tar

            #proj = model.langModel.proj_out(context.float())            
            out, _ = super(type(model), model).forward(context, ins)
            loss = masked_MSE_Loss(out, target, mask) 
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

def get_model_contexts(model, num_contexts, target_embedding_layer, task_file, self_supervised, lang_init=False, foldername='_ReLU128_5.7'):
    if target_embedding_layer.isnumeric(): 
        context_dim = model.langModel.intermediate_lang_dim
    elif target_embedding_layer =='full': 
        context_dim = model.langModel.out_dim
    all_contexts = np.empty((16, num_contexts, context_dim))

    supervised_str = ''
    if not self_supervised:
        supervised_str = '_supervised'
    

    model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
    model.to(device)

    if lang_init: 
        instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth=target_embedding_layer)
        batched_reps = np.repeat(np.mean(instruct_reps, axis=1), num_contexts).reshape(16, num_contexts, context_dim)
        batched_reps+= np.random.randn(16, num_contexts, context_dim)
        tensor_reps = torch.tensor(batched_reps, device=device)
        lang_init_str = 'lang_init'
    else: 
        lang_init_str = ''

    for i, task in enumerate(Task.TASK_LIST):     
        if lang_init: 
            context = nn.Parameter(tensor_reps[i, ...], requires_grad=True)
        else: 
            context = nn.Parameter(torch.randn((num_contexts, context_dim), device=device))

        opt= optim.Adam([context], lr=5e-3, weight_decay=0.0)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)

        streamer = TaskDataSet(foldername+'/training_data', batch_len = num_contexts, num_batches = 500, task_ratio_dict={task:1})
        streamer.data_to_device(device)

        contexts =_train_context_(model, streamer, 12, opt, sch, context, self_supervised)
        all_contexts[i, ...] = contexts


    filename=foldername+'/swap_holdouts/'+task_file + '/'+model.model_name+'/contexts/'+model.__seed_num_str__+lang_init_str+supervised_str
    pickle.dump(all_contexts, open(filename+'_context_vecs'+str(context_dim), 'wb'))
    pickle.dump(model._correct_data_dict, open(filename+'_context_correct_data'+str(context_dim), 'wb'))
    pickle.dump(model._loss_data_dict, open(filename+'_context_loss_data'+str(context_dim), 'wb'))



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
        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ALL_MODEL_PARAMS.keys(), ['Multitask']+all_swaps))

        for config in to_train: 
            print(config)
            seed_num, model_params_key, task_file = config 
            model, _, _, _ = config_model_training(model_params_key)

            model.set_seed(seed_num)
            model.to(device)
            for self_supervised in [False, True]:
                supervised_str = ''
                if not self_supervised:
                    supervised_str = '_supervised'
    
                try: 
                    filename=model_file+'/swap_holdouts/'+task_file + '/'+model.model_name+'/contexts/'+model.__seed_num_str__+supervised_str+'_context_vecs20'
                    pickle.load(open(filename, 'rb'))
                    print(filename+' already trained')
                except FileNotFoundError:
                    get_model_contexts(model, 256, 'full', task_file, self_supervised)

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
                    holdout_file = get_holdout_file(holdouts)
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
            holdout_file = get_holdout_file(holdouts)

            try:
                pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'_tuned'+'/seed'+str(seed_num)+'_training_correct', 'rb'))
                print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)
                continue
            except FileNotFoundError: 
                print(config)
                model, _, _, _ = config_model_training(model_params_key)
                tune_model(model, holdouts)

            
    if train_mode == 'train': 
        holdout_type = 'swap_holdouts'
        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ['sbertNet', 'gptNet',  'simpleNet', 'bow20Net', 'bertNet'], [['Multitask']]))

        for cur_train in to_train:      
            seed_num, model_params_key, holdouts = cur_train
            holdout_file = get_holdout_file(holdouts)

            try: 
                pickle.load(open(model_file+'/'+holdout_type+'/'+holdout_file+'/'+model_params_key+'/seed'+str(seed_num)+'_training_loss', 'rb'))
                print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)

                last_holdouts = holdouts
                continue
            except FileNotFoundError:
                print(cur_train)
                data = TaskDataSet(data_folder= model_file+'/training_data', holdouts=holdouts)
                data.data_to_device(device)

                model, _, _, epoch = config_model_training(model_params_key)
                model.set_seed(seed_num)

                opt, _ = init_optimizer(model, 0.001, [])
                sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
                #train 
                train_model(model, data, epoch, opt, sch)
                #save
                model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
                model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)

