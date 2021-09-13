from matplotlib.pyplot import stem
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rnn_models import InstructNet, SimpleNet
from nlp_models import SBERT, BERT, GPT, BoW
from task import Task
from data import TaskDataSet
from utils import isCorrect, train_instruct_dict, task_swaps_map
from model_analysis import task_eval, get_instruct_reps

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
    if testing and isinstance(model, InstructNet): 
        model.langModel.eval()
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

def _train_context(model, data_streamer, epochs, init_context = None, context_dim = 768): 
    model.freeze_weights()
    
    if init_context is None: 
        init_context = np.zeros(size=context_dim)

    init_context += np.random.normal(size=context_dim)
    context = nn.Parameter(torch.Tensor(init_context).unsqueeze(0))
    opt= optim.Adam([context], lr=0.01, weight_decay=0.00)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs-3, epochs-2, epochs-1], gamma=0.5)

    for i in range(epochs): 
        print('epoch', i)
        data_streamer.shuffle_stream_order()
        for j, data in enumerate(data_streamer.stream_batch()): 

            ins, tar, mask, tar_dir, task_type = data

            opt.zero_grad()
            batch_context = context.repeat(ins.shape[0], 1).to(device)
            proj = model.langModel.proj_out(batch_context)
            out, _ = super(type(model), model).forward(proj, ins)
            loss = masked_MSE_Loss(out, tar, mask) 
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


def get_model_contexts(model, num_contexts, filename, init_avg=False):
    all_contexts = np.empty((16, num_contexts, 768))
    model.load_model('_ReLU128_5.7/single_holdouts/Multitask')
    for i, task in enumerate(Task.TASK_LIST):     
        contexts = np.empty((num_contexts, 768))
        streamer = TaskDataSet(filename+'/training_data', num_batches = 500, task_ratio_dict={task:1})
        streamer.data_to_device(device)

        if init_avg: 
            instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
            init_context = np.mean(np.mean(instruct_reps, axis = 0), axis=0)
        else: 
            init_context = np.zeros(768)

        for j in range(num_contexts): 
            contexts[j, :]=_train_context(model, streamer, 5, init_context = init_context)

        all_contexts[i, ...] = contexts
        model._correct_data_dict[task] = np.array(model._correct_data_dict[task]).reshape(num_contexts, -1)
        model._loss_data_dict[task] = np.array(model._correct_data_dict[task]).reshape(num_contexts, -1)

    pickle.dump(all_contexts, open(filename+'/single_holdouts/Multitask/' + model.model_name+'/'+model.__seed_num_str__+'_context_vecs', 'wb'))
    pickle.dump(model._correct_data_dict, open(filename+'/single_holdouts/Multitask/' + model.model_name+'/'+model.__seed_num_str__+'_context_holdout_correct_data', 'wb'))
    pickle.dump(model._loss_data_dict, open(filename+'/single_holdouts/Multitask/' + model.model_name+'/'+model.__seed_num_str__+'_context_holdout_loss_data', 'wb'))


training_lists_dict={
'single_holdouts' :  [[item] for item in Task.TASK_LIST.copy()+['Multitask']],
'dual_holdouts' : [['RT Go', 'Anti Go'], ['Anti MultiDM', 'DM'], ['COMP1', 'MultiCOMP2'], ['DMC', 'DNMS']],
'aligned_holdouts' : [['Anti DM', 'Anti MultiDM'], ['COMP1', 'MultiCOMP1'], ['DMS', 'DNMS'],['Go', 'RT Go']],
'swap_holdouts' : [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['RT Go', 'DNMC'], ['DM', 'MultiCOMP2'], ['MultiDM', 'DNMS'], ['Anti MultiDM', 'COMP1'], ['COMP2', 'DMS'], ['Anti Go', 'MultiCOMP1']]
}


ALL_MODEL_PARAMS = {
    'sbertNet_tuned': {'model': InstructNet, 
                    'langModel': SBERT,
                    'model_name': 'sbertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                    'epochs': 35
                },

    'sbertNet': {'model': InstructNet, 
                'langModel': SBERT,
                'model_name': 'sbertNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 30
                },
    
    'bertNet_tuned': {'model': InstructNet, 
                    'langModel': BERT,
                    'model_name': 'bertNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25], 'langLR': 1e-4},
                    'epochs': 35
                },

    'bertNet': {'model': InstructNet, 
                'langModel': BERT,
                'model_name': 'bertNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 30
                },

    'gptNet_tuned': {'model': InstructNet, 
                    'langModel': GPT,
                    'model_name': 'gptNet_tuned',
                    'langModel_params': {'out_dim': 20, 'train_layers': []},
                    'opt_params': {'lr':0.001, 'milestones':[15, 20, 22, 25], 'langLR': 1e-5},
                    'epochs': 30
                },

    'gptNet': {'model': InstructNet, 
                'langModel': GPT,
                'model_name': 'gptNet',
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]}, 
                'epochs': 30
                },
    
    'bowNet': {'model': InstructNet, 
                'langModel': BoW,
                'model_name': 'bowNet',
                'langModel_params': {'out_dim': None}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 30
                },

    'simpleNet': {'model': SimpleNet, 
                'model_name': 'simpleNet',
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 30
                }

}


def config_model_training(key): 
    params = ALL_MODEL_PARAMS[key]

    if params['model'] is InstructNet:
        model = params['model'](params['langModel'](**params['langModel_params']), 128, 1, torch.relu)
    else:
        model = params['model'](128, 1, torch.relu, use_ortho_rules=True)

    model.model_name = params['model_name']

    opt, sch = init_optimizer(model, **params['opt_params'])
    epochs = params['epochs']

    return model, opt, sch, epochs

if __name__ == "__main__":
    model_file = '_ReLU128_5.7'

    train_mode = str(sys.argv[1])
    print('Mode: ' + train_mode + '\n')

    if train_mode == 'train_contexts': 
        seeds = [1, 0, 2, 3, 4]
        to_train = list(itertools.product(seeds, ['sbertNet'], ['Multitask']))

        for config in to_train: 
            print(config)
            seed_num, model_params_key, holdouts = config 
            model, _, _, _ = config_model_training(model_params_key)
            model.model_name += '_tuned'
            model.set_seed(seed_num)
            model.to(device)

            get_model_contexts(model, 5, model_file)

    if train_mode == 'test': 
        holdout_type = 'swap_holdouts'
        instruct_mode = 'swap'
        seeds = [0, 1, 2, 3, 4]
        to_test = list(itertools.product(ALL_MODEL_PARAMS.keys(), seeds, training_lists_dict['swap_holdouts']))


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

        #to_tune = list(itertools.product(['gptNet'], seeds, training_lists_dict['swap_holdouts']))

        to_tune = [('sbertNet', 0, ['DM', 'MultiCOMP2']), 
                    ('sbertNet', 1, ['RT Go', 'DNMC']), 
                    ('sbertNet', 3, ['Anti Go', 'MultiCOMP1']), 
                    ('sbertNet', 4, ['Anti Go', 'MultiCOMP1']), 
                    ('sbertNet', 4, ['COMP2', 'DMS'])
        ]

        for config in to_tune: 
            model_params_key, seed_num, holdouts = config

            if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
            else: holdout_file = holdouts[0]
            holdout_file = holdout_file.replace(' ', '_')
            print(holdout_file)

            try:
                pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'_tuned'+'/seed'+str(seed_num)+'_training_correct', 'rb'))
                print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)
                continue
            except FileNotFoundError: 
                print(config)
                model, _, _, _ = config_model_training(model_params_key)
                opt, sch = init_optimizer(model, 5*1e-4, [], langLR= 5*1e-5)
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

    if train_mode == 'train': 
        holdout_type = 'swap_holdouts'

        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ALL_MODEL_PARAMS.keys(), training_lists_dict[holdout_type]))
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

            try: 
                pickle.load(open(model_file+'/'+holdout_type+'/'+holdout_file+'/'+model_params_key+'/seed'+str(seed_num)+'_training_loss', 'rb'))
                print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)

                last_holdouts = holdouts
                continue
            except FileNotFoundError:
                print(cur_train)
                        #if its a new training task, make the new data 
                if holdouts == last_holdouts and data is not None: 
                    pass 
                else: 
                    if holdouts == ['Multitask']: data = TaskDataSet(data_folder= model_file+'/training_data')
                    else: data = TaskDataSet(data_folder= model_file+'/training_data', holdouts=holdouts)
                    data.data_to_device(device)

                model, opt, sch, epochs = config_model_training(model_params_key)
                model.set_seed(seed_num)
                model.to(device)

                #train 
                train_model(model, data, epochs, opt, sch)

                #save
                model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
                model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)

                #to check if you should make new data 
                last_holdouts = holdouts
