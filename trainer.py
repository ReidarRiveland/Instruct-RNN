import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rnn_models import InstructNet, SimpleNet
from nlp_models import SBERT, BERT, GPT, BoW
from task import Task
from data import TaskDataSet
from utils import isCorrect, train_instruct_dict
from model_analysis import get_instruct_reps

import itertools
import pickle
import sys

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
device = torch.device(0)

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
    
    
def train_model(model, streamer, epochs, optimizer, scheduler): 
    model.to(device)
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


        if scheduler is not None: 
            scheduler.step()    

def test_model(model, holdouts_test, repeats=5, foldername = '_ReLU128_24.7/single_holdouts', save=False): 
        holdout_file = holdouts_test.replace(' ', '_')
        for _ in range(repeats): 
            model.load_model(foldername+'/'+holdout_file)
            opt, _ = init_optimizer(model, 0.001, [])

            data = TaskDataSet(data_folder = '_ReLU128_24.7/training_data', batch_len=256, num_batches=100, task_ratio_dict={holdouts_test:1})
            data.data_to_device(device)
            train_model(model, data, 1, opt, None)
        
        correct_perf = np.mean(np.array(model._correct_data_dict[holdouts_test]).reshape(repeats, -1), axis=0)
        loss_perf = np.mean(np.array(model._loss_data_dict[holdouts_test]).reshape(repeats, -1), axis=0)

        if save: 
            pickle.dump(correct_perf, open(foldername +'/'+holdout_file + '/' + model.model_name+'/'+model.__seed_num_str__+'_holdout_correct', 'wb'))
            pickle.dump(loss_perf, open(foldername +'/'+holdout_file + '/' + model.model_name+'/'+model.__seed_num_str__+'_holdout_loss', 'wb'))

        return correct_perf, loss_perf

def train_context(model, data_streamer, epochs, model_load_file, init_context = None, context_dim = 768): 
    model_load_file = model_load_file.replace(' ', '_')
    model.load_model(model_load_file)
    model.freeze_weights()
    
    if init_context is None: 
        init_context = np.zeros(size=context_dim)

    init_context += np.random.normal(size=context_dim)
    context = nn.Parameter(torch.Tensor(init_context).unsqueeze(0))
    opt= optim.Adam([context], lr=0.01, weight_decay=0.00)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs-1, epochs-2], gamma=0.5)

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

'''
init_avg = False
model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
model.set_seed(0) 
model.to(device)
for task in Task.TASK_LIST:     
    task_file = task.replace(' ', '_')
    contexts = np.empty((15, 768))
    streamer = TaskDataSet('_ReLU128_5.7/training_data', num_batches = 500, task_ratio_dict={task:1})
    streamer.data_to_device(device)

    if init_avg: 
        instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
        init_context = np.mean(np.mean(instruct_reps, axis = 0), axis=0)
    else: 
        init_context = np.zeros(768)

    for j in range(15): 
        contexts[j, :]=train_context(model, streamer, 3, init_context = init_context, model_load_file='_ReLU128_5.7/single_holdouts/'+task_file)
    pickle.dump(contexts, open('_ReLU128_5.7/single_holdouts/'+task_file+'/sbertNet_layer_11/context_vecs', 'wb'))
    pickle.dump(np.array(model._correct_data_dict[task]).reshape(15, -1), open('_ReLU128_5.7/single_holdouts/'+task_file+'/sbertNet_layer_11/context_holdout_correct_data', 'wb'))
    pickle.dump(np.array(model._loss_data_dict[task]).reshape(15, -1), open('_ReLU128_5.7/single_holdouts/'+task_file+'/sbertNet_layer_11/context_holdout_loss_data', 'wb'))
'''


training_lists_dict={
'single_holdouts' :  [[item] for item in Task.TASK_LIST.copy()+['Multitask']],
'dual_holdouts' : [['RT Go', 'Anti Go'], ['Anti MultiDM', 'DM'], ['COMP1', 'MultiCOMP2'], ['DMC', 'DNMS']],
'aligned_holdouts' : [['Anti DM', 'Anti MultiDM'], ['COMP1', 'MultiCOMP1'], ['DMS', 'DNMS'],['Go', 'RT Go']],
'swap_holdouts' : [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['RT Go', 'COMP1']]
}


ALL_MODEL_PARAMS = {
    'sbertNet_layer_11': {'model': InstructNet, 
                    'langModel': SBERT,
                    'langModel_params': {'out_dim': 20, 'train_layers': ['11']},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                    'epochs': 30
                },

    'sbertNet': {'model': InstructNet, 
                'langModel': SBERT,
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 35
                },
    
    'bertNet_layer_11': {'model': InstructNet, 
                    'langModel': BERT,
                    'langModel_params': {'out_dim': 20, 'train_layers': ['11']},
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25], 'langLR': 1e-4},
                    'epochs': 30
                },

    'bertNet': {'model': InstructNet, 
                'langModel': BERT,
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 35
                },

    # 'GPT NET LAYER 11': {'model': InstructNet, 
    #                 'langModel': GPT,
    #                 'langModel_params': {'out_dim': 20, 'train_layers': ['11']},
    #                 'opt_params': {'lr':0.001, 'milestones':[15, 20, 22, 25], 'langLR': 1e-5},
    #                 'epochs': 30
    #             },

    'gptNet': {'model': InstructNet, 
                'langModel': GPT,
                'langModel_params': {'out_dim': 20, 'train_layers': []}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]}, 
                'epochs': 35
                },
    
    'bowNet': {'model': InstructNet, 
                'langModel': BoW,
                'langModel_params': {'out_dim': None}, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 30
                },

    'simpleNet': {'model': SimpleNet, 
                'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                'epochs': 30
                }
}

def config_model_training(key): 
    params = ALL_MODEL_PARAMS[key]

    if params['model'] is InstructNet:
        model = params['model'](params['langModel'](**params['langModel_params']), 128, 1, torch.relu)
    else:
        model = params['model'](128, 1, torch.relu)

    opt, sch = init_optimizer(model, **params['opt_params'])
    epochs = params['epochs']

    return model, opt, sch, epochs


if __name__ == "__main__":
    train_or_test = 'train'

    if train_or_test == 'test': 
        seeds = [0, 1, 2, 3, 4]
        to_test = list(itertools.product(seeds, ALL_MODEL_PARAMS.keys(), Task.TASK_LIST))
        for config in to_test: 
            seed_num, model_params_key, holdouts = config
            try:
                holdout_file = holdouts.replace(' ', '_')
                pickle.load(open('_ReLU128_24.7/single_holdouts' +'/'+holdout_file + '/' + model_params_key+'/seed'+str(seed_num)+'_holdout_correct', 'rb'))
                print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)
                continue
            except FileNotFoundError: 
                model, _, _, _ = config_model_training(model_params_key)
                model.set_seed(seed_num)
                model.to(device)
                test_model(model, holdouts, save=True)

    if train_or_test == 'train': 
        seeds = [0, 1, 2, 3, 4]
        holdout_type = 'swap_holdouts'
        to_train = list(itertools.product(seeds, ALL_MODEL_PARAMS.keys(), training_lists_dict[holdout_type]))

        last_holdouts = None
        data = None
        for cur_train in to_train:      
            #get the seed, holdout task, and model to train 
            seed_num, model_params_key, holdouts = cur_train

            #checkpoint the model training 

            #format save file name 
            if isinstance(holdouts, list): holdout_file = '_'.join(holdouts)
            else: holdout_file = holdouts
            holdout_file = holdout_file.replace(' ', '_')

            #build model from params 

            try: 
                pickle.load(open('_ReLU128_24.7/'+holdout_type+'/'+holdout_file+'/'+model_params_key+'/seed'+str(seed_num)+'_training_loss', 'rb'))
                print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)

                last_holdouts = holdouts
                continue
            except FileNotFoundError:
                print(cur_train)
                        #if its a new training task, make the new data 
                if holdouts == last_holdouts and data is not None: 
                    pass 
                else: 
                    if holdouts == 'Multitask': data = TaskDataSet(data_folder= '_ReLU128_24.7/training_data')
                    else: data = TaskDataSet(data_folder= '_ReLU128_24.7/training_data', holdouts=holdouts)
                    data.data_to_device(device)

                model, opt, sch, epochs = config_model_training(model_params_key)
                model.set_seed(seed_num)
                model.to(device)

                #train 
                train_model(model, data, epochs, opt, sch)

                #save
                model.save_model('_ReLU128_24.7/single_holdouts/'+holdout_file)
                model.save_training_data('_ReLU128_24.7/single_holdouts/'+holdout_file)

                #to check if you should make new data 
                last_holdouts = holdouts