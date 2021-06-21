from data import TaskDataSet
import numpy as np

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import isCorrect


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
    
    
def train(model, streamer, epochs, optimizer, scheduler): 
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



if __name__ == "__main__":

    import torch.nn as nn
    from rnn_models import InstructNet, SimpleNet
    from nlp_models import SBERT, BERT, GPT, BoW
    from task import Task
    import itertools
    import pickle

    ALL_MODEL_PARAMS = {
        'SBERT NET LAYER 11': {'model': InstructNet, 
                        'langModel': SBERT,
                        'langModel_params': {'out_dim': 20, 'train_layers': ['11']},
                        'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                        'epochs': 30
                    },

        'SBERT NET': {'model': InstructNet, 
                    'langModel': SBERT,
                    'langModel_params': {'out_dim': 20, 'train_layers': [], 'output_nonlinearity': nn.Identity()}, 
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                    'epochs': 30
                    },
        
        'BERT NET LAYER 11': {'model': InstructNet, 
                        'langModel': BERT,
                        'langModel_params': {'out_dim': 20, 'train_layers': ['11']},
                        'opt_params': {'lr':0.001, 'milestones':[5, 10, 15, 20, 25], 'langLR': 1e-4},
                        'epochs': 30
                    },

        'BERT NET': {'model': InstructNet, 
                    'langModel': BERT,
                    'langModel_params': {'out_dim': 20, 'train_layers': [], 'output_nonlinearity': nn.Identity()}, 
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                    'epochs': 30
                    },

        'GPT NET LAYER 11': {'model': InstructNet, 
                        'langModel': GPT,
                        'langModel_params': {'out_dim': 20, 'train_layers': ['11']},
                        'opt_params': {'lr':0.001, 'milestones':[5, 10, 15, 20, 25], 'langLR': 1e-5},
                        'epochs': 30
                    },

        'GPT NET': {'model': InstructNet, 
                    'langModel': GPT,
                    'langModel_params': {'out_dim': 20, 'train_layers': [], 'output_nonlinearity': nn.Identity()}, 
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]}, 
                    'epochs': 30
                    },
        
        'BoW NET': {'model': InstructNet, 
                    'langModel': BoW,
                    'langModel_params': {'out_dim': None}, 
                    'opt_params': {'lr':0.001, 'milestones':[10, 15, 20, 25]},
                    'epochs': 30
                    },

        'SIMPLE NET': {'model': SimpleNet, 
                    'opt_params': {'lr':0.001, 'milestones':[5, 10, 15, 20, 25]},
                    'epochs': 30
                    }
    }

    single_holdouts = Task.TASK_LIST.copy()
    dual_holdouts = [['RT Go', 'Anti Go'], ['Anti MultiDM', 'DM'], ['COMP1', 'MultiCOMP2'], ['DMC', 'DNMS']]
    aligned_holdouts = [['Anti DM', 'Anti MultiDM'], ['COMP1', 'MultiCOMP1'], ['DMS', 'DNMS'],['Go', 'RT Go']]
    swap_holdouts = [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['RT Go', 'COMP1']]

    seeds = [2, 3, 4]
    to_train = list(itertools.product(seeds, single_holdouts, ALL_MODEL_PARAMS.keys()))

    logged_checkpoint = pickle.load(open('_ReLU128_14.6/single_holdouts/logged_train_checkpoint', 'rb'))
    last_holdouts = None

    for cur_train in to_train[to_train.index(logged_checkpoint):]:      
        #get the seed, holdout task, and model to train 
        i, holdouts, model_params_key = cur_train

        #checkpoint the model training 
        pickle.dump(cur_train, open('_ReLU128_14.6/single_holdouts/logged_train_checkpoint', 'wb'))
        
        #if its a new training task, make the new data 
        if holdouts == last_holdouts: 
            pass 
        else: 
            data = TaskDataSet(holdouts=[holdouts])
            data.data_to_device(device)

        #format save file name 
        if isinstance(holdouts, list): holdout_file = '_'.join(holdouts)
        else: holdout_file = holdouts
        holdout_file = holdout_file.replace(' ', '_')

        #build model from params 
        params = ALL_MODEL_PARAMS[model_params_key]
        if params['model'] is InstructNet:
            model = params['model'](params['langModel'](**params['langModel_params']), 128, 1, torch.relu)
        else:
            model = params['model'](128, 1, torch.relu)
        opt, sch = init_optimizer(model, **params['opt_params'])
        model.model_name += ('_seed' + str(i))
        model.to(device)
        
        #train 
        train(model, data, params['epochs'], opt, sch)

        #save
        torch.save(model.state_dict(), '_ReLU128_14.6/single_holdouts/'+holdout_file+'/'+model.model_name+'.pt')
        model.save_training_data(holdout_file, '_ReLU128_14.6/single_holdouts/', model.model_name)

        #to check if you should make new data 
        last_holdouts = holdouts