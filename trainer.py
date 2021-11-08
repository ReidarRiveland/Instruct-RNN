from matplotlib.pyplot import axis, stem
import numpy as np
from numpy.random import randn

import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim

from task import Task
from data import TaskDataSet
from utils import get_holdout_file, isCorrect, train_instruct_dict, training_lists_dict, get_holdout_file, all_swaps
from model_analysis import task_eval, get_instruct_reps, get_model_performance

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

#def trim_data_for_retune(model): 


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

    
def train_model(model, streamer, epochs, optimizer, scheduler, print_eval=False, testing=False, checkpoint_for_tuning=None): 
    model.to(device)
    model.train()

    try:
        model.langModel.eval()
    except: 
        pass

    step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-10, epochs-3, epochs-2], gamma=0.5)


    batch_len = streamer.batch_len 
    for i in range(epochs):

        if i == epochs-checkpoint_for_tuning: 
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
                print(model.check_model_training(0.95))
            if j%100 == 0: 
                if print_eval:
                    for holdout in streamer.holdouts: 
                        frac_correct = task_eval(model, holdout, 128)
                        print(holdout + ' holdout performance: '+str(frac_correct) + '\n')


        if scheduler is not None: 
            scheduler.step()    
        if not testing: 
            step_scheduler.step()
        

    return model.check_model_training(0.95)


def tune_model(model, holdouts, epochs): 
    data = TaskDataSet(data_folder= '_ReLU128_5.7/training_data', holdouts=holdouts)
    data.data_to_device(device)
    model.set_seed(seed_num)
    opt, _ = init_optimizer(model, 5*1e-4, [-1], langLR= 5*1e-5)
    sch = optim.lr_scheduler.ExponentialLR(opt, 0.9)

    if 'tuned' not in model.model_name: model.model_name = model.model_name+'_tuned'
    model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
    model.load_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
    model.langModel.train_layers=['11', '10', '9']
    model.langModel.init_train_layers()

    model.to(device)
    is_tuned = train_model(model, data, epochs, opt, sch)
    return is_tuned


def test_model(model, holdouts_test, repeats=5, foldername = '_ReLU128_5.7', holdout_type = 'single_holdouts', save=False): 
    holdout_file = get_holdout_file(holdouts_test)
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

def train_context(model, data_streamer, epochs, opt, sch, context, self_supervised): 
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
        if sch is not None:                
            sch.step()

    return context.squeeze().detach().cpu().numpy()

def test_context(model, holdouts_test, repeats=5, foldername = '_ReLU128_5.7', holdout_type = 'single_holdouts', save=False): 
    holdout_file = get_holdout_file(holdouts_test)
    for holdout in holdouts_test: 
        for _ in range(repeats): 
            model.load_model(foldername+'/'+holdout_type+'/'+holdout_file)
            context = nn.Parameter(torch.randn((128, 20), device=device))

            opt = optim.Adam([
                    {'params' : model.parameters()},
                    {'params' : [context], 'lr': 1e-3},
                ], lr=0.001)
            sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
            data = TaskDataSet(data_folder = foldername+'/training_data', batch_len=128, num_batches=500, task_ratio_dict={holdout:1})
            data.data_to_device(device)
            self_super = False
            train_context(model, data, 8, opt, sch, context, self_super)
        
        correct_perf = np.mean(np.array(model._correct_data_dict[holdout]).reshape(repeats, -1), axis=0)
        loss_perf = np.mean(np.array(model._loss_data_dict[holdout]).reshape(repeats, -1), axis=0)

        if save: 
            filename=foldername + '/' + holdout_type +'/'+holdout_file + '/' + model.model_name+'/context_test'+holdout.replace(' ', '_')+'_'+model.__seed_num_str__
            pickle.dump(correct_perf, open(filename+'_holdout_correct', 'wb'))
            pickle.dump(loss_perf, open(filename+'_holdout_loss', 'wb'))
            print(filename)

    return correct_perf, loss_perf

def get_model_contexts(model, num_contexts, target_embedding_layer, task_file, self_supervised, lang_init=False, foldername='_ReLU128_5.7'):
    try: 
        if target_embedding_layer.isnumeric(): 
            context_dim = model.langModel.intermediate_lang_dim
        elif target_embedding_layer =='full': 
            context_dim = model.langModel.out_dim
    except: 
        context_dim = 20

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
    filename=foldername+'/swap_holdouts/'+task_file + '/'+model.model_name+'/contexts/'+model.__seed_num_str__

    for i, task in enumerate(Task.TASK_LIST):     
        if lang_init: 
            context = nn.Parameter(tensor_reps[i, ...], requires_grad=True)
        else: 
            context = nn.Parameter(torch.randn((num_contexts, context_dim), device=device))

        opt= optim.Adam([context], lr=8e-2, weight_decay=0.0)
        #sch = optim.lr_scheduler.ExponentialLR(opt, 0.99)
        sch = optim.lr_scheduler.MultiStepLR(opt, [10, 12, 14], 0.1)


        streamer = TaskDataSet(foldername+'/training_data', batch_len = num_contexts, num_batches = 250, task_ratio_dict={task:1})
        streamer.data_to_device(device)

        contexts =train_context(model, streamer, 16, opt, sch, context, self_supervised)
        pickle.dump(contexts, open(filename+'/'+task+supervised_str+'_context_vecs'+str(context_dim), 'wb'))
        pickle.dump(model._correct_data_dict, open(filename+'/'+task+supervised_str+'_context_correct_data'+str(context_dim), 'wb'))
        pickle.dump(model._loss_data_dict, open(filename+'/'+task+supervised_str+'_context_loss_data'+str(context_dim), 'wb'))
        print('saved: '+filename)
        model.reset_training_data()



if __name__ == "__main__":
    model_file = '1_ReLU128_5.7'

    train_mode = str(sys.argv[1])
    #train_mode = 'check_train'
    
    print('Mode: ' + train_mode + '\n')

    if train_mode == 'train_contexts': 
        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ['sbertNet_tuned', 'simpleNet'], ['Multitask']+all_swaps))
        print(to_train)
        for config in to_train: 
            seed_num, model_params_key, task_file = config 
            model, _, _, _ = config_model_training(model_params_key)
            torch.manual_seed(seed_num)
            model.set_seed(seed_num)
            model.to(device)
            for self_supervised in [False, True]:
                supervised_str = ''
                if not self_supervised:
                    supervised_str = '_supervised'
    
                print(str(config) + supervised_str) 

                # try: 
                #     filename=model_file+'/swap_holdouts/'+task_file + '/'+model.model_name+'/contexts/'+model.__seed_num_str__+'/'+supervised_str+'_context_correct_data20'
                #     pickle.load(open(filename, 'rb'))
                #     print(filename+' already trained')
                #except FileNotFoundError:
                get_model_contexts(model, 256, 'full', task_file, self_supervised)

    if train_mode == 'test_contexts': 
        holdout_type = 'swap_holdouts'
        seeds = [0, 1, 2, 3, 4]
        to_test = list(itertools.product(ALL_MODEL_PARAMS.keys(), seeds,  training_lists_dict['swap_holdouts']))

        for config in to_test: 
            print(config)
            model_params_key, seed_num, holdouts = config
            torch.manual_seed(seed_num)
            for holdout in holdouts:
                try:
                    holdout_file = get_holdout_file(holdouts)
                    pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'/contexts_test'+holdout.replace(' ', '_')+'_seed'+str(seed_num)+'_holdout_correct', 'rb'))
                    print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout)
                    continue
                except FileNotFoundError: 
                    model, _, _, _ = config_model_training(model_params_key)
                    model.set_seed(seed_num)
                    model.to(device)
                    test_context(model, holdouts, foldername= model_file, holdout_type = holdout_type, save=True)
                  

    if train_mode == 'test': 
        holdout_type = 'swap_holdouts'
        instruct_mode = ''
        seeds = [0, 1, 2, 3, 4]
        to_test = list(itertools.product(['simpleNet'], seeds, [['Go', 'Anti DM'], ['Anti RT Go', 'DMC']]))

        for instruct_mode in ['', 'swap']:
            for config in to_test: 
                print(config)
                model_params_key, seed_num, holdouts = config
                torch.manual_seed(seed_num)
                for holdout in holdouts:
                    # try:
                    #     holdout_file = get_holdout_file(holdouts)
                    #     pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'/'+instruct_mode+holdout.replace(' ', '_')+'_seed'+str(seed_num)+'_holdout_correct', 'rb'))
                    #     print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout)
                    #     continue
                    # except FileNotFoundError: 
                    model, _, _, _ = config_model_training(model_params_key)
                    model.set_seed(seed_num)
                    model.instruct_mode = instruct_mode
                    model.model_name 
                    model.to(device)
                    test_model(model, holdouts, foldername= model_file, holdout_type = holdout_type, save=True)
                
    if train_mode == 'fine_tune': 
        holdout_type = 'swap_holdouts'
        #seeds = [0, 1, 2, 3, 4]
        #to_tune = list(itertools.product(['gptNet'], seeds, [['Multitask']]))
        to_tune =[
        #             ('gptNet', 0, ['Go', 'Anti DM'])]
        #('gptNet', 0, ['Anti RT Go', 'DMC']),
        #             ('gptNet', 0, ['Anti MultiDM', 'COMP1']),
        #             ('gptNet', 0, ['COMP2', 'DMS']),
        #             ('gptNet', 1, ['Anti RT Go', 'DMC']),
        #             ('gptNet', 1, ['RT Go', 'DNMC']),
        #     ('gptNet', 1, ['Anti MultiDM', 'COMP1']),
        #     ('gptNet', 1, ['COMP2', 'DMS']),
        #     ('gptNet', 1, ['Anti Go', 'MultiCOMP1']),
        #     ('gptNet', 2, ['RT Go', 'DNMC']),
        #     ('gptNet', 2, ['COMP2', 'DMS']),
        #     ('gptNet', 3, ['Anti RT Go', 'DMC']),
        #     ('gptNet', 3, ['RT Go', 'DNMC']),
        #     ('gptNet', 3, ['Anti MultiDM', 'COMP1']),
        #     ('gptNet', 4, ['Go', 'Anti DM']),
        #     ('gptNet', 4, ['DM', 'MultiCOMP2']),
        #     ('gptNet', 4, ['COMP2', 'DMS']),
        #     ('gptNet', 4, ['Anti Go', 'MultiCOMP1']),
            ('bertNet', 1, ['MultiDM', 'DNMS']),
            # ('bertNet', 1, ['Anti MultiDM', 'COMP1']),
            # ('bertNet', 1, ['Anti Go', 'MultiCOMP1']),
            # ('sbertNet', 0, ['Anti RT Go', 'DMC']),
            # ('sbertNet', 0, ['Anti Go', 'MultiCOMP1']),
            # ('sbertNet', 2, ['RT Go', 'DNMC']),
            # ('sbertNet', 2, ['Anti Go', 'MultiCOMP1']),
            # ('sbertNet', 4, ['RT Go', 'DNMC']),
        #     ('gptNet', 0, ['Multitask']),
        #     ('gptNet', 1, ['Multitask']),
        #     ('gptNet', 2, ['Multitask']),
        #     ('gptNet', 4, ['Multitask'])
        # ]
        ]
        inspection_list = []
        for config in to_tune: 
            model_params_key, seed_num, holdouts = config
            torch.manual_seed(seed_num)
            holdout_file = get_holdout_file(holdouts)

            # try:
            #     pickle.load(open(model_file+'/'+holdout_type +'/'+holdout_file + '/' + model_params_key+'_tuned'+'/seed'+str(seed_num)+'_training_correct', 'rb'))
            #     print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)
            #     continue
            #except FileNotFoundError: 
            print(config)
            model, _, _, _ = config_model_training(model_params_key)
            if holdouts == ['Multitask']: 
                eps = 10
            else: 
                eps = 5 

            is_tuned = tune_model(model, holdouts, eps)
            if not is_tuned: 
                inspection_list.append(config)
            else: 
                model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
                model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
        print(inspection_list)
            
    if train_mode == 'train': 
        holdout_type = 'swap_holdouts'
        #seeds = [2]
        #to_train = list(itertools.product(seeds, ['simpleNet'], [['Go', 'Anti DM'], ['Anti RT Go', 'DMC']]))
        to_train=[('', 'gptNet', 0, ['Go', 'Anti DM']),
                    ('original', 'gptNet', 0, ['DM', 'MultiCOMP2']),
                    ('original', 'gptNet', 0, ['MultiDM', 'DNMS']),
                    ('', 'gptNet', 0, ['Anti MultiDM', 'COMP1']),
                    ('', 'gptNet', 0, ['COMP2', 'DMS']),
                    ('', 'gptNet', 1, ['DM', 'MultiCOMP2']),
                    ('original', 'gptNet', 1, ['MultiDM', 'DNMS']),
                    ('', 'gptNet', 1, ['Anti MultiDM', 'COMP1']),
                    ('', 'gptNet', 1, ['COMP2', 'DMS']),
                    ('original', 'gptNet', 2, ['Go', 'Anti DM']),
                    ('original', 'gptNet', 2, ['Anti RT Go', 'DMC']),
                    ('', 'gptNet', 2, ['RT Go', 'DNMC']),
                    ('original', 'gptNet', 2, ['MultiDM', 'DNMS']),
                    ('original', 'gptNet', 2, ['Anti MultiDM', 'COMP1']),
                    ('', 'gptNet', 2, ['COMP2', 'DMS']),
                    ('original', 'gptNet', 3, ['Go', 'Anti DM']),
                    ('', 'gptNet', 3, ['Anti RT Go', 'DMC']),
                    ('original', 'gptNet', 3, ['MultiDM', 'DNMS']),
                    ('', 'gptNet', 4, ['Go', 'Anti DM']),
                    ('original', 'gptNet', 4, ['Anti RT Go', 'DMC']),
                    ('original', 'gptNet', 4, ['MultiDM', 'DNMS']),
                    ('original', 'gptNet', 4, ['Anti MultiDM', 'COMP1']),
                    ('', 'gptNet', 4, ['COMP2', 'DMS']),
                    ('original', 'sbertNet', 0, ['Go', 'Anti DM']),
                    ('original', 'sbertNet', 0, ['Anti MultiDM', 'COMP1']),
                    ('original', 'sbertNet', 2, ['Go', 'Anti DM']),
                    ('original', 'sbertNet', 4, ['Anti RT Go', 'DMC']),
                    ('', 'sbertNet', 4, ['RT Go', 'DNMC']),
                    ('original', 'sbertNet', 4, ['DM', 'MultiCOMP2']),
                    ('original', 'bertNet', 0, ['COMP2', 'DMS']),
                    ('original', 'bertNet', 1, ['Go', 'Anti DM']),
                    ('original', 'bertNet', 1, ['RT Go', 'DNMC']),
                    ('', 'bertNet', 1, ['Anti Go', 'MultiCOMP1']),
                    ('original', 'bertNet', 2, ['RT Go', 'DNMC']),
                    ('original', 'bertNet', 2, ['COMP2', 'DMS']),
                    ('original', 'bertNet', 3, ['DM', 'MultiCOMP2']),
                    ('original', 'bertNet', 4, ['RT Go', 'DNMC'])]
        #to_train=[('original', 'bertNet', 2, ['Multitask'])]
            
        inspection_list = []
        for cur_train in to_train:      
            _, model_params_key, seed_num, holdouts = cur_train
            torch.manual_seed(seed_num)
            holdout_file = get_holdout_file(holdouts)

            # try: 
            #     pickle.load(open(model_file+'/'+holdout_type+'/'+holdout_file+'/'+model_params_key+'/seed'+str(seed_num)+'_training_loss', 'rb'))
            #     print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)

            #     last_holdouts = holdouts
            #     continue
            # except FileNotFoundError:
            print(cur_train)
            data = TaskDataSet(data_folder= '_ReLU128_5.7/training_data', holdouts=holdouts)
            data.data_to_device(device)

            model, _, _, epoch = config_model_training(model_params_key)
            model.set_seed(seed_num)

            opt, _ = init_optimizer(model, 0.001, [])
            sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)

            #train 
            if holdouts == ['Multitask']: 
                eps = 55
                checkpoint=-10

            else: 
                eps = 35 
                checkpoint=-5

            print(eps)

            finished_training = train_model(model, data, eps, opt, sch, checkpoint_for_tuning=checkpoint)

            if not finished_training: 
                inspection_list.append(cur_train)
            #save
            model.save_model(model_file+'/'+holdout_type+'/'+holdout_file)
            model.save_training_data(model_file+'/'+holdout_type+'/'+holdout_file)
        
        print(inspection_list)

    if train_mode == 'check_train':
        model_file='_ReLU128_5.7'
        holdout_type = 'swap_holdouts'
        seeds = [0, 1, 2, 3, 4]
        from utils import all_models
        #to_check = list(itertools.product(['gptNet', 'sbertNet', 'bertNet'], seeds, training_lists_dict['swap_holdouts']))
        to_check = list(itertools.product(['gptNet_tuned', 'sbertNet_tuned', 'bertNet_tuned'], seeds, [['Multitask']]))


        test_dict = {}
        to_retrain = []
        for config in to_check: 
            model_params_key, seed_num, holdouts = config
            torch.manual_seed(seed_num)
            holdout_file = get_holdout_file(holdouts) 
            print(config)
            model, _, _, _ = config_model_training(model_params_key)
            model.to(device)
            model.set_seed(seed_num)
            try: 
                version_str=''
                model.load_model(model_file+'/'+holdout_type+'/'+holdout_file)
            except: 
                version_str='original'
                print('testing original model')
                model.load_model('_ReLU128_5.7/'+holdout_type+'/'+holdout_file)

            perf = get_model_performance(model, 3).round(2)
            try:
                holdout_indices = [Task.TASK_LIST.index(holdouts[0]), Task.TASK_LIST.index(holdouts[1])]
            except ValueError: 
                holdout_indices=[]
            check_array=np.delete(perf, holdout_indices)
            passed_test = all(check_array>=0.95)
            print(perf)
            print(passed_test)
            test_dict[version_str+str(model_params_key)+ str(seed_num)+str(holdouts)] = perf
            if not passed_test: 
                to_retrain.append((version_str, model_params_key, seed_num, holdouts))
        print(test_dict)


    if train_mode == 'train_Go_Only': 
        holdout_type = 'swap_holdouts'
        seeds = [2]
        to_train = list(itertools.product(seeds, ['simpleNet']))

        for cur_train in to_train:      
            seed_num, model_params_key = cur_train
            torch.manual_seed(seed_num)

            # try: 
            #     pickle.load(open(model_file+'/'+holdout_type+'/'+holdout_file+'/'+model_params_key+'/seed'+str(seed_num)+'_training_loss', 'rb'))
            #     print(model_params_key+'_seed'+str(seed_num)+' already trained for ' + holdout_file)

            #     last_holdouts = holdouts
            #     continue
            # except FileNotFoundError:
            print(cur_train)
            data = TaskDataSet(data_folder = model_file+'/training_data', batch_len=128, num_batches=500, task_ratio_dict={'MultiDM':1})
            data.data_to_device(device)

            model, _, _, epoch = config_model_training(model_params_key)
            model.set_seed(seed_num)

            opt, _ = init_optimizer(model, 0.001, [])
            sch = optim.lr_scheduler.ExponentialLR(opt, 0.95)
            #train 
            train_model(model, data, 3, opt, sch)
            #save
            model.save_model(model_file+'/'+holdout_type+'/MultiDM_Only')
            model.save_training_data(model_file+'/'+holdout_type+'/MultiDM_Only')
