
import numpy as np

from utils import isCorrect, get_holdout_file, training_lists_dict, all_swaps, train_instruct_dict
from model_trainer import masked_MSE_Loss, config_model, ALL_MODEL_PARAMS
from data import TaskDataSet
from task import Task

import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim


import pickle
import itertools
import sys

device = torch.device(0)

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

def test_context(model, holdouts_test, foldername, repeats=5, holdout_type = 'swap_holdouts', save=False): 
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
            data = TaskDataSet(batch_len=128, num_batches=500, task_ratio_dict={holdout:1})
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

    filename=foldername+'/swap_holdouts/'+task_file + '/'+model.model_name+'/contexts/'+model.__seed_num_str__

    for i, task in enumerate(Task.TASK_LIST):     

        context = nn.Parameter(torch.randn((num_contexts, context_dim), device=device))

        opt= optim.Adam([context], lr=8e-2, weight_decay=0.0)
        #sch = optim.lr_scheduler.ExponentialLR(opt, 0.99)
        sch = optim.lr_scheduler.MultiStepLR(opt, [10, 12, 14], 0.1)


        streamer = TaskDataSet(batch_len = num_contexts, num_batches = 250, task_ratio_dict={task:1})
        streamer.data_to_device(device)

        contexts =train_context(model, streamer, 16, opt, sch, context, self_supervised)
        pickle.dump(contexts, open(filename+'/'+task+supervised_str+'_context_vecs'+str(context_dim), 'wb'))
        pickle.dump(model._correct_data_dict, open(filename+'/'+task+supervised_str+'_context_correct_data'+str(context_dim), 'wb'))
        pickle.dump(model._loss_data_dict, open(filename+'/'+task+supervised_str+'_context_loss_data'+str(context_dim), 'wb'))
        print('saved: '+filename)
        model.reset_training_data()

if __name__ == "__main__":
    model_file = '_ReLU128_4.11'

    train_mode = str(sys.argv[1])

    if train_mode == 'train_contexts': 
        seeds = [0, 1, 2, 3, 4]
        to_train = list(itertools.product(seeds, ['sbertNet_tuned', 'simpleNet'], ['Multitask']+all_swaps))
        print(to_train)
        for config in to_train: 
            seed_num, model_params_key, task_file = config 
            model, _, _, _ = config_model(model_params_key)
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
                    model, _, _, _ = config_model(model_params_key)
                    model.set_seed(seed_num)
                    model.to(device)
                    test_context(model, holdouts, foldername= model_file, holdout_type = holdout_type, save=True)
                  
