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

class ContextTrainer(): 
    def __init__(self, model, context_dim, load_file): 
        self.context_dim  = context_dim
        self.load_file = load_file
        self.foldername = '_ReLU128_4.11/swap_holdouts/'+load_file
        model.load_model(self.load_file)
        model.to(device)

        self.supervised_str = None

    def check_trained(self, task):
        filename = self.foldername + '/'+model.model_name+'/contexts/'+model.__seed_num_str__
        pickle.load(open(filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim), 'rb'))
        print(filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim))
        print('contexts already trained')
    
    def save_contexts(self, contexts, task): 
        filename = self.foldername + '/'+model.model_name+'/contexts/'+model.__seed_num_str__
        pickle.dump(contexts, open(filename+task+self.supervised_str+'_context_vecs'+str(self.context_dim), 'wb'))
        pickle.dump(model._correct_data_dict, open(filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim), 'wb'))
        pickle.dump(model._loss_data_dict, open(filename+task+self.supervised_str+'_context_loss_data'+str(self.context_dim), 'wb'))
        print('saved: '+filename+' '+task)

    def init_trainer_opts(self, context, lr, gamma):
        opt= optim.Adam([context], lr=lr, weight_decay=0.0)
        sch = optim.lr_scheduler.ExponentialLR(opt, gamma)
        return opt, sch

    def get_all_contexts(self, model, num_contexts, self_supervised):
        inspection_list = []

        if not self_supervised: self.supervised_str = '_supervised'
        else: self.supervised_str = ''

        for task in Task.TASK_LIST:     
            try:
                self.check_trained(task)
                continue
            except FileNotFoundError: 
                context = nn.Parameter(torch.randn((num_contexts, self.context_dim), device=device))
                opt, sch = self.init_trainer_opts(context, 8e-2, 0.99)
                streamer = TaskDataSet(batch_len = num_contexts, num_batches = 350, task_ratio_dict={task:1})
                contexts, is_trained = train_context(model, streamer, 30, opt, sch, context, self_supervised)

                if is_trained:
                    self.save_contexts(contexts, task)
                else:
                    inspection_list.append(task)

                model.reset_training_data()
                print(inspection_list)
        return inspection_list

    def train_context(self, model, data_streamer, epochs, opt, sch, context): 
        model.freeze_weights()
        model.eval()
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt,milestones=[epochs-2, epochs-1], gamma=0.1)

        for i in range(epochs): 
            print('epoch', i)
            data_streamer.shuffle_stream_order()
            for j, data in enumerate(data_streamer.stream_batch()): 

                ins, tar, mask, tar_dir, task_type = data

                opt.zero_grad()
        
                if self.supervised_str=='supervised':
                    target = tar
                else: 
                    task_info = model.get_task_info(ins.shape[0], task_type)
                    target, _ = model(task_info, ins.to(device))

                if self.context_dim == 768: 
                    context = model.langModel.proj_out(context.float())            
                out, _ = super(type(model), model).forward(context, ins.to(device))
                loss = masked_MSE_Loss(out, target.to(device), mask.to(device)) 
                loss.backward()

                opt.step()

                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                model._loss_data_dict[task_type].append(loss.item())
                model._correct_data_dict[task_type].append(frac_correct)
                if j%50 == 0:
                    print(task_type)
                    print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                    print('Frac Correct ' + str(frac_correct) + '\n')
            
            if i>5 and model.check_model_training(0.91, 5):
                return context.squeeze().detach().cpu().numpy(), True
            if sch is not None:                
                sch.step()
            step_scheduler.step()
        is_trained = model.check_model_training(0.91, 5)
        return context.squeeze().detach().cpu().numpy(), is_trained



def train_context(model, data_streamer, epochs, opt, sch, context, self_supervised): 
    model.freeze_weights()
    model.eval()
    step_scheduler = optim.lr_scheduler.MultiStepLR(opt,milestones=[epochs-2, epochs-1], gamma=0.1)

    for i in range(epochs): 
        print('epoch', i)
        data_streamer.shuffle_stream_order()
        for j, data in enumerate(data_streamer.stream_batch()): 

            ins, tar, mask, tar_dir, task_type = data

            opt.zero_grad()
    
            if self_supervised:
                task_info = model.get_task_info(ins.shape[0], task_type)
                target, _ = model(task_info, ins.to(device))
            else: 
                target = tar

            proj = model.langModel.proj_out(context.float())            
            out, _ = super(type(model), model).forward(proj, ins.to(device))
            loss = masked_MSE_Loss(out, target.to(device), mask.to(device)) 
            loss.backward()

            opt.step()

            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
            model._loss_data_dict[task_type].append(loss.item())
            model._correct_data_dict[task_type].append(frac_correct)
            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                print('Frac Correct ' + str(frac_correct) + '\n')
        
        if i>5 and model.check_model_training(0.91, 5):
            return context.squeeze().detach().cpu().numpy(), True
        if sch is not None:                
            sch.step()
        step_scheduler.step()
    is_trained = model.check_model_training(0.91, 5)
    return context.squeeze().detach().cpu().numpy(), is_trained

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

def get_all_contexts(model, num_contexts, target_embedding_layer, task_file, self_supervised, tasks_to_train=Task.TASK_LIST, foldername='_ReLU128_4.11'):
    try: 
        if target_embedding_layer.isnumeric(): 
            context_dim = model.langModel.intermediate_lang_dim
        elif target_embedding_layer =='full': 
            context_dim = model.langModel.out_dim
    except: 
        context_dim = 20

    inspection_list = []

    supervised_str = ''
    if not self_supervised:
        supervised_str = '_supervised'
    

    model.load_model(foldername+'/swap_holdouts/'+task_file)
    model.to(device)

    filename=foldername+'/swap_holdouts/'+task_file + '/'+model.model_name+'/contexts/'+model.__seed_num_str__

    for task in tasks_to_train:     
        try:
            pickle.load(open(filename+task+supervised_str+'_context_correct_data'+str(context_dim), 'rb'))
            print(filename+task+supervised_str+'_context_correct_data'+str(context_dim))
            print('contexts already trained')
            continue
        except FileNotFoundError: 
            context = nn.Parameter(torch.randn((num_contexts, context_dim), device=device))

            opt= optim.Adam([context], lr=8*1e-2, weight_decay=0.0)
            sch = optim.lr_scheduler.ExponentialLR(opt, 0.99)

            streamer = TaskDataSet(batch_len = num_contexts, num_batches = 350, task_ratio_dict={task:1})

            contexts, is_trained = train_context(model, streamer, 30, opt, sch, context, self_supervised)
            if is_trained:
                pickle.dump(contexts, open(filename+task+supervised_str+'_context_vecs'+str(context_dim), 'wb'))
                pickle.dump(model._correct_data_dict, open(filename+task+supervised_str+'_context_correct_data'+str(context_dim), 'wb'))
                pickle.dump(model._loss_data_dict, open(filename+task+supervised_str+'_context_loss_data'+str(context_dim), 'wb'))
                print('saved: '+filename+' '+task)
            else:
                inspection_list.append(task)
            model.reset_training_data()
            print(inspection_list)
    return inspection_list

def get_all_contexts_set(to_get):
    inspection_dict = {}
    for config in to_get: 
        model_params_key, seed_num, tasks = config 
        task_file = get_holdout_file(tasks)
        model = config_model(model_params_key)
        torch.manual_seed(seed_num)
        model.set_seed(seed_num)
        trainer=ContextTrainer(model, 768, task_file)
        for self_supervised in [False, True]:
            supervised_str = ''
            if not self_supervised:
                supervised_str = '_supervised'

            print(str(config) + supervised_str) 
            inspection_list = trainer.get_all_contexts(model, 128, self_supervised)
            inspection_dict[model.model_name+model.__seed_num_str__+supervised_str] = inspection_list
    return inspection_dict

if __name__ == "__main__":
    model_file = '_ReLU128_4.11'

    #train_mode = str(sys.argv[1])
    train_mode = 'train_contexts'
    if train_mode == 'train_contexts': 
        holdout_type = 'swap_holdouts'
        seeds = [0, 1, 2, 3, 4]
        to_train_contexts = list(itertools.product(['sbertNet_tuned'], [0, 1], training_lists_dict['swap_holdouts']))
        print(to_train_contexts)
        inspection_dict = get_all_contexts_set(to_train_contexts)
        print(inspection_dict)

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
                  
