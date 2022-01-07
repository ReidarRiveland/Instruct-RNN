import numpy as np

from utils import isCorrect, get_holdout_file, training_lists_dict, all_swaps, train_instruct_dict
from model_trainer import masked_MSE_Loss, config_model, ALL_MODEL_PARAMS
from data import TaskDataSet
from task import Task

import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim

from jit_GRU import CustomGRU

import pickle
import itertools
import sys
from model_analysis import get_instruct_reps
from collections import defaultdict

device = torch.device(0)

class ContextTrainer(): 
    def __init__(self, model, context_dim, load_file): 
        self.context_dim  = context_dim
        self.load_file = load_file
        self.foldername = '_ReLU128_4.11/swap_holdouts/'+load_file
        self.model = model
        self.model.load_model(self.foldername)
        self.model.to(device)
        self.filename = self.foldername + '/'+self.model.model_name+'/contexts/'+self.model.__seed_num_str__+'_'

        self.supervised_str = None

    def check_trained(self, task):
        pickle.load(open(self.filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim), 'rb'))
        print(self.filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim))
        print('contexts already trained')
    
    def save_contexts(self, contexts, task): 
        pickle.dump(contexts, open(self.filename+task+self.supervised_str+'_context_vecs'+str(self.context_dim), 'wb'))
        pickle.dump(self.model._correct_data_dict, open(self.filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim), 'wb'))
        pickle.dump(self.model._loss_data_dict, open(self.filename+task+self.supervised_str+'_context_loss_data'+str(self.context_dim), 'wb'))
        print('saved: '+self.filename+' '+task)

    def init_trainer_opts(self, context, lr, gamma):
        opt= optim.Adam([context], lr=lr, weight_decay=0.0)
        sch = optim.lr_scheduler.ExponentialLR(opt, gamma)
        return opt, sch

    def train_context(self, data_streamer, epochs, opt, sch, context): 
        self.model.freeze_weights()
        self.model.eval()
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
                    task_info = self.model.get_task_info(ins.shape[0], task_type)
                    target, _ = self.model(task_info, ins.to(device))

                if self.context_dim == 768: 
                    projected = self.model.langModel.proj_out(context.float())           
                else:
                    projected = context 
                out, _ = super(type(self.model), self.model).forward(projected, ins.to(device))
                loss = masked_MSE_Loss(out, target.to(device), mask.to(device)) 
                loss.backward()

                opt.step()

                frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
                self.model._loss_data_dict[task_type].append(loss.item())
                self.model._correct_data_dict[task_type].append(frac_correct)
                if j%50 == 0:
                    print(task_type)
                    print(j, ':', self.model.model_name, ":", "{:.2e}".format(loss.item()))
                    print('Frac Correct ' + str(frac_correct) + '\n')
            
            if i>5 and self.model.check_model_training(0.92, 3):
                return context.squeeze().detach().cpu().numpy(), True
            if sch is not None:                
                sch.step()
            step_scheduler.step()
        is_trained = self.model.check_model_training(0.92, 3)
        return context.squeeze().detach().cpu().numpy(), is_trained


    def get_all_contexts(self, num_contexts):
        inspection_list = []
        for task in Task.TASK_LIST:     
            try:
                pickle.load(open(self.filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim), 'rb'))
                print(self.filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim))
                print('contexts already trained')
                continue
            except FileNotFoundError: 
                context = nn.Parameter(torch.randn((num_contexts, self.context_dim), device=device))

                opt= optim.Adam([context], lr=5*1e-3, weight_decay=0.0)
                sch = optim.lr_scheduler.ExponentialLR(opt, 0.98)

                streamer = TaskDataSet(batch_len = num_contexts, num_batches = 800, task_ratio_dict={task:1})

                contexts, is_trained = self.train_context(streamer, 100, opt, sch, context)
                if is_trained:
                    pickle.dump(contexts, open(self.filename+task+self.supervised_str+'_context_vecs'+str(self.context_dim), 'wb'))
                    pickle.dump(self.model._correct_data_dict, open(self.filename+task+self.supervised_str+'_context_correct_data'+str(self.context_dim), 'wb'))
                    pickle.dump(self.model._loss_data_dict, open(self.filename+task+self.supervised_str+'_context_loss_data'+str(self.context_dim), 'wb'))
                    print('saved: '+self.filename+' '+task)
                else:
                    inspection_list.append(task)
                self.model.reset_training_data()
                print(inspection_list)
        return inspection_list

class ContextNetwork(nn.Module): 
    def __init__(self, hidden_size, sm_model, depth):
        super(ContextNetwork, self).__init__()
        if depth.isnumeric(): 
            self.target_dim = 768
        elif depth == 'full': 
            self.target_dim = 20
        self._loss_data_dict = defaultdict(list)
        self._correct_data_dict = defaultdict(list)
        self.sm_model = sm_model
        self.sm_model.freeze_weights()
        self.sm_model.eval()
        self.hidden_size = hidden_size
        self.gru = CustomGRU(128+65, hidden_size, 1, activ_func=torch.relu)
        self.out = nn.Linear(hidden_size, self.target_dim)

    def forward(self, s_input, sm_hidden, h0): 
        ins = torch.cat((s_input, sm_hidden))
        rnn_hid, _ = self.gru(ins, h0)
        outs = self.out(rnn_hid)
        return outs, rnn_hid
    
    def train_context(self, data_streamer, epochs, opt, sch, context): 
        criterion = nn.MSELoss(reduction='mean')
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt,milestones=[epochs-2, epochs-1], gamma=0.1)
        h0 = torch.full((1, data_streamer.batch_len, self.hidden_size), 0.1, device=device)
        for i in range(epochs): 
            print('epoch', i)
            data_streamer.shuffle_stream_order()
            for j, data in enumerate(data_streamer.stream_batch()): 
                ins, tar, _, tar_dir, task_type = data
                opt.zero_grad()
        
                task_info = self.sm_model.get_task_info(ins.shape[0], task_type)
                instruct_embedded = self.langModel(task_info)
                _, sm_rnn_hid = self.sm_model.forward(instruct_embedded, ins)
                outs, _ = self.forward(ins, sm_rnn_hid, h0)

                loss = criterion(outs, instruct_embedded) 
                loss.backward()

                opt.step()
                self.self._loss_data_dict[task_type].append(loss.item())

                if j%50 == 0:
                    frac_correct = self.test_context(ins, outs, tar, tar_dir)
                    self.sm_model._correct_data_dict[task_type].append(frac_correct)
                    self._correct_data_dict[task_type].append(frac_correct)

                    print(task_type)
                    print(j, ':', self.model.model_name, ":", "{:.2e}".format(loss.item()))
                    print('Frac Correct ' + str(frac_correct) + '\n')
            
            if sch is not None: sch.step()
            step_scheduler.step()

            if i>5 and self.model.check_model_training(0.92, 3):
                return context.squeeze().detach().cpu().numpy(), True

        is_trained = self.model.check_model_training(0.92, 3)
        return context.squeeze().detach().cpu().numpy(), is_trained

    def test_context(self, ins, context, tar, tar_dir): 
        if self.target_dim == 768: 
            projected = self.model.langModel.proj_out(context.float())           
        else:
            projected = context 
        out, _ = super(type(self.sm_model), self.sm_model).forward(projected, ins.to(device))
        frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
        return frac_correct


task_file = 'Go_Anti_DM'
model = config_model('sbertNet_tuned')
model.set_seed(0)
model.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

contextNet = ContextNetwork(128, model, '12')

def get_all_contexts_set(to_get):
    inspection_dict = {}
    for config in to_get: 
        model_params_key, seed_num, tasks = config 
        task_file = get_holdout_file(tasks)
        model = config_model(model_params_key)
        torch.manual_seed(seed_num)
        model.set_seed(seed_num)
        trainer=ContextTrainer(model, 20, task_file)
        for self_supervised in [False, True]:
            trainer.supervised_str = ''
            if not self_supervised:
                trainer.supervised_str = '_supervised'

            print(str(config) + trainer.supervised_str) 
            inspection_list = trainer.get_all_contexts(128)
            inspection_dict[model.model_name+model.__seed_num_str__+trainer.supervised_str] = inspection_list
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
                  
