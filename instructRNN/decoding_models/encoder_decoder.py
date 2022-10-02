
import torch
import torch.nn as nn 
import numpy as np
import pickle
import itertools

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from instructRNN.tasks.tasks import TASK_LIST, construct_trials
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import inv_train_instruct_dict, get_instructions

class EncoderDecoder(nn.Module): 
    def __init__(self, sm_model, decoder): 
        super(EncoderDecoder, self).__init__()
        self.sm_model = sm_model
        self.decoder = decoder
        self.contexts = None
        self.sm_model.eval()
        self.decoder.eval()

    def load_model_componenets(self, load_folder, seed):
        self.sm_model.load_model(load_folder, suffix='_seed'+str(seed))
        self.decoder.load_model(load_folder, suffix='_seed'+str(seed))
        self.init_context_set(load_folder+'/'+self.sm_model.model_name, seed)

    def init_context_set(self, file_name, seed, verbose=False):
        context_dim = self.sm_model.langModel.LM_intermediate_lang_dim
        all_contexts = np.empty((len(TASK_LIST), 100, context_dim))
        for i, task in enumerate(TASK_LIST):
            try: 
                filename = file_name+'/contexts/seed'+str(seed)+'_'+task+'_context_vecs'+str(context_dim)
                task_contexts = pickle.load(open(filename, 'rb'))
                all_contexts[i, ...]=task_contexts[:100, :]
            except FileNotFoundError: 
                if verbose:
                    print(filename)
                    print('no contexts for '+task+' for model file')

        self.contexts = all_contexts

    def to(self, cuda_device): 
        self.sm_model.to(cuda_device)
        self.decoder.to(cuda_device)

    def decode_set(self, num_trials, num_repeats = 1, from_contexts=False, tasks=TASK_LIST, t=120): 
        decoded_set = {}
        confusion_mat = np.zeros((len(tasks), len(TASK_LIST)+1))
        for _ in range(num_repeats): 
            for i, task in enumerate(tasks): 
                tasks_decoded = defaultdict(list)

                ins, _, _, _, _ = construct_trials(task, num_trials)

                if from_contexts: 
                    task_info = torch.Tensor(self.contexts[i, ...]).to(self.sm_model.__device__)
                    _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), context=task_info)
                else: 
                    task_info = get_instructions(num_trials, task, None)
                    _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), task_info)
                
                decoded_sentences = self.decoder.decode_sentence(sm_hidden[:,0:t, :]) 

                for instruct in decoded_sentences:
                    try: 
                        decoded_task = inv_train_instruct_dict[instruct]
                        tasks_decoded[decoded_task].append(instruct)
                        confusion_mat[i, TASK_LIST.index(decoded_task)] += 1
                    except KeyError:
                        tasks_decoded['other'].append(instruct)
                        confusion_mat[i, -1] += 1
                decoded_set[task] = tasks_decoded

        return decoded_set, confusion_mat
    
    def plot_confuse_mat(self, num_trials, num_repeats, tasks=TASK_LIST, from_contexts=False, confusion_mat=None, fmt='g'): 
        if confusion_mat is None:
            decoded_set, confusion_mat = self.decode_set(num_trials, tasks=tasks, num_repeats = num_repeats, from_contexts=from_contexts)
        res=sns.heatmap(confusion_mat, linewidths=0.5, linecolor='black', mask=confusion_mat == 0, xticklabels=TASK_LIST+['other'], yticklabels=TASK_LIST, annot=True, cmap='Blues', fmt=fmt, cbar=False)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
        plt.show()
        return decoded_set

    def test_partner_model(self, partner_model, num_repeats=1, from_context=True, tasks=TASK_LIST, decoded_dict=None, mode='instructions'): 
        partner_model.eval()
        if decoded_dict is None: 
            decoded_dict, _ = self.decode_set(100, from_contexts=from_context)

        with torch.no_grad():
            perf_array = np.full((num_repeats, len(tasks)), np.nan)
            for i in range(num_repeats): 
                for j, task in enumerate(tasks):
                    print(task)
                    ins, targets, _, target_dirs, _ = construct_trials(task, 100)
                    if mode == 'others': 
                        try:
                            task_info = list(np.random.choice(decoded_dict[task]['other'], 100))
                            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), task_info)
                        except ValueError:
                            continue
                    elif mode == 'instructions':
                        task_info = list(itertools.chain.from_iterable([value for value in decoded_dict[task].values()]))
                        out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), task_info)
                    elif mode == 'context':
                        task_index = TASK_LIST.index(task)
                        task_info = self.contexts[task_index, ...]
                        out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), context=torch.Tensor(task_info).to(partner_model.__device__))                        

                    task_perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

                    perf_array[i, j] = task_perf
        return perf_array

    def _partner_model_over_t(self, partner_model, task, from_contexts=True, t_set = [120]): 
        ins, targets, _, target_dirs, _ = construct_trials(task, 128)

        if from_contexts: 
            task_index = TASK_LIST.index(task)
            task_info = torch.Tensor(self.contexts[task_index, :128, :]).to(self.sm_model.__device__)
            _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), context=task_info)
        else: 
            task_info = self.sm_model.get_task_info(128, task)
            _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), task_info)

        instruct_list = []
        task_perf_list = []
        for t in t_set: 
            instruct = list(self.decoder.decode_sentence((sm_hidden[:, 0:t,:])))
            instruct_list.append(instruct)
            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), instruct)
            task_perf_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
        
        return task_perf_list, instruct_list

