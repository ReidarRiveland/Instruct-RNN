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
from instructRNN.decoding_models.decoder_models import DecoderRNN


class EncoderDecoder(nn.Module): 
    def __init__(self, sm_model, decoder): 
        super(EncoderDecoder, self).__init__()
        self.sm_model = sm_model
        self.decoder = decoder
        self.contexts = None
        self.sm_model.eval()
        self.decoder.eval()

        if hasattr(decoder, 'tokenizer'):
            decoder.tokenizer.index2word[2] = '<|endoftext|>'

    def load_model_componenets(self, load_folder, seed, tasks=TASK_LIST, with_holdout=False, decode_embeddings=False):
        suffix = '_seed'+str(seed)

        if with_holdout: 
            holdout_suffix = '_wHoldout'
        else: 
            holdout_suffix = ''

        self.sm_model.load_model('7.20models/'+'/'.join(load_folder.split('/')[1:]), suffix=suffix)
        self.decoder.load_model(load_folder, suffix=suffix+holdout_suffix)
        self.init_context_set('7.20models/'+'/'.join(load_folder.split('/')[1:]), seed, tasks)

    def init_context_set(self, file_name, seed, tasks, verbose=True):
        context_dim = 64
        all_contexts = np.full((len(TASK_LIST), 25, context_dim), np.nan)
        for i, task in enumerate(tasks):
            filename = file_name+'contexts/seed'+str(seed)+'_'+task+'test_context_vecs'+str(context_dim)
            is_trained_filename = file_name+'contexts/seed'+str(seed)+'_'+task+'test_context_vecs'+str(context_dim)

            task_contexts = pickle.load(open(filename, 'rb'))
            all_contexts[TASK_LIST.index(task), ...]=task_contexts[:25, :]

        self.contexts = all_contexts

    def to(self, cuda_device): 
        self.sm_model.to(cuda_device)
        self.decoder.to(cuda_device)

    def decode_set(self, num_trials, num_repeats = 1, from_contexts=False, tasks=TASK_LIST): 
        shallow_decoded_set = {}
        rich_decoded_set = {}
        confusion_mat = np.zeros((len(tasks), len(TASK_LIST)+1))
        with torch.no_grad():
            for _ in range(num_repeats): 
                for i, task in enumerate(tasks): 
                    tasks_decoded = defaultdict(list)

                    ins, _, _, _, _ = construct_trials(task, num_trials, noise=0)

                    if from_contexts and self.decoder.decode_embeddings: 
                        sm_hidden = torch.Tensor(self.contexts[TASK_LIST.index(task), :25, :]).repeat(2,1).to(self.sm_model.__device__)
                    elif from_contexts: 
                        task_info = torch.Tensor(self.contexts[TASK_LIST.index(task), :25, :]).repeat(2,1).to(self.sm_model.__device__)
                        _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), info_embedded=task_info)
                    else: 
                        task_info = get_instructions(num_trials, task, None)
                        _, sm_hidden = self.sm_model(torch.Tensor(ins).to(self.sm_model.__device__), task_info)
                    
                    decoded_sentences = self.decoder.decode_sentence(sm_hidden) 
                    shallow_decoded_set[task] = decoded_sentences
                    
                    if isinstance(self.decoder, DecoderRNN):
                        for instruct in decoded_sentences:
                            try: 
                                decoded_task = inv_train_instruct_dict[instruct]
                                tasks_decoded[decoded_task].append(instruct)
                                confusion_mat[i, TASK_LIST.index(decoded_task)] += 1
                            except KeyError:
                                tasks_decoded['other'].append(instruct)
                                confusion_mat[i, -1] += 1
                        rich_decoded_set[task] = tasks_decoded
        return shallow_decoded_set, rich_decoded_set, confusion_mat

