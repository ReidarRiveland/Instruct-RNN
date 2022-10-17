import os
from pip import main
import torch
import numpy as np
import pickle
import itertools
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.tasks import SWAPS_DICT, TASK_LIST, construct_trials
from instructRNN.tasks.task_criteria import isCorrect

def get_decoded_set(foldername, model_name, seeds=range(5), from_contexts=False, sm_holdouts=False, decoder_holdouts=False, with_dropout=False, save=False): 
    if sm_holdouts: assert 'swap' in foldername
    
    if sm_holdouts:
        exp_dict = SWAPS_DICT
        sm_str = 'holdout'
    else: 
        exp_dict = {'Multitask': TASK_LIST}
        sm_str = 'multi'

    full_shallow_set = {}
    full_rich_set = {}
    full_confuse_mat = np.full((len(seeds), len(TASK_LIST), len(TASK_LIST)+1), np.nan)
    sm_model = make_default_model(model_name)
    if with_dropout:
        rnn_decoder = DecoderRNN(128, drop_p=0.0)
    else: 
        rnn_decoder = DecoderRNN(256, drop_p=0.0)

    rnn_decoder.eval()
    encoder = EncoderDecoder(sm_model, rnn_decoder)
    full_all_contexts = np.full((len(seeds), len(TASK_LIST), 64, sm_model.langModel.LM_intermediate_lang_dim), np.nan)
    encoder.to(0)

    for i, seed in enumerate(seeds): 
        shallow_decoded_set = {}
        rich_decoded_set = {}
        confuse_mat = np.full((len(TASK_LIST), len(TASK_LIST)+1), np.nan)
        all_contexts = np.full(full_all_contexts.shape[1:], np.nan)
        for swap_label, swap_tasks in exp_dict.items(): 
            print(swap_label)
            encoder.load_model_componenets(foldername+'/'+swap_label+'/'+model_name+'/', seed, swap_tasks, with_holdout=decoder_holdouts, with_dropout=with_dropout)
            for task in swap_tasks:
                all_contexts[TASK_LIST.index(task), ...] = encoder.contexts[TASK_LIST.index(task), ...]
            _shallow, _rich, _confuse_mat = encoder.decode_set(50, 1, tasks=swap_tasks, from_contexts=from_contexts)
            shallow_decoded_set = {**_shallow, **shallow_decoded_set}
            rich_decoded_set = {**_rich, **rich_decoded_set}
            confuse_mat[[TASK_LIST.index(task) for task in swap_tasks], :] = _confuse_mat

        full_shallow_set[seed] = shallow_decoded_set
        full_rich_set[seed] = rich_decoded_set
        full_confuse_mat[i, ...] = confuse_mat
        full_all_contexts[i, ...] = all_contexts

    if save:
        file_path = foldername+'/decoder_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        if decoder_holdouts: decoder_str='holdout'
        else: decoder_str = 'multi'

        pickle.dump(full_rich_set, open(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_instructs_dict', 'wb'))
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_confuse_mat.npy',full_confuse_mat)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_contexts.npy', full_all_contexts)

    return full_rich_set, full_shallow_set, full_confuse_mat, full_all_contexts

def _test_partner_model(partner_model, decoded_dict, num_trials, contexts, tasks): 
    partner_model.eval()
    instruct_perf_array = np.full((len(tasks)), np.nan)
    other_perf_array = np.full((len(tasks)), np.nan)
    contexts_perf_array = np.full((len(tasks)), np.nan)

    with torch.no_grad():
        for i, task in enumerate(tasks):
            print(task)
            ins, targets, _, target_dirs, _ = construct_trials(task, num_trials)
        
            if contexts is not None: 
                task_info = torch.Tensor(contexts[TASK_LIST.index(task), 0:num_trials, ...]).to(partner_model.__device__)
                out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), context = task_info)
                contexts_perf_array[i] = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

            try:
                task_info = list(np.random.choice(decoded_dict[task]['other'], num_trials))
                out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), instruction = task_info)
                other_perf_array[i] = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
            except ValueError:
                pass

            task_instructs = list(itertools.chain.from_iterable([value for value in decoded_dict[task].values()]))
            task_info = list(np.random.choice(task_instructs, num_trials))
            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), instruction = task_info)

            instruct_perf_array[i] = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

    return instruct_perf_array, other_perf_array, contexts_perf_array


def test_partner_model(load_folder, model_name, decoded_dict, contexts = None, num_trials=50, tasks = TASK_LIST, partner_seed=3): 
    partner_model = make_default_model(model_name)
    partner_model.load_state_dict(torch.load(load_folder+'/Multitask/'+model_name+'/'+model_name+'_seed'+str(partner_seed)+'.pt'), strict=False)
    return _test_partner_model(partner_model, decoded_dict, num_trials, contexts, tasks)

def test_multi_partner_perf(foldername, model_name, num_trials=50, tasks = TASK_LIST, partner_seeds=range(5), decoder_holdouts=False, sm_holdouts= False, save=False):
    file_path = foldername+'/decoder_perf/'+model_name
    
    if decoder_holdouts: decoder_str='holdout'
    else: decoder_str = 'multi'
    
    if sm_holdouts: sm_str = 'holdout'
    else: sm_str = 'multi'

    full_decoded_dict = pickle.load(open(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_instructs_dict', 'rb'))
    contexts = np.load(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_contexts.npy')

    instruct_perf_array = np.full((len(full_decoded_dict), len(partner_seeds), len(TASK_LIST)), np.nan)
    other_perf_array = np.full((len(full_decoded_dict), len(partner_seeds), len(TASK_LIST)), np.nan)
    contexts_perf_array = np.full((len(full_decoded_dict), len(partner_seeds),len(TASK_LIST)), np.nan)
    for i, decoded_set in enumerate(full_decoded_dict.values()):
        for j, partner_seed in enumerate(partner_seeds):
            print('processing seed ' + str(j))
            instruct_perf, other_perf, contexts_perf = test_partner_model(foldername, model_name, decoded_set, contexts=contexts[i, ...], \
                                                    num_trials=num_trials, tasks=tasks, partner_seed=partner_seed)
            instruct_perf_array[i,j, ...] = instruct_perf
            other_perf_array[i,j, ...] = other_perf
            contexts_perf_array[i, j, ...] = contexts_perf

    if save:
        file_path = foldername+'/decoder_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_decoder_multi_partner_all_perf',instruct_perf_array)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_decoder_multi_partner_other_perf',other_perf_array)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_decoder_multi_partner_context_perf',contexts_perf_array)
    return instruct_perf_array, other_perf_array, contexts_perf_array

def test_holdout_partner_perf(foldername, model_name, num_trials = 50, partner_seeds = range(5), decoder_holdouts=False, sm_holdouts= False, save=False):
    file_path = foldername+'/decoder_perf/'+model_name
    if decoder_holdouts: decoder_str='holdout'
    else: decoder_str = 'multi'
    
    if sm_holdouts: sm_str = 'holdout'
    else: sm_str = 'multi'

    full_decoded_dict = pickle.load(open(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_instructs_dict', 'rb'))
    contexts = np.load(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_contexts.npy')

    instruct_perf_array = np.full((len(partner_seeds), len(full_decoded_dict), len(TASK_LIST)), np.nan)
    other_perf_array = np.full((len(partner_seeds), len(full_decoded_dict),  len(TASK_LIST)), np.nan)
    contexts_perf_array = np.full(( len(partner_seeds), len(full_decoded_dict), len(TASK_LIST)), np.nan)
    
    partner_model = make_default_model(model_name)
    for i, seed in enumerate(partner_seeds):
        for holdout_file, holdouts in SWAPS_DICT.items():
            print('processing '+ holdout_file)
            partner_model.load_model('7.20models/swap_holdouts/'+holdout_file+'/'+partner_model.model_name, suffix='_seed'+str(seed))
            for j, item in enumerate(full_decoded_dict.items()): 
                decoder_seed, decoded_set = item
                if decoder_seed == seed: 
                    continue
                instruct_perf, other_perf, contexts_perf = _test_partner_model(partner_model, decoded_set, contexts=contexts[j, ...], \
                                                num_trials=num_trials, tasks=holdouts)

                instruct_perf_array[i, j, [TASK_LIST.index(task) for task in holdouts]] = instruct_perf
                other_perf_array[i, j, [TASK_LIST.index(task) for task in holdouts]] = other_perf
                contexts_perf_array[i, j, [TASK_LIST.index(task) for task in holdouts]] = contexts_perf

    if save:
        file_path = foldername+'/decoder_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_decoder_holdout_partner_all_perf',instruct_perf_array)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_decoder_holdout_partner_other_perf',other_perf_array)
        np.save(file_path+'/sm_'+sm_str+'decoder_'+decoder_str+'_decoder_holdout_partner_context_perf',contexts_perf_array)

    return instruct_perf_array, other_perf_array, contexts_perf_array


def decoder_pipline(foldername, model_name, decoder_holdout=False, sm_holdout=False, seeds=range(5)): 
    get_decoded_set(foldername, model_name, seeds=seeds, from_contexts=True, decoder_holdouts=decoder_holdout, sm_holdouts=sm_holdout, save=True)
    test_multi_partner_perf(foldername, model_name, seeds=seeds, decoder_holdouts=decoder_holdout, sm_holdouts=sm_holdout, save=True)
    test_holdout_partner_perf(foldername, model_name, seeds=seeds, decoder_holdouts=decoder_holdout, sm_holdouts=sm_holdout, save=True)

