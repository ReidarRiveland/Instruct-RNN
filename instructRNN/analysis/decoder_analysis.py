from multiprocessing import context
import torch
import numpy as np
import itertools
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.tasks import SWAPS_DICT, TASK_LIST, construct_trials
from instructRNN.tasks.task_criteria import isCorrect

def get_holdout_decoded_set(foldername, model_name, seeds=range(5), from_contexts=False, with_holdouts=False, with_dropout=False): 
    full_shallow_set = {}
    full_rich_set = {}
    full_confuse_mat = np.full((len(seeds), len(TASK_LIST), len(TASK_LIST)+1), np.nan)
    sm_model = make_default_model(model_name)
    rnn_decoder = DecoderRNN(256, drop_p=0.0)
    encoder = EncoderDecoder(sm_model, rnn_decoder)
    full_all_contexts = np.full((len(seeds), len(TASK_LIST), 100, sm_model.langModel.LM_intermediate_lang_dim), np.nan)
    encoder.to(0)

    for i, seed in enumerate(seeds): 
        shallow_decoded_set = {}
        rich_decoded_set = {}
        confuse_mat = np.full((len(TASK_LIST), len(TASK_LIST)+1), np.nan)
        all_contexts = np.full(full_all_contexts.shape[1:], np.nan)
        for swap_label, swap_tasks in SWAPS_DICT.items(): 
            print(swap_label)
            encoder.load_model_componenets(foldername+'/'+swap_label+'/'+model_name+'/', seed, swap_tasks, with_holdout=with_holdouts, with_dropout=with_dropout)
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


def test_partner_model(model_name, decoded_dict, contexts = None, num_trials=50, tasks = TASK_LIST, partner_seed=3): 
    partner_model = make_default_model(model_name)
    partner_model.load_state_dict(torch.load('7.20models/multitask_holdouts/Multitask/'+model_name+'/'+model_name+'_seed'+str(partner_seed)+'.pt'), strict=False)
    return _test_partner_model(partner_model, decoded_dict, num_trials, contexts, tasks)

def test_multi_partner_perf(partner_model_name, full_decoded_dict, contexts, num_trials=50, tasks = TASK_LIST, partner_seeds=range(5)):
    instruct_perf_array = np.full((len(full_decoded_dict), len(partner_seeds), len(TASK_LIST)), np.nan)
    other_perf_array = np.full((len(full_decoded_dict), len(partner_seeds), len(TASK_LIST)), np.nan)
    contexts_perf_array = np.full((len(full_decoded_dict), len(partner_seeds),len(TASK_LIST)), np.nan)
    for i, decoded_set in enumerate(full_decoded_dict.values()):
        for j, partner_seed in enumerate(partner_seeds):
            print('processing seed ' + str(j))
            instruct_perf, other_perf, contexts_perf = test_partner_model(partner_model_name, decoded_set, contexts=contexts[i, ...], \
                                                    num_trials=num_trials, tasks=tasks, partner_seed=partner_seed)
            instruct_perf_array[i,j, ...] = instruct_perf
            other_perf_array[i,j, ...] = other_perf
            contexts_perf_array[i, j, ...] = contexts_perf
    return instruct_perf_array, other_perf_array, contexts_perf_array

def test_holdout_partner_perf(load_folder, model_name, full_decoded_dict, contexts, num_trials = 50, partner_seeds = range(5)):
    if 'swap' in load_folder:
        exp_dict = SWAPS_DICT

    instruct_perf_array = np.full((len(partner_seeds), len(full_decoded_dict), len(TASK_LIST)), np.nan)
    other_perf_array = np.full((len(partner_seeds), len(full_decoded_dict),  len(TASK_LIST)), np.nan)
    contexts_perf_array = np.full(( len(partner_seeds), len(full_decoded_dict), len(TASK_LIST)), np.nan)
    
    partner_model = make_default_model(model_name)
    for i, seed in enumerate(partner_seeds):
        for holdout_file, holdouts in exp_dict.items():
            print('processing '+ holdout_file)
            partner_model.load_model(load_folder+'/'+holdout_file+'/'+partner_model.model_name, suffix='_seed'+str(seed))
            for j, item in enumerate(full_decoded_dict.items()): 
                decoder_seed, decoded_set = item
                if decoder_seed == seed: 
                    continue
                instruct_perf, other_perf, contexts_perf = _test_partner_model(partner_model, decoded_set, contexts=contexts[j, ...], \
                                                num_trials=num_trials, tasks=holdouts)

                instruct_perf_array[i, j, [TASK_LIST.index(task) for task in holdouts]] = instruct_perf
                other_perf_array[i, j, [TASK_LIST.index(task) for task in holdouts]] = other_perf
                contexts_perf_array[i, j, [TASK_LIST.index(task) for task in holdouts]] = contexts_perf

    return instruct_perf_array, other_perf_array, contexts_perf_array


