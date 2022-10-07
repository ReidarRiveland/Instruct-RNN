import torch
import numpy as np
import itertools
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.tasks import SWAPS_DICT, TASK_LIST, construct_trials
from instructRNN.tasks.task_criteria import isCorrect


# model_name = 'clipNet_lin'
# sm_model = make_default_model(model_name)
# rnn_decoder = DecoderRNN(256, drop_p=0.0)
# encoder = EncoderDecoder(sm_model, rnn_decoder)
# encoder.to(0)

# seed = 0
# swap_label= 'swap0'
# tasks = SWAPS_DICT[swap_label]
# encoder.load_model_componenets('7.20models/swap_holdouts/'+swap_label+'/'+model_name+'/', seed, tasks)



# encoder.contexts[TASK_LIST.index(tasks[0])]



def get_holdout_decoded_set(foldername, model_name, seed, from_contexts=False): 
    shallow_decoded_set = {}
    rich_decoded_set = {}
    confuse_mat = np.full((len(TASK_LIST), len(TASK_LIST)+1), np.nan)
    sm_model = make_default_model(model_name)
    rnn_decoder = DecoderRNN(256, drop_p=0.0)
    encoder = EncoderDecoder(sm_model, rnn_decoder)
    encoder.to(0)

    for swap_label, swap_tasks in SWAPS_DICT.items(): 
        print(swap_label)
        encoder.load_model_componenets(foldername+'/'+swap_label+'/'+model_name+'/', seed, swap_tasks)
        _shallow, _rich, _confuse_mat = encoder.decode_set(50, 1, tasks=swap_tasks, from_contexts=from_contexts)
        shallow_decoded_set = {**_shallow, **shallow_decoded_set}
        rich_decoded_set = {**_rich, **rich_decoded_set}
        confuse_mat[[TASK_LIST.index(task) for task in swap_tasks], :] = _confuse_mat

    return rich_decoded_set, shallow_decoded_set, confuse_mat

def test_partner_model(model_name, decoded_dict, num_trials=50, tasks = TASK_LIST): 
    partner_model = make_default_model(model_name)
    partner_model.load_state_dict(torch.load('7.20models/multitask_holdouts/Multitask/'+model_name+'/'+model_name+'_seed0.pt'))
    partner_model.eval()
    instruct_perf_array = np.full((len(TASK_LIST)), np.nan)
    other_perf_array = np.full((len(TASK_LIST)), np.nan)
    with torch.no_grad():
        for task in tasks:
            print(task)
            ins, targets, _, target_dirs, _ = construct_trials(task, num_trials)
        
            try:
                task_info = list(np.random.choice(decoded_dict[task]['other'], num_trials))
                out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), instruction = task_info)
                other_perf_array[TASK_LIST.index(task)] = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
            except ValueError:
                pass

            task_instructs = list(itertools.chain.from_iterable([value for value in decoded_dict[task].values()]))
            task_info = list(np.random.choice(task_instructs, num_trials))
            out, _ = partner_model(torch.Tensor(ins).to(partner_model.__device__), instruction = task_info)

            instruct_perf_array[TASK_LIST.index(task)] = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

    return instruct_perf_array, other_perf_array


#get_holdout_decoded_set('7.20models/swap_holdouts', 'clipNet_lin', 0, from_contexts=True)