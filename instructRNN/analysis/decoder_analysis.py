import torch
import numpy as np
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model
from instructRNN.tasks.tasks import SWAPS_DICT, TASK_LIST

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
        encoder.load_model_componenets(foldername+'/'+swap_label+'/'+model_name+'/', seed)
        _shallow, _rich, _confuse_mat = encoder.decode_set(50, 1, tasks=swap_tasks, from_contexts=from_contexts)
        shallow_decoded_set = {**_shallow, **shallow_decoded_set}
        rich_decoded_set = {**_rich, **rich_decoded_set}
        confuse_mat[[TASK_LIST.index(task) for task in swap_tasks], :] = _confuse_mat

    return rich_decoded_set, shallow_decoded_set, confuse_mat

model_name = 'clipNet_lin'
load_str = '7.20models/swap_holdouts'

rich, shallow, conduse = get_holdout_decoded_set(load_str, 'clipNet_lin', 1, from_contexts=True)
