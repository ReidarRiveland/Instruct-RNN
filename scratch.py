from collections import defaultdict
from math import inf
import pickle
from model_trainer import config_model
from re import I
from matplotlib.cbook import flatten

from matplotlib.pyplot import axis
from numpy.core.fromnumeric import size, var
from numpy.lib.function_base import append
from numpy.ma import cos
import transformers
from nlp_models import GPT, SBERT, BERT
from rnn_models import InstructNet, SimpleNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_layer_sim_scores, get_hid_var_group_resp, get_hid_var_resp, get_all_CCGP
import numpy as np
from utils import train_instruct_dict, task_swaps_map, all_models
from task import DM
from plotting import plot_RDM, plot_rep_scatter, plot_CCGP_scores, plot_model_response, plot_hid_traj_quiver, plot_dPCA, plot_neural_resp, plot_trained_performance, plot_tuning_curve
import torch

from task import Task, make_test_trials

all_perf_dict = {}
for model_name in all_models:
    print(model_name)
    perf_array = np.empty((5, 16))
    for i in range(5): 
        print(i)
        model = config_model(model_name)
        model.instruct_mode = 'validation'
        model.set_seed(i)
        model.load_model('_ReLU128_4.11/swap_holdouts/Multitask')
        perf=get_model_performance(model, 3)
        print(perf)
        perf_array[i, :]=perf
    all_perf_dict[model_name]=perf_array

foldername = '_ReLU128_4.11/swap_holdouts'

all_perf_dict

import pickle
pickle.dump(all_perf_dict, open(foldername+'/Multitask/val_perf_dict', 'wb'))

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")


output_sequences = model.generate(input_ids=inputs['input_ids'], do_sample=True)

generated_sequence = output_sequences[0]
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
text

# Remove the batch dimension when returning multiple sequences
if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()


generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[: text.find(tokenizer.eos_token_id) if tokenizer.eos_token_id else None]

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    

    generated_sequences.append(generated_sequence)
        




[tokenizer.convert_ids_to_tokens(_id) for _id in inputs['input_ids']]


inputs['input_ids']

tokenizer.decoder.get(262)
tokenizer.convert_ids_to_tokens(15496)

inputs

outputs = model(**inputs, labels=inputs["input_ids"])

len(outputs)
loss = outputs.loss
logits = outputs.logits
logits.shape


from model_trainer import config_model
sbert_net = config_model('sbertNet_tuned')
sbert = sbert_net.langModel

x =['hi there']

tokens = sbert.tokenizer(x, return_tensors='pt', padding=True)


out = sbert.transformer(**tokens)
out.last_hidden_state.shape
sbert.tokenizer.special_tokens_map