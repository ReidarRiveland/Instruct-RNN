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



cur_sentences = "Hello, my dog is"
for i in range(20): 
    inputs = tokenizer(cur_sentences, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    scores = torch.softmax(logits, dim=-1)
    append_word = tokenizer.batch_decode(torch.max(scores, 2).indices[:, -1])
    cur_sentences+=append_word[0]

cur_sentences