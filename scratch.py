from nlp_models import SBERT
from rnn_models import InstructNet, SimpleNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores
import numpy as np
import torch.optim as optim
from utils import sort_vocab, train_instruct_dict
from task import DM

from matplotlib.pyplot import get
from numpy.lib import utils
import torch
import torch.nn as nn

#SimpleNet
model1 = SimpleNet(128, 1, use_ortho_rules=True)
all_sim_scores1 = get_sim_scores(model1, 'Multitask', 'task')


model1.langModel is None


#sbertNet_layer_11
model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model.model_name += '_tuned'
model.set_seed(2)
model.load_model('_ReLU128_5.7/single_holdouts/Multitask')

hasattr(model, 'langModel')

from plotting import plot_model_response

from task import Comp, make_test_trials

trials, var = make_test_trials('DMS', 'diff_direction', 0)

['respond in the first direction']*100

plot_model_response(model, trials, plotting_index = 99, instructions=['respond to the first direction']*100)
