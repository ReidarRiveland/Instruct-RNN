import torch
import numpy as np
from instructRNN.models.full_models import SimpleNetPlus
from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, construct_trials
from instructRNN.plotting.plotting import plot_scatter
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import one_hot_input_rule
from instructRNN.analysis.model_analysis import get_rule_embedder_reps

TASK_LIST
SWAP_LIST[2]

EXP_FILE = '3.16models/swap_holdouts'
simpleNet = SimpleNetPlus()
holdouts_file = 'swap2'

simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')

#plot_scatter(simpleNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], dims=3)

rule_encoding = get_rule_embedder_reps(simpleNet, depth='full')
rule_encoding.shape
transform1 = (rule_encoding[TASK_LIST.index('AntiGoMod1')] - rule_encoding[TASK_LIST.index('GoMod1')])
transform2 = (rule_encoding[TASK_LIST.index('AntiGoMod2')] - rule_encoding[TASK_LIST.index('Go')])

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(np.array([transform1.mean(0), transform2.mean(0)]))

np.dot



import itertools

for x_task in TASK_LIST: 
    for y_task in TASK_LIST: 



rule_encoding



plot_scatter(simpleNet, ['AntiGo', 'AntiGoMod1', 'Go', 'GoMod1'], dims=3, rep_depth='full', transform = transform[None, ], transform_task='AntiGo', num_trials=100)

ins, targets, _, target_dirs, _ = construct_trials('AntiGo', 100)
simpleNet.to('cpu')
out, _ = simpleNet(torch.Tensor(ins), context=torch.tensor(transform))
np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
