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

plot_scatter(simpleNet, ['GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], dims=3)
plot_scatter(simpleNet, ['GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], dims=3, rep_depth='full')


rule_encoding = get_rule_embedder_reps(simpleNet)
comp_rule = rule_encoding[0,None]




ins, targets, _, target_dirs, _ = construct_trials('Go', 50)
out, _ = simpleNet(torch.Tensor(ins), context=torch.tensor(rule_encoding[0, None]).repeat(50, 1))
np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
