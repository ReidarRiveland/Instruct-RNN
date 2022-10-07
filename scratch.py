from asyncio import all_tasks
from inspect import stack
from cv2 import threshold
import numpy as np
import scipy
import sklearn
from sqlalchemy import asc
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.models.full_models import *
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval

from instructRNN.tasks.tasks import *
import torch
from instructRNN.models.full_models import *
from instructRNN.instructions.instruct_utils import get_instructions
from instructRNN.plotting.plotting import *
from instructRNN.analysis.decoder_analysis import *


var_exp, thresholds = load_holdout_dim_measures('7.20models/swap_holdouts', 'clipNet_lin', ['task'], verbose=True)

np.mean(var_exp, axis=(0,1,2))

np.mean(thresholds)


model_list = ['clipNet_lin', 'sbertNet_lin', 'bertNet_lin',  'simpleNet']





plot_layer_dim(model_list, 'task')















decoded_set = get_holdout_decoded_set('7.20models/swap_holdouts', 'clipNet_lin', 4, from_contexts=True)

































EXP_FILE = '7.20models/multitask_holdouts'
simpleNet = GPTNet_lin(rnn_hidden_dim=256)
simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed4')



var_exp, thresholds = load_holdout_dim_measures('7.20models/swap_holdouts', 'clipNet_lin', [str(x) for x in range(1, 13)] + ['full', 'task'], verbose=True)


clip_task, clip_dich = load_multi_ccgp('clipNet_lin')
simple_task, simple_dich = load_multi_ccgp('simpleNet')

np.mean(clip_dich, axis=0)

np.mean(simple_dich, axis=0)



np.mean(thresholds[:, -1, :])
np.mean(thresholds[:, -2:, :], axis=(0, 2))

EXP_FILE = '7.20models/swap_holdouts'
GPTNet_lin = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 15, [130]*6, 'direction')


get_holdout_decoded_set('7.20models/swap_holdouts', 'clipNet_lin', 2, from_contexts=True)


# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*gptNetXL_FOR_TUNING*' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin


#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models

#rename 'seed5' seed0 *; rename 'seed6' seed1 *; rename 'seed7' seed2 *; rename 'seed8' seed3 *; rename 'seed9' seed4 *