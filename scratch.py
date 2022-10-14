from asyncio import all_tasks
from inspect import stack
from click import style
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



# decoded_set = get_holdout_decoded_set('7.20models/swap_holdouts', 'clipNet_lin', [0, 1, 3], from_contexts=True)

# holdout_perf_array = test_holdout_partner_perf('7.20models/swap_holdouts', 'clipNet_lin', decoded_set[0], decoded_set[-1], partner_seeds=range(5))

# multi_perf_array = test_multi_partner_perf('clipNet_lin', decoded_set[0], decoded_set[-1])



# plot_partner_perf_lolli(load_str='holdout', plot_holdouts=True, plot_multi_only=False)



# from sklearn.metrics.pairwise import cosine_similarity





EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

SWAP_LIST[-1]

cluster_dict, cluster_labels, sorted_indices= plot_task_var_heatmap('7.20models/swap_holdouts/swap9', 'clipNet_lin', 2)

cluster_dict[7][5]

unit=119
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod2','diff_strength', unit, num_trials=25, smoothing=1)

plot_tuning_curve(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], unit, [80]*4, num_trials=25, smoothing=1)







reps = get_instruct_reps(clipNet.langModel, depth='12', instruct_mode='combined')

rep_reshaped = reps.reshape(-1, 512)
sim_score = np.corrcoef(rep_reshaped)
sim_score
(sim_score-np.min(sim_score, axis=0))/(np.max(sim_score, axis=0)-np.min(sim_score, axis=0))

plot_RDM(sim_score, sns.color_palette("inferno", as_cmap=True))



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



# rsync -a  -P --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/clipNet_lin/ /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/clipNet_lin
