import numpy as np
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.models.full_models import *
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval

from instructRNN.tasks.tasks import *
import torch
from instructRNN.models.full_models import SBERTNet, BoWNet
from instructRNN.instructions.instruct_utils import get_instructions
from instructRNN.plotting.plotting import *



EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

plot_scatter(clipNet, ['DM', 'DMMod1', 'DMMod2', 'AntiDM', 'AntiDMMod1', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, epoch='0', instruct_mode='combined')

reps = get_task_reps(clipNet, num_trials = 100,  main_var=True, instruct_mode='combined')

dich_scores, holdouts =  get_dich_CCGP(reps, DICH_DICT['dich2'], holdouts_involved=['AntiDMMod1'])
np.mean(holdouts)


EXP_FILE = '7.20models/swap_holdouts'
bertNet = BERTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
bertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+bertNet.model_name, suffix='_seed0')

SWAP_LIST[0]

data = HoldoutDataFrame('7.20models', 'swap', 'bertNet_lin', seeds=range(5), mode='combined')
mean, _ = data.avg_seeds(k_shot=0)

list(zip(TASK_LIST, np.round(mean, 3)))


plot_scatter(bertNet, ['DM', 'DMMod1', 'DMMod2', 'AntiDM', 'AntiDMMod1', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)

reps = get_task_reps(bertNet, num_trials = 100,  main_var=True, instruct_mode='combined')
DICH_DICT['dich6']

dich_scores, holdouts =  get_dich_CCGP(reps, DICH_DICT['dich2'], holdouts_involved=['AntiDMMod1'])
holdouts

epoch='25'

berttask_scores, bert_dich_scores = get_holdout_CCGP('7.20models/swap_holdouts', 'bertNet_lin', 0, epoch)
cliptask_scores, clip_dich_scores = get_holdout_CCGP('7.20models/swap_holdouts', 'clipNet_lin', 1)
sbertNettask_scores, sbertNet_dich_scores = get_holdout_CCGP('7.20models/swap_holdouts', 'sbertNet_lin', 1)
simpletask_scores, simple_dich_scores = get_holdout_CCGP('7.20models/swap_holdouts', 'simpleNet', 0, epoch)

clip_dich_scores
sbertNet_dich_scores


stacked = np.stack((berttask_scores, cliptask_scores, simpletask_scores, sbertNettask_scores))

stacked = np.stack((bert_dich_scores, cliptask_scores, simpletask_scores, sbertNettask_scores))


(stacked-np.min(stacked, axis=0))/(np.max(stacked, axis=0)-np.min(stacked, axis=0))

(np.max(stacked)-np.min(stacked))

normalized_ccgp = (stacked-np.min(stacked, axis=0))/(np.max(stacked, axis=0)-np.min(stacked, axis=0))

normalized_ccgp

np.mean(normalized_ccgp, axis=1)


task_scores[TASK_LIST.index('AntiMultiDur1')]

list(zip(TASK_LIST, task_scores))


DICH_DICT['dich2']
np.mean(dich_scores[1,:])

np.mean(task_scores)







# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*gptNetXL_FOR_TUNING*' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin


#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models

rename 'seed5' seed0 *; rename 'seed6' seed1 *; rename 'seed7' seed2 *; rename 'seed8' seed3 *; rename 'seed9' seed4 *