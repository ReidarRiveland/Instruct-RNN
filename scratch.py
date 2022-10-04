from asyncio import all_tasks
from inspect import stack
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


from sklearn.preprocessing import normalize


EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

reps = get_instruct_reps(clipNet.langModel, depth='12')

from sklearn.metrics.pairwise import cosine_similarity
reps.shape
reps.reshape(-1, 512).shape
sim_score = np.corrcoef(reps.reshape(-1, 512))

normalized = np.divide(np.subtract(sim_score, np.min(sim_score, axis=0)[None, :]), (np.max(sim_score, axis=0)-np.min(sim_score, axis=0)[None, :]))
normalized = np.divide(np.subtract(sim_score, np.min(sim_score, axis=1)[:, None]), (np.max(sim_score, axis=1)-np.min(sim_score, axis=1)[:, None]))

normalized = np.divide(np.subtract(sim_score, np.min(sim_score)), (np.max(sim_score, axis=1)-np.min(sim_score)))


normalized = normalize(sim_score)
plot_RDM(normalized, cmap=sns.color_palette('inferno', as_cmap=True))




#############################


EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

EXP_FILE = '7.20models/swap_holdouts'
bertNet = BERTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
bertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+bertNet.model_name, suffix='_seed0')


EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


holdout_perf = eval_model_0_shot(clipNet, '7.20models', 'swap_holdouts', 4)


reps = get_task_reps(clipNet)

reps.shape


def get_all_ccgps(model_name , seeds, use_mean=False):
    all_task = np.empty((len(seeds), len(TASK_LIST)))
    all_dich = np.empty((len(seeds), len(DICH_DICT)))
    all_full = np.empty((len(seeds), 50, len(DICH_DICT)))

    for i, seed in enumerate(seeds):
        print(seed)
        task_score, dich_score, full_array = get_holdout_CCGP('7.20models/swap_holdouts', model_name, seed, use_mean=use_mean)
        all_task[i, ...] = task_score
        all_dich[i, ...] = dich_score
        all_full[i, ...] = full_array
    
    return all_task, all_dich, all_full



clip_task_scores, clip_dich_scores, clip_full_array = get_all_ccgps('clipNet_lin', [0])
bert_task_scores, bert_dich_scores, bert_full_array = get_all_ccgps('bertNet_lin', range(0, 4))
simple_task_scores, simple_dich_scores, simple_full_array = get_all_ccgps('simpleNet', range(0, 4))
sbert_task_scores, sbert_dich_scores, sbert_full_array = get_all_ccgps('sbertNet_lin', [5])




stacked = np.stack((clip_task_scores, sbert_task_scores, bert_task_scores, simple_task_scores))

stacked = np.stack((clip_dich_scores, sbert_dich_scores, bert_dich_scores, simple_dich_scores))



stacked = np.stack((clip_task_scores, sbert_task_scores))

stacked = np.stack((clip_dich_scores, sbert_dich_scores))


stacked = np.mean(stacked, axis=(1,2))
stacked

stacked = np.stack((clip_dich_scores, sbert_dich_scores, bert_dich_scores, simple_dich_scores))


clip_full_array[0, 1, :]

sbert_full_array[0, 1, :]


(stacked-np.min(stacked, axis=0))/(np.max(stacked, axis=0)-np.min(stacked, axis=0))

(np.max(stacked)-np.min(stacked))

normalized_ccgp = (stacked-np.min(stacked, axis=0))/(np.max(stacked, axis=0)-np.min(stacked, axis=0))

normalized_ccgp = (stacked-0.5)/(0.5)


normalized_ccgp

np.mean(normalized_ccgp, axis=2)








# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*gptNetXL_FOR_TUNING*' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin


#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models

#rename 'seed5' seed0 *; rename 'seed6' seed1 *; rename 'seed7' seed2 *; rename 'seed8' seed3 *; rename 'seed9' seed4 *