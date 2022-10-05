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


reps = get_task_reps(clipNet, num_trials=50, instruct_mode='combined', noise=0.0)
lang_reps = get_instruct_reps(clipNet.langModel, depth='12', instruct_mode='combined')


pcs, var_exp = reduce_rep(lang_reps, pcs=range(reps.shape[-1]))

np.sum(var_exp[:1])





def get_model_sv(model, layer='task'):
    if layer == 'task':
        reps = get_task_reps(model, num_trials=50, instruct_mode='combined', noise=0.0)
    else: 
        print('here')
        reps = get_instruct_reps(model.langModel, depth=layer, instruct_mode='combined')

    _, sv, _ = LA.svd(reps.reshape(-1, reps.shape[-1]))
    return sv

def get_model_spectrum(model, layer='task'):
    if layer == 'task':
        reps = get_task_reps(model, num_trials=50, instruct_mode='combined', noise=0.0)
    else: 
        print('here')
        reps = get_instruct_reps(model.langModel, depth=layer, instruct_mode='combined')

    spectrum = LA.eigvals(np.corrcoef(reps.reshape(-1, reps.shape[-1])))
    return spectrum

spectrum = get_model_spectrum(clipNet)


def get_dim_across_models(load_folder, model_name, seeds, layer, mode='spectrum'):
    model = make_default_model(model_name)
    if 'swap' in load_folder: 
        exp_dict = SWAPS_DICT

    if layer == 'task': 
        rep_dim = model.rnn_hidden_dim
    elif layer.isnumeric(): 
        rep_dim = model.langModel.LM_intermediate_lang_dim
    elif layer == 'bow': 
        rep_dim = len(sort_vocab())
    else: 
        rep_dim = model.langModel.LM_out_dim 

    sv_array = np.empty((len(seeds), len(exp_dict), 50**2))
    for i, seed in enumerate(seeds):
        for j, holdout_label in enumerate(exp_dict.keys()):
            model.load_model(EXP_FILE+'/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
            if mode == 'spectrum': 
                sv = get_model_spectrum(model, layer=layer)
            else: 
                sv = get_model_sv(model, layer=layer)
            sv_array[i, j, :] = sv

    return sv_array


clip_sv_array = get_dim_across_models(EXP_FILE, 'clipNet_lin', [0], layer='task')
sbert_sv_array = get_dim_across_models(EXP_FILE, 'sbertNet_lin', [5], layer='task')
bert_sv_array = get_dim_across_models(EXP_FILE, 'bertNet_lin', [0], layer='task')
simpleNet_sv_array = get_dim_across_models(EXP_FILE, 'simpleNet', [0], layer = 'task')

np.mean(clip_sv_array, axis=(0, 1))


#########################################

import matplotlib.pyplot as plt
plt.plot(np.mean(clip_sv_array, axis=(0, 1))[:20])
plt.plot(np.mean(sbert_sv_array, axis=(0, 1))[:20])
plt.plot(np.mean(bert_sv_array, axis=(0, 1))[:20])
plt.plot(np.mean(simpleNet_sv_array, axis=(0, 1))[:20])
plt.legend(['clip', 'sbert', 'bert', 'simpleNet'])
plt.show()


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


#############

from instructRNN.tasks.tasks import TASK_LIST, DICH_DICT
import numpy as np
folder_name = '7.20models/swap_holdouts/CCGP_scores'
model_list = ['clipNet_lin', 'sbertNet_lin', 'bertNet_lin', 'bowNet_lin', 'simpleNet']
seeds = range(5)
layer = 'full'
task_holdout_array = np.empty((len(model_list), len(seeds), len(TASK_LIST)))
dich_holdout_array = np.empty((len(model_list), len(seeds), len(DICH_DICT)))

for i, model_name in enumerate(model_list):
    for j, seed in enumerate(seeds):
        if model_name is 'sbertNet_lin': seed +=5
        task_load_str = folder_name+'/'+model_name+'/layer'+layer+'_task_holdout_seed'+str(seed)+'.npy'
        dich_load_str = folder_name+'/'+model_name+'/layer'+layer+'_dich_holdout_seed'+str(seed)+'.npy'
        task_arr = np.load(open(task_load_str, 'rb'))
        dich_arr = np.load(open(dich_load_str, 'rb'))
        task_holdout_array[i, j, :] = task_arr
        dich_holdout_array[i, j, :] = dich_arr


np.mean(task_holdout_array[1, ...])

np.mean(task_holdout_array, axis=(1,2))




# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*gptNetXL_FOR_TUNING*' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin


#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models

#rename 'seed5' seed0 *; rename 'seed6' seed1 *; rename 'seed7' seed2 *; rename 'seed8' seed3 *; rename 'seed9' seed4 *