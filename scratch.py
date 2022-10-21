from tkinter import font
import numpy as np
import scipy
import sklearn
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

# EXP_FILE = '7.20models/swap_holdouts'
# clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
# holdouts_file = 'swap9'
# clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')


def get_cluster_corr(foldername, model_name, seed, save=False): 
    if 'swap' in foldername: 
        exp_dict = SWAPS_DICT
    else:
        exp_dict = {'Multitask': TASK_LIST}

    holdout_corr_mat = np.full((len(TASK_LIST), len(TASK_LIST)), np.nan)

    for label, holdouts in exp_dict.items():
        print('Processing '+label)
        task_var, _, _, _ = get_cluster_info(foldername+'/'+label, model_name, seed)
        corr_mat = np.corrcoef(task_var.T)
        task_indices = [TASK_LIST.index(task) for task in holdouts]
        holdout_corr_mat[task_indices, :] = corr_mat[task_indices, :]

    if save: 
        file_path = foldername+'/cluster_measures/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/corr_mat_'+str(seed), holdout_corr_mat)

    return holdout_corr_mat


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def test_cluster_corr(model_list, seed): 
    corr_list = []
    for model_name in model_list:
        get_cluster_corr('7.20models/swap_holdouts', model_name, seed, save=True)
        get_cluster_corr('7.20models/multitask_holdouts', model_name, seed, save=True)

        holdout_corr = np.load('7.20models/swap_holdouts/cluster_measures/'+model_name+'/corr_mat_'+str(seed)+'.npy')
        multi_corr = np.load('7.20models/multitask_holdouts/cluster_measures/'+model_name+'/corr_mat_'+str(seed)+'.npy')
        corr_list.append(scipy.stats.pearsonr(holdout_corr.flatten(), multi_corr.flatten()))

    return corr_list


def get_cluster_count(foldername, model_name, seed, save=False): 
    if 'swap' in foldername: 
        exp_dict = SWAPS_DICT
    else:
        exp_dict = {'Multitask': TASK_LIST}

    holdout_count_mat = np.full((len(TASK_LIST), len(TASK_LIST)), np.nan)

    for label, holdouts in exp_dict.items():
        print('Processing '+label)
        task_var, cluters_dict, cluster_labels, sorted_indices = get_cluster_info(foldername+'/'+label, model_name, seed)
        cluster_arr = np.empty((len(TASK_LIST), len(cluters_dict)))
        for i in range(len(cluters_dict)): 
            cluster_arr[ :, i ] = np.mean(task_var[cluters_dict[i], :], axis=0)>0.5

        sym_mat = np.matmul(cluster_arr, cluster_arr.T)

        task_indices = [TASK_LIST.index(task) for task in holdouts]
        holdout_count_mat[task_indices, :] = sym_mat[task_indices, :]

    if save: 
        file_path = foldername+'/cluster_measures/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/count_mat_'+str(seed), holdout_count_mat)

    return holdout_count_mat



holdout_indices = [TASK_LIST.index(task) for task in SWAP_LIST[-1]]

holdout_indices

task_var, clusters_dict, cluster_labels, sorted_indices = get_cluster_info('7.20models/swap_holdouts/swap9', model_name, seed)
multi_task_var, multi_clusters_dict, cluster_labels, sorted_indices = get_cluster_info('7.20models/multitask_holdouts/Multitask', model_name, seed)




res = sns.heatmap(np.corrcoef(task_var.T)[34, :], yticklabels=TASK_LIST, vmin=0)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 5)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 4, rotation=90)
plt.show()










model_name = 'clipNet_lin'
seed=0

holdout_corr = get_cluster_corr('7.20models/swap_holdouts', model_name, seed, save=True)
multi_corr = get_cluster_corr('7.20models/multitask_holdouts', model_name, seed, save=True)


def normalize(mat, axis=0):
    return (mat-np.min(mat, axis=axis))/(np.max(mat, axis=axis)-np.min(mat, axis=axis))

normalize(holdout_corr)

res = sns.heatmap(normalize(multi_corr), xticklabels = TASK_LIST, yticklabels=TASK_LIST, vmin=0)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 5)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 4, rotation=90)
plt.show()

scipy.stats.pearsonr(normalize(holdout_corr, axis=0).flatten(), normalize(multi_corr, axis=0).flatten())





















EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

hasattr(clipNet, 'langModel')

hid_mean = get_task_reps(clipNet, epoch=None, num_trials=50, tasks=['AntiDMMod2'], num_repeats=20, main_var=True)[0,...]

unit=119
cmap = plt.get_cmap('seismic') 
fig, axn = plt.subplots()
ylim = np.max(hid_mean[..., unit])
for i in range(hid_mean.shape[0]):
    axn.plot(hid_mean[i, :, unit], c = cmap(i/hid_mean.shape[0]))

plt.plot(hid_mean[:, 140, 119])

plt.show()







cluster_dict[7][5]

unit=119
plot_neural_resp(clipNet, 'DMMod1', 'diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=50, smoothing=1)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod2','diff_strength', unit, num_trials=50, smoothing=1)

plot_tuning_curve(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], unit, [140]*4, num_trials=50, smoothing=1)







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



# rsync -a  -P --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/simpleNet/ /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/simpleNet


