from tkinter import font
import numpy as np
import scipy
import sklearn
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval, get_multi_comp_perf

from instructRNN.tasks.tasks import *
import torch
from instructRNN.instructions.instruct_utils import get_instructions
from instructRNN.plotting.plotting import *
from instructRNN.analysis.decoder_analysis import *
from collections import Counter
from instructRNN.models.full_models import *


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

model = GPTNetXL_lin()

get_n_params(model.langModel.transformer)

perf = np.load(open('7.20models/swap_holdouts/decoder_perf/clipNet_lin/test_sm_holdout_decoder_holdout_partner_holdout_all_perf.npy', 'rb'))
np.nanmean(perf)


# perf = np.load('7.20models/swap_holdouts/decoder_perf/simpleNet/test_sm_holdout_decoder_holdout_partner_multi_all_perf.npy')
# #perf.shape

model_name = 'simpleNet'
decoder_pipeline('7.20models/swap_holdouts', model_name, sm_holdout=True)







perf.shape











def get_val_perf(foldername, model_name, seed, num_repeats = 5, batch_len=100, save=False): 
    model = full_models.make_default_model(model_name)
    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    perf_array = get_model_performance(model, num_repeats=num_repeats, batch_len=batch_len, instruct_mode='')
    if save:
        file_path = foldername+'/val_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+model_name+'_val_perf_seed'+str(seed), perf_array)

    return perf_array

perf_list = []
for i in range(5): 
    perf_list.append(np.mean(get_val_perf('7.20models/multitask_holdouts', 'simpleNet', i)))

from instructRNN.trainers.mem_net_trainer import MemNet

def eval_memNet_multi_perf(model_name, foldername, exp_type, seed, holdout_label, **trial_kwargs):
    if 'swap' in exp_type: 
        exp_dict = SWAPS_DICT

    perf_array = np.full(len(TASK_LIST), np.NaN)
    model_folder = foldername+'/'+exp_type+'_holdouts/'+holdout_label+'/'+model_name

    model = full_models.make_default_model(model_name)
    model.load_model(model_folder, suffix='_seed'+str(seed))

    memNet = MemNet(64, 516)
    memNet.load_state_dict(torch.load(model_folder+'/mem_net/'+model_name+'_'+'seed'+str(seed)+'_memNet.pt'))

    memNet.to(device)
    model.to(device)

    with torch.no_grad():
        for task in TASK_LIST:
            print(task)
            ins, tar, mask, tar_dir, task_type = construct_trials(task, 50, return_tensor=True)
            mem_out, hid = memNet(ins.float().to(device), tar.float().to(device))
            contexts = mem_out[:, -1, :]
            perf_array[TASK_LIST.index(task)] = task_eval(model, task, 50, context=contexts)

    return perf_array

perf_array = eval_memNet_multi_perf('clipNet_lin', '7.20models', 'swap', 0, 'swap1')

def eval_memNet_holdout_perf(model_name, foldername, exp_type, seed, **trial_kwargs):
    if 'swap' in exp_type: 
        exp_dict = SWAPS_DICT

    perf_array = np.full(len(TASK_LIST), np.NaN)
    model = full_models.make_default_model(model_name)
    memNet = MemNet(64, 516)

    memNet.to(device)
    model.to(device)

    with torch.no_grad():
        for holdout_label, tasks in exp_dict.items(): 
            model_folder = foldername+'/'+exp_type+'_holdouts/'+holdout_label+'/'+model.model_name
            print(holdout_label)
            model.load_model(model_folder, suffix='_seed'+str(seed))
            memNet.load_state_dict(torch.load(model_folder+'/mem_net/'+model_name+'_'+'seed'+str(seed)+'_memNet.pt'))

            for task in tasks: 
                print(task)
                ins, tar, mask, tar_dir, task_type = construct_trials(task, 50, return_tensor=True)
                mem_out, hid = memNet(ins.float().to(device), tar.float().to(device))
                contexts = mem_out[:, -1, :]
                perf_array[TASK_LIST.index(task)] = task_eval(model, task, 50, context=contexts)

    return perf_array



perf_array = eval_memNet_holdout_perf('clipNet_lin', '7.20models', 'swap', 0)








##DM
unit=119
plot_neural_resp(clipNet, 'DMMod1', 'diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=50, smoothing=1)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod2','diff_strength', unit, num_trials=50, smoothing=1)

plot_tuning_curve(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], unit, [140]*4, num_trials=25, smoothing=1, min_coh=0.01, max_coh=0.3)

unit=175
plot_neural_resp(clipNet, 'DMMod1', 'diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod2','diff_strength', unit, num_trials=25, smoothing=1)

unit=41
plot_neural_resp(clipNet, 'DMMod1', 'diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1)

plot_tuning_curve(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], unit, [149]*4, num_trials=25, smoothing=1, min_coh=0.01, max_coh=0.3)

##ANTI GO

unit=39
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], unit, [145]*4, num_trials=80, smoothing=1, min_coh=0.01, max_coh=0.5)


###COMP
EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap6'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

unit=82
plot_neural_resp(clipNet, 'COMP2', 'diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'AntiCOMP2','diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'COMP1','diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'AntiCOMP1','diff_strength', unit, num_trials=25)


##matching
EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap1'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

##2
unit=147
plot_tuning_curve(clipNet, ['DMS', 'DNMS', 'DMC', 'DNMC'], unit, [149]*4, num_trials=50, smoothing=1.0)

trials = DMS(100, main_var=True)
trials.plot_trial(5)









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










reps = get_instruct_reps(clipNet.langModel, depth='12', instruct_mode='combined')

rep_reshaped = reps.reshape(-1, 512)
sim_score = np.corrcoef(rep_reshaped)
sim_score
(sim_score-np.min(sim_score, axis=0))/(np.max(sim_score, axis=0)-np.min(sim_score, axis=0))

plot_RDM(sim_score, sns.color_palette("inferno", as_cmap=True))




# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*context#' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin


#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models



# rsync -a  -P --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/simpleNet/ /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/simpleNet


