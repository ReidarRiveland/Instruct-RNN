# from tkinter import font
# import numpy as np
# import scipy
# import sklearn
# from instructRNN.instructions.instruct_utils import get_task_info
# from instructRNN.tasks.task_criteria import isCorrect
# from instructRNN.models.full_models import *
# from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval

# from instructRNN.tasks.tasks import *
# import torch
# from instructRNN.models.full_models import *
# from instructRNN.instructions.instruct_utils import get_instructions
# from instructRNN.plotting.plotting import *
# from instructRNN.analysis.decoder_analysis import *


# EXP_FILE = '7.20models/swap_holdouts'
# clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
# holdouts_file = 'swap1'
# clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed0')

from instructRNN.plotting.plotting import *
clipNet_lin = eval_model_exemplar('clipNet_lin', '7.20models', 'swap', 0)




pickle.load(open('7.20models/swap_holdouts/swap0/clipNet_lin/contexts/seed0_AntiDMMod2exemplar_context_vecs512', 'rb')).shape
















holdouts = SWAPS_DICT[holdouts_file]
task_indices = [TASK_LIST.index(task) for task in TASK_LIST if task not in holdouts]
reps = get_instruct_reps(clipNet.langModel)

torch.std(task_info_basis)

task_info_basis = torch.tensor(np.mean(reps, axis=1)[task_indices, :])+torch.randn(45, 64)*0.8

comp_vec = pickle.load(open('7.20models/swap_holdouts/swap1/clipNet_lin/lin_comp/seed0_COMP1Mod1_chk_comp_vecs', 'rb'))

lin = nn.Linear(45, 1)
lin.load_state_dict(comp_vec[0])


contexts = lin(task_info_basis.float().T)
contexts.shape

task = 'COMP1Mod1'
batch_size=50

ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)


out, _ = clipNet(torch.Tensor(ins), context = contexts.T.repeat(50, 1))
np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))



reps = get_instruct_reps(clipNet.langModel)

holdouts = SWAPS_DICT[holdouts_file]
holdouts

rule_basis = torch.tensor(np.mean(reps, axis=1)[[TASK_LIST.index(task) for task in TASK_LIST if task not in holdouts], :])

context = torch.randn(45)

torch.matmul(context, rule_basis.float())


simpleNet = SimpleNet()
simpleNet.rule_transform


plot_comp_bar('7.20models', 'swap', ['simpleNet', 'clipNet_lin'])

load_holdout_ccgp('7.20models/swap_holdouts', 'clipNet_lin', ['task'], range(5))

multi_ccgp = load_multi_ccgp('clipNet_lin')[0]
np.mean(multi_ccgp)

data = load_perf(['clipNet_lin'], mode='multi_comp')
np.mean(data)


plot_task_var_heatmap('7.20models/swap_holdouts/swap9', 'clipNet_lin', 0)


simple_perf = eval_model_exemplar('clipNet_lin', '7.20models', 'swap', 0)

np.mean(simple_perf)
np.mean(perf)


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


