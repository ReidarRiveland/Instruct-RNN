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



decoded_set = get_holdout_decoded_set('7.20models/swap_holdouts', 'clipNet_lin', [0, 1, 3], from_contexts=True)

holdout_perf_array = test_holdout_partner_perf('7.20models/swap_holdouts', 'clipNet_lin', decoded_set[0], decoded_set[-1], partner_seeds=range(5))

multi_perf_array = test_multi_partner_perf('clipNet_lin', decoded_set[0], decoded_set[-1])



def plot_partner_perf_lolli(load_str='holdout', plot_holdouts=False, plot_multi_only=False):
    to_plot_colors = [('All Decoded', '#0392cf'), ('Novel Decoded', '#7bc043'), ('Embedding', '#edc951')]
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(16, 8))
    plt.suptitle('Partner Model Performance on Decoded Instructions')

    axn.set_ylabel('Task', size=8, fontweight='bold')
    axn.set_xlabel('Performance', size=8, fontweight='bold')

    ind = np.arange(len(TASK_LIST))

    if plot_multi_only:
        mode_list = [('multi_', 'solid', 'o')]
    else: 
        mode_list = [('holdout_', 'dashed', 'D'), ('multi_', 'solid', 'o')]

    for mode in mode_list:
        perf_data = np.load(mode[0]+load_str+'_decoder_perf.npy')
        for i in range(len(perf_data)): 
            perf = np.nanmean(perf_data[i], axis=(0,1))
            print(perf.shape)
            axn.scatter(perf[::-1], ind+1, marker=mode[2], s = 2, color=to_plot_colors[i][1])
            axn.vlines(np.nanmean(perf), 0, len(TASK_LIST)+1, color=to_plot_colors[i][1], linestyle=mode[1], linewidth=0.8)

    axn.tick_params('y', bottom=False, top=False)
    axn.set_yticks(range(len(TASK_LIST)+3))
    axn.set_yticklabels(['']+TASK_LIST[::-1] + ['', ''], fontsize=4) 
    axn.set_xticks(np.linspace(0, 1, 11))
    
    patches = []
    if plot_holdouts: 
        data = HoldoutDataFrame('7.20models', 'swap', 'clipNet_lin', mode='combined')
        zero_shot, std = data.avg_seeds(k_shot=0)
        axn.vlines(np.nanmean(zero_shot), 0, len(TASK_LIST)+1, color=MODEL_STYLE_DICT['clipNet_lin'][0], linestyle='dashed', linewidth=0.8)
        axn.scatter(zero_shot[::-1], ind+1, marker='D', s = 2, color=MODEL_STYLE_DICT['clipNet_lin'][0])
        patches.append(Line2D([0], [0], label = 'Instructions', color= MODEL_STYLE_DICT['clipNet_lin'][0], marker = 's', linestyle = 'None', markersize=4))


    for style in to_plot_colors:
        patches.append(Line2D([0], [0], label = style[0], color= style[1], marker = 's', linestyle='None', markersize=4))


    patches.append(Line2D([0], [0], label = 'Multitask Partners', color= 'grey', marker = 'o', linestyle = 'None', markersize=4))
    patches.append(Line2D([0], [0], label = 'Multitask Partner', color= 'grey', linestyle='solid', markersize=4))

    patches.append(Line2D([0], [0], label = 'Holdout Partners', color= 'grey', marker = 'D', linestyle = 'None', markersize=2))
    patches.append(Line2D([0], [0], label = 'Holdout Partners', color= 'grey', linestyle='dashed', markersize=2))




    axn.legend(handles = patches, fontsize='x-small')
    
    axn.set_xticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)], fontsize=5)
    axn.set_ylim(-0.2, len(TASK_LIST)+1)
    axn.set_xlim(0, 1.01)

    plt.show()


plot_partner_perf_lolli(load_str='holdout', plot_holdouts=True, plot_multi_only=False)



from sklearn.metrics.pairwise import cosine_similarity





EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')


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
