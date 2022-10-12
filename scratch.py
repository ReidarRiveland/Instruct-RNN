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



decoded_set = get_holdout_decoded_set('7.20models/swap_holdouts', 'clipNet_lin', 0, from_contexts=True, with_holdouts=True)



plot_decoding_confuse_mat(decoded_set[-1], linewidth=0.25, annot_kws={'fontsize':4.5})



decoded_set[0]['AntiCOMP2']['other']

perf_array = test_partner_model('clipNet_lin', decoded_set[0], num_trials = 50, tasks=decoded_set[0].keys(), contexts=decoded_set[-1], partner_seed=3)



np.mean(perf_array[2])

list(zip(TASK_LIST, perf_array[0]))



def plot_partner_perf_lolli(foldername, exp_type, model_list, perf_type='correct', mode='', seeds=range(5)):
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(4, 14))

    width = 1/(len(model_list)+1)
    ind = np.arange(len(TASK_LIST))

    for i, model_name in enumerate(model_list): 
        data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, seeds=seeds)
        zero_shot, std = data.avg_seeds(k_shot=0)
        axn.scatter( zero_shot[::-1], (ind+(width/2))+(i*width), marker='o', s = 2, color=MODEL_STYLE_DICT[model_name][0])
        #axn.scatter( zero_shot[::-1], ind, marker='o', s = 3, color=MODEL_STYLE_DICT[model_name][0])
        axn.hlines((ind+(width/2))+(i*width), xmin=zero_shot[::-1]-std[::-1], xmax=np.min((np.ones_like(std), zero_shot[::-1]+std[::-1]), axis=0), color=MODEL_STYLE_DICT[model_name][0], linewidth=0.2)
        #axn.hlines(ind, xmin=zero_shot[::-1]-std[::-1], xmax=zero_shot[::-1]+std[::-1], color=MODEL_STYLE_DICT[model_name][0], linewidth=0.4)

    axn.set_yticks(ind)
    axn.set_yticklabels('')
    axn.tick_params(axis='y', which='minor', bottom=False)
    axn.set_yticks(ind+0.5, minor=True)
    axn.set_yticklabels(TASK_LIST[::-1], fontsize=4, minor=True) 
    axn.set_xticks(np.linspace(0, 1, 11))

    axn.set_xticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)], fontsize=5)
    axn.set_ylim(-0.15, len(TASK_LIST))
    plt.show()
















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