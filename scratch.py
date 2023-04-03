import torch
import numpy as np
from instructRNN.models.full_models import SimpleNetPlus
from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, SWAPS_DICT, construct_trials
from instructRNN.plotting.plotting import plot_scatter
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import one_hot_input_rule
from instructRNN.analysis.model_analysis import get_rule_embedder_reps
from instructRNN.trainers.mem_net_trainer import MemNet
import instructRNN.models.full_models as full_models

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt

from instructRNN.data_loaders.perfDataFrame import *
from instructRNN.plotting.plotting import *


model_list = ['simpleNet', 'sbertNet_lin']
data = PerfDataFrame('7.20models', 'swap', 'sbertNet_lin', mode='all_holdout_comp')




total_combos_data = np.full((50, len(model_list)), np.NaN)
threshold=0.8
for i, model_name in enumerate(model_list):  
    total_combos_data[:, i]=data.data

# normalized = np.divide(np.subtract(total_combos_data, np.min(total_combos_data, axis=0)[None, :]), 
#                             (np.max(total_combos_data, axis=0)[None,:]-np.min(total_combos_data, axis=0)[None, :]))


# np.sum(data.data.mean(0)>threshold, axis=-1)

def plot_all_comp_holdout_lolli_v(foldername, exp_type, model_list, marker = 'o', 
                                    mode='all_holdout_comp', threshold=0.8,  seeds=range(5)):
    with plt.rc_context({'axes.grid.axis': 'y'}):
        total_combos_data = np.full((50, len(model_list)), np.NaN)

        fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(11, 4))

        width = 1/(len(model_list)+1)
        ind = np.arange(len(TASK_LIST))

        axn.set_xticks(ind)
        axn.set_xticklabels('')
        axn.tick_params(axis='x', which='minor', bottom=False)
        axn.set_xticks(ind+0.75, minor=True)
        axn.set_xticklabels(TASK_LIST, fontsize=6, minor=True, rotation=45, ha='right', fontweight='bold') 
        axn.set_xlim(-0.15, len(ind))

        # axn.set_yticks(np.linspace(0, 1, 11))
        # axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)], fontsize=8)
        axn.set_ylim(0.0, 100)
        # axn.set_ylabel('Percent Correct', size=8, fontweight='bold')

        for i, model_name in enumerate(model_list):  
            data = PerfDataFrame(foldername, exp_type, model_name, mode = mode, seeds=seeds)
            total_combos_data[:, i]=np.sum(data.data.mean(0)>threshold, axis=-1)


        # normalized = np.divide(np.subtract(total_combos_data, np.min(total_combos_data, axis=1)[:, None]), 
        #                             (np.max(total_combos_data, axis=1)[:, None]-np.min(total_combos_data, axis=1)[:, None]))
        normalized = total_combos_data
        #normalized = normalize(total_combos_data, axis=1, norm='l2')

        for i, model_name in enumerate(model_list):
            color = MODEL_STYLE_DICT[model_name][0]     

            axn.axhline(normalized[:, i].mean(), color=color, linewidth=1.0, alpha=0.8, zorder=0)

            x_mark = (ind+(width/2))+(i*width)
            axn.scatter(x_mark,  normalized[:, i], color=color, s=3, marker=marker)
            axn.vlines(x_mark, ymin=0, ymax=normalized[:, i], color=color, linewidth=0.5)

        fig.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list], loc=5, title='Models', title_fontsize = 'x-small', fontsize='x-small')        

        plt.tight_layout()
        return fig, axn

plot_all_comp_holdout_lolli_v('7.20models', 'swap', ['simpleNet', 'simpleNetPlus', 'bowNet_lin', 'sbertNet_lin', 'clipNet_lin', 'bertNet_lin'], threshold=0.8)

plt.show()





plot_all_task_lolli_v('3.16models', 'swap', ['simpleNetPlus'], mode='memNet')
plt.show()





TASK_LIST
SWAP_LIST[3]

# EXP_FILE = '3.16models/swap_holdouts'
# simpleNet = SimpleNetPlus(rule_encoder_hidden=128)
# holdouts_file = 'swap3'
# simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')
# memNet = MemNet(64, 256)
# memNet.load_state_dict(torch.load('3.16models/swap_holdouts/swap0/simpleNetPlus/mem_net/simpleNetPlus_seed0_memNet.pt'))


def eval_mem_net_holdouts(model_name, folder_name, batch_size=100): 
    if 'swap' in folder_name: 
        exp_dict = SWAPS_DICT
    perf_array = np.full((5, len(TASK_LIST)), np.NaN)
    model = full_models.make_default_model(model_name)
    with torch.no_grad():
        for seed in range(5):
            for holdout_label, tasks in exp_dict.items(): 
                model.load_model(folder_name+'/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
                memNet = MemNet(64, 256)
                memNet.load_state_dict(torch.load(folder_name+'/'+holdout_label+'/'+model.model_name+'/mem_net/'+model.model_name+'_seed'+str(seed)+'_memNet.pt'))
                for task in tasks: 
                    print(task)
                    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)
                    _, hid = model(torch.Tensor(ins), torch.Tensor(one_hot_input_rule(100, task)))
                    inferred_embed, _ = memNet(torch.tensor(ins), torch.tensor(targets), hid)
                    out, hid = model(torch.Tensor(ins), info_embedded=inferred_embed[:, -1, :])
                    perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
                    perf_array[seed, TASK_LIST.index(task)] = perf

    return perf_array


perf_arr = eval_mem_net_holdouts('simpleNetPlus', '3.16models/swap_holdouts')

np.save(open('simpleNetPlus_memNet_holdout_perf.npy', 'wb'), perf_arr)