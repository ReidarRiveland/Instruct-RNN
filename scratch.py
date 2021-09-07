from nlp_models import SBERT
from rnn_models import InstructNet, SimpleNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores
import numpy as np
import torch.optim as optim
from utils import sort_vocab, train_instruct_dict
from task import DM

from matplotlib.pyplot import get, ylim
from numpy.lib import utils
import torch
import torch.nn as nn


#sbertNet_layer_11
model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model.model_name += '_tuned'
model.set_seed(1)
model.load_model('_ReLU128_5.7/swap_holdouts/Anti_Go_MultiCOMP1')
reduced_reps = reduce_rep(get_task_reps(model))[0]

from plotting import plot_trained_performance
from model_analysis import get_model_performance

perf = get_model_performance(model, 24)
plot_trained_performance({'sbertNet_tuned':perf})

reduced_reps_10 = reduce_rep(get_task_reps(model), dim=5)[0]


# hasattr(model, 'langModel')

# from plotting import plot_model_response, plot_rep_scatter

from task import Comp, make_test_trials, Task

# trials, var = make_test_trials('DMS', 'diff_direction', 0)

# ['respond in the first direction']*100

# plot_model_response(model, trials, plotting_index = 99, instructions=['respond to the first direction']*100)




from plotting import task_colors, two_line_instruct, mpatches, plt, itertools
from utils import comp_input_rule, get_input_rule

model1 = SimpleNet(128, 1, use_ortho_rules=True)
model1.set_seed(1)
model1.load_model('_ReLU128_5.7/single_holdouts/Multitask')
reduced_reps1 = reduce_rep(get_task_reps(model1))[0]

scores = get_total_para_score(model, 'RT_Go', 'task')
np.mean(scores, axis=0)

from model_analysis import g get_all_parallelogram_scores

sbertNet_all_scores, sbertNet_holdout_scores = get_all_holdout_para_scores(model)
simpleNet_all_scores, simpleNet_holdout_scores = get_all_holdout_para_scores(model1)

simpleNet_zero_shots = model_data_dict['simpleNet'][:, :, 0]
zero_shots = model_data_dict['sbertNet_tuned'][:, :, 0]



from sklearn import preprocessing
#normed_scores =preprocessing.normalize(np.concatenate((sbertNet_holdout_scores.flatten(), simpleNet_holdout_scores.flatten())).reshape(1, -1))
normed_scores =np.concatenate((sbertNet_holdout_scores.flatten(), simpleNet_holdout_scores.flatten()))
all_zero_shots = np.concatenate((zero_shots.flatten(),  simpleNet_zero_shots.flatten()))

from task import Task

para_dict = {'sbertNet_tuned': dict(zip(Task.TASK_LIST, np.mean(sbertNet_holdout_scores, axis=0))), 'simpleNet' : dict(zip(Task.TASK_LIST, np.mean(simpleNet_holdout_scores, axis=0)))}

from plotting import MODEL_STYLE_DICT, task_list, Line2D, mpatches

def plot_para_scores(all_perf_dict, save_file=None):
    barWidth = 0.15
    for i, item in enumerate(all_perf_dict.items()):  
        model_name, perf_dict = item
        values = list(perf_dict.values())
        len_values = len(holdout_types)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        if '_layer_11' in model_name: 
            mark_size = 4
        else: 
            mark_size = 3
        plt.plot(r, [1.05]*2, marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
        plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white')
    plt.ylim(0, 6)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], holdout_types)
    plt.tight_layout()
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys())[:-1]]
    #Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))

    plt.legend(handles=Patches)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

sbertNet_all_scores.shape
simpleNet_all_scores.shape
holdout_types = ['Holdouts', 'All Tasks']
np.mean(sbertNet_holdout_scores)
np.mean(simpleNet_holdout_scores, axis=(0,1))
dict(zip(holdout_types, [np.mean(sbertNet_holdout_scores), np.mean(sbertNet_all_scores)]))
para_dict = {'sbertNet_tuned': dict(zip(holdout_types, [np.mean(sbertNet_holdout_scores), np.mean(sbertNet_all_scores)])), 
                'simpleNet' : dict(zip(holdout_types, [np.mean(simpleNet_holdout_scores), np.mean(simpleNet_all_scores)]))}
para_dict

plot_para_scores(para_dict)


np.corrcoef(np.concatenate((sbertNet_holdout_scores.flatten(), simpleNet_holdout_scores.flatten())), 
            np.concatenate((zero_shots.flatten(),  simpleNet_zero_shots.flatten())))


np.corrcoef(normed_scores, 
            np.concatenate((zero_shots.flatten(),  simpleNet_zero_shots.flatten())))

np.corrcoef(simpleNet_holdout_scores.flatten(), simpleNet_zero_shots.flatten())


import matplotlib.pyplot as plt
plt.scatter(np.concatenate((sbertNet_holdout_scores.flatten(), simpleNet_holdout_scores.flatten())), 
            np.concatenate((zero_shots.flatten(),  simpleNet_zero_shots.flatten())))

from plotting import task_colors

task_colors
import itertools

tasks_and_seeds*2

normed_scores.shape
zero_shots.shape

import pickle
pickle.dump(sbertNet_all_scores, open('_ReLU128_5.7/sbert_tuned_para_scores', 'wb'))
pickle.dump(simpleNet_all_scores, open('_ReLU128_5.7/simpleNet_para_scores', 'wb'))


np.mean(scores1, axis=0)





#sbertNet_layer_11
model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model.model_name += '_tuned'
model.set_seed(1)
model.load_model('_ReLU128_5.7/single_holdouts/Multitask')

reduced_reps = reduce_rep(get_task_reps(model))[0]
reduced_reps.shape

from utils import comp_input_rule, get_input_rule
from task import Task


get_parallel_score(reduced_reps)
get_parallel_score(reduced_reps1)