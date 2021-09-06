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



def plot_rep_scatter(reps_reduced, tasks_to_plot, annotate_tuples=[], annotate_args=[], swapped_tasks= [], save_file=None): 
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*reps_reduced.shape[1] for task in tasks_to_plot]))
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]

    reps_to_plot = reps_reduced[task_indices, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=35)
    ax.scatter(np.mean(reps_to_plot, axis=1)[:, 0], np.mean(reps_to_plot, axis=1)[:, 1], c = [task_colors[task] for task in tasks_to_plot], s=35, marker='X', edgecolors='white')
    
    centroid_reps = np.mean(reps_reduced, axis=1)
    comp_task_rule = comp_input_rule(1, tasks_to_plot[0]).squeeze()
    parallel_point = np.matmul(comp_task_rule, centroid_reps)
    task_rule = get_input_rule(1, tasks_to_plot[0], instruct_mode=None).numpy().squeeze()
    identical_point = np.matmul(task_rule, centroid_reps)
    plt.scatter(parallel_point[0], parallel_point[1], color='black', s=50)
    plt.scatter(identical_point[0], identical_point[1], color='black', s=50, marker='^')


    if len(swapped_tasks)>0: 
        ax.scatter(reps_reduced[-1, :, 0], reps_reduced[-1, :, 1], c = [task_colors[swapped_tasks[0]]]*reps_reduced.shape[1], marker='x')
    for i, indices in enumerate(annotate_tuples): 
        task_index, instruct_index = indices 
        plt.annotate(str(1+instruct_index)+'. '+two_line_instruct(train_instruct_dict[tasks_to_plot[task_index]][instruct_index]), xy=(flattened_reduced[int(instruct_index+(task_index*15)), 0], flattened_reduced[int(instruct_index+(task_index*15)), 1]), 
                    xytext=annotate_args[i], size = 8, arrowprops=dict(arrowstyle='->'), textcoords = 'offset points')

    plt.xlabel("PC 1", fontsize = 12)
    plt.ylabel("PC 2", fontsize = 12)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]
    plt.legend(handles=Patches, fontsize='medium')

    if save_file is not None: 
        plt.savefig('figs/'+save_file)

    plt.show()

plot_rep_scatter(reduced_reps, Task.TASK_GROUP_DICT['Delay'])




scores = get_total_para_score(model, 'RT_Go', 'task')
np.mean(scores, axis=0)

def get_all_holdout_para_scores(model): 
    all_para_scores = np.empty((16, 5, 4))
    for i, task in enumerate(Task.TASK_LIST): 
        print(task)
        task_file = task.replace(' ', '_')
        scores = get_total_para_score(model, task_file, 'task')
        print(scores)
        all_para_scores[i, ...] = scores
    return all_para_scores

from model_analysis import get_all_holdout_para_scores

sbertNet_all_scores = get_all_holdout_para_scores(model)
simpleNet_all_scores = get_all_holdout_para_scores(model1)

simpleNet_zero_shots = model_data_dict['simpleNet'][:, :, 0]
zero_shots = model_data_dict['sbertNet_tuned'][:, :, 0]


def get_holdout_para_scores(all_scores):
    holdout_scores = np.empty((5, 16))
    for i in range(16): 
        j=int(np.floor(i/4))
        holdout_scores[:, i]=all_scores[i, :, j]
    return holdout_scores

sbertNet_holdout_scores = get_holdout_para_scores(sbertNet_all_scores)
simpleNet_holdout_scores = get_holdout_para_scores(simpleNet_all_scores)

from sklearn import preprocessing
normed_scores =preprocessing.normalize(np.concatenate((sbertNet_holdout_scores.flatten(), simpleNet_holdout_scores.flatten())).reshape(1, -1))
all_zero_shots = np.concatenate((zero_shots.flatten(),  simpleNet_zero_shots.flatten()))



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
tasks_and_seeds = list(itertools.chain.from_iterable([[color]*5 for color in task_colors.values()]))
tasks_and_seeds*2

normed_scores.shape
zero_shots.shape

m,b = np.polyfit(normed_scores.squeeze(), all_zero_shots.squeeze(), 1) 

coef = np.polyfit(normed_scores.squeeze(), all_zero_shots.squeeze(), 1) 
poly1d_fn = np.poly1d(coef) 

x = np.arange(0, 0.25,0.001)

plt.plot(x, poly1d_fn(x), '--k')

plt.scatter(normed_scores, 
            all_zero_shots,
            c = ['purple']*16*5+['blue']*16*5, edgecolors=tasks_and_seeds*2)
r_score = np.round(np.corrcoef(normed_scores, all_zero_shots)[0, 1], 3)
plt.ylim(-0.05, 1.05)
plt.text(0.20, 0.95, str('r = '+str(r_score)))

plt.show()

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