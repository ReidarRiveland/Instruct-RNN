from collections import defaultdict
from re import I
from matplotlib.cbook import flatten

from matplotlib.pyplot import axis
from numpy.core.fromnumeric import size, var
from numpy.lib.function_base import append
from numpy.ma import cos
from nlp_models import SBERT, BERT
from rnn_models import InstructNet, SimpleNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_layer_sim_scores, get_hid_var_group_resp, get_hid_var_resp, get_all_CCGP
import numpy as np
from utils import train_instruct_dict, task_swaps_map
from task import DM
from plotting import plot_RDM, plot_rep_scatter, plot_CCGP_scores, plot_model_response, plot_hid_traj_quiver, plot_dPCA, plot_neural_resp, plot_trained_performance, plot_tuning_curve
import torch

from task import Task, make_test_trials


from trainer import config_model_training

from plotting import MODEL_STYLE_DICT, Line2D, plt

from plotting import plot_CCGP_scores

from utils import all_models



unit=10
var_of_insterest = 'diff_strength'

model = InstructNet(SBERT(20, train_layers=[], reducer=torch.mean), 128, 1)
model.model_name += '_tuned'
model.set_seed(1)
swapped = 'Anti DM'
#multitask
model.set_seed(1)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)

perf = get_model_performance(model, 1)

perf

trials = plot_tuning_curve(model, Task.TASK_GROUP_DICT['DM'], var_of_insterest, unit, 1, [115]*4, swapped_task=swapped)








plot_CCGP_scores(all_models, 'task_stim_start_')

keys_list = ['all_CCGP', 'holdout_CCGP']



plot_CCGP_scores(['sbertNet_tuned', 'sbertNet', 'bertNet_tuned','bertNet', 'gptNet_tuned', 'gptNet', 'bowNet', 'simpleNet'], 'task_stim_start_')


for swap_bool in [False, True]: 
    for model_name in ['sbertNet_tuned', 'sbertNet', 'bertNet_tuned','bertNet', 'gptNet_tuned', 'gptNet', 'bowNet', 'simpleNet']:
        print(model_name)
        model, _, _, _ = config_model_training(model_name)
        get_all_CCGP(model, 'task', swap=True)






import pickle
from sklearn.manifold import TSNE
data_X =[]
for model_name in ['sbertNet_tuned', 'sbertNet', 'bertNet_tuned','bertNet', 'gptNet_tuned', 'gptNet']:
    all_sims = pickle.load(open('_ReLU128_5.7/swap_holdouts/Multitask/'+model_name+'/all_RDM_scores', 'rb'))
    for rdm in list(all_sims.values())[:-3]: 
        data_X.append(rdm.flatten())

import pickle
all_sims = pickle.load(open('_ReLU128_5.7/swap_holdouts/Multitask/sbertNet_tuned/all_RDM_scores', 'rb'))

for i, sim in enumerate(all_sims.items()): 
    if i<13:rep_type = 'lang'
    else: rep_type ='task'
    print(sim[0])
    plot_RDM(np.mean(sim[1], axis=0), rep_type, plot_title=str(sim[0]))



len(data_X)

X = np.array(data_X, dtype=np.float32)
X

X_embedded = TSNE(n_components=2, learning_rate=10).fit_transform(data_X)

from plotting import plt
colors = ['purple']*24+['green']*24+['red']*24
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
plt.show()

import pickle
all_sims = pickle.load(open('_ReLU128_5.7/swap_holdouts/Multitask/val_performance', 'rb'))

from plotting import plot_trained_performance

plot_trained_performance(all_sims)


#PLot validation instructions

model = InstructNet(SBERT(20, train_layers=[], reducer=torch.mean), 128, 1)

model1 = SimpleNet(128, 1, use_ortho_rules=True)

model.model_name += '_tuned'


swapped = 'COMP2'

model.set_seed(1)

task_file = task_swaps_map[swapped]
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)


task_file
model.load_model('_ReLU128_5.7/swap_holdouts/Multitask')




# reps, _ = get_task_reps(model, epoch='stim_start', stim_start_buffer=0, swapped_tasks=[swapped])
# reduced_reps = reduce_rep(reps)
# plot_rep_scatter(reduced_reps[0], Task.TASK_GROUP_DICT['COMP'])


i=12

instruct_index = 6

lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='full')

lang_rep_reduced, _ = reduce_rep(lang_reps)
plot_rep_scatter(lang_rep_reduced, Task.TASK_GROUP_DICT['DM'], annotate_tuples=[(1,5), (1, 1)], annotate_args=[(-100, 30), (-300, -50)])


trials, var_of_insterest = make_test_trials('COMP2', 'diff_strength', 0, num_trials=1)
plot_model_response(model, trials, instructions=[train_instruct_dict['COMP2'][5]])






#SimpleNet
model1 = SimpleNet(128, 1, use_ortho_rules=True)

swapped = 'DNMS'
#multitask
model1.set_seed(1)
task_file = task_swaps_map[swapped]
model1.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
#task scatter
reps1, _ = get_task_reps(model1, epoch='stim_start', stim_start_buffer=0)
reduced_reps1, _ = reduce_rep(reps1, dim=3)    
reduced_reps1.shape

plot_rep_scatter(reduced_reps1[..., 1:3], Task.TASK_GROUP_DICT['Delay'])



#task scatter
reduced_reps1 = reduce_rep(get_task_reps(model1))[0]

plot_rep_scatter(reduced_reps1, Task.TASK_GROUP_DICT['Go'])


#task RDM
all_sim_scores1 = get_layer_sim_scores(model1, 'Multitask', 'task')
plot_RDM(np.mean(all_sim_scores1, axis=0), 'task')


model = InstructNet(SBERT(20, train_layers=[], reducer=torch.mean), 128, 1)
model.model_name += '_tuned'
model.__seed_num_str__
for i, task in enumerate(Task.TASK_LIST): 
    swapped = task
    #multitask
    model.set_seed(1)
    task_file = task_swaps_map[swapped]
    task_file
    model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
    perf = get_model_performance(model, 1)
    print(task+' '+str(perf[i]))
    


model.instruct_mode

#task scatter
reps, _ = get_task_reps(model, epoch='stim_start', stim_start_buffer=0)
reduced_reps, _ = reduce_rep(reps, dim=3)
reduced_reps.shape
plot_rep_scatter(reduced_reps[..., 1:3], Task.TASK_GROUP_DICT['COMP'])

lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='full')
np.mean(lang_reps[7, ...], axis=0)





'Anti DM' in ['Anti DM']

plot_dPCA(model1, ['DM', 'Anti DM'], swapped_tasks=[])

task_grou_trajs = get_hid_var_group_resp(model, 'DM', 'diff_strength', swapped_tasks=['Anti DM'])
plot_hid_traj_quiver(task_grou_trajs, 'DM', [0, 1, 2, 3,4], [0], [1], context_task='Anti DM')


#lang scatter
lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer', swapped_tasks=[swapped])
lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
lang_rep_reduced, _ = reduce_rep(lang_reps)
lang_rep_reduced.shape
plot_rep_scatter(lang_rep_reduced, Task.TASK_GROUP_DICT['COMP'], swapped_tasks=[swapped])


#lang RDM
all_sim_scores = get_layer_sim_scores(model, 'Anti_Go', 'lang')
plot_RDM(np.mean(all_sim_scores, axis=0), 'lang')


model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model.model_name += '_tuned'
model.set_seed(0)

import pickle
from sklearn.metrics.pairwise import cosine_similarity

all_cos_sim = np.empty((16, 16))
all_lang_reps = np.empty((16, 15, 20))
all_contexts = np.empty((16, 256, 20))
for i, task in enumerate(Task.TASK_LIST): 
    swapped = task
    #multitask
    task_file = task_swaps_map[swapped]
    task_file
    model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
    model.instruct_mode=''
    lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='full')
    all_lang_reps[i, ...] = lang_reps[i, ...]
    foldername = '_ReLU128_5.7/swap_holdouts/'
    contexts = pickle.load(open(foldername+task_file+'/sbertNet_tuned/contexts/seed0_context_vecs20', 'rb'))
    all_contexts[i, ...] = contexts[i, ...]
    cos_sim= cosine_similarity(np.mean(lang_reps, axis=1), np.mean(contexts, axis=1), dense_output=False)
    all_cos_sim[i, :] = cos_sim[i, :]


lang_reps[7, ...]
cos_sim= cosine_similarity(np.mean(lang_reps, axis=1), np.mean(contexts, axis=1))
plt.show()

    
plot_RDM(all_cos_sim, 'task')


from plotting import task_colors, itertools, mpatches
def plot_rep_scatter(reps_reduced, tasks_to_plot, annotate_tuples=[], annotate_args=[], swapped_tasks= [], save_file=None): 
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.xlabel("PC 1", fontsize = 12)
    plt.ylabel("PC 2", fontsize = 12)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]
    reps_to_plot = reps_reduced[task_indices, 15:270, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*(270-15) for task in tasks_to_plot]))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1],  color='white', marker='o', edgecolors=colors_to_plot, s=25)
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*15 for task in tasks_to_plot]))
    reps_to_plot = reps_reduced[task_indices, 0:15, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=25)


    plt.legend(handles=Patches, fontsize='medium')

    if save_file is not None: 
        plt.savefig('figs/'+save_file)

    plt.show()


for i, task in enumerate(Task.TASK_LIST): 
    swapped = task
    #multitask
    task_file = task_swaps_map[swapped]
    task_file
    model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
    model.instruct_mode=''
    lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='full')
    foldername = '_ReLU128_5.7/swap_holdouts/'
    contexts = pickle.load(open(foldername+task_file+'/sbertNet_tuned/contexts/seed0_context_vecs20', 'rb'))
    all_reps = np.concatenate((lang_reps, contexts), axis=1)
    lang_rep_reduced, _ = reduce_rep(all_reps)
    index = int(np.floor(i/4))
    plot_rep_scatter(lang_rep_reduced, list(Task.TASK_GROUP_DICT.values())[index])

list(Task.TASK_GROUP_DICT.values())[0]

unit = 10
var_of_insterest = 'diff_strength'


model = InstructNet(SBERT(20, train_layers=[], reducer=torch.mean), 128, 1)
model.model_name += '_tuned'
model.set_seed(0)
swapped = 'Anti DM'
#multitask
model.set_seed(1)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)

perf = get_model_performance(model, 1)
print(task+' '+str(perf[i]))
    



trial = plot_neural_resp(model, 'Go', var_of_insterest, unit, 0)

trial = plot_neural_resp(model, 'Anti DM', var_of_insterest, unit, 0)

plot_model_response(model1, trial)

trial.plot_trial(0)

plot_tuning_curve(model, Task.TASK_GROUP_DICT['DM'], var_of_insterest, 110, 1, [115]*4, swapped_task=swapped)

model.set_seed(0)
model.load_model('_ReLU128_5.7/single_holdouts/Anti_DM')




unit = 110
task_group = 'DM'
task_var = 'diff_strength'
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][0], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][1], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][2], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][3], task_var, unit, 1)

plot_tuning_curve(model, Task.TASK_GROUP_DICT[task_group], task_var, unit, 1, [95]*4, swapped_task=None)

model.instruct_mode


