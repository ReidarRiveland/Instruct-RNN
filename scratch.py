from collections import defaultdict
from re import I
from matplotlib.cbook import flatten

from matplotlib.pyplot import axis
from numpy.core.fromnumeric import size
from numpy.lib.function_base import append
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

keys_list = ['all_CCGP', 'holdout_CCGP']



plot_CCGP_scores(['sbertNet_tuned', 'sbertNet', 'bertNet_tuned','bertNet', 'gptNet_tuned', 'gptNet', 'bowNet', 'simpleNet'], 'task_stim_start_')


for swap_bool in [False, True]: 
    for model_name in ['sbertNet_tuned', 'sbertNet', 'bertNet_tuned','bertNet', 'gptNet_tuned', 'gptNet', 'bowNet', 'simpleNet']:
        print(model_name)
        model, _, _, _ = config_model_training(model_name)
        get_all_CCGP(model, 'task', swap=True)





for i, j in zip(instructs, instructs_reps): 
    print(i)
    print(j.shape)

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
    if i ==9: 
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

swapped = 'Anti Go'
#multitask
model1.set_seed(4)
task_file = task_swaps_map[swapped]
model1.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
#task scatter
reps1, _ = get_task_reps(model1, epoch='stim_start', stim_start_buffer=0, swapped_tasks=[swapped])
reduced_reps1 = reduce_rep(reps1)    
plot_rep_scatter(reduced_reps1[0], Task.TASK_GROUP_DICT['Go'], swapped_tasks=[swapped])


#task scatter
reduced_reps1 = reduce_rep(get_task_reps(model1))[0]

plot_rep_scatter(reduced_reps1, Task.TASK_GROUP_DICT['Go'])


#task RDM
all_sim_scores1 = get_layer_sim_scores(model1, 'Multitask', 'task')
plot_RDM(np.mean(all_sim_scores1, axis=0), 'task')


model = InstructNet(SBERT(20, train_layers=[], reducer=torch.mean), 128, 1)
model.model_name += '_tuned'
swapped = 'DMC'
#multitask
model.set_seed(4)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)


#task scatter
reps, _ = get_task_reps(model, epoch='stim_start', stim_start_buffer=0, swapped_tasks=[swapped])
reduced_reps = reduce_rep(reps)
plot_rep_scatter(reduced_reps[0], Task.TASK_GROUP_DICT['Delay'], swapped_tasks=[swapped])





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
swapped = 'Anti DM'
#multitask
model.set_seed(0)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
model.instruct_mode=''


unit = 12
var_of_insterest = 'diff_strength'
plot_neural_resp(model, 'DNMS', var_of_insterest, unit, 1)
plot_neural_resp(model, 'DMS', var_of_insterest, unit, 1)



plot_tuning_curve(model, Task.TASK_GROUP_DICT['DM'], var_of_insterest, unit, 1, [115]*4, swapped_task=swapped)

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


