from re import I
from nlp_models import SBERT, BERT
from rnn_models import InstructNet, SimpleNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores, get_hid_var_group_resp, get_hid_var_resp
import numpy as np
from utils import train_instruct_dict, task_swaps_map
from task import DM
from plotting import plot_RDM, plot_rep_scatter, plot_CCGP_scores, plot_model_response, plot_hid_traj, plot_holdout_curves, plot_dPCA, plot_neural_resp, plot_tuning_curve
import torch

from task import Task, make_test_trials

model = InstructNet(BERT('bert', 20, train_layers=[], reducer=torch.mean), 128, 1)


model.model_name += '_tuned'


swapped = 'COMP2'

model.set_seed(2)

task_file = task_swaps_map[swapped]
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)

model.load_model('_ReLU128_5.7/single_holdouts/Multitask')


# reps, _ = get_task_reps(model, epoch='stim_start', stim_start_buffer=0, swapped_tasks=[swapped])
# reduced_reps = reduce_rep(reps)
# plot_rep_scatter(reduced_reps[0], Task.TASK_GROUP_DICT['COMP'])



all_sim_scores = get_sim_scores(model, 'Multitask', 'lang', depth='12', use_cos_sim=True)
plot_RDM(np.mean(all_sim_scores, axis=0), 'lang', cmap='Blues')


i=12

instruct_index = 6

lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='full')
lang_rep_reduced, _ = reduce_rep(lang_reps)
plot_rep_scatter(lang_rep_reduced, Task.TASK_GROUP_DICT['COMP'], annotate_tuples=[(1,5), (1, 1)], annotate_args=[(-100, 30), (-300, -50)])

trials, var_of_insterest = make_test_trials('COMP2', 'diff_strength', 0, num_trials=1)

plot_model_response(model, trials, instructions=[train_instruct_dict['COMP2'][5]])

task_grou_trajs = get_hid_var_group_resp(model, 'COMP', 'diff_strength', swapped_tasks=['COMP1'])

plot_hid_traj(task_grou_trajs, 'COMP', [0, 1, 2, 3], [0], [1, 5], context_task=1, annotate_tuples=[(1,5), (1, 1)])




for i in range(9, 13): 
    print(i)
    sim_scores = get_sim_scores(model, 'Multitask', rep_type='lang', depth=str(i))
    plot_RDM(np.mean(sim_scores, axis=0), 'lang')


    model_data_dict['sbertNet_tuned'][0, 4]
    model_data_dict = plot_holdout_curves('_ReLU128_5.7/swap_holdouts/', ['simpleNet'], 'correct', 'avg_holdout', range(5), instruction_mode = 'swap', smoothing = 0.01)

    for name, perf in model_data_dict.items():
        try:
            print(name, str(list(np.mean(perf, axis=(0,1))>0.99).index(True)))
        except: 
            print(name, str(-1))



    np.mean(model_data_dict['sbertNet_tuned'], axis=(0,1))
    zero_shot = model_data_dict['sbertNet_tuned'][0, :, 0]
    model_data_dict['sbertNet_tuned'].shape
    

    ###fig3###



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
all_sim_scores1 = get_sim_scores(model1, 'Multitask', 'task')
plot_RDM(np.mean(all_sim_scores1, axis=0), 'task')


model = InstructNet(SBERT(20, train_layers=[], reducer=torch.mean), 128, 1)
model.model_name += '_tuned'
swapped = 'Anti Go'
#multitask
model.set_seed(1)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)

len(model.langModel.state_dict().keys())

#task scatter
reps, _ = get_task_reps(model, epoch='stim_start', stim_start_buffer=0, swapped_tasks=[swapped])
reduced_reps = reduce_rep(reps)
plot_rep_scatter(reduced_reps[0], Task.TASK_GROUP_DICT['Go'], swapped_tasks=[swapped])

all_sim_scores = get_sim_scores(model, 'Anti_Go', 'task')
plot_RDM(np.mean(all_sim_scores, axis=0), 'task')



'Anti DM' in ['Anti DM']

plot_dPCA(model1, ['DM', 'Anti DM'], swapped_tasks=[])

task_grou_trajs = get_hid_var_group_resp(model, 'DM', 'diff_strength', swapped_tasks=['Anti DM'])
plot_hid_traj(task_grou_trajs, 'DM', [0, 1, 2, 3,4], [0], [1], context_task='Anti DM')


#lang scatter
lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer', swapped_tasks=[swapped])
lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
lang_rep_reduced, _ = reduce_rep(lang_reps)
lang_rep_reduced.shape
plot_rep_scatter(lang_rep_reduced, Task.TASK_GROUP_DICT['COMP'], swapped_tasks=[swapped])

#task RDM
swapped = 'DNMS'
#multitask
model.set_seed(0)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
model.instruct_mode=''

unit = 124
var_of_insterest = 'diff_direction'
trials = plot_neural_resp(model, 'C', var_of_insterest, unit, 1)
trials.plot_trial(0)
plot_neural_resp(model, 'DNMS', var_of_insterest, unit, 1)
trials, var_of_insterest = make_test_trials('Go', 'direction', 0, num_trials=6)
plot_tuning_curve(model, Task.TASK_GROUP_DICT['Delay'], var_of_insterest, unit, 1, [115]*4, swapped_task=swapped)

model.set_seed(0)
model.load_model('_ReLU128_5.7/single_holdouts/Anti_DM')

unit = 110
task_group = 'DM'
task_var = 'diff_strength'
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][0], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][1], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][2], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][3], task_var, unit, 1)

plot_tuning_curve(model, Task.TASK_GROUP_DICT[task_group], task_var, unit, 1, [115]*4, swapped_task=None)

model.instruct_mode


#lang RDM
all_sim_scores = get_sim_scores(model, 'Anti_Go', 'lang')
plot_RDM(np.mean(all_sim_scores, axis=0), 'lang')



model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
model.model_name += '_tuned'
swapped = 'DNMS'
#multitask
model.set_seed(0)
task_file = task_swaps_map[swapped]
task_file
model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
model.instruct_mode=''

unit = 124
var_of_insterest = 'diff_direction'
trials = plot_neural_resp(model, 'C', var_of_insterest, unit, 1)
trials.plot_trial(0)
plot_neural_resp(model, 'DNMS', var_of_insterest, unit, 1)
trials, var_of_insterest = make_test_trials('Go', 'direction', 0, num_trials=6)
plot_tuning_curve(model, Task.TASK_GROUP_DICT['Delay'], var_of_insterest, unit, 1, [115]*4, swapped_task=swapped)

model.set_seed(0)
model.load_model('_ReLU128_5.7/single_holdouts/Anti_DM')

unit = 110
task_group = 'DM'
task_var = 'diff_strength'
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][0], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][1], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][2], task_var, unit, 1)
plot_neural_resp(model, Task.TASK_GROUP_DICT[task_group][3], task_var, unit, 1)

plot_tuning_curve(model, Task.TASK_GROUP_DICT[task_group], task_var, unit, 1, [115]*4, swapped_task=None)

model.instruct_mode

