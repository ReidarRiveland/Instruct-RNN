# import numpy as np

# from plotting import plot_RDM, plot_avg_curves, plot_k_shot_learning, plot_model_response, plot_task_curves, plot_trained_performance, plot_rep_scatter, plot_tuning_curve, plot_neural_resp, plot_CCGP_scores, plot_hid_traj, plot_val_performance, plot_RDM
# from utils.task_info_utils import train_instruct_dict, test_instruct_dict
# from model_analysis import get_layer_sim_scores, get_model_performance, get_multitask_val_performance, get_task_reps, reduce_rep, get_instruct_reps, get_hid_var_group_resp
# import pickle
# from task import Task


# foldername = '_ReLU128_4.11/swap_holdouts'


# plot_k_shot_learning(all_models, '_ReLU128_4.11/swap_holdouts')


# #Figure 1
# data_dict = plot_task_curves('_ReLU128_4.11/swap_holdouts', all_models[::-1],'correct', )

# data_dict = plot_task_curves('_ReLU128_4.11/swap_holdouts', ['simpleNet'],'correct')


# data_dict = plot_task_curves('_ReLU128_4.11/swap_holdouts', all_models[::-1],'correct')


# np.mean(data_dict['sbertNet_tuned'][''][0, :, 8:12, :], axis=(0,1))
# np.std(np.mean(data_dict['sbertNet'][''][0, :, 8:12, :], axis=1), axis=0)

# zero_shot_dict = {}
# for model_name in all_models: 
#     zero_shot_perf = data_dict[model_name][''][0, :, :, 0]
#     print(model_name)
#     zero_shot_dict[model_name]=zero_shot_perf

# np.mean(zero_shot_dict['gptNet_tuned'])

# model_name = 'simpleNet'
# np.mean(data_dict[model_name][''][0, :, 1, 0])

# plot_trained_performance(zero_shot_dict)


# all_val_perf = pickle.load(open(foldername+'/Multitask/val_perf_dict', 'rb'))
# del all_val_perf['simpleNet']
# plot_trained_performance(all_val_perf)



# #Figure 2
# data_dict = plot_avg_curves('_ReLU128_4.11/swap_holdouts', all_models[::-1],'correct', split_axes=True, plot_swaps=True)





# #Figure 3
# from task import Task
# def make_rep_scatter(model, task_to_plot=Task.TASK_GROUP_DICT['Go'], swapped_tasks = []): 
#     model_reps, _ = get_task_reps(model, epoch='stim_start', swapped_tasks=swapped_tasks)
#     reduced_reps, _ = reduce_rep(model_reps)
#     plot_rep_scatter(reduced_reps, task_to_plot, swapped_tasks=swapped_tasks)

# sbert_tuned_multi = config_model('sbertNet_tuned')
# sbert_tuned_multi.set_seed(3)
# sbert_tuned_multi.load_model('_ReLU128_4.11/swap_holdouts/Multitask')

# sim_scores = get_layer_sim_scores(sbert_tuned_multi, 'Multitask', foldername, use_cos_sim=True)
# plot_RDM(np.mean(sim_scores, axis=0), 'lang', cmap='Blues')

# simple_multi = config_model('simpleNet')
# simple_multi.set_seed(1)
# simple_multi.load_model('_ReLU128_4.11/swap_holdouts/Multitask')
# make_rep_scatter(simple_multi)


# sbert_tuned_anti_go = config_model('sbertNet_tuned')
# sbert_tuned_anti_go.model_name
# sbert_tuned_anti_go.set_seed(2)
# task_file = task_swaps_map['Anti Go']
# sbert_tuned_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

# make_rep_scatter(sbert_tuned_anti_go)


# simple_anti_go = config_model('simpleNet')
# simple_anti_go.set_seed(2)
# task_file = task_swaps_map['DM']
# task_file
# simple_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)
# make_rep_scatter(simple_anti_go)

# hid_reps = get_hid_var_group_resp(sbert_tuned_anti_go, 'Go', 'direction')
# plot_hid_traj(hid_reps, 'Go', [0, 1, 2, 3], [0], [0], s=5)

# sbert_tuned_anti_go.langModel.transformer

# model_reps = get_instruct_reps(sbert_tuned_anti_go.langModel, train_instruct_dict, depth='12')
# reduced_reps, _ = reduce_rep(model_reps)
# plot_rep_scatter(reduced_reps, Task.TASK_GROUP_DICT['Go'])



# sbert_tuned_anti_go = config_model('sbertNet_tuned')
# sbert_tuned_anti_go.model_name
# sbert_tuned_anti_go.set_seed(2)
# task_file = task_swaps_map['Anti Go']
# sbert_tuned_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)
# make_rep_scatter(sbert_tuned_anti_go)


# simple_anti_go = config_model('simpleNet')
# simple_anti_go.set_seed(2)
# task_file = task_swaps_map['DM']
# task_file
# simple_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)
# make_rep_scatter(simple_anti_go)


# plot_CCGP_scores(all_models, rep_type_file_str='task_stim_start_', plot_swaps=True)

# #single neurons 

# import torch
# device = torch.device(0)


# sbert_tuned_anti_go = config_model('sbertNet_tuned')
# sbert_tuned_anti_go.model_name
# sbert_tuned_anti_go.set_seed(3)
# task_file = task_swaps_map['Anti Go']
# sbert_tuned_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

# var_of_insterest = 'direction'
# unit = 86
# plot_neural_resp(sbert_tuned_anti_go, 'Anti Go', var_of_insterest, unit, 1)
# plot_tuning_curve(sbert_tuned_anti_go, Task.TASK_GROUP_DICT['Go'], var_of_insterest, unit, 1, [115]*4)


# sbert_tuned_anti_dm = config_model('sbertNet_tuned')
# sbert_tuned_anti_dm.set_seed(3)
# task_file = task_swaps_map['Anti DM']
# sbert_tuned_anti_dm.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

# var_of_insterest = 'diff_strength'
# unit = 11
# plot_neural_resp(sbert_tuned_anti_dm, 'Anti DM', var_of_insterest, unit, 1)
# plot_neural_resp(sbert_tuned_anti_dm, 'DM', var_of_insterest, unit, 1)
# plot_tuning_curve(sbert_tuned_anti_dm, Task.TASK_GROUP_DICT['DM'], var_of_insterest, unit, 1, [119]*4)


# sbert_tuned_comp = config_model('sbertNet_tuned')
# sbert_tuned_comp.set_seed(1)
# task_file = task_swaps_map['COMP1']
# sbert_tuned_comp.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

# sbert_tuned_comp.to(device)

# var_of_insterest = 'diff_strength'
# unit = 42
# plot_tuning_curve(sbert_tuned_comp, Task.TASK_GROUP_DICT['COMP'][::-1], var_of_insterest, unit, 1, [115]*4, num_repeats=200)




# sbert_tuned_dms = config_model('sbertNet_tuned')
# sbert_tuned_dms.set_seed(0)
# task_file = task_swaps_map['DNMS']
# sbert_tuned_dms.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

# sbert_tuned_dms.to(device)

# var_of_insterest = 'diff_direction'
# unit = 96
# plot_neural_resp(sbert_tuned_dms, 'DMS', var_of_insterest, unit, 1)
# plot_neural_resp(sbert_tuned_dms, 'DNMS', var_of_insterest, unit, 1)

# plot_tuning_curve(sbert_tuned_dms, Task.TASK_GROUP_DICT['Delay'], var_of_insterest, unit, 1, [119]*4, num_repeats=20)

# #COMP COMPARISON





# #use actual directions


# from task import make_test_trials
# from model_analysis import get_hid_var_resp
# from plotting import plot_model_response

# def special_comp_resp(model):     
#     task_infos = ['select the first stimulus if it is presented with higher intensity than the second stimulus otherwise do not respond',
#                     'select the second stimulus if it is presented with the higher intensity than the first stimulus otherwise do not respond',
#                     train_instruct_dict['MultiCOMP1'][0],
#                     train_instruct_dict['MultiCOMP2'][0]]

#     task_group_hid_traj = np.empty((4, 1, 1, 120, 128))
#     for i, task in enumerate(Task.TASK_GROUP_DICT['COMP']):
#         trials, vars = make_test_trials(task, 'diff_strength', 1, num_trials=1, sigma_in=0.1)
#         _, hid_mean = get_hid_var_resp(model, task, trials, num_repeats=3, task_info=[task_infos[i]])
#         task_group_hid_traj[i, 0,  ...] = hid_mean
#         if i<2: 
#             plot_model_response(model, trials, instructions=[task_infos[i]])
#     return task_group_hid_traj

# #untuned_hid_reps = get_hid_var_group_resp(sbert_comp, 'COMP', 'diff_direction')
# untuned_hid_reps = special_comp_resp(sbert_comp)
# plot_hid_traj(untuned_hid_reps, 'COMP', [0, 1], [0], [0], s=5)

# hid_reps = special_comp_resp(sbert_tuned_comp)
# plot_hid_traj(hid_reps, 'COMP', [0, 1], [0], [0], s=5)





# from model_analysis import get_instruct_reps
# from utils.utils import task_swaps_map

# plot_list = [('Anti Go', 'Go'), ('Anti DM', 'DM'), ('COMP2', 'COMP'), ('DMC', 'Delay')]

# model = config_model('gptNet_tuned')

# for task, group in plot_list:     
#     holdout_task = task
#     task_group = group
#     seed = 0

#     task_file = task_swaps_map[holdout_task]

#     model.set_seed(seed)
#     model.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

#     lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='12', swapped_tasks=[holdout_task])
#     reduced, _ = reduce_rep(lang_reps)
#     plot_rep_scatter(reduced,  Task.TASK_GROUP_DICT[task_group], swapped_tasks=[holdout_task] )


# #make_rep_scatter(sbert_tuned_anti_go, task_to_plot = Task.TASK_GROUP_DICT['Delay'], swapped_tasks=[holdout_task])





# def plot_lang_RDM(model, layer):    
#     sim_scores = get_layer_sim_scores(model, layer)
#     plot_RDM(sim_scores, 'task', cmap='Reds')


# exp_file = '_ReLU128_4.11/swap_holdouts/Multitask'




from models.full_models import make_default_model
from model_analysis import get_layer_sim_scores
import seaborn as sns
import matplotlib.pyplot as plt
from task import Task
from model_analysis import get_task_reps, reduce_rep, get_instruct_reps
from plotting import plot_rep_scatter
import pickle
import numpy as np 
import torch
import itertools
from utils.utils import task_colors
import matplotlib.patches as mpatches
from matplotlib import colors, cm, markers, use 
from matplotlib.lines import Line2D

exp_file = '_ReLU128_4.11/swap_holdouts/Multitask'

def _rep_scatter(reps_reduced, task, ax, **scatter_kwargs): 
    task_reps = reps_reduced[Task.TASK_LIST.index(task), ...]
    ax.scatter(task_reps[:, 0], task_reps[:, 1], s=25, edgecolors=[task_colors[task]]*task_reps.shape[0], **scatter_kwargs)
    patch = Line2D([0], [0], label = task, linestyle='None', markersize=8, **scatter_kwargs)
    return patch

def _group_rep_scatter(reps_reduced, task_to_plot, ax, **scatter_kwargs): 
    Patches = []
    for task in task_to_plot: 
        if bool(scatter_kwargs): 
             _ = _rep_scatter(reps_reduced, task, ax, **scatter_kwargs)
        else:
            patch = _rep_scatter(reps_reduced, task, ax, color = task_colors[task], marker='o')
            Patches.append(patch)
    return Patches

def plot_lang_vs_context(model, contexts, tasks_to_plot, mode='task'): 
    if mode == 'task': 
        context_reps = get_task_reps(model, epoch='stim_start', num_trials = 128, contexts=torch.Tensor(contexts))
        wInstruct_reps = get_task_reps(model, epoch='stim_start', num_trials = 128)
        reduced, _ = reduce_rep(np.concatenate((context_reps, wInstruct_reps)))
        context_reps_reduced = reduced[0:16, ...]
        wInstruct_reps_reduced = reduced[16:,...]
    elif mode == 'lang': 
        wInstruct_reps = get_instruct_reps(model.langModel)
        modulo = contexts.shape[1]-contexts.shape[1]%wInstruct_reps.shape[1]
        context_reps = contexts[:, :modulo, :].reshape(-1, wInstruct_reps.shape[1], wInstruct_reps.shape[-1])
        reduced, _ = reduce_rep(np.concatenate((context_reps, wInstruct_reps)))
        context_reps_reduced = reduced[0:-16, ...].reshape(16, -1, 2)
        wInstruct_reps_reduced = reduced[-16:,...]

    _, ax = plt.subplots(figsize=(6,6))
    _ = _group_rep_scatter(context_reps_reduced, tasks_to_plot,  ax, color='white', marker='X')

    Patches = _group_rep_scatter(wInstruct_reps_reduced, tasks_to_plot,   ax)
    Patches.append((Line2D([0], [0], linestyle='None', marker='X', color='grey', label='Contexts', 
                    markerfacecolor='white', markersize=8)))
    plt.legend(handles=Patches, fontsize='medium')
    plt.show()

def load_contexts(model_name, seed):
    all_contexts = np.empty((16, 128, 20))
    for i, task in enumerate(Task.TASK_LIST): 
        filename = exp_file+'/'+model_name+'/contexts/seed'+str(seed)+task+'supervised_context_vecs20'
        task_contexts = pickle.load(open(filename, 'rb'))
        all_contexts[i, ...]=task_contexts[:128, :]
    return all_contexts


model_name = 'sbertNet_tuned'
seed = 0
contexts = load_contexts(model_name, seed)
model = make_default_model(model_name)
model.load_model(exp_file+'/'+model_name, suffix='_seed'+str(seed))

plot_lang_vs_context(model, contexts, Task.TASK_GROUP_DICT['Delay'], mode='task')


#Figure 3
from task import Task
def make_rep_scatter(model, task_to_plot=Task.TASK_GROUP_DICT['Go'], swapped_tasks = []): 
    model_reps, _ = get_task_reps(model, epoch='stim_start', swapped_tasks=swapped_tasks)
    reduced_reps, _ = reduce_rep(model_reps)
    plot_rep_scatter(reduced_reps, task_to_plot, swapped_tasks=swapped_tasks)

