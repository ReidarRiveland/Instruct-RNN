import numpy as np

from plotting import plot_avg_curves, plot_task_curves, plot_trained_performance, plot_rep_scatter, plot_tuning_curve, plot_neural_resp, plot_CCGP_scores
from utils import all_models, task_swaps_map
from model_trainer import config_model
from model_analysis import get_model_performance, get_multitask_val_performance, get_task_reps, reduce_rep
import pickle
from task import Task

foldername = '_ReLU128_4.11/swap_holdouts'

task_file = task_swaps_map['DM']
data_dict = plot_task_curves('_ReLU128_4.11/swap_holdouts', ['sbertNet_tuned', 'bertNet_tuned', 'sbertNet'],'correct')
data_dict = plot_task_curves('_ReLU128_4.11/swap_holdouts', ['sbertNet_tuned', 'simpleNet', 'gptNet'],'correct', seeds=[1, 2, 3])
data_dict = plot_avg_curves('_ReLU128_4.11/swap_holdouts', ['sbertNet_tuned', 'simpleNet', 'bertNet', 'gptNet'],'correct', split_axes=True)
data_dict = plot_avg_curves('_ReLU128_4.11/swap_holdouts', ['simpleNet', 'bertNet_tuned', 'sbertNet_tuned'],'correct', split_axes=True)
data_dict = plot_avg_curves('_ReLU128_4.11/swap_holdouts', ['simpleNet', 'sbertNet_tuned'],'correct', split_axes=True, plot_swaps=True)
data_dict = plot_avg_curves('_ReLU128_4.11/swap_holdouts', all_models,'correct', split_axes=True)

data_dict = plot_task_curves('_ReLU128_4.11/swap_holdouts', all_models[::-1],'correct', train_folder='Multitask')


np.mean()
np.mean(data_dict['sbertNet_tuned'][''][0, ...], axis=(0,1))


all_val_perf = pickle.load(open(foldername+'/Multitask/val_perf_dict', 'rb'))
del all_val_perf['simpleNet']
plot_trained_performance(all_val_perf)


#Figure 3
def make_rep_scatter(model, task_to_plot=Task.TASK_GROUP_DICT['Go']): 
    model_reps, _ = get_task_reps(model, epoch='stim_start')
    reduced_reps, _ = reduce_rep(model_reps)
    plot_rep_scatter(reduced_reps, task_to_plot)

sbert_tuned_multi = config_model('sbertNet_tuned')
sbert_tuned_multi.set_seed(3)
sbert_tuned_multi.load_model('_ReLU128_4.11/swap_holdouts/Multitask')


from model_analysis import get_layer_sim_scores
sim_scores = get_layer_sim_scores(sbert_tuned_multi, 'Multitask', '_ReLU128_4.11/swap_holdouts', use_cos_sim=True)

from plotting import plot_RDM

plot_RDM(np.mean(sim_scores, axis=0), 'lang', cmap='Blues')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


make_rep_scatter(sbert_tuned_multi)

simple_multi = config_model('simpleNet')
simple_multi.set_seed(1)
simple_multi.load_model('_ReLU128_4.11/swap_holdouts/Multitask')
make_rep_scatter(simple_multi)

simple_multi.instruct_mode='masked'

perf = get_model_performance(simple_multi, 3)

plot_trained_performance({'simpleNet': np.expand_dims(perf, 0)})

task_rule = np.random.rand(128, 20)
task_rule


perf

np.mean(perf)

sbert_tuned_anti_go = config_model('sbertNet_tuned')
sbert_tuned_anti_go.model_name
sbert_tuned_anti_go.set_seed(2)
task_file = task_swaps_map['Anti Go']
sbert_tuned_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)
make_rep_scatter(sbert_tuned_anti_go)


simple_anti_go = config_model('simpleNet')
simple_anti_go.set_seed(2)
task_file = task_swaps_map['DM']
task_file
simple_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(simple_anti_go.rule_transform[1, :].repeat(120, 1).numpy(), cmap = 'Reds')
plt.show()


make_rep_scatter(simple_anti_go)

simple_anti_go.instruct_mode='masked'
perf = get_model_performance(simple_anti_go, 3)
plot_trained_performance({'simpleNet': np.expand_dims(perf, 0)})

np.mean(perf)


plot_CCGP_scores(all_models, rep_type_file_str='task_stim_start_', plot_swaps=True)

#single neurons 
sbert_tuned_anti_go = config_model('sbertNet_tuned')
sbert_tuned_anti_go.model_name
sbert_tuned_anti_go.set_seed(3)
task_file = task_swaps_map['Anti Go']
sbert_tuned_anti_go.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

var_of_insterest = 'direction'
unit = 86
plot_neural_resp(sbert_tuned_anti_go, 'Anti Go', var_of_insterest, unit, 1)
plot_tuning_curve(sbert_tuned_anti_go, Task.TASK_GROUP_DICT['Go'], var_of_insterest, unit, 1, [115]*4)


sbert_tuned_anti_dm = config_model('sbertNet_tuned')
sbert_tuned_anti_dm.set_seed(3)
task_file = task_swaps_map['Anti DM']
sbert_tuned_anti_dm.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

var_of_insterest = 'diff_strength'
unit = 11
plot_neural_resp(sbert_tuned_anti_dm, 'Anti DM', var_of_insterest, unit, 1)
plot_neural_resp(sbert_tuned_anti_dm, 'DM', var_of_insterest, unit, 1)

plot_tuning_curve(sbert_tuned_anti_dm, Task.TASK_GROUP_DICT['DM'], var_of_insterest, unit, 1, [119]*4)


sbert_tuned_anti_dm = config_model('sbertNet_tuned')
sbert_tuned_anti_dm.set_seed(4)
task_file = task_swaps_map['Anti DM']
sbert_tuned_anti_dm.load_model('_ReLU128_4.11/swap_holdouts/'+task_file)

var_of_insterest = 'diff_strength'
unit = 8
plot_tuning_curve(sbert_tuned_anti_dm, Task.TASK_GROUP_DICT['DM'], var_of_insterest, unit, 1, [115]*4)



