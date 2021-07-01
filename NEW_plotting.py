from numpy.core.fromnumeric import size
from model_analysis import get_hid_var_resp
from task import Task, make_test_trials
task_list = Task.TASK_LIST

from model_analysis import get_hid_var_resp

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pickle
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import colors, cm, markers 
from matplotlib.lines import Line2D
from matplotlib import rc


task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

task_group_dict = {'Go': ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'],
                'DM': ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], 
                'COMP': ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'],
                'Delay': ['DMS', 'DNMS', 'DMC', 'DNMC']}

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'bowNet': ('orange', None), 'gptNet': ('red', '^'), 'gptNet_layer_11': ('red', '.'), 
                        'bertNet': ('green', '^'), 'bertNet_layer_11': ('green', '.'), 'sbertNet': ('purple', '^'), 'sbertNet_layer_11': ('purple', '.')}


def plot_single_seed_training(foldername, holdout, model_list, train_data_type, seed, smoothing=0.1):
    seed = '_seed' + str(seed)
    task_file = holdout.replace(' ', '_')
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(19, 12))
    for model_name in model_list: 
        try: 
            training_data = pickle.load(open(foldername+task_file+'/'+model_name+seed+'_training_'+train_data_type, 'rb'))
        except FileNotFoundError: 
            print('No training data for '+ model_name + seed)
            continue 
        for i, ax in enumerate(axn.flat):
            task_to_plot = task_list[i]
            if task_to_plot == holdout: continue
            ax.set_ylim(-0.05, 1.15)
            smoothed_perf = gaussian_filter1d(training_data[task_to_plot], sigma=smoothing)
            ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=10, markevery=250)
            ax.set_title(task_to_plot)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.55), title='Models', title_fontsize=12)
    fig.suptitle('Training for '+holdout+' Holdout', size=16)
    plt.show()

# plot_single_seed_training('_ReLU128_14.6/single_holdouts/', 'DMC', MODEL_STYLE_DICT.keys(), 'correct', 2, smoothing = 5)


def plot_single_seed_holdout(foldername, model_list, train_data_type, seed, smoothing=0.1):
    seed = '_seed' + str(seed)
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(19, 12))
    for model_name in model_list: 
        for i, ax in enumerate(axn.flat):
            ax.set_ylim(-0.05, 1.15)
            task_to_plot = task_list[i]
            task_file = task_to_plot.replace(' ', '_')
            try:
                training_data = pickle.load(open(foldername+task_file+'/'+model_name+seed+'_holdout_'+train_data_type, 'rb'))
            except FileNotFoundError: 
                print('No training data for '+ model_name + seed+' '+task_to_plot)
                continue 
            smoothed_perf = gaussian_filter1d(training_data, sigma=smoothing)
            ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=8, markevery=25)
            ax.set_title(task_to_plot)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.55), title='Models', title_fontsize=12)
    fig.suptitle('Performance on Heldout Tasks', size=16)
    plt.show()

def plot_avg_seed_holdout(foldername, model_list, train_data_type, seed, smoothing=0.1):
    rc('font', weight='bold')

    seed = '_seed' + str(seed)
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(14, 10))
    axn.set_ylim(-0.05, 1.05)
    for model_name in model_list: 
        training_data = np.empty((len(task_list), 100))
        for i, task in enumerate(task_list):
            task_file = task.replace(' ', '_')
            try:
                training_data[i, :] = pickle.load(open(foldername+task_file+'/'+model_name+seed+'_holdout_'+train_data_type, 'rb'))
            except FileNotFoundError: 
                print('No training data for '+ model_name + seed+' '+task)
                continue 
        smoothed_perf = gaussian_filter1d(np.mean(training_data, axis=0), sigma=smoothing)
        axn.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=8, markevery=20)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.75, 0.25), title='Models', title_fontsize=12)
    fig.suptitle('Avg. Performance on Heldout Tasks', size=16)
    axn.xaxis.set_tick_params(labelsize=20)
    axn.yaxis.set_tick_params(labelsize=20)

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()
    return training_data

# plot_avg_seed_holdout('_ReLU128_14.6/single_holdouts/', model_list, 'correct', 2, smoothing = 0.01)
# plot_single_seed_holdout('_ReLU128_14.6/single_holdouts/', model_list, 'correct', 2, smoothing = 0.01)

def plot_trained_performance(all_perf_dict):
    barWidth = 0.15
    for i, item in enumerate(all_perf_dict.items()):  
        model_name, perf_dict = item
        values = list(perf_dict.values())
        len_values = len(task_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        if '_layer_11' in model_name: 
            mark_size = 4
        else: 
            mark_size = 3
        plt.plot(r, [1.05]*16, marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
        plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white')

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth+0.25 for r in range(len_values)], task_list)
    plt.tight_layout()
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys())[:-1]]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    plt.legend(handles=Patches)
    plt.show()


def plot_rep_scatter(reps_reduced, tasks_to_plot): 
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*reps_reduced.shape[1] for task in tasks_to_plot]))
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]
    reps_to_plot = reps_reduced[task_indices, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=35)

    plt.xlabel("PC 1", fontsize = 18)
    plt.ylabel("PC 2", fontsize = 18)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]
    plt.legend(handles=Patches)
    plt.show()

def plot_RDM(avg_reps, cmap=sns.color_palette("rocket_r", as_cmap=True)):
    opp_task_list = Task.TASK_LIST.copy()
    opp_task_list[1], opp_task_list[2] = opp_task_list[2], opp_task_list[1]

    avg_reps[[1,2], :] = avg_reps[[2,1], :] 
    sim_scores = 1-np.corrcoef(avg_reps)
    sns.set(font_scale=0.65)
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(9, 7))
    map = sns.heatmap(sim_scores, yticklabels = opp_task_list, xticklabels= opp_task_list, 
                        cmap=cmap, vmin=0, vmax=1, ax=axn, annot_kws={"size": 8})

    for i in range(4):
        plt.axhline(y = 4*i, xmin=i/4, xmax=(i+1)/4, color = 'k',linewidth = 3)
        plt.axhline(y = 4*(i+1), xmin=i/4, xmax=(i+1)/4, color = 'k',linewidth = 3)  
        plt.axvline(x = 4*i, ymin=1-i/4, ymax = 1-(i+1)/4, color = 'k',linewidth = 3)
        plt.axvline(x = 4*(i+1), ymin=1-i/4, ymax = 1-(i+1)/4, color = 'k',linewidth = 3)

    plt.show()


def make_tuning_curve(model, tasks, task_variable, unit, mod, times): 
    if task_variable == 'direction': 
        labels = ["0", "$2\pi$"]
        plt.xticks([0, np.pi, 2*np.pi], labels=['0', '$\pi$', '$2\pi$'])
    elif task_variable == 'diff_direction':
        labels = ["$\pi$", "0"]
    elif task_variable == 'strength':
        labels = ["0.3", "1.8"]
    elif task_variable =='diff_strength': 
        labels = ["delta -0.5", "delta 0.5"]
    y_max = 1.0
    for i, task in enumerate(tasks): 
        time = times[i]
        trials, var_of_interest = make_test_trials(task, task_variable, mod)
        _, hid_mean = get_hid_var_resp(model, task, trials)
        neural_resp = hid_mean[:, time, unit]
        plt.plot(var_of_interest, neural_resp, color=task_colors[task])
        y_max = max(y_max, neural_resp.max())
    plt.title('Tuning curve for Unit' + str(unit) + ' at time ' +str(time))
    plt.ylim(0, y_max+0.15)
    plt.xlabel(task_variable.replace('_', ' '))
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks]
    plt.legend(handles=Patches)
    plt.show()
    return trials

def plot_neural_resp(model, task_type, task_variable, unit, mod):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    trials, _ = make_test_trials(task_type, task_variable, mod)
    _, hid_mean = get_hid_var_resp(model, task_type, trials)
    if task_variable == 'direction': 
        labels = ["0", "$2\pi$"]
        cmap = plt.get_cmap('twilight') 
    elif task_variable == 'diff_direction':
        labels = ["$\pi$", "0"]
        cmap = plt.get_cmap('twilight') 
    elif task_variable == 'strength':
        labels = ["0.3", "1.8"]
        cmap = plt.get_cmap('plasma') 
    elif task_variable =='diff_strength': 
        labels = ["delta -0.5", "delta 0.5"]
        cmap = plt.get_cmap('plasma') 
    cNorm  = colors.Normalize(vmin=0, vmax=100)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    fig, axn = plt.subplots()
    ylim = np.max(hid_mean[:,:,unit])
    for i in [x*4 for x in range(25)]:
        plot = plt.plot(hid_mean[i, :, unit], c = scalarMap.to_rgba(i))
    plt.vlines(100, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    if 'RT' in task_type: 
        plt.xticks([100], labels=['Stim. Onset/Reponse'])
    else:
        plt.xticks([20, 100], labels=['Stim. Onset', 'Reponse'])
        plt.vlines(20, -1.5, ylim+0.15, colors='k', linestyles='dashed')

    # elif 'DM' in task_type:
    #     plt.vlines(20, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    #     plt.vlines(60, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    #     plt.xticks([20, 60, 100], labels=['Stim. 1 Onset', 'Stim. 2 Onset', 'Reponse'])



    axn.set_ylim(0, ylim+0.15)
    cbar = plt.colorbar(scalarMap, orientation='vertical', label = task_variable.replace('_', ' '), ticks = [0, 100])
    plt.title(task_type + ' response for Unit' + str(unit))
    cbar.set_ticklabels(labels)
    plt.show()
    return trials


model_list = list(MODEL_STYLE_DICT.keys())[0:3] + list(MODEL_STYLE_DICT.keys())[4:]
model_list

from task import make_test_trials
from model_analysis import get_hid_var_resp

from rnn_models import InstructNet, SimpleNet
from nlp_models import SBERT, BERT
from data import TaskDataSet
from utils import train_instruct_dict
import torch
from model_analysis import get_instruct_reps, get_hid_var_resp, get_task_reps, reduce_rep
from utils import train_instruct_dict
from mpl_toolkits.mplot3d import Axes3D

model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
#model = SimpleNet(128, 1)
model.model_name+='_seed2'

model.load_model('_ReLU128_14.6/single_holdouts/Anti_DM')
#model.to(torch.device(0))


task_group_hid_traj = np.empty((4, 15, 120, 128))
for i, task in enumerate(task_group_dict['Go']): 
    if 'RT' in task: 
        trials, vars = make_test_trials('RT Go', 'direction', 1, num_trials=1)
    else: 
        trials, vars = make_test_trials('Go', 'direction', 1, num_trials=1)
    for j, instruct in enumerate(train_instruct_dict[task]): 
        _, hid_mean = get_hid_var_resp(model, task, trials, num_repeats=5, task_info=[instruct])
        task_group_hid_traj[i, j,  ...] = np.squeeze(hid_mean)


task_group_hid_traj = np.empty((4, 15, 120, 128))

for i, task in enumerate(task_group_dict['DM']): 
    if 'Multi' in task: 
        trials, vars = make_test_trials('MultiDM', 'diff_strength', 1, num_trials=1)
    else: 
        trials, vars = make_test_trials('DM', 'diff_strength', 1, num_trials=1)
    for j, instruct in enumerate(train_instruct_dict[task]): 
        _, hid_mean = get_hid_var_resp(model, task, trials, num_repeats=5, task_info=[instruct])
        task_group_hid_traj[i, j,  ...] = np.squeeze(hid_mean)

trials, vars = make_test_trials('MultiDM', 'diff_strength', 1, num_trials=1)
trials, vars = make_test_trials('DM', 'diff_strength', 1, num_trials=1)

trials.plot_trial(0)


trial_epoch_size = 100

task_group = 'DM'

from sklearn.decomposition import PCA
embedder = PCA(n_components=3)

embedded = embedder.fit_transform(task_group_hid_traj.reshape(-1, 128)).reshape(4, 15, 120, 3)



alphas = np.linspace(0.5, 1.0, num=120)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


for i in range(4):
    for instruct_index in range(15): 
        ax.scatter(embedded[i, instruct_index, :, 0], embedded[i, instruct_index, :, 1], embedded[i, instruct_index, :,2], color = task_colors[list(task_group_dict[task_group])[i]], s=10, alpha=alphas)
        ax.scatter(embedded[i, instruct_index, 0, 0], embedded[i, instruct_index, 0, 1], embedded[i, instruct_index, 0,2],  s = trial_epoch_size, color='white', edgecolor= task_colors[list(task_group_dict[task_group])[i]], marker='*')
        ax.scatter(embedded[i, instruct_index, 119, 0], embedded[i, instruct_index, 119, 1], embedded[i, instruct_index, 119,2],  s = trial_epoch_size, color='white', edgecolor= task_colors[list(task_group_dict[task_group])[i]], marker='o')

        #if i%2 == 0: 
        ax.scatter(embedded[i, instruct_index, 19, 0], embedded[i, instruct_index, 19, 1], embedded[i, instruct_index, 19,2], s=trial_epoch_size, color='white', edgecolor= task_colors[list(task_group_dict[task_group])[i]], marker = 'X')
        ax.scatter(embedded[i, instruct_index, 99, 0], embedded[i, instruct_index, 99, 1], embedded[i, instruct_index, 99,2], s=trial_epoch_size, color='white', edgecolor= task_colors[list(task_group_dict[task_group])[i]], marker = 'P')
        # else: 
        #     ax.scatter(embedded[i, instruct_index, 99, 0], embedded[i, instruct_index, 99, 1], embedded[i, instruct_index, 99,2],  s=trial_epoch_size, color='white', edgecolor= task_colors[list(task_group_dict['Go'])[i]], marker = 'X')
        #     ax.scatter(embedded[i, instruct_index, 99, 0], embedded[i, instruct_index, 99, 1], embedded[i, instruct_index, 99,2],  s=trial_epoch_size, color='white', edgecolor= task_colors[list(task_group_dict['Go'])[i]], marker = 'P')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_zlim(-6, 6)

marker_list = [('*', 'Start'), ('X', 'Stim. Onset'), ('P', 'Resp.'), ('o', 'End')]
marker_patches = [(Line2D([0], [0], linestyle='None', marker=marker[0], color='grey', label=marker[1], 
                markerfacecolor='white', markersize=8)) for marker in marker_list]
Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in task_group_dict['DM']]
plt.legend(handles=Patches+marker_patches, fontsize = 'x-small')

plt.show()








from dPCA import dPCA



trials, var_of_insterest = make_test_trials('DM', 'diff_strength', 0, num_trials=1)
var_of_insterest
hid_resp, mean_hid_resp = get_hid_var_resp(model, 'DM', trials, num_repeats=3)

# # trial-average data
# R = mean(trialR,0)

# # center data
# R -= mean(R.reshape((N,-1)),1)[:,None,None]

reshape_mean_hid_resp = mean_hid_resp.T.swapaxes(-1, 1)
reshape_hid_resp = hid_resp.swapaxes(1, 2).swapaxes(-1, 1)

np.expand_dims(reshape_mean_hid_resp, -1).shape

#reshape_mean_hid_resp -= np.mean(mean_hid_resp.reshape((128, -1)), 1)[:, None, None]

dpca = dPCA.dPCA(labels='std',regularizer='auto')
dpca.protect = ['t']

Z = dpca.fit_transform(np.expand_dims(reshape_mean_hid_resp, -1), np.expand_dims(reshape_hid_resp, -1))


time = np.arange(120)

plt.figure(figsize=(16,7))
plt.subplot(131)

for s in range(6):
    plt.plot(time,Z['st'][0,s])

plt.title('1st mixing component')

plt.subplot(132)

for s in range(6):
    plt.plot(time,Z['t'][0,s])

plt.title('1st time component')
    
plt.subplot(133)
for s in range(6):
    plt.plot(time,Z['s'][0,s])

plt.title('1st Decision Variable component')
    

plt.figlegend(['delta'+ str(num) for num in np.round(var_of_insterest, 2)], loc=5)

plt.show()
