from task import Task
task_list = Task.TASK_LIST

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pickle
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import colors, cm 

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


model_list = list(MODEL_STYLE_DICT.keys())[0:3] + list(MODEL_STYLE_DICT.keys())[4:]
model_list

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
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()
    return training_data

# plot_avg_seed_holdout('_ReLU128_14.6/single_holdouts/', model_list, 'correct', 2, smoothing = 0.01)
# plot_single_seed_holdout('_ReLU128_14.6/single_holdouts/', model_list, 'correct', 2, smoothing = 0.01)

def plot_trained_performance(perf_dict):
    barWidth = 0.1
    for i, model in enumerate(model_list):  
        values = list(perf_dict.values())
        len_values = len(task_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        plt.bar(r, values, width =barWidth, label = model_list[i])

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], task_list)
    plt.legend()
    plt.show()

def plot_rep_scatter(reps_reduced, tasks_to_plot): 
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*reps_reduced.shape[1] for task in tasks_to_plot]))
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]
    reps_to_plot = reps_reduced[task_indices, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=35)

    plt.xlabel("PC 1", fontsize = 18)
    plt.ylabel("PC 2", fontsize = 18)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]
    plt.legend(handles=Patches, loc=7)
    plt.show()

def plot_RDM(avg_reps, cmap=sns.color_palette("rocket_r", as_cmap=True)):
    opp_task_list = Task.TASK_LIST.copy()
    opp_task_list[1], opp_task_list[2] = opp_task_list[2], opp_task_list[1]

    avg_reps[[1,2], :] = avg_reps[[2,1], :] 
    sim_scores = 1-np.corrcoef(avg_reps)

    map = sns.heatmap(sim_scores, yticklabels = opp_task_list, xticklabels= opp_task_list, 
                        cmap=cmap, vmin=0, vmax=1)

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





if __name__ == "__main__":


    from task import make_test_trials
    from model_analysis import get_hid_var_resp

    from rnn_models import InstructNet, SimpleNet
    from nlp_models import SBERT, BERT
    from data import TaskDataSet
    from utils import train_instruct_dict
    import torch


    model = InstructNet(BERT(20, train_layers=['11']), 128, 1)
    #model = SimpleNet(128, 1)
    model.model_name+='_seed2'

    model.load_model('_ReLU128_14.6/single_holdouts/Anti_Go')
    model.to(torch.device(0))

    for task in task_group_dict['Go']:
        plot_neural_resp(model, task, 'direction', 108, 1)

    plot_neural_resp(model, 'Go', 'direction', 110, 0)
    plot_neural_resp(model, 'Anti Go', 'direction', 110, 1)
    plot_neural_resp(model, 'Anti RT Go', 'direction', 99, 1)




    make_tuning_curve(model, task_group_dict['Go'], 'direction', 110, 1, [115]*4)

    make_tuning_curve(model, ['Go', 'RT Go', 'Anti RT Go' ], 'direction', 110, 1, [115]*4)
    make_tuning_curve(model, 'RT Go', 'direction', 112, 1, 60)

    make_tuning_curve(model, ['Go'], 'direction', 108, 1, [30]*4)
    make_tuning_curve(model, 'Anti Go', 'direction', 108, 1, 0)
    make_tuning_curve(model, 'Anti RT Go', 'direction', 108, 1, 60)
