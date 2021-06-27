from numpy.random import seed
from task import Task, construct_batch
task_list = Task.TASK_LIST
from collections import defaultdict
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pickle

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import colors, cm 


task_group_colors = defaultdict(dict)
task_group_colors['Go'] = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange'}
task_group_colors['Decision Making'] = { 'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow'}
task_group_colors['Comparison'] = { 'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold'}
task_group_colors['Delay'] = { 'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

ALL_STYLE_DICT = {'simpleNet': ('blue', None), 'bowNet': ('orange', None), 'gptNet': ('red', '^'), 'gptNet_layer_11': ('red', '.'), 
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
            ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=10, markevery=250)
            ax.set_title(task_to_plot)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.55), title='Models', title_fontsize=12)
    fig.suptitle('Training for '+holdout+' Holdout', size=16)
    plt.show()

plot_single_seed_training('_ReLU128_14.6/single_holdouts/', 'DMC', ALL_STYLE_DICT.keys(), 'correct', 2, smoothing = 5)


model_list = list(ALL_STYLE_DICT.keys())[0:3] + list(ALL_STYLE_DICT.keys())[4:]
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
            ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=8, markevery=25)
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
        axn.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=8, markevery=20)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.75, 0.25), title='Models', title_fontsize=12)
    fig.suptitle('Avg. Performance on Heldout Tasks', size=16)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()
    return training_data

plot_avg_seed_holdout('_ReLU128_14.6/single_holdouts/', model_list, 'correct', 2, smoothing = 0.01)
plot_single_seed_holdout('_ReLU128_14.6/single_holdouts/', model_list, 'correct', 2, smoothing = 0.01)
















def plot_trained_performance(model_list):
    barWidth = 0.1
    for i, model in enumerate(model_list):  
        perf_dict = get_performance(model, 5)
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

def plot_model_response(model, task_type, trials = None, trial_num = 0, instruct = None):
    model.eval()
    with torch.no_grad(): 
        if trials == None: 
            task = construct_batch(task_type, 1)

        tar = task.targets
        ins = task.inputs

        if instruct is not None: task_info = [instruct]
        else: task_info = model.get_task_info(ins.shape[0], task_type)
        
        out, hid = model(task_info, ins)

        correct = isCorrect(out, torch.Tensor(tar), task.target_dirs)[trial_num]
        out = out.detach().cpu().numpy()[trial_num, :, :]
        hid = hid.detach().cpu().numpy()[trial_num, :, :]

        try: 
            task_info_embedding = model.langModel(task_info).unsqueeze(1).repeat(1, ins.shape[1], 1)
        except: 
            task_info_embedding = task_info.unsqueeze(1).repeat(1, ins.shape[1], 1)

        fix = ins[trial_num, :, 0:1]            
        mod1 = ins[trial_num, :, 1:1+Task.STIM_DIM]
        mod2 = ins[trial_num, :, 1+Task.STIM_DIM:1+(2*Task.STIM_DIM)]

        to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, task_info_embedding.T, tar[trial_num, :, :].T, out.squeeze().T]
        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 2, 5, 5])
        ylabels = ['fix.', 'mod. 1', 'mod. 2', 'Task Info', 'Target', 'Response']

        fig, axn = plt.subplots(6,1, sharex = True, gridspec_kw=gs_kw)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title(task_type +' trial info; correct: ' + str(correct))
            if i == 5: 
                ax.set_xlabel('time')

        plt.show()


