from ast import Mod
from cmath import nan
from math import exp
from pyexpat import model
from random import seed
from turtle import color
from xml.dom.expatbuilder import theDOMImplementation
import matplotlib
from pyparsing import col
from instructRNN.analysis.model_analysis import *
from instructRNN.tasks.tasks import TASK_LIST
from instructRNN.data_loaders.perfDataFrame import HoldoutDataFrame, TrainingDataFrame
from instructRNN.tasks.task_criteria import isCorrect

from instructRNN.instructions.instruct_utils import train_instruct_dict, test_instruct_dict
from instructRNN.tasks.task_factory import STIM_DIM
from instructRNN.instructions.instruct_utils import get_task_info

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import colors, cm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib
import torch

from scipy.ndimage import gaussian_filter1d
import warnings

plt.style.use('ggplot')

Blue = '#1C3FFD'
lightBlue='#ADD8E6'
lightRed = '#FF6D6A'
Green = '#45BF55'
Red = '#FF3131'
Orange = '#FFA500'
Yellow = '#FFEE58'
Purple = '#800080'


MODEL_STYLE_DICT = {'simpleNet': (Blue, None, 'simpleNet'), 'simpleNetPlus': (Blue, '+', 'simpleNetPlus'), 
                    'comNet': (lightBlue, 'None', 'comNet'), 'comNetPlus': (lightBlue, '+', 'comNetPlus'), 
                    #'clipNet': (Yellow, None), 'clipNet_tuned': (Yellow, 'v'), 
                    'clipNet_lin': (Yellow, 'None', 'clipNet'), 'clipNet_lin_tuned': (Yellow, 'v', 'clipNet (tuned)'), 
                    #'bowNet': (Orange, None), 
                    'bowNet_lin': (Orange, None, 'bowNet'), 
                    'gptNet_lin': (lightRed, None, 'gptNet'), 'gptNet_lin_tuned': (lightRed, 'v','gptNet (tuned)'), 
                    'gptNetXL_lin': (Red, None, 'gptNetXL'), 'gptNetXL_lin_tuned': (Red, None, 'gptNetXL (tuned)'), 
                    #'bertNet': (Green, None), 'bertNet_tuned': (Green, 'v'),  
                    'bertNet_lin': (Green, None, 'bertNet'), 'bertNet_lin_tuned': (Green, 'v', 'bertNet (tuned)'),  
                    #'sbertNet': (Purple, None), 'sbertNet_tuned': (Purple, 'v'),
                    'sbertNet_lin': (Purple, None, 'sbertNet'), 'sbertNet_lin_tuned': (Purple, 'v', 'sbertNet (tuned)')}

def get_task_color(task): 
    index = TASK_LIST.index(task)
    return plt.get_cmap('Paired')(index%12)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from matplotlib import rc
plt.rcParams["font.family"] = "serif"


def split_axes():
    inset1_lims = (-1, 10)
    inset2_lims = (80, 99)
    gs_kw = dict(width_ratios=[inset1_lims[1]-inset1_lims[0],inset2_lims[1]-inset2_lims[0]], height_ratios=[1])
    fig,(axn,ax2) = plt.subplots(1,2,sharey=True, facecolor='w',  gridspec_kw=gs_kw, figsize =(6, 4))

    axn.set_xlim(inset1_lims)
    ax2.set_xlim(inset2_lims)

    ax2.yaxis.set_visible(False)

    # hide the spines between ax and ax2
    axn.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    axn.set_ylim(0.0, 1.0)

    axn.xaxis.set_tick_params(labelsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)

    axn.yaxis.set_tick_params(labelsize=10)            
    axn.set_yticks(np.linspace(0, 1, 11))
    return fig, axn, ax2

def _plot_performance_curve(avg_perf, std_perf, plt_ax, model_name, **plt_args):
        color = MODEL_STYLE_DICT[model_name][0] 
        marker = MODEL_STYLE_DICT[model_name][1]
        plt_ax.fill_between(np.linspace(0, avg_perf.shape[-1], avg_perf.shape[-1]), np.min(np.array([np.ones(avg_perf.shape[-1]), avg_perf+std_perf]), axis=0), 
                                        avg_perf-std_perf, color = color, alpha= 0.1)

        plt_ax.plot(avg_perf, color = color, marker=marker, **plt_args)

def _make_model_legend(model_list): 
    Patches = []
    for model_name in model_list: 
        color = MODEL_STYLE_DICT[model_name][0]
        marker = MODEL_STYLE_DICT[model_name][1]
        if marker is None: marker = 's'
        Patches.append(Line2D([], [], linestyle='None', marker=marker, color=color, label=model_name, 
                    markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=4))
    plt.legend(handles=Patches)

def plot_avg_holdout_curve(foldername, exp_type, model_list, perf_type='correct', mode = '', plot_swaps = False, seeds=range(5), split=False):
    if split: 
        fig, axn, ax2 = split_axes()
    else: 
        fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(4, 4))
        fig.suptitle('Performance on Novel Tasks')
        axn.set_ylim(0.0, 1.0)
        axn.set_ylabel('Percent Correct', size=8, fontweight='bold')
        axn.set_xlabel('Exposures to Novel Task', size=8, fontweight='bold')

        axn.xaxis.set_tick_params(labelsize=8)

        axn.yaxis.set_tick_params(labelsize=8)
        axn.yaxis.set_major_locator(MaxNLocator(10)) 
        axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)]) 

        #axn.set_yticks(np.linspace(0, 100, 11))

    axn.spines['top'].set_visible(False)
    axn.spines['right'].set_visible(False)

    for model_name in model_list:
        data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, mode = mode, seeds=seeds)
        mean, std = data.avg_tasks()
        _plot_performance_curve(mean, std, axn, model_name, linestyle='-', linewidth=0.8, markevery=10, markersize=1.5)

        if plot_swaps: 
            data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, seeds=seeds,  mode='swap')
            mean, std = data.avg_tasks()
            _plot_performance_curve(mean, std, axn, model_name, linestyle='--', linewidth=0.8, markevery=10, markersize=1.5)

        if split:
            _plot_performance_curve(mean, std, ax2, model_name, linestyle='-', linewidth=0.8, markevery=10, markersize=1.5)
            if plot_swaps: 
                data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, seeds=seeds,  mode='swap')
                mean, std = data.avg_tasks()
                _plot_performance_curve(mean, std, ax2, model_name, linestyle='--', linewidth=0.8, markevery=10, markersize=1.5)

    fig.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list], loc=4, title='Models', title_fontsize = 'x-small', fontsize='x-small')        
    plt.show()

def plot_0_shot_task_hist(foldername, exp_type, model_list, perf_type='correct', mode='', seeds=range(5)): 
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(4, 4))
    fig.suptitle('Zero-Shot Performance Across Tasks')

    axn.set_ylabel('Percent Correct', size=8, fontweight='bold')
    axn.set_xlabel('Number of Tasks', size=8, fontweight='bold')

    thresholds = np.linspace(0.1, 1.0, 10)
    thresholds0 = np.linspace(0.0, 0.9, 10)

    width = 1/(len(model_list)+1)
    ind = np.arange(10)

    axn.set_yticks(ind+0.5, minor=True)
    axn.set_yticklabels([f'{x:.0%}>{y:.0%}' for x,y in list(zip(thresholds, thresholds0))], fontsize=5, minor=True) 
    

    axn.set_yticks(np.arange(11))
    axn.yaxis.set_ticks_position('none') 

    axn.set_yticklabels('') 

    for i, model_name in enumerate(model_list):
        data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, mode=mode, seeds=seeds)
        mean, _ = data.avg_seeds(k_shot=0)
        bins = np.zeros(10)
        for perf in mean: 
            bins[int(np.floor((perf*10)-1e-5))]+=1
        axn.barh((ind+(width/2))+(i*width), bins, width, color=MODEL_STYLE_DICT[model_name][0], align='edge', alpha=0.6)

    plt.show()


def plot_all_curves(dataframe, axn, **plt_args):
    for j, task in enumerate(TASK_LIST):
        ax = axn.flat[j]
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(task, size=6, pad=1)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=10)
        mean, std = dataframe.avg_seeds(task=task)
        _plot_performance_curve(mean, std, ax, dataframe.model_name, **plt_args)


def plot_all_holdout_curves(foldername, exp_type, model_list,  mode = '', perf_type='correct', seeds=range(5), plot_swap=False):
    fig, axn = plt.subplots(5,10, sharey = True, sharex=True, figsize =(8, 8))

    fig.suptitle('Avg. Performance on Heldout Tasks', size=14)        

    for model_name in model_list:
        data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, mode = '', seeds=seeds)
        plot_all_curves(data, axn, linewidth = 0.6, linestyle = '-', alpha=1, markersize=0.8, markevery=10)
        if plot_swap:
            data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, seeds=seeds, mode='swap')
            plot_all_curves(data, axn, linewidth = 0.6, linestyle = '--', alpha=1, markersize=0.8, markevery=10)

    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.6), title='Models', title_fontsize = 'small', fontsize='x-small')        
    plt.show()

def plot_all_training_curves(foldername, exp_type, holdout_file, model_list, perf_type='correct', plot_swaps = False, seeds=range(5)):
    fig, axn = plt.subplots(5,10, sharey = True, sharex=True, figsize =(8, 8))

    fig.suptitle('Avg. Performance on Heldout Tasks', size=14)        

    for model_name in model_list:
        try:
            data = TrainingDataFrame(foldername, exp_type, holdout_file, model_name, perf_type=perf_type, seeds=seeds)
        except:
            data = TrainingDataFrame(foldername, exp_type, holdout_file, model_name, file_suffix = '_FOR_TUNING', perf_type=perf_type, seeds=seeds)
        plot_all_curves(data, axn, linewidth = 0.6, linestyle = '-', alpha=1, markersize=0.8, markevery=5)

    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.6), title='Models', title_fontsize = 'small', fontsize='x-small')        
    plt.show()

def plot_trained_performance(all_perf_dict):
    barWidth = 0.1
    for i, model_name in enumerate(all_perf_dict.keys()):  
        perf = all_perf_dict[model_name]
        values = list(np.mean(perf, axis=0))
        std = np.std(perf, axis=0)
        
        len_values = len(TASK_LIST)
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
        # plt.plot(r, [1.05]*16, marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, markeredgecolor = MODEL_STYLE_DICT[model_name][0], color='white', markersize=mark_size)
        # plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white', hatch='/')
        #cap error bars at perfect performance 
        error_range= (std, np.where(values+std>1, (values+std)-1, std))
        print(error_range)
        markers, caps, bars = plt.errorbar(r, values, yerr = error_range, elinewidth = 0.5, capsize=1.0, linestyle="", alpha=0.8, color = 'black', markersize=1)

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth+0.25 for r in range(len_values)], TASK_LIST, fontsize='xx-small', fontweight='bold')
    plt.tight_layout()
    _make_model_legend(all_perf_dict.keys())
    plt.show()

def plot_model_response(model, trials, plotting_index = 0, instructions = None, save_file=None):
    model.eval()
    with torch.no_grad(): 

        tar = trials.targets
        ins = trials.inputs

        if instructions is not None: 
            assert len(instructions) == trials.num_trials, 'instructions do not equally number of trials'
            task_info = instructions
            is_task_instruct = all([instruct in train_instruct_dict or instruct in test_instruct_dict for instruct in instructions])
            if not is_task_instruct: warnings.warn('Not all instructions correspond to given task!')
        else: 
            task_info = get_task_info(ins.shape[0], trials.task_type, model.info_type)
        
        out, hid = model(torch.Tensor(ins), task_info)

        correct = isCorrect(out, torch.Tensor(tar), trials.target_dirs)[plotting_index]
        out = out.detach().cpu().numpy()[plotting_index, :, :]
        hid = hid.detach().cpu().numpy()[plotting_index, :, :]

        fix = ins[plotting_index, :, 0:1]            
        mod1 = ins[plotting_index, :, 1:1+STIM_DIM]
        mod2 = ins[plotting_index, :, 1+STIM_DIM:1+(2*STIM_DIM)]

        to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, tar[plotting_index, :, :].T, out.squeeze().T]
        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 5, 5])
        ylabels = ['fix.', 'mod. 1', 'mod. 2', 'Target', 'Response']

        fig, axn = plt.subplots(5,1, sharex = True, gridspec_kw=gs_kw, figsize=(4,3))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1.3, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i], fontweight = 'bold', fontsize=6)
            if i == 0: 
                ax.set_title(trials.task_type +' trial info; correct: ' + str(correct))
            if i == 5: 
                ax.set_xlabel('time')
                ax.xaxis.set_ticks(np.arange(0, 120, 5))
                ax.set_xticklabels(np.arange(0, 120, 5), fontsize=16)

                ax.tick_params(axis='x', labelsize= 6)

        if save_file is not None: 
            plt.savefig('figs/'+save_file)
        plt.show()

def _rep_scatter(reps_reduced, task, ax, dims, pcs, **scatter_kwargs): 
    #task_reps = reps_reduced[TASK_LIST.index(task), ...]
    task_reps = reps_reduced
    task_color = get_task_color(task)
    if dims ==2: 
        ax.scatter(task_reps[:, 0], task_reps[:, 1], s=10, c = [task_color]*task_reps.shape[0], **scatter_kwargs)
    else: 
        ax.scatter(task_reps[:, 0], task_reps[:, 1], task_reps[:,2], s=10, c = [task_color]*task_reps.shape[0], **scatter_kwargs)
    patch = Line2D([0], [0], label = task, color= task_color, linestyle='None', markersize=8, **scatter_kwargs)
    return patch

def _group_rep_scatter(reps_reduced, task_to_plot, ax, dims, pcs, **scatter_kwargs): 
    Patches = []
    for task in task_to_plot: 
        patch = _rep_scatter(reps_reduced[TASK_LIST.index(task), ...], task, ax, dims, pcs, marker='o', **scatter_kwargs)
        Patches.append(patch)
    return Patches

def plot_scatter(model, tasks_to_plot, rep_depth='task', dims=2, pcs=None, num_trials =50, epoch= 'stim_start', instruct_mode = None, **scatter_kwargs): 
    if pcs is None: 
        pcs = range(dims)

    if rep_depth == 'task': 
        reps = get_task_reps(model, epoch=epoch, num_trials = num_trials, main_var=True, instruct_mode=instruct_mode)
    elif rep_depth != 'task': 
        reps = get_instruct_reps(model.langModel, depth=rep_depth, instruct_mode=instruct_mode)
    reduced, _ = reduce_rep(reps, pcs=pcs)

    fig = plt.figure(figsize=(14, 14))
    if dims==2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection='3d')

    Patches = _group_rep_scatter(reduced, tasks_to_plot, ax, dims, pcs, **scatter_kwargs)
    Patches.append((Line2D([0], [0], linestyle='None', marker='X', color='grey', label='Contexts', 
                    markerfacecolor='white', markersize=8)))
    ax.set_xlabel('PC '+str(pcs[0]))
    ax.set_xticklabels([])
    ax.set_ylabel('PC '+str(pcs[1]))
    ax.set_yticklabels([])
    if dims==3: 
        ax.set_zlabel('PC '+str(pcs[2]))
        ax.set_zticklabels([])
    plt.legend(handles=Patches, fontsize='small')
    plt.show()

def plot_hid_traj(model, tasks_to_plot, trial_indices = [0], pcs=range(3), **scatter_kwargs): 
    Patches = []
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    reps = get_task_reps(model, epoch=None, num_trials = 10, max_var=True)
    reduced, _ = reduce_rep(reps, pcs=pcs)
    for task in tasks_to_plot: 
        task_index = TASK_LIST.index(task)
        task_color = get_task_color(task)
        for trial_index in trial_indices: 
            ax.scatter(reduced[task_index, trial_index, 1:, 0], reduced[task_index, trial_index, 1:, 1], reduced[task_index, trial_index, 1:, 2], 
                            color = task_color, **scatter_kwargs)
            ax.scatter(reduced[task_index, trial_index, 0, 0], reduced[task_index, trial_index, 0, 1], reduced[task_index, trial_index, 0, 2], 
                            color='white', edgecolor= task_color, marker='*', s=10)
            ax.scatter(reduced[task_index, trial_index, TRIAL_LEN-1, 0], reduced[task_index, trial_index, TRIAL_LEN-1, 1], reduced[task_index, trial_index, TRIAL_LEN-1, 2], 
                            color='white', edgecolor= task_color, marker='o', s=10)
            ax.scatter(reduced[task_index, trial_index, 129, 0], reduced[task_index, trial_index, 129, 1], reduced[task_index, trial_index, 129, 2], 
                            color='white', edgecolor= task_color, marker = 'P', **scatter_kwargs)

            if 'COMP' in task: 
                ax.scatter(reduced[task_index, trial_index, 89, 0], reduced[task_index, trial_index, 89, 1], reduced[task_index, trial_index, 89, 2], 
                                    color='white', edgecolor= task_color, marker = 'X', **scatter_kwargs)
            if 'RT' in task: 
                ax.scatter(reduced[task_index, trial_index, 129, 0], reduced[task_index, trial_index, 129, 1], reduced[task_index, trial_index, 129, 2], 
                            color='white', edgecolor= task_color, marker = 'X', **scatter_kwargs)
            else: 
                ax.scatter(reduced[task_index, trial_index, 29, 0], reduced[task_index, trial_index, 29, 1], reduced[task_index, trial_index, 29, 2], 
                            color='white', edgecolor= task_color, marker = 'X', s=10)

        Patches.append(Line2D([], [], linestyle='None', marker='.', color=task_color, label=task, markersize=4))
                
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_zlim(-6, 6)
    marker_list = [('*', 'Start'), ('X', 'Stim. Onset'), ('P', 'Resp.'), ('o', 'End')]
    marker_patches = [(Line2D([0], [0], linestyle='None', marker=marker[0], color='grey', label=marker[1], 
            markerfacecolor='white', markersize=8)) for marker in marker_list]
    Patches += marker_patches
    plt.legend(handles = Patches, fontsize = 'x-small')
    plt.tight_layout()
    plt.show()

def plot_RDM(sim_scores,  cmap=sns.color_palette("rocket_r", as_cmap=True), plot_title = 'RDM', save_file=None):
    number_reps=sim_scores.shape[1]/len(TASK_LIST)

    _, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(10, 8))
    sns.heatmap(sim_scores, yticklabels = '', xticklabels= '', 
                        cmap=cmap, vmin=0, vmax=1, ax=axn, cbar_kws={'label': '1-r'})

    for i, task in enumerate(TASK_LIST):
        plt.text(-2, (number_reps/2+number_reps*i), task, va='center', ha='right', size=4)
        plt.text(number_reps/2+number_reps*i, number_reps*(len(TASK_LIST)), task, va='top', ha='center', rotation='vertical', size=5)
    plt.title(plot_title, fontweight='bold', fontsize=12)

    if save_file is not None: 
        plt.savefig(save_file, dpi=400)

    plt.show()

def plot_tuning_curve(model, tasks, unit, times, var_of_interest, num_trials=100, num_repeats=5, smoothing = 1e-7): 
    y_max = 1.0
    hid_mean = get_task_reps(model, epoch=None, num_trials=num_trials, tasks=tasks, num_repeats=num_repeats, main_var=True)
    for i, task in enumerate(tasks): 
        time = times[i]
        neural_resp = hid_mean[i, :, time, unit]        
        plt.plot(var_of_interest, gaussian_filter1d(neural_resp, smoothing), color=get_task_color(task))
        y_max = max(y_max, neural_resp.max())

    plt.title('Tuning curve for Unit ' + str(unit) + ' at time ' +str(time))
    plt.ylim(-0.05, y_max+0.15)
    Patches = [mpatches.Patch(color=get_task_color(task), label=task) for task in tasks]
    plt.legend(handles=Patches)
    plt.show()


def plot_neural_resp(model, task, task_variable, unit, num_trials=100, num_repeats = 10, save_file=None):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    hid_mean = get_task_reps(model, epoch=None, num_trials=num_trials, tasks=[task], num_repeats=num_repeats, main_var=True)[0,...]

    if task_variable == 'direction' or task_variable=='diff_direction': 
        labels = ["0", "$2\pi$"]
        cmap = plt.get_cmap('twilight') 
    elif task_variable == 'strength':
        labels = ["0.3", "1.8"]
        cmap = plt.get_cmap('plasma') 
    elif task_variable =='diff_strength': 
        labels = ["delta -0.5", "delta 0.5"]
        cmap = plt.get_cmap('plasma') 

    cNorm  = colors.Normalize(vmin=0, vmax=num_trials)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    fig, axn = plt.subplots()
    print(hid_mean.shape)
    ylim = np.max(hid_mean[..., unit])
    for i in range(hid_mean.shape[0]):
        axn.plot(hid_mean[i, :, unit], c = scalarMap.to_rgba(i))


    plt.xticks([30, 130], labels=['Stim. Onset', 'Reponse'])
    plt.vlines(130, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    plt.vlines(30, -1.5, ylim+0.15, colors='k', linestyles='dashed')

    axn.set_ylim(0, ylim+0.15)
    cbar = plt.colorbar(scalarMap, orientation='vertical', label = task_variable.replace('_', ' '), ticks = [0, 100])
    #plt.title(task + ' response for Unit' + str(unit))
    cbar.set_ticklabels(labels)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return axn

def plot_layer_ccgp(model_list): 
    layer_list = [str(x) for x in range(1, 13)] + ['full', 'task']
    _, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(4, 4))
    axn.set_ylim(0.45, 1.0)
    axn.set_xticks(range(len(layer_list)))
    axn.set_xticklabels(layer_list)
    for model_name in model_list:
        all_holdout_ccgp = np.full((50, len(layer_list)), np.nan)
        folder_path = '7.20models/swap_holdouts/CCGP_scores/'+model_name
        for i, layer in enumerate(layer_list): 
            try: 
                all_holdout_ccgp[:, i] = np.load(folder_path+'/layer'+layer+'_task_holdout_seed0.npy')
            except FileNotFoundError: 
                print('no data for layer {} for model {} seed '.format(layer, model_name))
        print(MODEL_STYLE_DICT[model_name][1] == None)
        axn.scatter(range(len(layer_list)), np.mean(all_holdout_ccgp, axis=0), marker='.', c=MODEL_STYLE_DICT[model_name][0])
    plt.show()

def plot_unit_clustering(load_folder, model_name, seed):
    norm_var, cluster_labels = get_cluster_info(load_folder, model_name, seed)
    tSNE = TSNE(n_components=2)
    fitted = tSNE.fit_transform(norm_var)
    plt.scatter(fitted[:, 0], fitted[:, 1], cmap = plt.cm.tab20, c = cluster_labels)
    plt.show()

def plot_task_var_heatmap(load_folder, model_name, seed, cmap = sns.color_palette("inferno", as_cmap=True)):
    task_var, _ = get_cluster_info(load_folder, model_name, seed)
    _, cluster_labels, sorted_indices = sort_units(task_var)
    label_list = [task for task in TASK_LIST if 'Con' not in task]
    res = sns.heatmap(task_var[sorted_indices, :].T, xticklabels = cluster_labels, yticklabels=label_list, vmin=0, cmap=cmap)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 6)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 4, rotation=0)
    plt.show()
        

def decoding_confuse_mat(confusion_mat, fmt='g'): 
    res=sns.heatmap(confusion_mat, linewidths=0.5, linecolor='black', mask=confusion_mat == 0, xticklabels=TASK_LIST+['other'], yticklabels=TASK_LIST, annot=True, cmap='Blues', fmt=fmt, cbar=False)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
    plt.show()
