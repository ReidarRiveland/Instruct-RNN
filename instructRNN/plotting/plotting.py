import matplotlib
from instructRNN.analysis.model_analysis import get_hid_var_resp, get_instruct_reps
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
import matplotlib
import torch

from sklearn.decomposition import PCA
import warnings

Blue = '#1C3FFD'
lightBlue='#ACF0F2'
Green = '#45BF55'
Red = '#94090D'
Orange = '#FA9600'
Yellow = '#FFEE58'
Purple = '#512DA8'

MODEL_STYLE_DICT = {'simpleNet': (Blue, None), 'simpleNetPlus': (Blue, '+'), 
                    'comNet': (lightBlue, 'None'), 'comNetPlus': (lightBlue, '+'), 
                    'clipNet': (Yellow, None), 'clipNet_tuned': (Yellow, 'v'), 
                    'bowNet': (Orange, None), 
                    'gptNetXL': (Red, None), 'gptNetXL_tuned': (Red, 'v'), 
                    'bertNet': (Green, None), 'bertNet_tuned': (Green, 'v'),  
                    'sbertNet': (Purple, None), 'sbertNet_tuned': (Purple, 'v'),
                    'sbertNet_lin': (Purple, 'X'), 'sbertNet_lin_tuned': (Purple, '*')}


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from matplotlib import rc
plt.rcParams["font.family"] = "serif"

def get_task_color(task, cmap=matplotlib.cm.nipy_spectral):
    norm = matplotlib.colors.Normalize(0, len(TASK_LIST))
    spacer = lambda x : int(np.floor(x/4)+((x%4)*4))
    return cmap(norm(spacer(TASK_LIST.index(task))))

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

    axn.set_ylim(-0.05, 1.05)

    axn.xaxis.set_tick_params(labelsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)

    axn.yaxis.set_tick_params(labelsize=10)            
    axn.set_yticks(np.linspace(0, 1, 11))
    return fig, axn, ax2

def _plot_performance_curve(avg_perf, std_perf, plt_ax, model_name, **plt_args):
        color = MODEL_STYLE_DICT[model_name][0] 
        marker = MODEL_STYLE_DICT[model_name][1]
        plt_ax.fill_between(np.linspace(0, avg_perf.shape[-1], avg_perf.shape[-1]), np.min(np.array([np.ones(avg_perf.shape[-1]), avg_perf+std_perf]), axis=0), 
                                        avg_perf-std_perf, color = color, alpha= 0.08)

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

def plot_avg_holdout_curve(foldername, exp_type, model_list,  perf_type='correct', plot_swaps = False, seeds=range(5), split=False):
    if split: 
        fig, axn, ax2 = split_axes()
    else: 
        fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))
        axn.set_ylim(-0.05, 1.05)
        axn.set_ylabel('Percent Correct', size=8, fontweight='bold')
        axn.set_xlabel('Training Exmaples', size=8, fontweight='bold')

        axn.xaxis.set_tick_params(labelsize=10)
        axn.yaxis.set_tick_params(labelsize=10)
        axn.set_yticks(np.linspace(0, 1, 11))

    for model_name in model_list:
        data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, seeds=seeds)
        mean, std = data.avg_tasks()
        _plot_performance_curve(mean, std, axn, model_name, linestyle='-', linewidth=0.8, markevery=10, markersize=1.5)

        if split:
            _plot_performance_curve(mean, std, ax2, model_name, linestyle='-', linewidth=0.8, markevery=10, markersize=1.5)
            # if plot_swaps:
            #     data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type, seeds=seeds)
            #     plt_args['linestyle'] = '--'
            #     _plot_performance_curve(np.mean(swap_data, axis = (0, 1)), None, ax2, model_name, plt_args=plt_args)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.6), title='Models', title_fontsize = 'small', fontsize='x-small')        
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

def plot_all_holdout_curves(foldername, exp_type, model_list,  perf_type='correct', plot_swaps = False, seeds=range(5)):
    fig, axn = plt.subplots(6,6, sharey = True, sharex=True, figsize =(8, 8))

    fig.suptitle('Avg. Performance on Heldout Tasks', size=14)        

    for model_name in model_list:
        data = HoldoutDataFrame(foldername, exp_type, model_name, perf_type=perf_type, seeds=seeds)
        plot_all_curves(data, axn, linewidth = 0.6, linestyle = '-', alpha=1, markersize=0.8, markevery=10)

    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.6), title='Models', title_fontsize = 'small', fontsize='x-small')        
    plt.show()

def plot_all_training_curves(foldername, exp_type, holdout_file, model_list, perf_type='correct', plot_swaps = False, seeds=range(5)):
    fig, axn = plt.subplots(6,6, sharey = True, sharex=True, figsize =(8, 8))

    fig.suptitle('Avg. Performance on Heldout Tasks', size=14)        

    for model_name in model_list:
        data = TrainingDataFrame(foldername, exp_type, holdout_file, model_name, perf_type=perf_type, seeds=seeds)
        plot_all_curves(data, axn, linewidth = 0.6, linestyle = '-', alpha=1, markersize=0.8, markevery=5)

    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.6), title='Models', title_fontsize = 'small', fontsize='x-small')        
    plt.show()


def plot_k_shot_learning(foldername, exp_type, model_list, ks=[0,1,3], seeds=range(5)): 
    barWidth = 0.1
    ks = [0, 1, 3]
    plt.figure(figsize=(3, 6))

    for i, model_name in enumerate(model_list):  
        data = HoldoutDataFrame(foldername, exp_type, model_name,  seeds=seeds)
    
        all_mean = []
        all_std = []
        for k in ks: 
            mean, std = data.avg_tasks(k_shot=k)
            all_mean.append(mean)
            all_std.append(std)


        std = np.array(all_std)
        len_values = len(ks)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]

        mark_size = 4
        plt.plot(r, [vals+0.08 for vals in all_mean], marker=MODEL_STYLE_DICT[model_name][1], linestyle="", 
            alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
        error_range= (std, np.where(all_mean+std>1, (all_mean+std)-1, std))

        markers, caps, bars = plt.errorbar(r, all_mean, yerr = error_range, elinewidth = 0.5, capsize=1.0, linestyle="", alpha=0.8, color = 'black', markersize=1)
        plt.bar(r, all_mean, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white')


    plt.ylim(0, 1.05)
    plt.xlim(-0.15, len(ks))
    plt.title('Few Shot Learning Performance', size=12)
    plt.xlabel('Training Exmaples', fontweight='bold', size=8)
    plt.yticks(np.linspace(0, 1, 11), size=8)
    r = np.arange(len_values)
    plt.xticks([r + barWidth + 0.2 for r in range(len_values)], [0, 1, 3], size=8)
    _make_model_legend(model_list)
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
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))
    #plt.legend()
    plt.show()



def plot_val_performance(all_perf_dict):
    barWidth = 0.1
    for i, model_name in enumerate(all_perf_dict.keys()):  
        perf = all_perf_dict[model_name]
        values = np.mean(perf)
        std = np.std(np.mean(perf, axis=1), axis=0)
        mark_size = 3
        if i == 0:
            r = np.arange(1)
        else:
            r = [x + barWidth for x in r]
        plt.plot(r, 0.01, marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
        plt.bar(r, -values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white')
        #cap error bars at perfect performance 
        error_range= -1*np.array([(std, np.where(values+std>1, (values+std)-1, std))]).T
        print(error_range)
        markers, caps, bars = plt.errorbar(r, -values, yerr = error_range, elinewidth = 0.5, capsize=1.0, linestyle="", alpha=0.8, color = 'black', markersize=1)

    plt.ylim(-0.15, 0.02)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    plt.tight_layout()
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))
    #plt.legend()
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
            task_info = get_task_info(ins.shape[0], trials.task_type, model.is_instruct)
        
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

def _rep_scatter(reps_reduced, task, ax, dims, **scatter_kwargs): 
    task_reps = reps_reduced[TASK_LIST.index(task), ...]
    task_color = get_task_color(task)
    if dims ==2: 
        ax.scatter(task_reps[:, 0], task_reps[:, 1], s=25, c = [task_color]*task_reps.shape[0], **scatter_kwargs)
    else: 
        ax.scatter(task_reps[:, 0], task_reps[:, 1], task_reps[:,2], s=25, c = [task_color]*task_reps.shape[0], **scatter_kwargs)
    patch = Line2D([0], [0], label = task, color= task_color, linestyle='None', markersize=8, **scatter_kwargs)
    return patch

def _group_rep_scatter(reps_reduced, task_to_plot, ax, dims, **scatter_kwargs): 
    Patches = []
    for task in task_to_plot: 
        patch = _rep_scatter(reps_reduced, task, ax, dims, marker='o', **scatter_kwargs)
        Patches.append(patch)
    return Patches

def plot_scatter(model, tasks_to_plot, rep_depth='task', dims=2): 
    if rep_depth == 'task': 
        reps = get_task_reps(model, epoch='stim_start', num_trials = 128)
    elif rep_depth is not 'task': 
        reps = get_instruct_reps(model.langModel, depth=rep_depth)
    reduced, _ = reduce_rep(reps, dim=dims)

    fig = plt.figure(figsize=(12, 12))
    if dims==2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection='3d')

    Patches = _group_rep_scatter(reduced, tasks_to_plot, ax, dims)
    Patches.append((Line2D([0], [0], linestyle='None', marker='X', color='grey', label='Contexts', 
                    markerfacecolor='white', markersize=8)))
    plt.legend(handles=Patches, fontsize='medium')
    plt.show()


def plot_hid_traj(task_group_hid_traj, task_group, task_indices, trial_indices, instruct_indices, s = 1, subtitle='', annotate_tuples = [], context_task=None, save_file=None): 
    alphas = np.linspace(0.8, 0.2, num=task_group_hid_traj.shape[2])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    embedder = PCA(n_components=3)
    marker_size=40
    for trial_index in trial_indices: 
        embedded = embedder.fit_transform(task_group_hid_traj[:,:,trial_index, :, : ].reshape(-1, 128)).reshape((*task_group_hid_traj[:,:,trial_index, :, : ].shape[0:-1], 3))
        for task_index in task_indices:
            try: 
                task = list(task_group_dict[task_group])[task_index]
                linestyle = 'solid'
            except IndexError: 
                task = context_task
                linestyle = 'dashed'
            for instruct_index in instruct_indices: 
                ax.scatter(embedded[task_index, instruct_index, 1:, 0], embedded[task_index, instruct_index, 1:, 1], embedded[task_index, instruct_index, 1:, 2], s=s, color = task_colors[task])

                ax.scatter(embedded[task_index, instruct_index, 0, 0], embedded[task_index, instruct_index, 0, 1], embedded[task_index, instruct_index, 0, 2],  
                            s = marker_size, color='white', edgecolor= task_colors[task], marker='*')

                ax.scatter(embedded[task_index, instruct_index, 119, 0], embedded[task_index, instruct_index, 119, 1], embedded[task_index, instruct_index, 119, 2],  
                            s = marker_size, color='white', edgecolor= task_colors[task], marker='o')
                ax.scatter(embedded[task_index, instruct_index, 99, 0], embedded[task_index, instruct_index, 99, 1], embedded[task_index, instruct_index, 99, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'P')
                if task_group == 'COMP': 
                    ax.scatter(embedded[task_index, instruct_index, 49, 0], embedded[task_index, instruct_index, 49, 1], embedded[task_index, instruct_index, 49, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'X')
                if 'RT' in task: 
                    ax.scatter(embedded[task_index, instruct_index, 99, 0], embedded[task_index, instruct_index, 99, 1], embedded[task_index, instruct_index, 99, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'X')
                else: 
                    ax.scatter(embedded[task_index, instruct_index, 19, 0], embedded[task_index, instruct_index, 19, 1], embedded[task_index, instruct_index, 19, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'X')
                if (task_index, trial_index, instruct_index) in annotate_tuples: 
                    offset = 0.25
                    instruction = str(1+instruct_index)+'. '+train_instruct_dict[task][instruct_index]
                    if len(instruction) > 90: 
                        instruction=two_line_instruct(instruction)
                    ax.text(embedded[task_index, instruct_index, 119, 0]+offset, embedded[task_index, instruct_index, 119, 1]+offset, embedded[task_index, instruct_index, 119, 2]+offset, 
                        instruction, size=8, zorder=50,  color='k') 
    ax.set_title(subtitle, fontsize='medium')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_zlim(-6, 6)
    marker_list = [('*', 'Start'), ('X', 'Stim. Onset'), ('P', 'Resp.'), ('o', 'End')]
    marker_patches = [(Line2D([0], [0], linestyle='None', marker=marker[0], color='grey', label=marker[1], 
                    markerfacecolor='white', markersize=8)) for marker in marker_list]
    try: 
        Patches = [mpatches.Patch(color=task_colors[task_group_dict[task_group][index]], label=task_group_dict[task_group][index]) for index in task_indices]
    except: 
        Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in task_group_dict[task_group]]
    plt.legend(handles=Patches+marker_patches, fontsize = 'x-small')
    plt.suptitle('Neural Hidden State Trajectories for ' + task_group + ' Tasks')
    plt.tight_layout()
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_hid_traj_quiver(task_group_hid_traj, task_group, task_indices, trial_indices, instruction_indices, subtitle='', annotate_tuples = [], context_task=None, save_file=None): 
    alphas = np.linspace(0.8, 0.2, num=task_group_hid_traj.shape[2])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    embedder = PCA(n_components=3)
    marker_size=40
    for trial_index in trial_indices: 
        embedded = embedder.fit_transform(task_group_hid_traj[:,:,trial_index, :, : ].reshape(-1, 128)).reshape((*task_group_hid_traj[:,:,trial_index, :, : ].shape[0:-1], 3))
        for task_index in task_indices:
            try: 
                task = list(task_group_dict[task_group])[task_index]
                linestyle = 'solid'
                task_color = task_colors[task]
            except IndexError: 
                task = context_task
                linestyle = 'solid'
                task_color = 'black'

            if task_index == context_task: instruct_indices = instruction_indices
            else: instruct_indices = [0]
            for instruct_index in instruct_indices: 
                ax.quiver(embedded[task_index, instruct_index, 1:, 0], embedded[task_index, instruct_index, 1:, 1], embedded[task_index, instruct_index, 1:, 2], 
                            np.diff(embedded[task_index, instruct_index, :, 0], axis=0), np.diff(embedded[task_index, instruct_index, :, 1], axis=0), np.diff(embedded[task_index, instruct_index, :, 2], axis=0),
                            length = 0.35, color = task_color, facecolor = 'white', arrow_length_ratio=0.5,  pivot='middle', linewidth=1, linestyle = linestyle)

                ax.scatter(embedded[task_index, instruct_index, 0, 0], embedded[task_index, instruct_index, 0, 1], embedded[task_index, instruct_index, 0, 2],  
                            s = marker_size, color='white', edgecolor= task_colors[task], marker='*')
                ax.scatter(embedded[task_index, instruct_index, 119, 0], embedded[task_index, instruct_index, 119, 1], embedded[task_index, instruct_index, 119, 2],  
                            s = marker_size, color='white', edgecolor= task_colors[task], marker='o')

                ax.scatter(embedded[task_index, instruct_index, 99, 0], embedded[task_index, instruct_index, 99, 1], embedded[task_index, instruct_index, 99, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'P')
                if task_group == 'COMP': 
                    ax.scatter(embedded[task_index, instruct_index, 59, 0], embedded[task_index, instruct_index, 59, 1], embedded[task_index, instruct_index, 59, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'X')

                if 'RT' in task: 
                    ax.scatter(embedded[task_index, instruct_index, 99, 0], embedded[task_index, instruct_index, 99, 1], embedded[task_index, instruct_index, 99, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'X')
                else: 
                    ax.scatter(embedded[task_index, instruct_index, 19, 0], embedded[task_index, instruct_index, 19, 1], embedded[task_index, instruct_index, 19, 2], 
                            s=marker_size, color='white', edgecolor= task_colors[task], marker = 'X')
                if (task_index, trial_index, instruct_index) in annotate_tuples: 
                    offset = 0.25
                    instruction = str(1+instruct_index)+'. '+train_instruct_dict[task][instruct_index]
                    if len(instruction) > 90: 
                        instruction=two_line_instruct(instruction)
                    ax.text(embedded[task_index, instruct_index, 119, 0]+offset, embedded[task_index, instruct_index, 119, 1]+offset, embedded[task_index, instruct_index, 119, 2]+offset, 
                        instruction, size=8, zorder=50,  color='k') 


    ax.set_title(subtitle, fontsize='medium')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_zlim(-6, 6)

    marker_list = [('*', 'Start'), ('X', 'Stim. Onset'), ('P', 'Resp.'), ('o', 'End')]
    marker_patches = [(Line2D([0], [0], linestyle='None', marker=marker[0], color='grey', label=marker[1], 
                    markerfacecolor='white', markersize=8)) for marker in marker_list]
    try: 
        Patches = [mpatches.Patch(color=task_colors[task_group_dict[task_group][index]], label=task_group_dict[task_group][index]) for index in task_indices]
    except: 
        Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in task_group_dict[task_group]]
    plt.legend(handles=Patches+marker_patches, fontsize = 'x-small')
    plt.suptitle('Neural Hidden State Trajectories for ' + task_group + ' Tasks')
    plt.tight_layout()
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()




def plot_RDM(sim_scores,  cmap=sns.color_palette("rocket_r", as_cmap=True), plot_title = 'RDM', save_file=None):
    number_reps=sim_scores.shape[1]/len(TASK_LIST)

    _, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(10, 8))
    sns.heatmap(sim_scores, yticklabels = '', xticklabels= '', 
                        cmap=cmap, vmin=0, vmax=1, ax=axn, cbar_kws={'label': '1-r'})

    for i, task in enumerate(TASK_LIST):
        plt.text(-2, (number_reps/2+number_reps*i), task, va='center', ha='right', size=5)
        plt.text(number_reps/2+number_reps*i, number_reps*(len(TASK_LIST)), task, va='top', ha='center', rotation='vertical', size=5)
    plt.title(plot_title, fontweight='bold', fontsize=12)

    if save_file is not None: 
        plt.savefig(save_file, dpi=400)

    plt.show()

    

def plot_tuning_curve(model, tasks, task_variable, unit, mod, times, num_repeats =10, swapped_task = None, save_file=None): 
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
        _, hid_mean = get_hid_var_resp(model, task, trials, num_repeats=num_repeats)
        neural_resp = hid_mean[:, time, unit]
        plt.plot(var_of_interest, neural_resp, color=task_colors[task])
        y_max = max(y_max, neural_resp.max())

    if swapped_task is not None: 
        time = times[i]
        trials, var_of_interest = make_test_trials(swapped_task, task_variable, mod)
        model.instruct_mode = 'swap'
        _, hid_mean = get_hid_var_resp(model, swapped_task, trials)
        neural_resp = hid_mean[:, time, unit]
        plt.plot(var_of_interest, neural_resp, color=task_colors[swapped_task], linestyle='--')
        y_max = max(y_max, neural_resp.max())
        model.instruct_mode = ''

    plt.title('Tuning curve for Unit' + str(unit) + ' at time ' +str(time))
    plt.ylim(-0.05, y_max+0.15)
    plt.xlabel(task_variable.replace('_', ' '))
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks]
    plt.legend(handles=Patches)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return trials


def plot_CCGP_scores(model_list, rep_type_file_str = '', plot_swaps=False):
    barWidth = 0.08
    Patches = []
    for i, model_name in enumerate(model_list):
        if '_tuned' in model_name: marker_shape = MODEL_STYLE_DICT[model_name][1]
        else: marker_shape='s'
        Patches.append(Line2D([0], [0], linestyle='None', marker=marker_shape, color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8))
        len_values = 2
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        if '_tuned' in model_name: 
            mark_size = 4
        else: 
            mark_size = 3

        if plot_swaps: mode_list = ['', '_swap']
        else: mode_list = ['']
        for j, swap_mode in enumerate(mode_list):
            values = np.full(2, np.NAN)
            spread_values = np.empty((len_values, 5))

            CCGP_score = np.load(open('_ReLU128_4.11/CCGP_measures/'+rep_type_file_str+model_name+swap_mode+'_CCGP_scores.npz', 'rb'))
            if swap_mode != '_swap': 
                print('all_CCGP')
                values[0] = np.mean(np.nan_to_num(CCGP_score['all_CCGP'][:, -1, :, :]))
                spread_values[0, :] = np.mean(np.nan_to_num(CCGP_score['all_CCGP'][:, -1, :, :]), axis=(1,2))

            values[1] = np.mean(np.nan_to_num(CCGP_score['holdout_CCGP']))
            spread_values[1, :] = np.mean(np.nan_to_num(CCGP_score['holdout_CCGP']), axis=(1,2))

            for k in range(2):
                markers, caps, bars = plt.errorbar(r[k], values[k], yerr = np.std(spread_values[k, :]), elinewidth = 0.5, capsize=1.0, marker=marker_shape, linestyle="", mfc = [None, 'white'][j], alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)

            [bar.set_alpha(0.2) for bar in bars]

    plt.hlines(0.5, 0, r[-1], linestyles='--', color='black')
    plt.ylim(0.45, 0.95)
    plt.title('CCGP Measures')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth +0.15 for r in range(len_values)], ['all CCGP', 'holdout CCGP'])
    #plt.yticks(np.linspace(0.4, 1, 6), size=8)

    plt.tight_layout()

    plt.legend(handles=Patches, fontsize=6, markerscale=0.5)

    plt.show()



def plot_neural_resp(model, task_type, task_variable, unit, mod, num_repeats = 10, save_file=None):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    trials, _ = make_test_trials(task_type, task_variable, mod)
    _, hid_mean = get_hid_var_resp(model, task_type, trials, num_repeats=num_repeats)
    if task_variable == 'direction' or task_variable=='diff_direction': 
        labels = ["0", "$2\pi$"]
        cmap = plt.get_cmap('twilight') 
    # elif task_variable == 'diff_direction':
    #     labels = ["$\pi$", "0"]
    #     cmap = plt.get_cmap('twilight') 
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
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return trials



if __name__ == "__main__":
    from multitasking_models.language_models import SBERT, BERT
    from multitasking_models.sensorimotor_models import InstructNet, SimpleNet
    from tasks_utils import train_instruct_dict
    from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores, get_hid_var_group_resp
    import numpy as np
    from tasks_utils import train_instruct_dict, task_swaps_map
    from tasks import DM



    #fig 2
    #plot_single_task_training('_ReLU128_5.7/single_holdouts/', 'Multitask', 'DM', ['sbertNet', 'bertNet', 'gptNet', 'simpleNet'], range(5))
    #plot_single_seed_training('_ReLU128_5.7/single_holdouts/', 'DMS', ['sbertNet', 'sbertNet_layer_11', 'simpleNet'], 'correct', 4)

