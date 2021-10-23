import enum
import matplotlib
from numpy.core.fromnumeric import ndim, size
from numpy.core.numeric import indices
from scipy.ndimage.measurements import label
from torch.nn.modules.container import T
from model_analysis import get_hid_var_group_resp, get_hid_var_resp, get_model_performance, get_instruct_reps
from task import Comp, Task, make_test_trials, construct_batch
task_list = Task.TASK_LIST
task_group_dict = Task.TASK_GROUP_DICT

from model_analysis import get_hid_var_resp
from utils import isCorrect, train_instruct_dict, test_instruct_dict, two_line_instruct, task_swaps_map, task_colors, MODEL_STYLE_DICT, all_swaps, load_training_data, load_holdout_data

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import itertools
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import colors, cm, markers, use 
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib.transforms as mtrans
import matplotlib
from dPCA import dPCA
import torch

from sklearn.decomposition import PCA
import warnings

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from matplotlib import rc
plt.rcParams["font.family"] = "serif"

all_models = ['sbertNet_tuned', 'sbertNet', 'bertNet_tuned', 'bertNet', 'gptNet_tuned', 'gptNet', 'bowNet', 'simpleNet']

foldername = '_ReLU128_5.7/swap_holdouts'
model_list = all_models

def _plot_performance_curve(avg_perf, std_dev_perf, plt_ax, model_name, plt_args): 
        if std_dev_perf is not None: 
            plt_ax.fill_between(np.linspace(0, avg_perf.shape[-1], avg_perf.shape[-1]), np.min(np.array([np.ones(avg_perf.shape[-1]), avg_perf+std_dev_perf]), axis=0), 
                                            avg_perf-std_dev_perf, color = MODEL_STYLE_DICT[model_name][0], alpha= 0.1)
        plt_ax.plot(avg_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], markeredgecolor='white', markeredgewidth=0.25, **plt_args)

def plot_avg_curves(foldername, model_list, correct_or_loss, seeds=np.array(range(5))):
    data_dict = load_holdout_data(foldername, model_list)
    if correct_or_loss == 'correct': data_type_index = 0
    else: data_type_index = 1
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))

    axn.set_ylim(-0.05, 1.05)
    axn.set_ylabel('Percent Correct', size=8, fontweight='bold')
    axn.set_xlabel('Training Exmaples', size=8, fontweight='bold')

    axn.xaxis.set_tick_params(labelsize=10)
    axn.yaxis.set_tick_params(labelsize=10)
    axn.set_yticks(np.linspace(0, 1, 11))
    plt_args={'linewidth' : 0.8, 'linestyle' : '-', 'alpha':1, 'markersize':4, 'markevery':10}

    for model_name in model_list:
        data = data_dict[model_name][''][data_type_index, seeds, ...]
        plt_args['linestyle'] = '-'
        _plot_performance_curve(np.mean(data, axis = (0, 1)), np.std(np.mean(data, axis = 1), 0), axn, model_name, plt_args=plt_args)

        swap_data = data_dict[model_name]['swap'][data_type_index, seeds, ...]
        plt_args['linestyle'] = '--'
        _plot_performance_curve(np.mean(swap_data, axis = (0, 1)), None, axn, model_name, plt_args=plt_args)

    plt.show()
    return data_dict

def plot_task_curves(foldername, model_list, correct_or_loss, train_folder=None, seeds=np.array(range(5))):
    if train_folder is None: 
        data_dict = load_holdout_data(foldername, model_list)
        marker_every=15
    else: 
        data_dict = load_training_data(foldername, model_list)
        marker_every=200

    if correct_or_loss == 'correct': data_type_index = 0
    else: data_type_index = 1
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(8, 8))

    fig.suptitle('Avg. Performance on Heldout Tasks', size=14)        

    plt_args={'linewidth' : 0.6, 'linestyle' : '-', 'alpha':1, 'markersize':3, 'markevery':marker_every}

    for model_name in model_list: 
        if train_folder is None: data = data_dict[model_name][''][data_type_index, seeds, ...]
        else: data = data_dict[model_name][data_type_index, seeds, list(all_swaps+['Multitask']).index(train_folder), ...]

        for j, task in enumerate(task_list):
            ax = axn.flat[j]
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(task, size=6, pad=1)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelsize=10)
            _plot_performance_curve(np.mean(data[:, j, :], axis = 0), np.std(data[:, j, :], axis=0), ax, model_name, plt_args=plt_args)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.6), title='Models', title_fontsize = 'small', fontsize='x-small')        
    plt.show()
    return data_dict

#plot_task_curves(foldername, ['sbertNet_tuned'],'correct', train_folder='Multitask')

#data_dict = plot_task_curves(foldername, all_models[::-1],'correct', train_folder='Multitask')



def plot_single_holdout_task(foldername, holdout, model_list, seeds, smoothing=0.1, save_file=None):
    task_file = task_swaps_map[holdout]
    fig, axn = plt.subplots(1,2, sharex=True, figsize =(9, 4))
    train_data_types = ['correct', 'loss']
    ylims = [(-0.05, 1.05), (0, 0.05)]
    ylabels = ['Fraction Correct', 'MSE Loss']
    for model_name in model_list: 
        for i, ax in enumerate(axn.flat):
            training_data = np.zeros((len(seeds), 100))
            for j in seeds: 
                seed = 'seed' + str(j)
                try: 
                    tmp_training_data = pickle.load(open(foldername+task_file+'/'+model_name+'/'+holdout+'_'+seed+'_holdout_'+train_data_types[i], 'rb'))
                    training_data[j, :]= tmp_training_data
                except FileNotFoundError: 
                    print('No training data for '+ model_name + seed)
                    print('\n'+ foldername+task_file+'/'+model_name+'/'+seed+'_holdout_'+train_data_types[i])
                    continue 
            ax.set_ylim(ylims[i])
            ax.set_ylabel(ylabels[i], fontweight='bold', size=8)

            avg_performance = np.mean(training_data, axis = 0)
            std_performance = np.std(training_data, 0)
            smoothed_perf = gaussian_filter1d(avg_performance, sigma=smoothing)

            ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), avg_performance+std_performance]), axis=0), 
                                        avg_performance-std_performance, color = MODEL_STYLE_DICT[model_name][0], alpha= 0.1)
            ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=6, markevery=20)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)

            
    fig.legend(labels=model_list, title='Models', loc=2,  bbox_to_anchor=(0.68, 0.88), title_fontsize = 'large', fontsize='medium')
    fig.suptitle('Learning Curves for '+holdout+ ' Heldout', size=16, fontweight='bold')
    trans = mtrans.blended_transform_factory(fig.transFigure,
                                                mtrans.IdentityTransform())
    txt = fig.text(.5, 25, "Training Examples", ha='center', size=10, fontweight='bold')
    txt.set_transform(trans)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_context_training(foldername, model_list, seed, smoothing=0.1, save_file=None):
    seed = 'seed' + str(seed)
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(19, 12))
    for model_name in model_list: 
        for i, ax in enumerate(axn.flat):
            task = 'Multitask'
            task_file = task.replace(' ', '_')
            try: 
                training_data = pickle.load(open(foldername+task_file+'/'+model_name+'/'+seed+'_context_holdout_correct_data', 'rb'))
            except FileNotFoundError: 
                print('No training data for '+ model_name + seed)
                print('\n'+ foldername+task_file+'/'+model_name+'/'+seed+'_context_holdout_correct_data')
                continue 
            ax.set_ylim(-0.05, 1.15)
            for j in range(5): 
                smoothed_perf = gaussian_filter1d(training_data[task_list[i]][j, :], sigma=smoothing)
                alpha = 0.1
                ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=alpha, markersize=10, markevery=250)
            ax.set_title(task)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.55), title='Models', title_fontsize=12)
    fig.suptitle('Training for Semantic Contexts', size=16)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

#model_data_dict = plot_context_training('_ReLU128_5.7/swap_holdouts/', ['bowNet'],  1, smoothing = 0.01)


def plot_tuned_vs_standard(model_data_dict): 
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(2, 5))
    for model_name in ['sbertNet', 'bertNet', 'gptNet']:
        for i in range(4):
            tuned_perf = np.mean(model_data_dict[model_name+'_tuned'], axis=0)[:,0]
            standard_perf =np.mean(model_data_dict[model_name], axis=0)[:,0]
            color = MODEL_STYLE_DICT[model_name][0]
            axn.plot([0, 1], [tuned_perf[i], standard_perf[i]], color=color, linewidth=0.8)
            axn.plot([0], tuned_perf[i], marker='v', mec=color, color=task_colors[task_group_dict['COMP'][i]])
            axn.plot([1], standard_perf[i], marker='o', mec=color, color=task_colors[task_group_dict['COMP'][i]])
    axn.set_ylim(0, 1.0)
    axn.set_xlim(-0.15, 1.15)
    axn.xaxis.set_tick_params(labelsize=8)
    axn.yaxis.set_tick_params(labelsize=8)
    axn.set_yticks(np.linspace(0, 1, 11))
    plt.show()


def plot_holdout_curves_split_axes(foldername, model_list, train_data_type, seeds, smoothing=0.1):
    #rc('font', weight='bold')
    instruction_mode = 'swap'
    inset1_lims = (0, 10)
    inset2_lims = (80, 100)
    gs_kw = dict(width_ratios=[inset1_lims[1]-inset1_lims[0],inset2_lims[1]-inset2_lims[0]], height_ratios=[1])
    fig,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w',  gridspec_kw=gs_kw, figsize =(6, 4))
    data_dict_list = []
    for k, mode in enumerate(['', 'swap']):
        model_data_dict = {}
        for model_name in model_list: 
            #training_data = np.empty((len(seeds), len(Task.TASK_LIST), 100))
            training_data = np.empty((len(seeds), 4, 100))

            for i, seed_num in enumerate(seeds):
                seed_name = 'seed' + str(seed_num)

                #for j, task in enumerate(task_list):
                for j, task in enumerate(['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2']):
                    holdout_file = task.replace(' ', '_')

                    if instruction_mode =='swap': 
                        task_file = task_swaps_map[task]
                        holdout_file += '_'
                    else: 
                        task_file = holdout_file
                        holdout_file = ''

                    try:
                        training_data[i, j, :] = pickle.load(open(foldername+'/'+task_file+'/'+model_name+'/'+mode+holdout_file+seed_name+'_holdout_'+train_data_type, 'rb'))
                    except FileNotFoundError: 
                        print('No training data for '+ model_name + ' '+seed_name+' '+task)
                        print(foldername+'/'+task_file+'/'+model_name+'/'+holdout_file+seed_name+'_holdout_'+train_data_type)
                        continue 

            avg_performance = np.mean(training_data, axis = (0, 1))
            std_performance = np.std(np.mean(training_data, axis = 1), 0)
            smoothed_perf = gaussian_filter1d(avg_performance, sigma=smoothing)
            if mode == '':
                ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), avg_performance+std_performance]), axis=0), 
                                            avg_performance-std_performance, color = MODEL_STYLE_DICT[model_name][0], alpha= 0.1)
                ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), avg_performance+std_performance]), axis=0), 
                                            avg_performance-std_performance, color = MODEL_STYLE_DICT[model_name][0], alpha= 0.1)
            ax.plot(smoothed_perf, linewidth = 0.8, linestyle = ['-', '--'][k], color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=4, markevery=3)
            ax2.plot(smoothed_perf, linewidth = 0.8, linestyle = ['-', '--'][k], color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=4, markevery=3)
            model_data_dict[model_name] = training_data
        data_dict_list.append(model_data_dict)
        
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.7, 0.48), title='Models', title_fontsize = 'small', fontsize='x-small')

    ax.set_xlim(inset1_lims)
    ax2.set_xlim(inset2_lims)

    ax2.yaxis.set_visible(False)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.set_ylim(-0.05, 1.05)

    ax.xaxis.set_tick_params(labelsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)

    ax.yaxis.set_tick_params(labelsize=10)            
    ax.set_yticks(np.linspace(0, 1, 11))


    fig.suptitle('Avg. Performance on Heldout Tasks', size=14)
    plt.show()

    return data_dict_list


def plot_k_shot_learning(model_data_dict_list, save_file=None): 
    barWidth = 0.1
    ks = [0, 1, 3]
    plt.figure(figsize=(3, 6))

    for index, model_data_dict in enumerate(model_data_dict_list):
        for i, item in enumerate(model_data_dict.items()):  
            model_name, perf = item
            values = list(np.mean(perf, axis=(0,1))[ks])
            spread = list(np.std(np.mean(perf, axis=1), axis=0)[ks])

            len_values = len(ks)
            if i == 0:
                r = np.arange(len_values)
            else:
                r = [x + barWidth for x in r]
            if '_layer_11' in model_name: 
                mark_size = 8
            else: 
                mark_size = 4
            # if index == 0: 
            #     plt.plot(r, [vals+0.1 for vals in values], marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
            #     plt.bar(r, values, yerr = spread, error_kw = {'elinewidth':0.3, 'capsize': 1.0}, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white', linestyle=['-','--'][index])
            # plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white', linestyle=['-','--'][index])
            if index == 0: 
                plt.plot(r, [vals+0.03 for vals in values], marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
                plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white', linestyle=['-','--'][index])
            plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white', linestyle=['-','--'][index])


    plt.ylim(0, 1.05)
    plt.title('Few Shot Learning Performance', size=12)
    plt.xlabel('Training Exmaples', fontweight='bold', size=8)
    plt.yticks(np.linspace(0, 1, 11), size=8)
    r = np.arange(len_values)
    plt.xticks([r + barWidth + 0.2 for r in range(len_values)], [0, 1, 3], size=8)
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(model_data_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))

    #plt.legend(handles=Patches)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_trained_performance(all_perf_dict):
    barWidth = 0.1
    for i, model_name in enumerate(['sbertNet_tuned', 'sbertNet', 'bertNet_tuned','bertNet', 'gptNet_tuned', 'gptNet', 'bowNet']):  
        perf = all_perf_dict[model_name]
        values = list(np.mean(perf, axis=1))
        std = np.std(perf, axis=1)
        
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
        #cap error bars at perfect performance 
        error_range= (std, np.where(values+std>1, (values+std)-1, std))
        print(error_range)
        markers, caps, bars = plt.errorbar(r, values, yerr = error_range, elinewidth = 0.5, capsize=1.0, linestyle="", alpha=0.8, color = 'black', markersize=1)

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth+0.25 for r in range(len_values)], task_list, fontsize='xx-small', fontweight='bold')
    plt.tight_layout()
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(all_perf_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))
    #plt.legend()
    plt.show()


def plot_model_response(model, trials, plotting_index = 0, instructions = None, save_file=None):
    assert isinstance(trials, Task)
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
            task_info = model.get_task_info(ins.shape[0], trials.task_type)
        
        out, hid = model(task_info, torch.Tensor(ins))

        correct = isCorrect(out, torch.Tensor(tar), trials.target_dirs)[plotting_index]
        out = out.detach().cpu().numpy()[plotting_index, :, :]
        hid = hid.detach().cpu().numpy()[plotting_index, :, :]

        try: 
            task_info_embedding = torch.Tensor(get_instruct_reps(model.langModel, {trials.task_type: task_info}, depth='12')).swapaxes(0, 1)
            task_info_embedding = task_info_embedding.repeat(1, ins.shape[1], 1)
        except: 
            task_info_embedding = torch.matmul(task_info, model.rule_transform).unsqueeze(1).repeat(1, ins.shape[1], 1)

        fix = ins[plotting_index, :, 0:1]            
        mod1 = ins[plotting_index, :, 1:1+Task.STIM_DIM]
        mod2 = ins[plotting_index, :, 1+Task.STIM_DIM:1+(2*Task.STIM_DIM)]

        to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, task_info_embedding[plotting_index, 0:119, :].T, tar[plotting_index, :, :].T, out.squeeze().T]
        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 2, 5, 5])
        ylabels = ['fix.', 'mod. 1', 'mod. 2', 'Task Info', 'Target', 'Response']

        fig, axn = plt.subplots(6,1, sharex = True, gridspec_kw=gs_kw, figsize=(4,3))
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

def plot_rep_scatter(reps_reduced, tasks_to_plot, annotate_tuples=[], annotate_args=[], swapped_tasks= [], save_file=None): 
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*reps_reduced.shape[1] for task in tasks_to_plot]))
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]

    reps_to_plot = reps_reduced[task_indices, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=25)
    ax.scatter(np.mean(reps_to_plot, axis=1)[:, 0], np.mean(reps_to_plot, axis=1)[:, 1], c = [task_colors[task] for task in tasks_to_plot], s=10, marker='D', edgecolors='white')


    for i, indices in enumerate(annotate_tuples): 
        task_index, instruct_index = indices 
        plt.annotate(str(1+instruct_index)+'. '+two_line_instruct(train_instruct_dict[tasks_to_plot[task_index]][instruct_index]), xy=(flattened_reduced[int(instruct_index+(task_index*15)), 0], flattened_reduced[int(instruct_index+(task_index*15)), 1]), 
                    xytext=annotate_args[i], size = 6, arrowprops=dict(arrowstyle='->'), textcoords = 'offset points')

    plt.xlabel("PC 1", fontsize = 12)
    plt.ylabel("PC 2", fontsize = 12)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]

    if len(swapped_tasks)>0: 
        ax.scatter(reps_reduced[-1, :, 0], reps_reduced[-1, :, 1], color='white', marker='o', edgecolors=[task_colors[swapped_tasks[0]]]*reps_reduced.shape[1], s=25)
        ax.scatter(np.mean(reps_reduced[-1, :, 0]), np.mean(reps_reduced[-1, :, 1]), c = task_colors[swapped_tasks[0]], s=10, marker='D', edgecolors='white')
        Patches.append((Line2D([0], [0], linestyle='None', marker='o', color=task_colors[swapped_tasks[0]], label='Instruction Swap', 
                markerfacecolor='white', markersize=8)))

    plt.legend(handles=Patches, fontsize='medium')

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



def plot_hid_traj(task_group_hid_traj, task_group, task_indices, trial_indices, instruction_indices, subtitle='', annotate_tuples = [], context_task=None, save_file=None): 
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
                task_color = 'white'

            if task_index == context_task: instruct_indices = instruction_indices
            else: instruct_indices = [0]
            for instruct_index in instruct_indices: 
                ax.scatter(embedded[task_index, instruct_index, 1:20, 0], embedded[task_index, instruct_index, 1:20, 1], embedded[task_index, instruct_index, 1:20, 2], color = task_color)

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



def plot_dPCA(model, tasks, swapped_tasks=[]):
    trials, var_of_insterest = make_test_trials('Anti DM', 'diff_strength', 0, num_trials=6)

    Z_dict = {}
    for task in tasks:
        if task in swapped_tasks:
            model.instruct_mode = 'swap'

        print(task, model.instruct_mode)
        hid_resp, mean_hid_resp = get_hid_var_resp(model, task, trials)

        reshape_mean_hid_resp = mean_hid_resp.T.swapaxes(-1, 1)
        reshape_hid_resp = hid_resp.swapaxes(1, 2).swapaxes(-1, 1)

        dpca = dPCA.dPCA(labels='st',regularizer='auto')
        dpca.protect = ['t']

        Z = dpca.fit_transform(reshape_mean_hid_resp, reshape_hid_resp)
        Z_dict[task] = Z
        model.instruct_mode = ''

    time = np.arange(120)
    plt.figure(figsize=(16,7))


    linestyle_list = ['-', '--']
    cmap = plt.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=0, vmax=5)

    for i, Z in enumerate(Z_dict.values()):
        plt.subplot(131)
        for s in range(6):
            plt.plot(time,Z['st'][1,s], linestyle=linestyle_list[i], c=cmap(norm(s)))
        plt.title('1st mixing component')

        plt.subplot(132)
        for s in range(6):
            plt.plot(time,Z['t'][0,s], linestyle=linestyle_list[i], c=cmap(norm(s)))
        plt.title('1st time component')

        plt.subplot(133)
        for s in range(6):
            plt.plot(time,Z['s'][0,s], linestyle=linestyle_list[i], c=cmap(norm(s)))
        plt.title('1st Decision Variable component')

    plt.figlegend(['delta '+ str(num) for num in np.round(var_of_insterest, 2)], loc=5)

    plt.show()


def plot_RDM(sim_scores, rep_type, cmap=sns.color_palette("rocket_r", as_cmap=True), plot_title = 'RDM', use_avg_reps = False, save_file=None):
    if rep_type == 'lang': label_buffer = 2
    if rep_type == 'task': label_buffer = 8
    rep_dim = sim_scores.shape[-1]
    number_reps=sim_scores.shape[1]
    

    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(10, 8))
    sns.heatmap(sim_scores, yticklabels = '', xticklabels= '', 
                        cmap=cmap, vmin=0, vmax=1, ax=axn, cbar_kws={'label': '1-r'})

    for i, task in enumerate(Task.TASK_LIST):
        plt.text(-2, label_buffer+number_reps/2+number_reps*i, task, ha='right', size=8, fontweight='bold')
        plt.text(-label_buffer + number_reps/2+number_reps*i, number_reps*16, task, va='top', rotation='vertical', size=8, fontweight='bold')
    plt.title(plot_title, fontweight='bold', fontsize=12)

    if save_file is not None: 
        plt.savefig('figs/'+save_file, dpi=600)

    plt.show()
    

def plot_tuning_curve(model, tasks, task_variable, unit, mod, times, swapped_task = None, save_file=None): 
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


def plot_CCGP_scores(model_list, rep_type_file_str = '', save_file=None):
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
    
        for j, swap_mode in enumerate(['', '_swap']):
            values = np.full(2, np.NAN)
            spread_values = np.empty((len_values, 5))

            CCGP_score = np.load(open('_ReLU128_5.7/CCGP_measures_new/'+rep_type_file_str+model_name+swap_mode+'_CCGP_scores.npz', 'rb'))
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
    plt.ylim(0.45, 1.05)
    plt.title('CCGP Measures')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth +0.15 for r in range(len_values)], ['all CCGP', 'holdout CCGP'])
    #plt.yticks(np.linspace(0.4, 1, 6), size=8)

    plt.tight_layout()

    plt.legend(handles=Patches, fontsize=6, markerscale=0.5)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_neural_resp(model, task_type, task_variable, unit, mod, save_file=None):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    trials, _ = make_test_trials(task_type, task_variable, mod)
    _, hid_mean = get_hid_var_resp(model, task_type, trials)
    if task_variable == 'direction' or 'diff_direction': 
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
    from nlp_models import SBERT, BERT
    from rnn_models import InstructNet, SimpleNet
    from utils import train_instruct_dict
    from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores, get_hid_var_group_resp
    import numpy as np
    from utils import train_instruct_dict, task_swaps_map
    from task import DM



    #fig 2
    #plot_single_task_training('_ReLU128_5.7/single_holdouts/', 'Multitask', 'DM', ['sbertNet', 'bertNet', 'gptNet', 'simpleNet'], range(5))
    #plot_single_seed_training('_ReLU128_5.7/single_holdouts/', 'DMS', ['sbertNet', 'sbertNet_layer_11', 'simpleNet'], 'correct', 4)

