from numpy.core.fromnumeric import size
from numpy.core.numeric import indices
from scipy.ndimage.measurements import label
from torch.nn.modules.container import T
from model_analysis import get_hid_var_resp, get_model_performance
from task import Task, make_test_trials, construct_batch
task_list = Task.TASK_LIST
task_group_dict = Task.TASK_GROUP_DICT

from model_analysis import get_hid_var_resp
from utils import isCorrect, train_instruct_dict, test_instruct_dict, two_line_instruct

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
import torch

from sklearn.decomposition import PCA
import warnings

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from matplotlib import rc
plt.rcParams["font.family"] = "serif"



task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'bowNet': ('orange', None), 'gptNet': ('red', '^'), 
                    'bertNet': ('green', '^'), 'bertNet_layer_11': ('green', '.'), 'sbertNet': ('purple', '^'), 'sbertNet_tuned': ('purple', 'v'), 'sbertNet_layer_11': ('purple', '.')}

all_models = ['sbertNet_layer_11', 'bertNet_layer_11', 'sbertNet',  'bertNet','gptNet', 'bowNet', 'simpleNet']

def plot_single_seed_training(foldername, holdout, model_list, train_data_type, seed, smoothing=0.1, save_file=None):
    seed = 'seed' + str(seed)
    task_file = holdout.replace(' ', '_')
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(19, 12))
    for model_name in model_list: 
        try: 
            training_data = pickle.load(open(foldername+task_file+'/'+model_name+'/'+seed+'_training_'+train_data_type, 'rb'))
        except FileNotFoundError: 
            print('No training data for '+ model_name + seed)
            print('\n'+ foldername+task_file+'/'+model_name+'/'+seed+'_training_'+train_data_type)
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
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_single_task_training(foldername, holdout, task, model_list, seeds, smoothing=0.1, save_file=None):
    task_file = holdout.replace(' ', '_')
    fig, axn = plt.subplots(1,2, sharex=True, figsize =(9, 3))
    train_data_types = ['correct', 'loss']
    ylims = [(-0.05, 1.05), (0, 0.05)]
    ylabels = ['Fraction Correct', 'MSE Loss']
    for model_name in model_list: 
        for i, ax in enumerate(axn.flat):
            training_data = np.zeros(750)
            for j in seeds: 
                seed = 'seed' + str(j)
                try: 
                    tmp_training_data = pickle.load(open(foldername+task_file+'/'+model_name+'/'+seed+'_training_'+train_data_types[i], 'rb'))
                    training_data += tmp_training_data[task][:750]
                except FileNotFoundError: 
                    print('No training data for '+ model_name + seed)
                    print('\n'+ foldername+task_file+'/'+model_name+'/'+seed+'_training_'+train_data_types[i])
                    continue 
            ax.set_ylim(ylims[i])
            ax.set_ylabel(ylabels[i], fontweight='bold')
            smoothed_perf = gaussian_filter1d(training_data/len(seeds), sigma=smoothing)
            ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=6, markevery=100)

            
    fig.legend(labels=model_list, title='Models', title_fontsize=12)
    fig.suptitle('Learning Curves for '+task + ' Task', size=16, fontweight='bold')
    trans = mtrans.blended_transform_factory(fig.transFigure,
                                                mtrans.IdentityTransform())
    txt = fig.text(.5, 25, "Training Examples", ha='center', size=12, fontweight='bold')
    txt.set_transform(trans)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()


def plot_single_holdout_task(foldername, holdout, model_list, seeds, smoothing=0.1, save_file=None):
    task_file = holdout.replace(' ', '_')
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
                    tmp_training_data = pickle.load(open(foldername+task_file+'/'+model_name+'/'+seed+'_holdout_'+train_data_types[i], 'rb'))
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



def plot_context_training(foldername, model_list, train_data_type, seed, smoothing=0.1, save_file=None):
    seed = 'seed' + str(seed)
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(19, 12))
    for model_name in model_list: 
        for i, ax in enumerate(axn.flat):
            task = task_list[i]
            task_file = task.replace(' ', '_')
            try: 
                training_data = pickle.load(open(foldername+task_file+'/'+model_name+'/context_holdout_correct_data', 'rb'))
            except FileNotFoundError: 
                print('No training data for '+ model_name + seed)
                print('\n'+ foldername+task_file+'/'+model_name+'/'+seed+'_training_'+train_data_type)
                continue 
            ax.set_ylim(-0.05, 1.15)
            for j in range(16): 
                if j == 15:
                    smoothed_perf = gaussian_filter1d(np.mean(training_data, axis=0), sigma=smoothing)
                    alpha = 1
                else: 
                    smoothed_perf = gaussian_filter1d(training_data[j, :], sigma=smoothing)
                    alpha = 0.1
                ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=alpha, markersize=10, markevery=250)
            ax.set_title(task)
    fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.55), title='Models', title_fontsize=12)
    fig.suptitle('Training for Semantic Contexts', size=16)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_holdout_curves(foldername, model_list, train_data_type, plot_type, seeds, smoothing=0.1, save_file=None):
    #rc('font', weight='bold')
    if plot_type == 'avg_holdout': fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))
    if plot_type == 'task_holdout': fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(19, 12))

    model_data_dict = {}

    for model_name in model_list: 
        training_data = np.empty((len(seeds), len(Task.TASK_LIST), 100))

        for i, seed_num in enumerate(seeds):
            seed_name = 'seed' + str(seed_num)

            for j, task in enumerate(task_list):
                task_file = task.replace(' ', '_')

                try:
                    training_data[i, j, :] = pickle.load(open(foldername+task_file+'/'+model_name+'/'+seed_name+'_holdout_'+train_data_type, 'rb'))
                except FileNotFoundError: 
                    print('No training data for '+ model_name + ' '+seed_name+' '+task)
                    continue 

        if plot_type == 'avg_holdout': 
            avg_performance = np.mean(training_data, axis = (0, 1))
            std_performance = np.std(np.mean(training_data, axis = 1), 0)
            smoothed_perf = gaussian_filter1d(avg_performance, sigma=smoothing)

            axn.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), avg_performance+std_performance]), axis=0), 
                                        avg_performance-std_performance, color = MODEL_STYLE_DICT[model_name][0], alpha= 0.1)
            axn.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=6, markevery=20)

            axn.set_ylim(-0.05, 1.05)
            axn.set_ylabel('Percent Correct', size=10, fontweight='bold')
            axn.set_xlabel('Training Exmaples', size=10, fontweight='bold')
            fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.55, 0.4), title='Models', title_fontsize = 'large', fontsize='medium')
            fig.suptitle('Avg. Performance on Heldout Tasks', size=20)
            axn.xaxis.set_tick_params(labelsize=10)
            axn.yaxis.set_tick_params(labelsize=10)
            axn.set_yticks(np.linspace(0, 1, 11))

        if plot_type == 'task_holdout':
            avg_performance = np.mean(training_data, axis = 0)
            std_performance = np.std(training_data, axis = 0)
            for k, ax in enumerate(axn.flat):
                ax.set_ylim(-0.05, 1.15)
                ax.set_title(task_list[k])
                smoothed_perf = gaussian_filter1d(avg_performance[k, :], sigma=smoothing)
                ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), avg_performance[k, :]+std_performance[k, :]]), axis=0), 
                                        avg_performance[k, :]-std_performance[k, :], color = MODEL_STYLE_DICT[model_name][0], alpha= 0.1)

                ax.plot(smoothed_perf, color = MODEL_STYLE_DICT[model_name][0], marker=MODEL_STYLE_DICT[model_name][1], alpha=1, markersize=8, markevery=20)
            fig.legend(labels=model_list, loc=2,  bbox_to_anchor=(0.9, 0.55), title='Models', title_fontsize='12')
            fig.suptitle('Performance on Heldout Tasks', size=16)
        
        model_data_dict[model_name] = avg_performance
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return model_data_dict

def plot_k_shot_learning(model_data_dict, save_file=None): 
    barWidth = 0.1
    ks = [0, 1, 3]
    plt.figure(figsize=(4, 10))

    for i, item in enumerate(model_data_dict.items()):  
        model_name, avg_perf = item
        values = list(avg_perf[ks])
        len_values = len(ks)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        if '_layer_11' in model_name: 
            mark_size = 8
        else: 
            mark_size = 6
        plt.plot(r, [1]*3, marker=MODEL_STYLE_DICT[model_name][1], linestyle="", alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
        plt.bar(r, values, width =barWidth, label = model_name, color = MODEL_STYLE_DICT[model_name][0], edgecolor = 'white')

    plt.ylim(0, 1.05)
    plt.title('Few Shot Learning Performance', size=12)
    plt.xlabel('Training Exmaples', fontweight='bold', size=8)
    plt.yticks(np.linspace(0, 1, 11), size=8)
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], [0, 1, 3], size=8)
    Patches = [(Line2D([0], [0], linestyle='None', marker=MODEL_STYLE_DICT[model_name][1], color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8)) for model_name in list(model_data_dict.keys()) if 'bert' in model_name or 'gpt' in model_name]
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['bowNet'][0], label='bowNet'))
    Patches.append(mpatches.Patch(color=MODEL_STYLE_DICT['simpleNet'][0], label='simpleNet'))

    #plt.legend(handles=Patches)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()


def plot_trained_performance(all_perf_dict, save_file=None):
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
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
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
            task_info_embedding = model.langModel(task_info).unsqueeze(1).repeat(1, ins.shape[1], 1)
        except: 
            task_info_embedding = task_info.unsqueeze(1).repeat(1, ins.shape[1], 1)

        fix = ins[plotting_index, :, 0:1]            
        mod1 = ins[plotting_index, :, 1:1+Task.STIM_DIM]
        mod2 = ins[plotting_index, :, 1+Task.STIM_DIM:1+(2*Task.STIM_DIM)]

        to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, task_info_embedding[plotting_index, :, :].T, tar[plotting_index, :, :].T, out.squeeze().T]
        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 2, 5, 5])
        ylabels = ['fix.', 'mod. 1', 'mod. 2', 'Task Info', 'Target', 'Response']

        fig, axn = plt.subplots(6,1, sharex = True, gridspec_kw=gs_kw, figsize=(12,8))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1.3, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i], fontweight = 'bold')
            if i == 0: 
                ax.set_title(trials.task_type +' trial info; correct: ' + str(correct))
            if i == 5: 
                ax.set_xlabel('time')
        
        

        if isinstance(model, InstructNet):
            trans = mtrans.blended_transform_factory(fig.transFigure,
                                                mtrans.IdentityTransform())
            txt = fig.text(.5, 25, "Instruction: \"" + str(task_info[plotting_index]) +"\"", ha='center', size=8, fontweight='bold')
            txt.set_transform(trans)
        if save_file is not None: 
            plt.savefig('figs/'+save_file)
        plt.show()

def plot_rep_scatter(reps_reduced, tasks_to_plot, annotate_tuples=[], annotate_args=[], save_file=None): 
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*reps_reduced.shape[1] for task in tasks_to_plot]))
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]
    reps_to_plot = reps_reduced[task_indices, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=35)
    for i, indices in enumerate(annotate_tuples): 
        task_index, instruct_index = indices 
        plt.annotate(str(1+instruct_index)+'. '+two_line_instruct(train_instruct_dict[tasks_to_plot[task_index]][instruct_index]), xy=(flattened_reduced[int(instruct_index+(task_index*15)), 0], flattened_reduced[int(instruct_index+(task_index*15)), 1]), 
                    xytext=annotate_args[i], size = 8, arrowprops=dict(arrowstyle='->'), textcoords = 'offset points')

    plt.xlabel("PC 1", fontsize = 18)
    plt.ylabel("PC 2", fontsize = 18)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]
    plt.legend(handles=Patches)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()

def plot_hid_traj(task_group_hid_traj, task_group, task_indices, trial_indices, instruct_indices, subtitle='', annotate_tuples = [], context_task=None, save_file=None): 
    alphas = np.linspace(0.8, 0.2, num=task_group_hid_traj.shape[2])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    embedder = PCA(n_components=3)
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
                ax.quiver(embedded[task_index, instruct_index, 1:, 0], embedded[task_index, instruct_index, 1:, 1], embedded[task_index, instruct_index, 1:, 2], 
                            np.diff(embedded[task_index, instruct_index, :, 0], axis=0), np.diff(embedded[task_index, instruct_index, :, 1], axis=0), np.diff(embedded[task_index, instruct_index, :, 2], axis=0),
                            length = 0.35, color = task_colors[task], arrow_length_ratio=0.3,  pivot='middle', linewidth=1, linestyle = linestyle)

                ax.scatter(embedded[task_index, instruct_index, 0, 0], embedded[task_index, instruct_index, 0, 1], embedded[task_index, instruct_index, 0, 2],  
                            s = 100, color='white', edgecolor= task_colors[task], marker='*')
                ax.scatter(embedded[task_index, instruct_index, 119, 0], embedded[task_index, instruct_index, 119, 1], embedded[task_index, instruct_index, 119, 2],  
                            s = 100, color='white', edgecolor= task_colors[task], marker='o')

                ax.scatter(embedded[task_index, instruct_index, 99, 0], embedded[task_index, instruct_index, 99, 1], embedded[task_index, instruct_index, 99, 2], 
                            s=100, color='white', edgecolor= task_colors[task], marker = 'P')
                if task_group == 'COMP': 
                    ax.scatter(embedded[task_index, instruct_index, 59, 0], embedded[task_index, instruct_index, 59, 1], embedded[task_index, instruct_index, 59, 2], 
                            s=100, color='white', edgecolor= task_colors[task], marker = 'X')

                if 'RT' in task: 
                    ax.scatter(embedded[task_index, instruct_index, 99, 0], embedded[task_index, instruct_index, 99, 1], embedded[task_index, instruct_index, 99, 2], 
                            s=100, color='white', edgecolor= task_colors[task], marker = 'X')
                else: 
                    ax.scatter(embedded[task_index, instruct_index, 19, 0], embedded[task_index, instruct_index, 19, 1], embedded[task_index, instruct_index, 19, 2], 
                            s=100, color='white', edgecolor= task_colors[task], marker = 'X')
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

def plot_RDM(input_reps, rep_type, similarity_scores = None, cmap=sns.color_palette("rocket_r", as_cmap=True), plot_title = 'RDM', use_avg_reps = False, save_file=None):
    if rep_type == 'lang': 
        rep_dim = 768
        number_reps=15
        label_buffer = 2
    if rep_type == 'task': 
        rep_dim = 128
        number_reps=100
        label_buffer = 8

    if similarity_scores is None: 
        reps = input_reps.copy()
        reps[[1,2], ...] = reps[[2,1], ...] 
        if use_avg_reps:
            reps = np.mean(reps, axis=1)
        reps = reps.reshape(-1, rep_dim)
        sim_scores = 1-np.corrcoef(reps)
    else: 
        sim_scores = similarity_scores.copy()
        # sim_scores.reshape(16, number_reps, -1)
        # sim_scores[[1,2], ...] = sim_scores[[2,1], ...] 
        # sim_scores.reshape(-1, number_reps*16)

    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))
    sns.heatmap(sim_scores, yticklabels = '', xticklabels= '',
                        cmap=cmap, vmin=0, vmax=1, ax=axn, annot_kws={"size": 4}, cbar_kws={'label': '1-r'})

    for i, task in enumerate(Task.opp_task_list):
        plt.text(-2, label_buffer+number_reps/2+number_reps*i, task, ha='right', size=6)
        plt.text(-label_buffer + number_reps/2+number_reps*i, number_reps*16, task, va='top', rotation='vertical', size=6)
    plt.title(plot_title, fontweight='bold', fontsize=12)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return sim_scores


def make_tuning_curve(model, tasks, task_variable, unit, mod, times, save_file=None): 
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
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return trials

def plot_neural_resp(model, task_type, task_variable, unit, mod, save_file=None):
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
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()
    return trials

if __name__ == "__main__":
    from nlp_models import SBERT
    from rnn_models import InstructNet, SimpleNet
    from utils import train_instruct_dict
    from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores
    import numpy as np
    import torch.optim as optim
    from utils import sort_vocab, train_instruct_dict
    from task import DM

    from matplotlib.pyplot import get
    from numpy.lib import utils
    import torch
    import torch.nn as nn



    trials = construct_batch('DM', 100)
    #plot_model_response(model, DM('DM', 10))
    #plot_single_seed_training('_ReLU128_5.7/single_holdouts/', 'DMS', ['sbertNet', 'sbertNet_layer_11', 'simpleNet'], 'correct', 4)
    model_data_dict = plot_holdout_curves('_ReLU128_5.7/single_holdouts/', ['sbertNet_tuned', 'sbertNet_layer_11', 'simpleNet'], 'correct', 'avg_holdout', range(5), smoothing = 0.01)
    model_data_dict

    #fig2
    model_data_dict = plot_holdout_curves('_ReLU128_5.7/single_holdouts/', ['sbertNet_layer_11', 'simpleNet'], 'correct', 'avg_holdout', range(5), smoothing = 0.01)
    plot_k_shot_learning(model_data_dict)

    ###fig3

    plot_single_holdout_task('_ReLU128_5.7/single_holdouts/', 'Anti Go', ['sbertNet_layer_11', 'simpleNet'], range(5), save_file='Anti_Go_holdout_curves.svg')


    #SimpleNet
    model1 = SimpleNet(128, 1, use_ortho_rules=True)
    all_sim_scores1 = get_sim_scores(model1, 'Multitask', 'task')
    pickle.dump(np.mean(all_sim_scores1, axis=0), open('_ReLU128_5.7/single_holdouts/simpleNet_Multitask_RDM', 'wb'))

    plot_RDM(None, 'task', similarity_scores= np.mean(all_sim_scores1, axis=0))

    reduced_reps1 = reduce_rep(get_task_reps(model1))
    plot_rep_scatter(reduced_reps1[0], ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'])

    #sbertNet_layer_11
    model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
    model.model_name += '_tuned'
    all_sim_scores = get_sim_scores(model, 'Multitask', 'lang')
    #pickle.dump(np.mean(all_sim_scores, axis=0), open('_ReLU128_5.7/single_holdouts/sbert11_Anti_Go_RDM', 'wb'))


    np.mean(all_sim_scores, axis=0).reshape()


    plot_RDM(None, 'lang', similarity_scores= np.mean(all_sim_scores, axis=0))

    reduced_reps = reduce_rep(get_task_reps(model))
    plot_rep_scatter(reduced_reps[0], ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'])


    #lang_reps
    model.set_seed(2)
    model.load_model('_ReLU128_5.7/single_holdouts/Anti_Go')

    lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
    lang_rep_reduced, _ = reduce_rep(lang_reps)
    plot_rep_scatter(lang_rep_reduced, ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'])
    plot_RDM(lang_reps, rep_type = 'lang')


    #Representation Dist.
    model1_multi_sims = pickle.load(open('_ReLU128_5.7/single_holdouts/simpleNet_Multitask_RDM', 'rb'))
    plot_RDM(None, 'task', similarity_scores= model1_multi_sims)

    model1_antigo_sims = pickle.load(open('_ReLU128_5.7/single_holdouts/simpleNet_Anti_Go_RDM', 'rb'))
    plot_RDM(None, 'task', similarity_scores= model1_antigo_sims)

    sbert11_multi_sims = pickle.load(open('_ReLU128_5.7/single_holdouts/sbert11_Multitask_RDM', 'rb'))
    plot_RDM(None, 'task', similarity_scores= sbert11_multi_sims)

    sbert11_antigo_sims = pickle.load(open('_ReLU128_5.7/single_holdouts/sbert11_Anti_Go_RDM', 'rb'))
    plot_RDM(None, 'task', similarity_scores= sbert11_antigo_sims)

    
    abs(np.linalg.norm(model1_multi_sims) - np.linalg.norm(model1_antigo_sims))
    abs(np.linalg.norm(sbert11_multi_sims) - np.linalg.norm(sbert11_antigo_sims))

    abs(np.linalg.norm(model1_multi_sims, ord=np.inf) - np.linalg.norm(model1_antigo_sims, ord = np.inf))
    abs(np.linalg.norm(sbert11_multi_sims, np.inf) - np.linalg.norm(sbert11_antigo_sims, np.inf))

    abs(np.linalg.norm(model1_multi_sims, 'nuc') - np.linalg.norm(model1_antigo_sims, 'nuc'))
    abs(np.linalg.norm(sbert11_multi_sims, 'nuc') - np.linalg.norm(sbert11_antigo_sims, 'nuc'))

    ###Figure 4
