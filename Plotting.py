import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from Task import Task
from CogModule import CogModule
from LangModule import swaps

from matplotlib import rc,rcParams
from pylab import *

task_list = Task.TASK_LIST

def label_plot(fig, Patches, Markers, legend_loc = (0.9, 0.3)): 
    arch_legend = plt.legend(handles=Patches, title = r"$\textbf{Language Module}$", bbox_to_anchor = legend_loc, loc = 'lower center')
    ax = plt.gca().add_artist(arch_legend)
    plt.legend(handles= Markers, title = r"$\textbf{Transformer Fine-Tuning}$", bbox_to_anchor = legend_loc, loc = 'upper center')
    fig.text(0.5, 0.04, 'Training Examples', ha='center')
    fig.text(0.04, 0.5, 'Fraction Correct', va='center', rotation='vertical')

def plot_avg_curves(model_dict, foldername, smoothing = 1, plot_background_traces=False): 


    # activate latex text rendering
    rc('text', usetex=True)
    #rc('axes', linewidth=2)
    rc('font', weight='bold')
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']



    fig, ax = plt.subplots(1,1, figsize =(12, 8))
    plt.suptitle(r'$\textbf{Avg. Performance over All Holdout Tasks}$')
    ax.set_ylim(-0.05, 1.15)
    cog = CogModule(model_dict)
    avg_perf_dict = {}
    for model_name in model_dict.keys(): 
        avg_perf_dict[model_name] = np.zeros(100)
    for holdout_task in task_list: 
        cog.reset_data()
        cog.load_training_data(holdout_task, foldername, 'holdout')
        for model_type in model_dict.keys():       
            train_data = cog.task_sorted_correct[model_type][holdout_task]
            avg_perf_dict[model_type]+=np.array(train_data)
            if plot_background_traces:
                smoothed_perf = gaussian_filter1d(train_data, sigma=smoothing)
                ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_type][0], linewidth = 2, marker=cog.ALL_STYLE_DICT[model_type][1], markersize=8, markevery=2, alpha=0.05)
    for model_type in model_dict.keys(): 
        train_data = avg_perf_dict[model_type]/len(task_list)
        smoothed_perf = gaussian_filter1d(train_data, sigma=smoothing)
        ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_type][0], linewidth = 2, marker=cog.ALL_STYLE_DICT[model_type][1], alpha=1, markersize=7, markevery=2)
    Patches, Markers = cog.get_model_patches()
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    # fig.text(0.5, 0.02, r'$\textbf{Training Examples}$', ha='center')
    # fig.text(0.02, 0.5, r'$\textbf{Fraction Correct}$', va='center', rotation='vertical')
    plt.legend(handles=Patches+Markers, loc = 'lower right', title = r"$\textbf{Language Module}$")
    plt.tight_layout()

    fig.show()


def plot_all_tasks_by_model(model_name, foldername, smoothing=1):
    fig, ax = plt.subplots(1,1)
    name_to_plot = CogModule.NAME_TO_PLOT_DICT[model_name]
    plt.suptitle(name_to_plot + ' Performance over Tasks')
    ax.set_ylim(-0.05, 1.15)
    model_dict = dict(zip([model_name], [None]))
    cog = CogModule(model_dict)

    cmap = matplotlib.cm.get_cmap('tab20')

    for i, holdout_task in enumerate(task_list): 
        cog.reset_data()
        cog.load_training_data(holdout_task, foldername, 'holdout')
        train_data = cog.task_sorted_correct[model_name][holdout_task]
        smoothed_perf = gaussian_filter1d(train_data, sigma=smoothing)
        ax.plot(smoothed_perf, color = cmap(i))

    Patches = [mpatches.Patch(color=cmap(d), label=task_list[d]) for d in range(len(task_list))]
    plt.legend(handles=Patches)

    fig.text(0.5, 0.04, 'Training Examples', ha='center')
    fig.text(0.04, 0.5, 'Fraction Correct', va='center', rotation='vertical')
    fig.show()

def plot_all_holdout_curves(model_dict, foldername, smoothing=1):
    cog = CogModule(model_dict)
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True)
    plt.suptitle('Holdout Learning for All Tasks')
    for i, ax in enumerate(axn.flat):
        ax.set_ylim(-0.05, 1.15)
        holdout_task = task_list[i]
        cog.load_training_data(holdout_task, foldername, 'holdout')
        for model_name in model_dict.keys(): 
            smoothed_perf = gaussian_filter1d(cog.task_sorted_correct[model_name][holdout_task][0:100], sigma=smoothing)
            ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_name][0], marker=cog.ALL_STYLE_DICT[model_name][1], alpha=1, markersize=5, markevery=3)
        ax.set_title(holdout_task)
    Patches, Markers = cog.get_model_patches()
    label_plot(fig, Patches, Markers, legend_loc=(1.3, 0.5))
    fig.show()

def plot_learning_curves(model_dict, tasks, foldername, comparison, dim, smoothing=1): 
    cog = CogModule(model_dict)
    fig, axn = plt.subplots(1,4, sharey = True, sharex=True, figsize = dim)
    legend_loc = (0.7, 0.3)


    # activate latex text rendering
    rc('text', usetex=True)
    #rc('axes', linewidth=2)
    rc('font', weight='bold')
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


    if comparison is None: 
        plt.suptitle(r"$\textbf{Holdout Learning Curves for 'Comparison' Tasks}$", fontsize = 16)
        load_list = [('holdout', '-')]
        mark = []
    elif comparison == 'shuffled': 
        plt.suptitle(r"$\textbf{Shuffled Instructions Comparisons}$", fontsize = 16)
        load_list = [('holdout', '-'), ('holdoutshuffled', '--')]
        mark = [Line2D([0], [0], color='grey', linestyle = load_list[1][1], label = 'Shuffled', markerfacecolor='grey', markersize=10)]
        # arch_legend = plt.legend(handles=[mark], loc = 'lower center', bbox_to_anchor = legend_loc)
        # ax = plt.gca().add_artist(arch_legend)
    
    elif comparison == 'comp': 
        plt.suptitle(r"$\textbf{Compositional One-Hot Comparisons}$", fontsize = 16)
        load_list = [('holdout', '-'), ('holdoutcomp', '--')]
        mark = [Line2D([0], [0], color='grey', linestyle = load_list[1][1], label = 'Compositional One-Hot', markerfacecolor='grey', markersize=10)]
        # arch_legend = plt.legend(handles=[mark], loc = 'lower center', bbox_to_anchor = legend_loc)
        # ax = plt.gca().add_artist(arch_legend)

    elif comparison == 'swapped': 
        plt.suptitle(r"$\textbf{Swapped Instructions Comparisons}$", fontsize = 16)
        load_list = [('holdout', '-'), ('swappedholdout', '--')]
        assert all([task in ['Go', 'Anti DM', 'Anti RT Go', 'DMC', 'RT Go', 'COMP2'] for task in tasks])
        mark = [Line2D([0], [0], color='grey', linestyle = load_list[1][1], label = 'Swapped', markerfacecolor='grey', markersize=10)]
        # arch_legend = plt.legend(handles=[mark], loc = 'lower center', bbox_to_anchor = legend_loc)
        # ax = plt.gca().add_artist(arch_legend)

    for i, task in enumerate(tasks): 
        ax = axn.flat[i]
        holdout_folder = task
        for load_type in load_list: 
            if load_type[0] == 'swappedholdout': 
                holdout_folder = ''.join([x for x in swaps if task in x][0]).replace(' ', '_')
            cog.load_training_data(holdout_folder, foldername, load_type[0])
            for model_name in model_dict.keys(): 
                if load_type[0] != 'holdout' and model_name == 'Model1' and comparison == 'shuffled': 
                    continue
                if load_type[0] == 'holdoutcomp' and model_name != 'Model1':
                    continue
                smoothed_perf = gaussian_filter1d(cog.task_sorted_correct[model_name][task][0:99], sigma=smoothing)
                ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_name][0], marker=cog.ALL_STYLE_DICT[model_name][1], linestyle = load_type[1], markevery=5)
        ax.set_title(task + ' Holdout')
        # if i == 0: 
        #     ax.set_ylabel('Fraction Correct')
    Patches, Markers = cog.get_model_patches()
    plt.legend(handles=Patches+Markers+mark, title = r"$\textbf{Language Module}$", loc = 'lower right')

    fig.text(0.5, 0.01, r'$\textbf{Training Examples}$', ha='center', fontsize = 14)
    fig.text(0.01, 0.5, r'$\textbf{Fraction Correct}$', va='center', rotation='vertical', fontsize = 14)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    fig.show()


# def plot_learning_curves(model_dict, tasks, foldername, comparison, smoothing=1): 
#     cog = CogModule(model_dict)
#     fig, axn = plt.subplots(3,1, sharey = True, sharex=True, figsize =(4, 12))
#     legend_loc = (1.1, 0.7)
#     if comparison is None: 
#         plt.suptitle(r"$\textbf{Holdout Learning Curves}$")
#         load_list = [('holdout', '-')]
#     elif comparison == 'shuffled': 
#         plt.suptitle(r"$\textbf{Shuffled Instructions Comparisons}$")
#         load_list = [('holdout', '-'), ('holdoutshuffled', '--')]
#         mark = Line2D([0], [0], color='grey', linestyle = load_list[1][1], label = 'Shuffled', markerfacecolor='grey', markersize=10)
#         arch_legend = plt.legend(handles=[mark], loc = 'lower center', bbox_to_anchor = legend_loc)
#         ax = plt.gca().add_artist(arch_legend)
    
#     elif comparison == 'comp': 
#         plt.suptitle(r"$\textbf{Compositional One-Hot Comparisons}$")
#         load_list = [('holdout', '-'), ('holdoutcomp', '--')]
#         mark = Line2D([0], [0], color='grey', linestyle = load_list[1][1], label = 'Compositional One-hot', markerfacecolor='grey', markersize=10)
#         arch_legend = plt.legend(handles=[mark], loc = 'lower center', bbox_to_anchor = legend_loc)
#         ax = plt.gca().add_artist(arch_legend)

#     elif comparison == 'swapped': 
#         plt.suptitle(r"$\textbf{Swapped Instructions Comparisons}$")
#         load_list = [('holdout', '-'), ('swappedholdout', '--')]
#         assert all([task in ['Go', 'Anti DM', 'Anti RT Go', 'DMC', 'RT Go', 'COMP2'] for task in tasks])
#         mark = Line2D([0], [0], color='grey', linestyle = load_list[1][1], label = 'Swapped', markerfacecolor='grey', markersize=10)
#         arch_legend = plt.legend(handles=[mark], loc = 'lower center', bbox_to_anchor = legend_loc)
#         ax = plt.gca().add_artist(arch_legend)

#     for i, task in enumerate(tasks): 
#         ax = axn.flat[i]
#         holdout_folder = task
#         for load_type in load_list: 
#             if load_type[0] == 'swappedholdout': 
#                 holdout_folder = ''.join([x for x in swaps if task in x][0]).replace(' ', '_')
#             cog.load_training_data(holdout_folder, foldername, load_type[0])
#             for model_name in model_dict.keys(): 
#                 if load_type[0] != 'holdout' and model_name == 'Model1' and comparison == 'shuffled': 
#                     continue
#                 if load_type[0] == 'holdoutcomp' and model_name != 'Model1':
#                     continue
#                 smoothed_perf = gaussian_filter1d(cog.task_sorted_correct[model_name][task][0:99], sigma=smoothing)
#                 ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_name][0], linestyle = load_type[1], markevery=5)
#         ax.set_title(task + ' Holdout')
#         # if i == 0: 
#         #     ax.set_ylabel('Fraction Correct')
#     Patches = []
#     Patches.append(mpatches.Patch(color='blue', label='One-Hot Task Vec.'))
#     Patches.append(mpatches.Patch(color='Red', label='GPT'))
#     Patches.append(mpatches.Patch(color='Purple', label='S-Bert'))
#     plt.legend(handles=Patches, title = r"$\textbf{Language Module}$")

#     fig.text(0.5, 0.01, r'$\textbf{Training Examples}$', ha='center')
#     fig.text(0.01, 0.5, r'$\textbf{Fraction Correct}$', va='center', rotation='vertical')
#     fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
#     fig.show()


def plot_side_by_side(ax_list, title, ax_titles): 
    num_trials = 250
    fig, axn = plt.subplots(len(ax_list), 1, figsize = (5, 12))
    plt.suptitle(r'$\textbf{PCA of Preparatory Sensory-Motor Activity (COMP2 Holdout)}$', fontsize=14, fontweight='bold')
    Patches = []
    for i, ax in enumerate(ax_list):
        ax = axn.flat[i]
        ax.set_title(ax_titles[i]) 
        scatter = ax_list[i]
            
        for j, color in enumerate(['Red', 'Green', 'Blue', 'Yellow']): 
            start = j*num_trials
            stop = start+num_trials
            ax.scatter(scatter[0][start:stop], scatter[1][start:stop], color=color, s=25)
            if i ==1: 
                Patches.append(mpatches.Patch(color = color, label = task_list[j+8]))
                ax.set_xlabel('PC2')

        ax.set_ylabel('PC1')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))        
    plt.legend(handles = Patches, loc='lower right')
    plt.show()
