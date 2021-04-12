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

def plot_hid_PCA_comparison(cogMod, tasks, holdout_task, ax_titles): 
    colors = ['Red']*250  + ['Yellow']*250 + ['Blue']*250 + ['Green']*250
    ax_list = []
    for model_name in cogMod.model_dict.keys(): 
        _, ax = cogMod.plot_task_rep(model_name, dim = 2, tasks = tasks, epoch = 'prep', avg_rep = False)
        ax_list.append(ax)

    fig, axn = plt.subplots(1, len(ax_list), figsize = (10, 4))
    plt.suptitle(r'$\textbf{PCA of Preparatory RNN Activity}$', fontsize=14, fontweight='bold')
    Patches = []
    for i, ax in enumerate(ax_list):
        scatter = ax_list[i]
        ax = axn.flat[i]
        ax.set_title(ax_titles[i]) 
        ax.scatter(scatter[0], scatter[1], color=colors, cmap = scatter[3], s=10)
        if i == 0: 
            ax.set_ylabel('PC1')
        ax.set_xlabel('PC2')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))        

    Patches = []
    for label, color in [('DM', 'Red'), ('Anti DM', 'Green'), ('MultiDM', 'Blue'), ('Anti MultiDM', 'Yellow')]:
        patch = mpatches.Patch(color=color, label=label)
        Patches.append(patch)

    plt.legend(handles = Patches)
    plt.show()



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














from Task import construct_batch, Go, DM
import torch
from sklearn.preprocessing import normalize
from LangModule import get_batch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Rectangle
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from NLPmodels import SBERT
from RNNs import instructNet, simpleNet
from LangModule import LangModule
from CogModule import CogModule

# foldername = '_ReLU128_dmStaggered'


# holdout = 'Anti RT Go'

# model_dict = {}
# #model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
# model1_name = 'ReLU128_/'+holdout+'/'+holdout+'_Model1.pt'
# model1_name = model1_name.replace(' ', '_')
# Model1 = simpleNet(81, 128, 1, 'relu')
# Model1.load_state_dict(torch.load(model1_name))
# model_dict['Model1'] = Model1

# ModelS = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
# ModelS_name = foldername +'/'+holdout+'/'+holdout+'_S-Bert_train.pt'
# ModelS_name = ModelS_name.replace(' ', '_')
# ModelS.load_state_dict(torch.load(ModelS_name))
# model_dict['S-Bert train'] = ModelS

# cog = CogModule(model_dict)

# cog.load_training_data(holdout, foldername, 'holdout')
# cog.plot_learning_curve('correct', holdout, smoothing=0.1)

def make_test_trials(task_type, task_variable, mod, instruct_mode, num_trials=100): 
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    intervals = np.empty((num_trials, 5), dtype=tuple)
    if task_variable == 'direction': 
        directions = np.linspace(0, 2*np.pi, num=num_trials)
        strengths = [1]* num_trials
        var_of_interest = directions
    elif task_variable == 'strength': 
        directions = np.array([np.pi+1] * num_trials)
        strengths = np.linspace(0.3, 1.8, num=num_trials)
        var_of_interest = strengths
    elif task_variable == 'diff_strength': 
        directions = np.array([np.pi] * num_trials)
        strengths = [1]* num_trials
        diff_strength = np.linspace(-0.5, 0.5, num=num_trials)
        var_of_interest = diff_strength
    elif task_variable == 'diff_direction': 
        diff_directions = np.linspace(0, np.pi, num=num_trials)
        strengths = [0.5] * num_trials
        var_of_interest = directions
    if task_type in ['Go', 'Anti Go', 'RT Go', 'Anti RT Go']:
        stim_mod_arr = np.empty((2, num_trials), dtype=list)
        for i in range(num_trials): 
            intervals[i, :] = ((0, 20), (20, 60), (60, 80), (80, 100), (100, 120))
            strength_dir = [(strengths[i], directions[i])]
            stim_mod_arr[mod, i] = strength_dir
            stim_mod_arr[((mod+1)%2), i] = None
        trials = Go(task_type, num_trials, intervals=intervals, stim_mod_arr=stim_mod_arr, directions=directions)
    if task_type in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']:
        stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)
        for i in range(num_trials): 
            intervals[i, :] = ((0, 20), (20, 60), (60, 80), (80, 100), (100, 120))
            if task_variable == 'diff_direction': 
                stim_mod_arr[0, mod, i] = [(1+(strengths[i]/2), np.pi)]
                stim_mod_arr[1 ,mod, i] = [(1-(strengths[i]/2), diff_directions[i])]
            elif task_variable == 'diff_strength':
                stim_mod_arr[0, mod, i] = [(strengths[i], directions[i])]
                stim_mod_arr[1 ,mod, i] = [(strengths[i]+diff_strength[i], directions[i]+np.pi)]
            else: 
                stim_mod_arr[0, mod, i] = [(strengths[i], directions[i])]
                stim_mod_arr[1 ,mod, i] = [(1, np.pi)]

            if 'Multi' in task_type: 
                stim_mod_arr[0, ((mod+1)%2), i] = stim_mod_arr[0, mod, i]
                stim_mod_arr[1, ((mod+1)%2), i] = stim_mod_arr[1, mod, i]
            else: 
                stim_mod_arr[0, ((mod+1)%2), i] = None
                stim_mod_arr[1, ((mod+1)%2), i] = None

            stim_mod_arr.shape
        trials = DM(task_type, num_trials, intervals=intervals, stim_mod_arr=stim_mod_arr, directions=directions)
    return trials, var_of_interest

def get_hid_var_resp(model, task_type, trials, num_repeats = 10, instruct_mode=None): 
    tar_dirs = trials.target_dirs
    tar = trials.targets
    num_trials = trials.inputs.shape[0]
    total_neuron_response = np.empty((num_repeats, 100, 120, 128))
    for i in range(num_repeats): 
        h0 = model.initHidden(num_trials, 0.1).to(device)
        model.eval()
        out, hid = cog._get_model_resp(model, num_trials, torch.Tensor(trials.inputs).to(device), task_type, instruct_mode)
        hid = hid.detach().cpu().numpy()
        total_neuron_response[i, :, :, :] = hid
    mean_neural_response = np.mean(total_neuron_response, axis=0)
    return mean_neural_response


def make_tuning_curve(model, task_type, task_variable, unit, time, mod, instruct_mode=None): 
    trials, var_of_interest = make_test_trials(task_type, task_variable, 1, instruct_mode=instruct_mode)
    hid = get_hid_var_resp(ModelS, task_type, trials, instruct_mode=instruct_mode)

    if task_variable == 'direction': 
        labels = ["0", "$2\pi$"]
        plt.xticks([0, np.pi, 2*np.pi], labels=['0', '$\pi$', '$2\pi$'])
    elif task_variable == 'diff_direction':
        labels = ["$\pi$", "0"]
    elif task_variable == 'strength':
        labels = ["0.3", "1.8"]
    elif task_variable =='diff_strength': 
        labels = ["delta -0.5", "delta 0.5"]

    neural_resp = hid[:, time, unit]

    plt.plot(var_of_interest, neural_resp)
    plt.title(task_type + ' tuning curve for Unit' + str(unit) + ' at time ' +str(time))
    plt.ylim(0, np.max(neural_resp)+0.15)
    plt.xlabel(task_variable)
    plt.show()
    return trials



def plot_neural_resp(model, task_type, task_variable, unit, mod, instruct_mode=None):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    trials, _ = make_test_trials(task_type, task_variable, mod, instruct_mode)
    hid = get_hid_var_resp(model, task_type, trials, instruct_mode=instruct_mode)
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
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    fig, axn = plt.subplots()
    ylim = np.max(hid[:,:,unit])
    for i in [x*4 for x in range(25)]:
        plot = plt.plot(hid[i, :, unit], c = scalarMap.to_rgba(i))
    plt.vlines(100, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    if 'RT' in task_type: 
        plt.xticks([100], labels=['Stim. Onset/Reponse'])

    elif 'DM' in task_type:
        plt.vlines(20, -1.5, ylim+0.15, colors='k', linestyles='dashed')
        plt.vlines(60, -1.5, ylim+0.15, colors='k', linestyles='dashed')
        plt.xticks([20, 60, 100], labels=['Stim. 1 Onset', 'Stim. 2 Onset', 'Reponse'])

    else: 
        plt.xticks([20, 100], labels=['Stim. Onset', 'Reponse'])
        plt.vlines(20, -1.5, ylim+0.15, colors='k', linestyles='dashed')

    axn.set_ylim(0, ylim+0.15)
    cbar = plt.colorbar(scalarMap, orientation='vertical', label = task_variable, ticks = [0, 100])
    plt.title(task_type + ' response for Unit' + str(unit))
    cbar.set_ticklabels(labels)
    plt.show()
    return trials


# unit = 110
# time = 60

# trials = plot_neural_resp(ModelS, 'Go', 'direction', unit, 1)
# trials = plot_neural_resp(ModelS, 'RT Go', 'direction', unit, 1)

# trials = plot_neural_resp(ModelS, 'Anti Go', 'direction', unit, 1)

# trials = plot_neural_resp(ModelS, holdout, 'direction', unit, 1)
# trials = plot_neural_resp(ModelS, holdout, 'direction', unit, 1, instruct_mode='instruct_swap')


# trials = make_tuning_curve(ModelS, 'Go', 'direction', unit, 99, 1, instruct_mode=None)
# trials = make_tuning_curve(ModelS, 'RT Go', 'direction', unit, 115, 1, instruct_mode=None)
# trials = make_tuning_curve(ModelS, 'Anti Go', 'direction', unit, 99, 1, instruct_mode=None)
# trials = make_tuning_curve(ModelS, 'Anti RT Go', 'direction', unit, 115, 1, instruct_mode=None)

# trials = make_tuning_curve(ModelS, holdout, 'direction', unit, 115, 1, instruct_mode='masked')



# cog.plot_response('S-Bert train', 'Anti RT Go', instruct_mode='masked')

# cog._plot_trained_performance()



# for task in ['Go', 'Anti Go', 'RT Go', 'Anti RT Go', 'DM']:
#     trials = plot_neural_resp([ModelS], task, 10, 'direction', 1)

# for task in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']:
#     trials = plot_neural_resp([Model1], task, 17, 'diff_strength', 1)


# trials.plot_trial(0)







# fig, axn = plt.subplots()
# plt.plot(np.linspace(0, 2*np.pi, num=num_trials), hid[:, 60, j], alpha = normed[i], c = cmap(normed[i]))
# axn.set_ylim(-1, 1)
# plt.show()



# torch.cat((torch.empty(100, 120, 81), torch.zeros(100, 120, 30)), dim=2).shape


# cog = CogModule(model_dict)
# cog.model_dict.keys()
# cog.load_models('Anti DM', foldername)

# from LangModule import train_instruct_dict
# from collections import defaultdict
# import seaborn as sns
# import matplotlib.pyplot as plt


# indices, reps = cog.model_dict['S-Bert train'].langMod._get_instruct_rep(train_instruct_dict)
# indices, reps = cog.model_dict['S-Bert_cat'].langMod._get_instruct_rep(train_instruct_dict)
# indices, reps = LangModule(SBERT(20))._get_instruct_rep(train_instruct_dict)

# rep_dict = defaultdict(list)
# for index, rep in list(zip(indices, reps)): 
#     rep_dict[task_list[index]].append(rep)

# sims = cosine_similarity(rep_dict['COMP1'], rep_dict['COMP2'])

# sims = cosine_similarity(np.array([np.mean(np.array(rep_dict['COMP1']), 0), np.mean(np.array(rep_dict['COMP2']), 0)]))

# sns.heatmap(sims, annot=True, vmin=0, vmax=1)
# plt.title('S-BERT (end-to-end)')
# plt.ylabel('COMP1 Instructions')
# plt.xlabel('COMP2 Instructions')
# plt.show()

# shuffled_dict = {}

# for task, instructs in train_instruct_dict.items(): 
#     instruction_list = []
#     for instruct in instructs: 
#         instruct = instruct.split()
#         shuffled = np.random.permutation(instruct)
#         instruct = ' '.join(list(shuffled))
#         instruction_list.append(instruct)
#     shuffled_dict[task] = instruction_list

# shuffled_dict['Go']


# indices, reps = cog.model_dict['S-Bert train'].langMod._get_instruct_rep(train_instruct_dict)
# shuffled_rep_dict = defaultdict(list)
# for index, rep in list(zip(indices, reps)): 
#     shuffled_rep_dict[task_list[index]].append(rep)
# shuffled_sims = cosine_similarity(rep_dict['DM'], rep_dict['DM'])

# sns.heatmap(shuffled_sims,  annot=True, vmin=0, vmax=1)
# plt.title('Language Representation Similarity Scores (S-BERT train)')
# plt.ylabel('COMP1 Instructions')
# plt.xlabel('COMP2 Instructions')
# plt.show()



# cog.load_models('Anti DM', foldername)


# from Task import construct_batch
# from CogModule import mask_input_rule, isCorrect
# import torch
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import matplotlib


# dim=2
# epoch = 'prep'
# avg_rep = False
# instruct_mode = None
# num_trials = 50
# model = cog.model_dict['Model1']
# holdout_task= None

# tasks = ['Go', 'Anti Go']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# task_reps = []
# correct_list  = []
# for task in task_list: 
#     trials = construct_batch(task, num_trials)
#     tar_dirs = trials.target_dirs
#     tar = trials.targets
#     ins = mask_input_rule(torch.Tensor(trials.inputs), num_trials, 120).to(device)
#     h0 = torch.FloatTensor(num_trials, 128).uniform_(-1, 1).unsqueeze(0).to(device)
#     out, hid = model(ins, h0)

#     correct_list += list(isCorrect(out, tar, tar_dirs))
#     hid = hid.detach().cpu().numpy()
#     epoch_state = []

#     for i in range(num_trials): 
#         if epoch.isnumeric(): 
#             epoch_index = int(epoch)
#             epoch_state.append(hid[i, epoch_index, :])
#         if epoch == 'stim': 
#             epoch_index = np.where(tar[i, :, 0] == 0.85)[0][-1]
#             epoch_state.append(hid[i, epoch_index, :])
#         if epoch == 'response':
#             epoch_state.append(hid[i, -1, :])
#         if epoch == 'input':
#             epoch_state.append(hid[i, 0, :])
#         if epoch == 'prep': 
#             epoch_index = np.where(ins[i, :, 18:]>0.25)[0][0]-1
#             epoch_state.append(hid[i, epoch_index, :])
    
#     if avg_rep: 
#         epoch_state = [np.mean(np.stack(epoch_state), axis=0)]
    
#     task_reps += epoch_state

# embedded = PCA(n_components=dim).fit_transform(task_reps)
# cmap = matplotlib.cm.get_cmap('tab20')

# if avg_rep: 
#     to_plot = np.stack([embedded[task_list.index(task), :] for task in tasks])
#     task_indices = np.array([task_list.index(task) for task in tasks]).astype(int)
#     marker_size = 100
# else: 
#     to_plot = np.stack([embedded[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials, :] for task in tasks]).reshape(len(tasks)*num_trials, dim)
#     correct = np.stack([correct_list[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials] for task in tasks]).flatten()
#     task_indices = np.array([[task_list.index(task)]*num_trials for task in tasks]).astype(int).flatten()
#     marker_size = 25
# tasks
# len(task_indices)
# dots = cmap(task_indices)
# correct = np.where(correct<1, 0.25, correct)
# dots[:, 3] = correct

# plt.scatter(to_plot[:, 0], to_plot[:, 1], c=dots, s=25)
# plt.xlabel("PC 1", fontsize = 18)
# plt.ylabel("PC 2", fontsize = 18)
# plt.show()

