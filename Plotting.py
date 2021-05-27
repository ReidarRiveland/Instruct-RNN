import numpy as np
import pickle
from collections import defaultdict
from numpy.lib.index_tricks import fill_diagonal
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import colors, cm 

import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter1d
import umap
from sklearn.manifold import TSNE

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import itertools

from Task import Task, construct_batch
from LangModule import swaps
from CogModule import CogModule

from matplotlib import rc,rcParams
from pylab import *

from Task import Go, DM

task_list = Task.TASK_LIST
from utils import ALL_STYLE_DICT, MARKER_DICT, MODEL_MARKER_DICT, NAME_TO_PLOT_DICT, task_group_colors, task_cmap, strip_model_name, get_model_patches, _label_plot, _collect_data_across_seeds

from RNNs import instructNet, simpleNet
from LangModule import LangModule
from NLPmodels import SBERT
from CogModule import CogModule

from scipy import stats




def plot_avg_holdout_curves(foldername, model_list, smoothing=0.01, seeds = list(range(5))): 
    all_correct, _, _, _ = _collect_data_across_seeds(foldername, model_list, seeds)
    seeds_summary_dict = {}
    seeds_avg_dict = {}
    for model_name in all_correct.keys(): 
        seeds_avg = np.empty((100, len(seeds)))
        for i in range(len(seeds)):
            model_task_data = np.empty((100, len(task_list)))
            for j, task in enumerate(task_list): 
                model_task_data[:, j] = all_correct[model_name][task][:, i]

            seeds_avg[:, i] = np.mean(model_task_data, axis=1)
        seeds_avg_dict[model_name] = seeds_avg
        seeds_summary_dict[model_name] = np.array([np.round(np.mean(seeds_avg_dict[model_name], axis=1), decimals=3), 
                                                        np.round(np.std(seeds_avg_dict[model_name], axis=1), decimals=3)])

    # activate latex text rendering
    rc('text', usetex=True)
    #rc('axes', linewidth=2)
    rc('font', weight='bold')
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    fig, ax = plt.subplots(1,1, figsize =(12, 8))
    plt.suptitle(r'$\textbf{Avg. Performance over All Holdout Tasks}$')

    ax.set_ylim(-0.05, 1.05)
    plt.yticks(np.arange(0, 1.1, 0.1))

    for model_name in model_list: 
        smoothed_perf = gaussian_filter1d(seeds_summary_dict[model_name][0], sigma=smoothing)
        ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), seeds_summary_dict[model_name][0]+seeds_summary_dict[model_name][1]]), axis=0), 
            seeds_summary_dict[model_name][0]-seeds_summary_dict[model_name][1], color = ALL_STYLE_DICT[model_name][0], alpha= 0.1)
        ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=5, markevery=3)
    Patches, Markers = get_model_patches(model_list)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.legend(handles=Patches+Markers, loc = 'lower right', title = r"$\textbf{Language Module}$")
    plt.tight_layout()
    plt.show()
    return seeds_summary_dict

def plot_all_holdout_curves(foldername, model_list, smoothing=0.01, seeds = list(range(5))): 
    all_correct, all_loss, all_summary_correct, all_summary_loss = _collect_data_across_seeds(foldername, model_list, seeds)

    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(14, 10))
    plt.suptitle(r'$\textbf{Holdout Learning for All Tasks}$')
    for i, ax in enumerate(axn.flat):
        ax.set_ylim(-0.05, 1.15)
        holdout_task = task_list[i]
        for model_name in model_list: 
            smoothed_perf = gaussian_filter1d(all_summary_correct[model_name][holdout_task][0], sigma=smoothing)
            ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), all_summary_correct[model_name][holdout_task][0]+all_summary_correct[model_name][holdout_task][1]]), axis=0), 
                all_summary_correct[model_name][holdout_task][0]-all_summary_correct[model_name][holdout_task][1], color =  ALL_STYLE_DICT[model_name][0], alpha= 0.1)
            ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=5, markevery=3)
        ax.set_title(holdout_task)

    Patches, Markers = get_model_patches(model_list)
    _label_plot(fig, Patches, Markers, legend_loc=(1.2, 0.5))
    plt.show()
    return all_summary_correct, all_summary_loss

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
                ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], linestyle = load_type[1], markevery=5)
        ax.set_title(task + ' Holdout')
        # if i == 0: 
        #     ax.set_ylabel('Fraction Correct')
    Patches, Markers = get_model_patches(model_dict.keys())
    plt.legend(handles=Patches+Markers+mark, title = r"$\textbf{Language Module}$", loc = 'lower right')

    fig.text(0.5, 0.01, r'$\textbf{Training Examples}$', ha='center', fontsize = 14)
    fig.text(0.01, 0.5, r'$\textbf{Fraction Correct}$', va='center', rotation='vertical', fontsize = 14)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    fig.show()

def plot_task_rep(model, epoch, reduction_method, z_score, num_trials = 250, dim = 2, holdout_task = None, tasks = task_list, avg_rep = False, Title=''): 
    if model.instruct_mode == 'comp': 
        assert holdout_task != None 
    # if not next(model.rnn.parameters()).is_cuda:
    #     model.to(device)

    assert epoch in ['input', 'stim', 'response', 'prep'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
    
    task_reps = []
    for task in task_list: 
        trials = construct_batch(task, num_trials)
        tar = trials.targets
        ins = trials.inputs

        if model.instruct_mode == 'comp': 
            if task == holdout_task: 
                out, hid = CogModule._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task)
            else: 
                out, hid = CogModule._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task)
        else: 
            out, hid = CogModule._get_model_resp(model, num_trials, torch.Tensor(ins).to(device), task)


        hid = hid.detach().cpu().numpy()
        epoch_state = []

        for i in range(num_trials): 
            if epoch.isnumeric(): 
                epoch_index = int(epoch)
                epoch_state.append(hid[i, epoch_index, :])
            if epoch == 'stim': 
                epoch_index = np.where(tar[i, :, 0] == 0.85)[0][-1]
                epoch_state.append(hid[i, epoch_index, :])
            if epoch == 'response':
                epoch_state.append(hid[i, -1, :])
            if epoch == 'input':
                epoch_state.append(hid[i, 0, :])
            if epoch == 'prep': 
                epoch_index = np.where(ins[i, :, 18:]>0.25)[0][0]-1
                epoch_state.append(hid[i, epoch_index, :])
        
        if avg_rep: 
            epoch_state = [np.mean(np.stack(epoch_state), axis=0)]
        
        task_reps += epoch_state


    if reduction_method == 'PCA': 
        embedder = PCA(n_components=dim)
    elif reduction_method == 'UMAP':
        embedder = umap.UMAP()
    elif reduction_method == 'tSNE': 
        embedder = TSNE(n_components=2)

    embedded = embedder.fit_transform(task_reps)

    if z_score: 
        embedded = stats.zscore(embedded)

    if reduction_method == 'PCA': 
        explained_variance = embedder.explained_variance_ratio_
    else: 
        explained_variance = None

    if avg_rep: 
        to_plot = np.stack([embedded[task_list.index(task), :] for task in tasks])
        task_indices = np.array([task_list.index(task) for task in tasks]).astype(int)
        marker_size = 100
    else: 
        to_plot = np.stack([embedded[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials, :] for task in tasks]).reshape(len(tasks)*num_trials, dim)
        task_indices = np.array([[task_list.index(task)]*num_trials for task in tasks]).astype(int).flatten()
        marker_size = 25

    cmap = matplotlib.cm.get_cmap('tab20')
    Patches = []
    if dim ==3: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = [to_plot[:, 0], to_plot[:, 1], to_plot[:,2], cmap(task_indices), cmap, marker_size]
        ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:,2], c = cmap(task_indices), cmap=cmap, s=marker_size)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    else:             
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = [to_plot[:, 0], to_plot[:, 1], cmap(task_indices), cmap, marker_size]
        # for i, color in enumerate(['Red', 'Blue', 'Green', 'Yellow']): 
        #     start = i*num_trials
        #     stop = start+num_trials
        #     plt.scatter(to_plot[:, 0][start:stop], to_plot[:, 1][start:stop], color=listset(task_indices), s=marker_size)
        #     Patches.append(mpatches.Patch(color = cmap(task_indices), label = task_list[list(set(task_indices))[i]]))
        ax.scatter(to_plot[:, 0], to_plot[:, 1], c = cmap(task_indices), cmap=cmap, s=marker_size)
        plt.xlabel("PC 1", fontsize = 18)
        plt.ylabel("PC 2", fontsize = 18)

    #plt.suptitle(r"$\textbf{PCA Embedding for Task Representation$", fontsize=18)
    plt.title(Title)
    digits = np.arange(len(tasks))
    plt.tight_layout()
    Patches = [mpatches.Patch(color=cmap(d), label=task_list[d]) for d in set(task_indices)]
    scatter.append(Patches)
    plt.legend(handles=Patches)
    #plt.show()
    return explained_variance, scatter

def plot_hid_PCA_comparison(cogMod, task_group, ax_titles, reduction_method = 'PCA', z_score = False): 
    plot_colors= list(itertools.chain.from_iterable([[color]*250 for color in task_group_colors[task_group].values()]))
    ax_list = []
    for model in cogMod.model_dict.values(): 
        _, ax = plot_task_rep(model, 'prep', reduction_method, z_score, dim = 2, tasks = list(task_group_colors[task_group].keys()), avg_rep = False)
        ax_list.append(ax)

    fig, axn = plt.subplots(1, len(ax_list), figsize = (10, 4))
    plt.suptitle(r'$\textbf{PCA of Preparatory RNN Activity}$', fontsize=14, fontweight='bold')
    Patches = []
    for i, ax in enumerate(ax_list):
        scatter = ax_list[i]
        ax = axn.flat[i]
        ax.set_title(ax_titles[i]) 
        ax.scatter(scatter[0], scatter[1], c=plot_colors, cmap = scatter[3], s=10)
        if i == 0: 
            ax.set_ylabel('PC1')
        ax.set_xlabel('PC2')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    Patches = []
    for label, color in task_group_colors[task_group].items():
        patch = mpatches.Patch(color=color, label=label)
        Patches.append(patch)

    plt.legend(handles = Patches)
    plt.show()

def plot_hid_traj(cogMod, tasks, dim, instruct_mode = None): 
    models = list(cogMod.model_dict.keys())
    if dim == 2: 
        fig, ax = plt.subplots()
    else: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    fig.suptitle('RNN Hidden State Trajectory')
    model_task_state_dict, intervals = get_hid_traj(cogMod, tasks, dim, instruct_mode)

    stim_onset = intervals[1][0]
    resp = intervals[-1][0]

    data_array = np.empty((len(models)* len(tasks)*dim), dtype=list)

    for i in range(data_array.shape[0]): 
        data_array[i] = [] 
    data_array = np.reshape(data_array, (len(models), len(tasks), dim))

    plot_list = []
    style_list = ['-', '--']
    for i in range(len(models)):
        for j in range (len(tasks)): 
            model_name = strip_model_name(models[i])
            if dim == 2: 
                plot_list.append(plt.plot([],[], label = model_name, color = ALL_STYLE_DICT[model_name][0], linestyle = style_list[j]))
            else:
                embedding_data = model_task_state_dict[models[i]][tasks[j]]
                plot_list.append(plt.plot(embedding_data[0, 0:1],embedding_data[1, 0:1], embedding_data[2, 0:1], color = ALL_STYLE_DICT[model_name][0], linestyle = style_list[j]))

    plot_array = np.array(plot_list).reshape((len(models), len(tasks)))

    def init():
        ax.set_xlim(-10, 10)
        ax.set_xlabel('PC 1')
        ax.set_ylim(-10, 10)
        ax.set_ylabel('PC 2')
        if dim == 3: 
            ax.set_zlim(-10, 10)
            ax.set_zlabel('PC 3')
        return tuple(plot_array.flatten())


    def update(i): 
        for j, model_name in enumerate(models): 
            for k, task in enumerate(tasks):
                embedding_data = model_task_state_dict[model_name][task]
                if dim ==3: 
                    plot_array[j][k].set_data(embedding_data[0:i, 0], embedding_data[0:i, 1])
                    plot_array[j][k].set_3d_properties(embedding_data[0:i, 2])
                else: 
                    data_array[j][k][0].append(embedding_data[i, 0])
                    data_array[j][k][1].append(embedding_data[i, 1])
                    plot_array[j][k].set_data(data_array[j][k][0], data_array[j][k][1])
        return tuple(plot_array.flatten())



    ani = animation.FuncAnimation(fig, update, frames=119,
                        init_func=init, blit=True)

    Patches, _ = get_model_patches(cogMod.model_dict.keys())

    for i, task in enumerate(tasks):
        Patches.append(Line2D([0], [0], linestyle=style_list[i], color='grey', label=task, markerfacecolor='grey', markersize=10))
    plt.legend(handles=Patches)
    plt.show()

    ax.clear()

def get_hid_traj(cogMod, tasks, dim, instruct_mode):
    with torch.no_grad(): 
        for model in cogMod.model_dict.values(): 
            if not next(model.parameters()).is_cuda:
                model.to(device)

        task_info_list = []
        for task in tasks: 
            trial = construct_batch(task, 1)
            task_info_list.append(trial.inputs)

        model_task_state_dict = {}
        for model_name, model in cogMod.model_dict.items(): 
            tasks_dict = {}
            for i, task in enumerate(tasks): 
                out, hid = cogMod._get_model_resp(model, 1, torch.Tensor(task_info_list[i]).to(device), task, instruct_mode)
                embedded = PCA(n_components=dim).fit_transform(hid.squeeze().detach().cpu())
                tasks_dict[task] = embedded
            model_task_state_dict[model_name] = tasks_dict
        return model_task_state_dict, trial.intervals[0]


def make_test_trials(task_type, task_variable, mod, num_trials=100): 
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    
    conditions_arr = np.empty((2, 2, 2, num_trials))
    intervals = np.empty((num_trials, 5), dtype=tuple)
    for i in range(num_trials): intervals[i, :] = ((0, 20), (20, 60), (60, 80), (80, 100), (100, 120))
    

    if task_variable == 'direction': 
        directions = np.linspace(0, 2*np.pi, num=num_trials)
        strengths = np.array([1]* num_trials)
        var_of_interest = directions

    elif task_variable == 'strength': 
        directions = np.array([np.pi] * num_trials)
        strengths = np.linspace(0.3, 1.8, num=num_trials)
        var_of_interest = strengths

    elif task_variable == 'diff_strength': 
        directions = np.array([[np.pi] * num_trials, [2*np.pi] * num_trials])
        fixed_strengths = np.array([1]* num_trials)
        diff_strength = np.linspace(-0.5, 0.5, num=num_trials)
        strengths = np.array([fixed_strengths, fixed_strengths-diff_strength])
        var_of_interest = diff_strength

    elif task_variable == 'diff_direction': 
        fixed_direction = np.array([np.pi] * num_trials)
        diff_directions = np.linspace(np.pi/4, 2*pi-np.pi/4, num=num_trials)
        directions = np.array([fixed_direction, fixed_direction-diff_directions])
        strengths = np.array([[1] * num_trials, [1.2] * num_trials])
        var_of_interest = diff_directions
    
    
    if task_type in ['Go', 'Anti Go', 'RT Go', 'Anti RT Go']:
        conditions_arr[mod, 0, 0, :] = directions
        conditions_arr[mod, 0, 1, :] = strengths
        conditions_arr[mod, 1, 0, :] = np.NaN
        conditions_arr[((mod+1)%2), :, :, :] = np.NaN
        trials = Go(task_type, num_trials, intervals=intervals, conditions_arr=conditions_arr)


    if task_type in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']:
        assert task_variable not in ['directions', 'strengths']
        conditions_arr[mod, :, 0, : ] = directions
        conditions_arr[mod, :, 1, : ] = strengths

        if 'Multi' in task_type: 
            conditions_arr[((mod+1)%2), :, :, :] = conditions_arr[mod, :, :, : ]
        else: 
            conditions_arr[((mod+1)%2), :, :, : ] = np.NaN

        trials = DM(task_type, num_trials, intervals=intervals, conditions_arr=conditions_arr)

    return trials, var_of_interest


def get_hid_var_resp(model, task_type, trials, num_repeats = 10, instruct_mode=None): 
    with torch.no_grad(): 
        num_trials = trials.inputs.shape[0]
        total_neuron_response = np.empty((num_repeats, num_trials, 120, 128))
        for i in range(num_repeats): 
            h0 = model.initHidden(num_trials, 0.1)
            model.eval()
            out, hid = CogModule._get_model_resp(model, num_trials, torch.Tensor(trials.inputs).to(device), task_type, instruct_mode)
            hid = hid.detach().cpu().numpy()
            total_neuron_response[i, :, :, :] = hid
        mean_neural_response = np.mean(total_neuron_response, axis=0)
    return total_neuron_response, mean_neural_response

def make_tuning_curve(model, task_type, task_variable, unit, time, mod, instruct_mode=None): 
    trials, var_of_interest = make_test_trials(task_type, task_variable, mod, instruct_mode=instruct_mode)
    hid = get_hid_var_resp(model, task_type, trials, instruct_mode=instruct_mode)

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
    plt.xlabel(task_variable.replace('_', ' '))
    plt.show()
    return trials

def plot_neural_resp(model, task_type, task_variable, unit, mod, instruct_mode=None):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    trials, _ = make_test_trials(task_type, task_variable, mod)
    _, hid = get_hid_var_resp(model, task_type, trials, instruct_mode=instruct_mode)
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
    ylim = np.max(hid[:,:,unit])
    for i in [x*4 for x in range(25)]:
        plot = plt.plot(hid[i, :, unit], c = scalarMap.to_rgba(i))
    plt.vlines(100, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    if 'RT' in task_type: 
        plt.xticks([100], labels=['Stim. Onset/Reponse'])

    # elif 'DM' in task_type:
    #     plt.vlines(20, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    #     plt.vlines(60, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    #     plt.xticks([20, 60, 100], labels=['Stim. 1 Onset', 'Stim. 2 Onset', 'Reponse'])

    plt.xticks([20, 100], labels=['Stim. Onset', 'Reponse'])
    plt.vlines(20, -1.5, ylim+0.15, colors='k', linestyles='dashed')

    axn.set_ylim(0, ylim+0.15)
    cbar = plt.colorbar(scalarMap, orientation='vertical', label = task_variable.replace('_', ' '), ticks = [0, 100])
    plt.title(task_type + ' response for Unit' + str(unit))
    cbar.set_ticklabels(labels)
    plt.show()
    return trials


foldername = '_ReLU128_12.4'
seed = '_seed'+str(0)
model_dict = {}
model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')
cog = CogModule(model_dict)
cog.load_models('DMC', foldername, seed)

cog._plot_trained_performance()


ModelS.langModel.out_dim

# plot_all_holdout_curves(foldername, ['S-Bert train', 'Model1'])
# plot_avg_holdout_curves(foldername, ['S-Bert train', 'Model1'])

# plot_hid_PCA_comparison(cog, 'Decision Making', ['', ''])

# plot_hid_traj(cog, ['Anti DM', 'DM'], 2, instruct_mode=None)

ModelS = model_dict['S-Bert train'+seed]

# ModelS.langMod.plot_embedding()

unit = 110
task_variable = 'diff_strength'

# trials = plot_neural_resp(ModelS, 'DM', task_variable, unit, 0)

# trials.intervals

trials = plot_neural_resp(ModelS, 'DM', task_variable, unit, 0)
# trials = plot_neural_resp(ModelS, 'MultiDM', task_variable, unit, 0)
# trials = plot_neural_resp(ModelS, 'Anti MultiDM', task_variable, unit, 0)

# trials.plot_trial(0)

# trials = plot_neural_resp(ModelS, holdout, 'direction', unit, 1, instruct_mode='instruct_swap')


# trials = make_tuning_curve(ModelS, 'DM', task_variable, unit, 115, 1, instruct_mode=None)
# trials = make_tuning_curve(ModelS, 'Anti DM', task_variable, unit, 115, 1, instruct_mode=None)
# trials = make_tuning_curve(ModelS, 'MultiDM', task_variable, unit, 50, 115, instruct_mode=None)
# trials = make_tuning_curve(ModelS, 'Anti MultiDM', task_variable, unit, 115, 1, instruct_mode=None)



#