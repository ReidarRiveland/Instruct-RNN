from instructRNN.analysis.model_analysis import *
from instructRNN.tasks.tasks import TASK_LIST
from instructRNN.data_loaders.perfDataFrame import PerfDataFrame
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import train_instruct_dict, test_instruct_dict
from instructRNN.tasks.task_factory import STIM_DIM
from instructRNN.instructions.instruct_utils import get_task_info

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import torch
from scipy.stats import sem

from scipy.ndimage import gaussian_filter1d
import warnings

from matplotlib import rc
plt.rcParams["font.family"] = "serif"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


Blue = '#1C3FFD'
lightBlue=	'#89cff0'
lightRed = '#FF6D6A'
Green = "#5A5C53"
Red = '#FF3131'
Orange = '#FFA500'
Yellow = '#FFEE58'
Purple = '#800080'
Grey = '#36454F'
lightGreen = '#90EE90'

MODEL_STYLE_DICT = {'simpleNet': (Blue, None, 'simpleNet'), 'simpleNetPlus': (Blue, None, 'simpleNetPlus'),  'combNet': (lightBlue, None, 'structureNet'), 'combNetPlus': (lightBlue, None, 'structureNetPlus'),
                    'clipNet_lin': ('Pink', None, 'clipNet'), 'clipNet_lin_tuned': (Purple, 'v', 'clipNet (tuned)'), 'clipNet': (Purple, None, 'clipNet'), 
                    'clipNetS_lin': (Purple, None, 'clipNet (S)'), 'clipNetS_lin_tuned': (Purple, None, 'clipNet (S)(tuned)'), 'clipNetS': (Purple, None, 'clipNet (S)'),
                    'bowNet_lin': (Yellow, None, 'bowNet'), 'bowNet': (Yellow, None, 'bowNet'), 'bowNetPlus': (Yellow, None, 'bowNetPlus'), 
                    'gptNet_lin': (lightRed, None, 'gptNet'), 'gptNet_lin_tuned': (lightRed, 'v','gptNet (tuned)'), 'gptNet': (lightRed, 'v','gptNet'),
                    'gptNetXL_lin': (Red, None, 'gptNet (XL)'), 'gptNetXL_lin_tuned': (Red, None, 'gptNet (XL)(tuned)'), 'gptNetXL': (Red, None, 'gptNet (XL)'), 
                    'gptNetXL_L_lin': (Red, 'D', 'gptNetXL (last)'), 
                    'gptNet_L_lin': (Red, 'D', 'gptNet (last)'), 
                    'bertNet_lin': (Orange, None, 'bertNet'), 'bertNet_lin_tuned': (Orange, 'v', 'bertNet (tuned)'), 'bertNet': (Orange, None, 'bertNet'), 'rawBertNet_lin': (Orange, None, 'rawBertNet'),
                    'sbertNetL_lin': (Green, None, 'sbertNet (L)'), 'sbertNetL_lin_tuned': (Green, None, 'sbertNet (L)(tuned)'), 'sbertNetL': (Green, None, 'sbertNet (L)'), 
                    'sbertNet_lin': (lightGreen, 'v', 'sbertNet'), 'sbertNet_lin_tuned': (lightGreen, 'v', 'sbertNet (tuned)'), 'sbertNet': (lightGreen, None, 'sbertNet'),
                    
                    }

def get_task_color(task): 
    index = TASK_LIST.index(task)
    return plt.get_cmap('Paired')(index%8)

def get_all_tasks_markers(task):
    marker_list = ['o', 'v', 'x', '+', 'D']
    group = INV_GROUP_DICT[task]
    group_index = list(TASK_GROUPS.keys()).index(group)
    task_index = TASK_GROUPS[group].index(task)

    marker = marker_list[group_index]
    color = plt.get_cmap('tab10')(group_index)
    return color, marker

def make_all_axes(xlim=None):
    fig, axn = plt.subplots(5,10, sharey = True, sharex=True, figsize =(8, 8))
    for j, task in enumerate(TASK_LIST):
        ax = axn.flat[j]
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(task, size=6, pad=1)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_yticks([0,1])
        ax.set_xticks(np.linspace(0, 3000, 4))
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in [0, 1]])
        if xlim is not None: 
            ax.set_xlim(xlim)
    return fig, axn

def make_avg_axes():
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(5, 4))
    axn.set_ylim(0.0, 1.0)
    axn.set_ylabel('Percent Correct', size=8, fontweight='bold')
    axn.set_xlabel('Exposures to Novel Task', size=8, fontweight='bold')

    axn.xaxis.set_tick_params(labelsize=8)

    axn.yaxis.set_tick_params(labelsize=8)
    axn.yaxis.set_major_locator(MaxNLocator(10)) 
    axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)]) 
    return fig, axn

def _plot_performance_curve(avg_perf, std_perf, plt_ax, color, zero_marker, **plt_args):
        plt_ax.fill_between(np.linspace(0, avg_perf.shape[-1], avg_perf.shape[-1]), np.min(np.array([np.ones(avg_perf.shape[-1]), avg_perf+std_perf]), axis=0), 
                                        avg_perf-std_perf, color = color, alpha= 0.1)
        plt_ax.plot(avg_perf, color=color, **plt_args, zorder=0)

def _plot_all_performance_curves(avg_perf, std_perf, plt_axn, color, zero_marker, **plt_args):
    for j, ax in enumerate(plt_axn.flat):
        _plot_performance_curve(avg_perf[j, :], std_perf[j, :], ax, color, zero_marker, **plt_args)

def plot_curves(foldername, exp_type, model_list, mode = '', training_file = '', avg=False, 
                    perf_type='correct', seeds=range(5), fig_axn=None, zero_marker = None, **curve_kwargs):
    if avg: 
        plt_func = _plot_performance_curve
        axes_func = make_avg_axes
        context_kwargs = {}
    else:
        plt_func = _plot_all_performance_curves
        axes_func = make_all_axes
        context_kwargs = {'axes.grid': False}

    with plt.rc_context(context_kwargs):
        if fig_axn is None: 
            fig, axn = axes_func()
        else: 
            fig, axn = fig_axn


        for model_name in model_list:
            color = MODEL_STYLE_DICT[model_name][0] 
            data = PerfDataFrame(foldername, exp_type, model_name, training_file = training_file, perf_type=perf_type, mode = mode, seeds=seeds)
            if avg: 
                mean, std = data.avg_tasks()
                
                axn.scatter(0, mean[0], color=color, s=5, marker=zero_marker)
            else: 
                mean, std = data.avg_seeds()
            if model_name in ['rawBertNet_lin', 'bowNetPlus', 'simpleNetPlus', 'combNetPlus', 'gptNetXL_L_lin']: linestyle = '--'
            else: linestyle = '-'
            plt_func(mean, std, axn, color, zero_marker, linestyle=linestyle, **curve_kwargs)

        fig.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list], loc=5, title='Models', title_fontsize = 'x-small', fontsize='x-small')   
        return fig, axn

def plot_context_curves(foldername, exp_type, model_list, mode = 'lin_comp', avg=False, 
                    perf_type='correct', seeds=range(5), axn=None, zero_marker = None, **curve_kwargs):
    if axn is None: 
        fig, axn = make_all_axes((-1, 250))

    for model_name in model_list:
        color = MODEL_STYLE_DICT[model_name][0] 
        data = PerfDataFrame(foldername, exp_type, model_name, perf_type=perf_type, mode = mode, seeds=seeds)
        reshaped_data = data.data

        for i in range(10):
            print(i)
            mean = reshaped_data[1, i, ...]
            std = np.zeros_like(mean)
            _plot_all_performance_curves(mean, std, axn, color, zero_marker, **curve_kwargs)


def plot_comp_dots(foldername, exp_type, model_list, mode, split_clauses = False, fig_axn=None, y_lim =(0.0, 1.0), **formatting):
    if fig_axn is None: 
        fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(3, 4))
    else: 
        fig, axn = fig_axn


    axn.set_ylim(y_lim)
    width = 1/(len(model_list)+2)

    for i, model_name in enumerate(model_list):
        color=MODEL_STYLE_DICT[model_name][0]
        data  = PerfDataFrame(foldername, exp_type, model_name, mode=mode)
        zero_shot = data.data[..., 0].mean(-1)

        x_mark = width+((i*1.05*width))
        for k, seed_point in enumerate(zero_shot):
            axn.scatter(x_mark+(k*0.01), seed_point, color=color, edgecolor='white', s=20.0, linewidth=0.5)

    axn.set_xticklabels('')
    axn.xaxis.set_ticks_position('none') 

    if not 'ccgp' in mode: 
        axn.set_ylabel('Percent Correct', size=8, fontweight='bold')
        axn.yaxis.set_tick_params(labelsize=8)
        axn.yaxis.set_major_locator(MaxNLocator(10)) 
        axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)]) 

    return fig, axn

def plot_all_models_task_dist(foldername, exp_type, model_list, k= 0, perf_type='correct', mode='', seeds=range(5), **kwargs): 
    fig, axn = plt.subplots(3, 3, sharey = True, sharex=True, figsize =(5, 4))
    fig.suptitle('Distribution of Performance on Novel Tasks')

    thresholds = np.linspace(0.1, 1.0, 10)
    thresholds0 = np.linspace(0.0, 0.9, 10)

    for i, ax in enumerate(axn.flatten()):
        try:
            model_name = model_list[i]
            data = PerfDataFrame(foldername, exp_type, model_name, perf_type=perf_type, mode=mode, seeds=seeds)
            perf = data.data[:, :, k]
            bins = np.zeros((len(range(5)), 10))
            for iy, ix in np.ndindex(perf.shape):
                bins[iy, int(np.floor((perf[iy, ix]*10)-1e-5))]+=1
            mean_bins = np.mean(bins, axis=0)
            std_bins = np.std(bins, axis=0)
            ax.set_title(MODEL_STYLE_DICT[model_name][2], fontsize=8)
            ax.set_ylim(0, 35)
            ax.set_yticks([0, 10, 20, 30])
            ax.set_yticklabels(['0', '10', '20', '30'])

            if i == 3: 
                ax.set_ylabel('Number of Tasks')
            if i > 3: 
                ax.set_xticks(np.arange(0, 10)+0.5)
                ax.set_xticklabels([f'{x:.0%}-{y:.0%}' for x,y in list(zip(thresholds0, thresholds))][::-1], fontsize=5, rotation='45', ha='right') 


            ax.bar(range(0,10), mean_bins[::-1], 1.0, color=MODEL_STYLE_DICT[model_name][0], align='edge', alpha=0.8, **kwargs)
        except IndexError: 
            ax.remove()

    return fig, axn

def plot_all_task_lolli_v(foldername, exp_type, model_list, marker = 'o', mode='', perf_type='correct',  seeds=range(5), **kwargs):
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(11, 4))

    width = 1/(len(model_list)+1)
    ind = np.arange(len(TASK_LIST))

    axn.set_xticks(ind)
    axn.set_xticklabels('')
    axn.tick_params(axis='x', which='minor', bottom=False)
    axn.set_xticks(ind+0.75, minor=True)
    axn.set_xticklabels(TASK_LIST, fontsize=6, minor=True, rotation=45, ha='right', fontweight='bold') 
    axn.set_xlim(-0.15, len(ind))

    axn.set_yticks(np.linspace(0, 1, 11))
    axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)], fontsize=8)
    axn.set_ylim(0.0, 1.01)
    axn.set_ylabel('Percent Correct', size=8, fontweight='bold')

    for i, model_name in enumerate(model_list):  
        color = MODEL_STYLE_DICT[model_name][0]     
        data = PerfDataFrame(foldername, exp_type, model_name, perf_type=perf_type, mode = mode, seeds=seeds)
        zero_shot, std = data.avg_seeds(k_shot=0)


        x_mark = (ind+(width/2))+(i*width)
        axn.scatter(x_mark,  zero_shot, color=color, s=3, marker=marker)
        if model_name in ['rawBertNet_lin', 'bowNetPlus', 'simpleNetPlus', 'combNetPlus']: linestyle = '--'
        else: linestyle = '-'
        axn.vlines(x_mark, ymin=zero_shot-std, ymax=np.minimum(np.ones_like(zero_shot), zero_shot+std), color=color, linewidth=0.8, linestyle=linestyle, **kwargs)
        axn.vlines(x_mark, ymin=0, ymax=zero_shot, color=color, linewidth=0.5, linestyle=linestyle, alpha=0.5, **kwargs)


        axn.axhline(np.mean(zero_shot), color=color, linewidth=1.0, alpha=0.8, zorder=0, linestyle=linestyle)

    fig.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list], loc=5, title='Models', title_fontsize = 'x-small', fontsize='x-small')        

    plt.tight_layout()
    return fig, axn

def plot_significance(t_mat, p_mat, model_list=None, fig_size = (6, 4), labelsize = 4, xticklabels=None, yticklabels=None, **heatmap_kwargs): 
    labels = []
    for p in p_mat.flatten(): 
        if p < 0.001: 
            labels.append('p<0.001')
        elif p == 1: 
            labels.append('')
        else: 
            labels.append('p={:.3f}'.format(p))

    if model_list is not None: 
        model_names = [MODEL_STYLE_DICT[model_name][2] for model_name in model_list]
        yticklabels = model_names
        xticklabels = model_names

    try:
        mask = np.zeros_like(t_mat)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = False
        t_mat = np.where(mask, np.full_like(t_mat, np.nan), t_mat)
        labels = np.array(labels).reshape((len(xticklabels), len(yticklabels)))

    except ValueError: 
        mask = None
        labels = np.array(labels)[None, :]

    with sns.plotting_context(rc={ 'xtick.labelsize': labelsize,'ytick.labelsize': labelsize}):
        fig, axn = plt.subplots(1, 1, figsize =fig_size)
        sns.heatmap(t_mat, linecolor='white', linewidths=1, cbar=True, fmt='', 
                    cbar_kws={'label': 't-value'}, mask=mask, ax=axn,
                    xticklabels=xticklabels, yticklabels=yticklabels, 
                    annot=labels,annot_kws={'fontweight': 'bold', 'fontsize':'6'}, 
                    **heatmap_kwargs)


def _rep_scatter(reps_reduced, task, ax, dims, pcs, **scatter_kwargs): 
    task_reps = reps_reduced
    if dims ==2: 
        ax.scatter(task_reps[:, 0], task_reps[:, 1], s=5, **scatter_kwargs)
    else: 
        ax.scatter(task_reps[:, 0], task_reps[:, 1], task_reps[:,2], **scatter_kwargs)

def _group_rep_scatter(reps_reduced, task_to_plot, ax, dims, pcs, **scatter_kwargs): 
    Patches = []
    for i, task in enumerate(task_to_plot): 
        if task_to_plot == TASK_LIST:
            task_color, marker = get_all_tasks_markers(task)
            marker = 'o'
        else: 
            task_color = get_task_color(task)
            marker = 'o'
        _rep_scatter(reps_reduced[TASK_LIST.index(task), ...], task, ax, dims, pcs, marker=marker, c = [task_color]*reps_reduced.shape[1], **scatter_kwargs)
        Patches.append(Line2D([0], [0], label = task, color= task_color, marker = marker, markersize=5, linestyle='None'))
    return Patches

def plot_scatter(model, tasks_to_plot, rep_depth='task', dims=2, num_trials = 50, epoch= 'stim_start', instruct_mode = 'combined', **scatter_kwargs): 
    with plt.style.context('ggplot'):
        pcs = range(dims)

        if rep_depth == 'task': 
            reps = get_task_reps(model, epoch=epoch, num_trials = num_trials, main_var=True, instruct_mode=instruct_mode)
        if rep_depth == 'rule': 
            assert model.model_name in ['combNet', 'simpleNet']
            reps = get_rule_reps(model)
        elif rep_depth != 'task': 
            reps = get_instruct_reps(model.langModel, depth=rep_depth, instruct_mode=instruct_mode)
        reduced, _ = reduce_rep(reps, pcs=pcs)

        fig = plt.figure(figsize=(14, 14))
        if dims==2:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')

        Patches = _group_rep_scatter(reduced, tasks_to_plot, ax, dims, pcs, **scatter_kwargs)

        ax.set_xlabel('PC '+str(pcs[0]))
        ax.set_xticklabels([])
        ax.set_ylabel('PC '+str(pcs[1]))
        ax.set_yticklabels([])
        if dims==3: 
            ax.set_zlabel('PC '+str(pcs[2]))
            ax.set_zticklabels([])
        plt.legend(handles=Patches, fontsize=5)
        plt.show()


def get_clause_diff(data, clause_indices=None, wo_clause_indices=None):
    if clause_indices is None:     
        clause_indices = [TASK_LIST.index(task) for task in COND_CLAUSE_LIST]
    
    if wo_clause_indices is None:
        wo_clause_indices = [TASK_LIST.index(task) for task in NONCOND_CLAUSE_LIST]

    clause_zero_shot = data.data[:,clause_indices, 0]
    wo_clause_zero_shot = data.data[:, wo_clause_indices, 0]
    zero_shot = clause_zero_shot.mean(-1)-wo_clause_zero_shot.mean(-1)
    return zero_shot, clause_zero_shot, wo_clause_zero_shot

def plot_clauses_dots(foldername, exp_type, model_list, mode='combined', fig_axn=None, y_lim =(0.0, 1.0), **formatting):
    t_value_arr = np.empty((len(model_list), len(model_list)))
    p_value_arr = np.empty((len(model_list), len(model_list)))

    if fig_axn is None: 
        fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(3, 4))
    else: 
        fig, axn = fig_axn

    axn.set_ylabel('Perforamance', size=8, fontweight='bold')
    axn.set_ylim(y_lim)
    width = 1/(len(model_list)+2)
    
    clause_arr = np.empty((len(model_list), 5, 30))
    wo_clause_arr = np.empty((len(model_list), 5, 20))

    within_p = []
    within_t = []

    for i, model_name in enumerate(model_list):
        color=MODEL_STYLE_DICT[model_name][0]
        data  = PerfDataFrame(foldername, exp_type, model_name, mode=mode)
        zero_shot, clause_zero_shot, wo_clause_zero_shot = get_clause_diff(data)
        clause_arr[i, ... ] = clause_zero_shot
        wo_clause_arr[i, ... ] = wo_clause_zero_shot
        
        x_mark = ((i*1.05*width))
        for k, seed_point in enumerate(zero_shot):
            axn.scatter(x_mark+(k*0.01), seed_point, color=color, s=20, edgecolor='white', linewidth=0.5)

        dummy_perfs = []
        for _ in range(50): 
            clause_indices = np.random.choice(list(range(50)), size = 30, replace=False)
            wo_clause_indices = [x for x in list(range(50)) if x not in clause_indices]
            dummy_perf, _, _ = get_clause_diff(data, clause_indices=clause_indices, wo_clause_indices=wo_clause_indices)
            dummy_perfs+=list(dummy_perf)
            axn.scatter(x_mark+(2*0.01), dummy_perf.mean(), color='white', edgecolor=color, s=3, alpha=0.25)

        t_stat, p_value =ttest_ind(dummy_perfs, zero_shot, equal_var=False)
        within_t.append(t_stat)
        within_p.append(p_value)
    

    clause_diff = clause_arr.mean(-1)-wo_clause_arr.mean(-1)
    for i, diff1 in enumerate(clause_diff): 
        for j, diff2 in enumerate(clause_diff): 
            t_stat, p_value = ttest_ind(diff1, diff2, equal_var=False)
        
            t_value_arr[i, j] = t_stat
            p_value_arr[i,j] = p_value

    axn.set_xticklabels('')
    axn.xaxis.set_ticks_position('none') 
    return fig, axn, (np.array(within_t), np.array(within_p)), (t_value_arr, p_value_arr)


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

def plot_tuning_curve(model, tasks, unit, times, num_trials=50, num_repeats=20, smoothing = 1e-7, max_coh=0.3, min_coh=0.05, **trial_kwargs): 
    fig, axn = plt.subplots(1,1, sharey = True, sharex=True, figsize =(6, 6))

    if 'Go' in tasks[0]:
        x_label = "direction"
        var_of_interest = np.linspace(0, np.pi*2, num_trials)
        axn.set_xlim(0.0, 2*np.pi)
        axn.set_xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
        axn.set_xticklabels(["0", r"$\frac{\pi}{2}$", "$\pi$", r"$\frac{3\pi}{2}$", "$2\pi$"]) 
    elif 'DMS' in tasks: 
        x_label = "diff direction"

        var_of_interest = np.linspace(-np.pi/2, (3/2)*np.pi, num_trials) 
        axn.set_xlim(-np.pi/2, (3/2)*np.pi)
        axn.set_xticks([-np.pi/2, 0, 0.5*np.pi, np.pi, (3/2)*np.pi,])
        axn.set_xticklabels([r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$-\frac{\pi}{2}$"]) 

    elif 'DM' in tasks[0] or 'COMP' in tasks[0]: 
        x_label = "coherence"

        var_of_interest = np.concatenate((np.linspace(-max_coh, -min_coh, num=int(np.ceil(num_trials/2))), 
                np.linspace(min_coh, max_coh, num=int(np.floor(num_trials/2)))))

    y_max=0.0
    hid_mean = get_task_reps(model, epoch=None, num_trials=num_trials, tasks=tasks, num_repeats=num_repeats, main_var=True, **trial_kwargs)
    for i, task in enumerate(tasks): 
        time = times[i]
        neural_resp = hid_mean[i, :, time, unit]        
        axn.plot(var_of_interest, gaussian_filter1d(neural_resp, smoothing), color=get_task_color(task))
        if np.max(neural_resp)>y_max: y_max = np.max(neural_resp)

    plt.suptitle('Tuning curve for Unit ' + str(unit))
    axn.set_ylim(-0.05, y_max+0.05)

    axn.set_ylabel('Unit Activity', size=8, fontweight='bold')
    axn.set_xlabel(x_label, size=8, fontweight='bold')

    Patches = [mpatches.Patch(color=get_task_color(task), label=task) for task in tasks]
    plt.legend(handles=Patches)
    plt.show()


def plot_neural_resp(model, task, task_variable, unit, num_trials=25, num_repeats = 10, cmap=sns.color_palette("inferno", as_cmap=True), **trial_kwargs):
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']
    hid_mean = get_task_reps(model, epoch=None, num_trials=num_trials, tasks=[task], num_repeats=num_repeats, main_var=True, **trial_kwargs)[0,...]

    if task_variable == 'direction' or task_variable=='diff_direction': 
        labels = ["0", "$2\pi$"]
        cmap = plt.get_cmap('twilight') 
    elif task_variable == 'strength':
        labels = ["0.3", "1.8"]
        cmap = plt.get_cmap('plasma') 
    elif task_variable =='diff_strength': 
        labels = [r"$\Delta$ -0.3", r"$\Delta$ 0.3"]
        cmap = plt.get_cmap('seismic') 

    mappable = cm.ScalarMappable(cmap=cmap)

    fig, axn = plt.subplots()
    ylim = np.max(hid_mean[..., unit])
    for i in range(hid_mean.shape[0]):
        axn.plot(hid_mean[i, :, unit], c = cmap(i/hid_mean.shape[0]))


    plt.xticks([30, 130], labels=['Stim. Onset', 'Response'])
    plt.vlines(130, -1.5, ylim+0.15, colors='k', linestyles='dashed')
    plt.vlines(30, -1.5, ylim+0.15, colors='k', linestyles='dashed')

    axn.set_ylim(0, ylim+0.025)
    cbar = plt.colorbar(mappable, orientation='vertical', label = task_variable.replace('_', ' '), ticks = [0, hid_mean.shape[0]])
    plt.suptitle(task + ' response for Unit' + str(unit))
    cbar.set_ticklabels(labels)

    plt.show()
    return axn

def plot_ccgp_corr(folder, exp_type, model_list):
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(4, 4))
    corr, p_val, ccgp, perf = get_perf_ccgp_corr(folder, exp_type, model_list)
    a, b = np.polyfit(perf.flatten(), ccgp.flatten(), 1)
    print(p_val)
    fig.suptitle('CCGP Correlates with Generalization')
    axn.set_ylim(0.475, 1)
    axn.set_ylabel('Holdout Task CCGP', size=8, fontweight='bold')
    axn.set_xlabel('Zero-Shot Performance', size=8, fontweight='bold')

    axn.spines['top'].set_visible(False)
    axn.spines['right'].set_visible(False)
    axn.set_axisbelow(True)
    axn.grid(visible=True, color='grey', linewidth=0.5)


    for i, model_name in enumerate(model_list):
        axn.scatter(perf[i, :], ccgp[i, :], marker='.', c=MODEL_STYLE_DICT[model_name][0])
    x = np.linspace(0, 1, 100)
    axn.text(0.01, 0.95, "$r^2=$"+str(round(corr, 3)), fontsize=7)
    axn.text(0.01, 0.93, "p<.001", fontsize=5)

    axn.plot(x, a*x+b, linewidth=0.8, linestyle='dotted', color='black')
    plt.show()

def plot_layer_ccgp(foldername,exp_type, model_list, fig_axn=None, seeds=range(5), ylim = (0.495, 1.05), **plt_kwargs): 
    if fig_axn is None: 
        fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))
        fig.suptitle('CCGP Across Model Hierarchy')
        axn.set_ylim(0.475, 1)
        axn.set_ylabel('Holdout Task CCGP', size=8, fontweight='bold')
        axn.set_xlabel('Model Layer', size=8, fontweight='bold')
    else: 
        fig, axn = fig_axn

    patches = []
    for model_name in model_list:
        color = MODEL_STYLE_DICT[model_name][0]
        holdout_ccgp = PerfDataFrame(foldername, exp_type, model_name, mode='layer_ccgp', seeds=seeds)
        axn.plot(range(14-len(holdout_ccgp.layer_list), 14), np.nanmean(holdout_ccgp.data, axis=(0,1)), marker='.', c=color, linewidth=0.8, **plt_kwargs)
        patches.append(Line2D([0], [0], label = MODEL_STYLE_DICT[model_name][2], color= color, marker = 'o', linestyle = 'None', markersize=4))

    axn.legend(handles = patches, fontsize='x-small')
    axn.set_xticklabels([str(x) for x in range(1, 13)] + ['embed', 'task']) 
    axn.set_ylim(*ylim)
    axn.set_xticks(range(14))
    return fig, axn

def plot_task_var_heatmap(load_folder, model_name, seed, cmap = sns.color_palette("inferno", as_cmap=True), cluster_info=None):
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(12, 4))

    if cluster_info == None:
        task_var, cluters_dict, cluster_labels, sorted_indices = get_cluster_info(load_folder, model_name, seed)
    else: 
        task_var, cluters_dict, cluster_labels, sorted_indices = cluster_info
        
    res = sns.heatmap(task_var[sorted_indices, :].T, xticklabels = cluster_labels, yticklabels=TASK_LIST, vmin=0, cmap=cmap, ax=axn)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 5)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 4, rotation=0)
    axn.set_ylabel('Tasks', size=8, fontweight='bold')
    axn.set_xlabel('Clustered Task Variance', size=8, fontweight='bold')
    plt.suptitle('Functional Clustering')
    plt.tight_layout()
    plt.show()
    return task_var, cluters_dict, cluster_labels, sorted_indices
        

def plot_decoding_confuse_mat(confusion_mat, cmap='Blues', cos_sim=False, **heatmap_args): 
    if cos_sim: 
        res=sns.heatmap(confusion_mat, xticklabels=TASK_LIST, yticklabels=TASK_LIST, cmap=cmap, **heatmap_args)
    else: 
        res=sns.heatmap(confusion_mat, mask=confusion_mat == 0, 
                            xticklabels=TASK_LIST+['novel'], yticklabels=TASK_LIST, cmap=cmap, **heatmap_args)
             
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 4, rotation=45, ha='right')
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 5)
    plt.show()

def _plot_partner_perf(foldername, model_name, axn, sm_holdout, decoder_holdout, decode_embeddings=False, **scatter_kwargs):
    mode_dict = {'All Instructions': ('all_perf', '#0392cf'), 'Novel Instructions': ('other_perf', '#7bc043'), 'Embeddings': ('context_perf','#edc951')}
    multi_holdout_formatting = {'multi': { 'edgecolor':'white'}, 'holdout': {'edgecolor':'white', 'marker':'d'}}

    if sm_holdout: 
        sm_str = 'holdout'
        folder = foldername+'/swap_holdouts'
    else: 
        sm_str = 'multi'
        folder = foldername+'/multitask_holdouts'

    if decoder_holdout: 
        decoder_str = 'holdout'
    else: 
        decoder_str = 'multi'

    if decode_embeddings: dec_emd_str = 'from_embeddings'
    else: dec_emd_str = ''
    
    axn.set_ylabel('Perforamance', size=8, fontweight='bold')
    width = 1/4
    means_list = []
    perf_list = []
    for i, mode_value in enumerate(mode_dict.values()):
        for j, values in enumerate(multi_holdout_formatting.items()):
            holdouts, formatting = values
            perf_data = np.load(folder+'/decoder_perf/'+model_name+'/test_sm_'+sm_str+'_decoder_'+decoder_str+'_partner_'+holdouts+'_'+mode_value[0]+dec_emd_str+'.npy')
            perf_list.append(perf_data.flatten())

            zero_shot = np.nanmean(perf_data)
            means_list.append(zero_shot)
            std = sem(perf_data.flatten(), nan_policy='omit')

            x_mark = ((i)+width)+((j*1.05*width))

            for k, seed_point in enumerate(np.nanmean(perf_data, axis=(1, 2))):
                axn.scatter(x_mark+(k*0.03), seed_point, color=mode_value[1], linewidth=0.5, **formatting, **scatter_kwargs)
                        
    axn.set_yticks(np.linspace(0, 1, 11))            
    axn.set_yticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 11)], fontsize=5) 
    axn.set_xticklabels('') 
    axn.xaxis.set_ticks_position('none') 
    patches = []
    for label, mode in mode_dict.items():
        patches.append(mpatches.Patch(label = label, facecolor= mode[1]))
    patches.append(Line2D([0], [0], label = 'Multitask Partner', color= 'gray', marker = 'o', markersize=5, linestyle='None'))
    patches.append(Line2D([0], [0], label = 'Holdout Partner', color= 'gray', marker = 'd', markersize=5, linestyle='None'))

    axn.set_ylim(0.0, 1.0)
    axn.set_xlim(0, 3)

    return patches, means_list, perf_list

def plot_partner_perf(foldername, model_name, decode_embeddings=False, figsize=(4,4), **scatter_kwargs):
    fig, axn = plt.subplots(2, 2, sharex=True, figsize =figsize)
    means_list = []
    perf_list = [] 

    patches, means, _perf_list = _plot_partner_perf(foldername, model_name, axn.flatten()[0], False, False, decode_embeddings=decode_embeddings, **scatter_kwargs)
    means_list.append(means)
    perf_list += _perf_list

    if not decode_embeddings: 
        axn.flatten()[1].axis('off')
        _, means, _perf_list = _plot_partner_perf(foldername, model_name, axn.flatten()[2], True, False, **scatter_kwargs)
        means_list.append(means)
        perf_list += _perf_list

        _, means, _perf_list = _plot_partner_perf(foldername, model_name, axn.flatten()[3], True, True, **scatter_kwargs)
        means_list.append(means)
        perf_list += _perf_list

    axn[0,0].legend(handles = patches, fontsize='x-small')

    t_value_arr = np.empty((18, 18))
    p_value_arr = np.empty((18, 18))
    for i, perf1 in enumerate(perf_list): 
        for j, perf2 in enumerate(perf_list):
            t_stat, p_value = ttest_ind(perf1, perf2, equal_var=False, nan_policy='omit')
        
            t_value_arr[i, j] = t_stat
            p_value_arr[i,j] = p_value

    plt.show()
    return means_list, (t_value_arr, p_value_arr)

def plot_model_response(out, hid, ins, tar, task_type, plotting_index = 0, cmap = 'Reds'):
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
        sns.heatmap(to_plot[i], yticklabels = False, cmap = cmap, ax=ax, cbar=i == 0, vmin=0, vmax=1.3, cbar_ax=None if i else cbar_ax)
        ax.set_ylabel(ylabels[i], fontweight = 'bold', fontsize=6)
        if i == 0: 
            ax.set_title(task_type +' trial info')
        if i == 5: 
            ax.set_xlabel('time')
            ax.xaxis.set_ticks(np.arange(0, 150, 5))
            ax.set_xticklabels(np.arange(0, 150, 5), fontsize=16)

            ax.tick_params(axis='x', labelsize= 6)


    plt.show()


def plot_trial(trials, index, cmap="Reds"):
    ins = trials.inputs
    tars = trials.targets
    fix = ins[index, :, 0:1]
    mod1 = ins[index, :, 1:task_factory.STIM_DIM+1]
    mod2 = ins[index, :, 1+task_factory.STIM_DIM:1+(2*task_factory.STIM_DIM)]
    tars = tars[index, :, :]

    to_plot = (fix.T, mod1.T, mod2.T, tars.T)

    gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 5])

    fig, axn = plt.subplots(4,1, sharex = True, gridspec_kw=gs_kw, figsize=(5, 3))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ylabels = ('fix.', 'mod. 1', 'mod. 2', 'Target')
    for i, ax in enumerate(axn.flat):
        sns.heatmap(to_plot[i], yticklabels = False, cmap = cmap, ax=ax, cbar=i == 0, vmin=0, vmax=1.8, cbar_ax=None if i else cbar_ax)

        ax.set_ylabel(ylabels[i])
        if i == 0: 
            ax.set_title('%r Trial Info' %trials.task_type)
        if i == 3: 
            ax.set_xlabel('time')
            ax.set_xticks(np.linspace(0, 150, 16))
            ax.set_xticklabels(np.linspace(0, 150, 16).astype(int), size=5)


    plt.show()
