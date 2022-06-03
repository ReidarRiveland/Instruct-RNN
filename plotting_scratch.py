
import seaborn as sns
import matplotlib.pyplot as plt
from tasks import Task
from model_analysis import get_task_reps, reduce_rep, get_instruct_reps
import pickle
import numpy as np 
import torch
import itertools
from tasks_utils import task_colors
import matplotlib
import matplotlib.patches as mpatches
from matplotlib import colors, cm, markers, use 
from models.full_models import SBERTNet
from matplotlib.lines import Line2D
from tasks import TASK_LIST

def get_task_color(task, cmap=matplotlib.cm.nipy_spectral):
    norm = matplotlib.colors.Normalize(0, len(TASK_LIST))
    spacer = lambda x : int(np.floor(x/4)+((x%4)*4))
    return cmap(norm(spacer(TASK_LIST.index(task))))


EXP_FILE = '6.4models/swap_holdouts'
sbertNet = SBERTNet()

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


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

plot_scatter(sbertNet, ['DM', 'Anti_DM', 'MultiDM', 'Anti_MultiDM'], dims=2)