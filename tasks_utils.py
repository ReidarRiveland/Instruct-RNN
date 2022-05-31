from tasks import TASK_LIST

def invert_task_dict(task_dict):
    inv_swap_dict = {}
    for k, v in task_dict.items():
        for task in v:
            inv_swap_dict[task] = k
    return inv_swap_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

SWAP_LIST = [
            ('RT_Go',  'Anti_Go_Mod2', 'DMS'),

            ('Anti_DelayGo', 'Go_Mod2', 'Anti_DM_Mod1', 'DelayMultiDM', 'COMP2'), 
            ('Anti_DelayDM', 'Go_Mod1', 'Anti_DM_Mod2', 'MultiCOMP1'), 
            ('DM', 'Anti_MultiDM',  'Anti_Go', 'DMC'),             

            ('DM_Mod2', 'Anti_RT_Go', 'DelayGo', 'MultiCOMP2'), 

            ('DelayDM',  'Anti_RT_DM', 'DNMC'), 
            
            ('Anti_DM', 'MultiDM', 'Go', 'DNMS'), 

            ('Anti_DelayMultiDM', 'RT_DM', 'Anti_Go_Mod1', 'DM_Mod1', 'COMP1')
            ]

ALIGNED_LIST = [
            ('DM', 'Anti_DM', 'MultiCOMP1', 'MultiCOMP2'), 
            ('Go', 'Anti_Go', 'COMP1_Mod1', 'COMP1_Mod2'), 
            ('DM_Mod1', 'DM_Mod2', 'COMP2_Mod1', 'COMP2_Mod2'), 
            ('Go_Mod1', 'Go_Mod2', 'ConDM', 'Anti_ConDM'), 
            ('Anti_Go_Mod1', 'Anti_Go_Mod2', 'DelayMultiDM', 'Anti_DelayMultiDM'), 
            ('DelayGo', 'Anti_DelayGo', 'ConMultiDM', 'Anti_ConMultiDM'), 
            ('MultiDM', 'Anti_MultiDM', 'DMS', 'DNMS'), 
            ('RT_DM', 'Anti_RT_DM', 'COMP1', 'COMP2'), 
            ('DelayDM', 'Anti_DelayDM', 'DMC', 'DNMC'), 
            ('RT_Go', 'Anti_RT_Go', 'Anti_DM_Mod1', 'Anti_DM_Mod2')
            ]


SWAPS_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], SWAP_LIST.copy()))
ALIGNED_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], ALIGNED_LIST.copy()))

INV_SWAPS_DICT = invert_task_dict(SWAPS_DICT)

ALIGNED_LIST=['Anti_ConDM', 'ConDM']

def get_swap_task(task):
    swap_label = INV_SWAPS_DICT[task]
    pos = SWAPS_DICT[swap_label].index(task)
    swap_index = (pos+1)%len(SWAPS_DICT[swap_label])
    return SWAPS_DICT[swap_label][swap_index]

task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'goldenrod', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'simpleNetPlus': ('blue', '+'), 'bowNet': ('orange', None), 'gptNet': ('red', None), 'gptNet_tuned': ('red', 'v'), 'bertNet_tuned': ('green', 'v'),
                    'bertNet': ('green', None), 'sbertNet': ('purple', None), 'sbertNet_tuned': ('purple', 'v')}



# import seaborn as sns
# import matplotlib.pyplot as plt
# from tasks import Task
# from model_analysis import get_task_reps, reduce_rep, get_instruct_reps
# import pickle
# import numpy as np 
# import torch
# import itertools
# from tasks_utils import task_colors
# import matplotlib
# import matplotlib.patches as mpatches
# from matplotlib import colors, cm, markers, use 
# from models.full_models import SBERTNet
# from matplotlib.lines import Line2D
# from tasks import TASK_LIST

# cmap = plt.get_cmap('hsv')
# cmap.set_clim=(0, 40)

# EXP_FILE = '5.30models/swap_holdouts'
# sbertNet = SBERTNet()

# holdouts_file = 'swap0'
# sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


# def _rep_scatter(reps_reduced, task, ax, **scatter_kwargs): 
#     task_reps = reps_reduced[TASK_LIST.index(task), ...]
#     ax.scatter(task_reps[:, 0], task_reps[:, 1], s=25, color = cmap(TASK_LIST.index(task)), **scatter_kwargs)
#     patch = Line2D([0], [0], label = task, linestyle='None', markersize=8, **scatter_kwargs)
#     return patch

# def _group_rep_scatter(reps_reduced, task_to_plot, ax, **scatter_kwargs): 
#     Patches = []
#     for task in task_to_plot: 
#         if bool(scatter_kwargs): 
#              _ = _rep_scatter(reps_reduced, task, ax, **scatter_kwargs)
#         else:
#             patch = _rep_scatter(reps_reduced, task, ax, marker='o')
#             Patches.append(patch)
#     return Patches

# def plot_scatter(model, tasks_to_plot, rep_depth='task'): 
#     if rep_depth == 'task': 
#         reps = get_task_reps(model, epoch='stim_start', num_trials = 32)
#     elif rep_depth is not 'task': 
#         reps = get_instruct_reps(model.langModel, depth=rep_depth)
#     reduced, _ = reduce_rep(reps)
#     _, ax = plt.subplots(figsize=(6,6))

#     Patches = _group_rep_scatter(reduced, tasks_to_plot,   ax)
#     Patches.append((Line2D([0], [0], linestyle='None', marker='X', color='grey', label='Contexts', 
#                     markerfacecolor='white', markersize=8)))
#     plt.legend(handles=Patches, fontsize='medium')
#     plt.show()

# plot_scatter(sbertNet, ['Go_Mod1', 'Go_Mod2', 'Anti_Go_Mod1', 'Anti_Go_Mod2'])