from collections import defaultdict

from Task import Task
task_list = Task.TASK_LIST

from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

import pandas as pd


ALL_STYLE_DICT = {'Model1': ('blue', None), 'Model1shuffled': ('blue', '+'), 'SIF':('brown', None), 'BoW': ('orange', None), 'GPT_cat': ('red', '^'), 'GPT train': ('red', '.'), 
                        'BERT_cat': ('green', '^'), 'BERT train': ('green', '.'), 'S-Bert_cat': ('purple', '^'), 'S-Bert train': ('purple', '.'), 'S-Bert' : ('purple', None), 
                        'InferSent train': ('yellow', '.'), 'InferSent_cat': ('yellow', '^'), 'Transformer': ('pink', '.')}
COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'GPT': 'red', 'BERT': 'green', 'S-Bert': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
MODEL_MARKER_DICT = {'SIF':None, 'BoW':None, 'shuffled':'+', 'cat': '^', 'train': '.', 'Transformer':'.'}
MARKER_DICT = {'^': 'task categorization', '.': 'end-to-end', '+':'shuffled'}
NAME_TO_PLOT_DICT = {'Model1': 'One-Hot Task Vec.','Model1shuffled': 'Shuffled One-Hot', 'SIF':'SIF', 'BoW': 'BoW', 'GPT_cat': 'GPT (task cat.)', 'GPT train': 'GPT (end-to-end)', 
                        'BERT_cat': 'BERT (task cat.)', 'BERT train': 'BERT (end-to-end)', 'S-Bert_cat': 'S-BERT (task cat.)', 'S-Bert train': 'S-BERT (end-to-end)',  
                        'S-Bert': 'S-BERT (raw)', 'InferSent train': 'InferSent (end-to-end)', 'InferSent_cat': 'InferSent (task cat.)', 'Transformer': 'Transformer (end-to-end)'}


task_group_colors = defaultdict(dict)
task_group_colors['Go'] = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange'}
task_group_colors['Decision Making'] = { 'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow'}
task_group_colors['Comparison'] = { 'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold'}
task_group_colors['Delay'] = { 'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

def task_cmap(array): 
    all_task_dict = {}
    for task_colors in task_group_colors.values(): 
        all_task_dict.update(task_colors)
    color_list = []

    for index in array: 
        color_list.append(all_task_dict[task_list[index]])

    return color_list


def strip_model_name(model_name): 
    try:
        stripped_name = model_name[:model_name.index('_seed')]
    except: 
        stripped_name = model_name
    return stripped_name

def get_model_patches(model_list): 
    Patches = []
    Markers = []
    color_dict = COLOR_DICT.copy()
    for model_name in model_list: 
        architecture_type = list(COLOR_DICT.keys())[np.where([model_name.startswith(key) for key in COLOR_DICT.keys()])[0][0]]
        try:
            color = color_dict.pop(architecture_type)
        except:
            continue
        if architecture_type == 'Model1': architecture_type = 'One-Hot Vec.'
        patch = mpatches.Patch(color=color, label=architecture_type)
        Patches.append(patch)

    for model_name in model_list: 
        print(strip_model_name(model_name))
        if strip_model_name(model_name) in ['Model1', 'BoW', 'SIF', 'S-Bert']: 
            continue
        where_array = np.array([model_name.find(key) for key in MODEL_MARKER_DICT.keys()])
        marker = MODEL_MARKER_DICT[list(MODEL_MARKER_DICT.keys())[np.where(where_array >= 0)[0][0]]]
        if any([marker == m.get_marker() for m in Markers]): 
            continue
        mark = Line2D([0], [0], marker=marker, color='w', label=MARKER_DICT[marker], markerfacecolor='grey', markersize=10)
        Markers.append(mark)

    return Patches, Markers

def _label_plot(fig, Patches, Markers, legend_loc = (0.9, 0.3)): 
    arch_legend = plt.legend(handles=Patches, title = r"$\textbf{Language Module}$", bbox_to_anchor = legend_loc, loc = 'lower center')
    ax = plt.gca().add_artist(arch_legend)
    plt.legend(handles= Markers, title = r"$\textbf{Transformer Fine-Tuning}$", bbox_to_anchor = legend_loc, loc = 'upper center')
    fig.text(0.5, 0.04, 'Training Examples', ha='center')
    fig.text(0.04, 0.5, 'Fraction Correct', va='center', rotation='vertical')



def _convert_data_to_pandas(foldername, model_list, seeds, name, multitask=False): 
    all_correct = defaultdict(np.array)
    all_loss = defaultdict(np.array)

    for model_name in model_list: 
        correct_data_dict = {task : np.full((1000, len(seeds)), np.NaN) for task in task_list}
        loss_data_dict = {task : np.full((1000, len(seeds)), np.NaN) for task in task_list}

        for holdout in task_list: 
            for i, seed_num in enumerate(seeds):
                
                seed = '_seed'+str(seed_num)
                if multitask: holdout_file = 'Multitask'
                else: holdout_file = holdout.replace(' ', '_')

                task_sorted_correct = pickle.load(open(foldername+'/'+holdout_file+'/'+seed+name+'_training_correct_dict', 'rb'))
                task_sorted_loss = pickle.load(open(foldername+'/'+holdout_file+'/'+seed+name+'_training_loss_dict', 'rb'))
                correct_data_dict[holdout][0:len(task_sorted_correct[model_name+seed][holdout]), i] = task_sorted_correct[model_name+seed][holdout]
                loss_data_dict[holdout][0:len(task_sorted_loss[model_name+seed][holdout]), i] = task_sorted_loss[model_name+seed][holdout]
                

                print(model_name+str(holdout)+str(seed))

        all_correct[model_name] = correct_data_dict
        all_loss[model_name] = loss_data_dict

    return pd.DataFrame(all_correct), pd.DataFrame(all_loss)


# seed = '_seed0'

# foldername = '_ReLU128_19.5' 
# modelSBERT_name = 'S-Bert train' 
# modelBERT_name = 'BERT train'
# modelBOW_name = 'BoW'
# model1_name = 'Model1'

# model_list= [modelBOW_name, model1_name, modelBERT_name, modelSBERT_name]
# correct_frame, loss_frame = _convert_data_to_pandas(foldername, model_list, [0, 1, 2, 3], '', True)

# task_sorted_correct = pickle.load(open(foldername+'/Go/_seed4_training_correct_dict', 'rb'))
# task_sorted_correct['S-Bert train_seed4']['Anti RT Go']


# np.mean(correct_frame['S-Bert train']['Anti RT Go'], axis=1)

# from scipy.ndimage.filters import gaussian_filter1d

# correct_data_dict = pd.DataFrame(columns=[0, 1, 2, 3, 4])
# correct_data_dict[0]=np.arange(797)
# correct_data_dict[1]=np.arange(800)





# def plot_training_across_seesds(model_list, smoothing=0.01): 

#     fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(14, 10))
#     plt.suptitle(r'$\textbf{Holdout Learning for All Tasks}$')
#     for i, ax in enumerate(axn.flat):
#         ax.set_ylim(-0.05, 1.15)
#         holdout_task = task_list[i]
#         for model_name in model_list: 
#             smoothed_perf = gaussian_filter1d(np.mean(correct_frame[model_name][holdout_task], axis=1), sigma=smoothing)
#             # ax.fill_between(np.linspace(0, num_trials, num_trials), np.min(np.array([np.ones(num_trials), all_summary_correct[model_name][holdout_task][0]+all_summary_correct[model_name][holdout_task][1]]), axis=0), 
#             #     all_summary_correct[model_name][holdout_task][0]-all_summary_correct[model_name][holdout_task][1], color =  ALL_STYLE_DICT[model_name][0], alpha= 0.1)
#             ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=5, markevery=3)
#         ax.set_title(holdout_task)

#     Patches, Markers = get_model_patches(model_list)
#     _label_plot(fig, Patches, Markers, legend_loc=(1.2, 0.5))
#     plt.show()

# plot_training_across_seesds(model_list, smoothing=1)

# task_sorted_correct = pickle.load(open(foldername+'/RT_Go/_seed1_training_correct_dict', 'rb'))

# task_sorted_correct['BERT train_seed1'][task][-100:-None]

# task_sorted_correct.keys()
# for task in task_list: 
#     all(np.array(task_sorted_correct['BERT train_seed1'][task][-100:])>0.95)
