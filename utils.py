from collections import defaultdict
import Task
task_list = Task.TASK_LIST
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import CogModule

ALL_STYLE_DICT = {'Model1': ('blue', None), 'Model1shuffled': ('blue', '+'), 'SIF':('brown', None), 'BoW': ('orange', None), 'GPT_cat': ('red', '^'), 'GPT train': ('red', '.'), 
                        'BERT_cat': ('green', '^'), 'BERT train': ('green', '+'), 'S-Bert_cat': ('purple', '^'), 'S-Bert train': ('purple', '.'), 'S-Bert' : ('purple', None), 
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

def _collect_data_across_seeds(foldername, model_list, seeds): 
    all_correct = defaultdict(np.array)
    all_loss = defaultdict(np.array)
    all_summary_correct = defaultdict(np.array)
    all_summary_loss = defaultdict(np.array)
    for model_name in model_list: 
        correct_data_dict = {task : np.zeros((100, len(seeds))) for task in task_list}
        correct_summary_dict = {}
        loss_data_dict = {task : np.zeros((100, len(seeds))) for task in task_list}
        loss_summary_dict = {}
        for holdout in task_list: 
            for i, seed_num in enumerate(seeds):
                seed = '_seed'+str(seed_num)
                model_dict = {}
                model_dict[model_name+seed] = None                
                cog = CogModule(model_dict)
                cog.load_training_data(holdout, foldername, seed+'holdout')
                correct_data_dict[holdout][:, i] = cog.task_sorted_correct[model_name+seed][holdout]
                loss_data_dict[holdout][:, i] = cog.task_sorted_loss[model_name+seed][holdout]

            correct_summary_dict[holdout]=np.array([np.mean(correct_data_dict[holdout], axis=1), np.std(correct_data_dict[holdout], axis=1)])
            loss_summary_dict[holdout]=np.array([np.mean(loss_data_dict[holdout], axis=1), np.std(loss_data_dict[holdout], axis=1)])

        all_correct[model_name] = correct_data_dict
        all_loss[model_name] = loss_data_dict
        all_summary_correct[model_name] = correct_summary_dict
        all_summary_loss[model_name] = loss_summary_dict


    return all_correct, all_loss, all_summary_correct, all_summary_loss
