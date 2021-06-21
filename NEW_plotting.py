from numpy.random import seed
from task import Task, construct_batch
task_list = Task.TASK_LIST
from collections import defaultdict
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import colors, cm 


task_group_colors = defaultdict(dict)
task_group_colors['Go'] = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange'}
task_group_colors['Decision Making'] = { 'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow'}
task_group_colors['Comparison'] = { 'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold'}
task_group_colors['Delay'] = { 'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

ALL_STYLE_DICT = {'simpleNet': ('blue', None), 'bowNet': ('orange', None), 'gptNet': ('red', '^'), 'gptNet_layer_11': ('red', '.'), 
                        'bertNet': ('green', '^'), 'bertNet_layer_11': ('green', '.'), 'sbertNet': ('purple', '^'), 'sbertNet_layer_11': ('purple', '.')}

COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'GPT': 'red', 'BERT': 'green', 'S-Bert': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
MODEL_MARKER_DICT = {'SIF':None, 'BoW':None, 'shuffled':'+', 'cat': '^', 'train': '.', 'Transformer':'.'}
MARKER_DICT = {'^': 'task categorization', '.': 'end-to-end', '+':'shuffled'}
NAME_TO_PLOT_DICT = {'Model1': 'One-Hot Task Vec.','Model1shuffled': 'Shuffled One-Hot', 'SIF':'SIF', 'BoW': 'BoW', 'GPT_cat': 'GPT (task cat.)', 'GPT train': 'GPT (end-to-end)', 
                        'BERT_cat': 'BERT (task cat.)', 'BERT train': 'BERT (end-to-end)', 'S-Bert_cat': 'S-BERT (task cat.)', 'S-Bert train': 'S-BERT (end-to-end)',  
                        'S-Bert': 'S-BERT (raw)', 'InferSent train': 'InferSent (end-to-end)', 'InferSent_cat': 'InferSent (task cat.)', 'Transformer': 'Transformer (end-to-end)'}

# import pandas as pd
import pickle
# task_file = 'Go'
# train_data_type = 'correct'
# model_string = 'simpleNet'
# seed = '_seed2'
# model_list = ALL_STYLE_DICT.keys()
# for model in model_list: 
#     df = pd.DataFrame(training_data.values()).T
#     df.columns = training_data.keys()


def plot_single_seed_training(foldername, holdout, model_list, train_data_type, seed, smoothing=0.1):
    seed = '_seed' + str(seed)
    task_file = holdout.replace(' ', '_')
    fig, axn = plt.subplots(4,4, sharey = True, sharex=True, figsize =(14, 10))
    for model_name in model_list: 
        training_data = pickle.load(open(foldername+task_file+'/'+model_name+seed+'_training_'+train_data_type+'dict', 'rb'))
        for i, ax in enumerate(axn.flat):
            ax.set_ylim(-0.05, 1.15)
            task_to_plot = task_list[i]
            
            if task_to_plot == holdout: continue
            smoothed_perf = gaussian_filter1d(training_data[task_to_plot], sigma=smoothing)
            ax.plot(smoothed_perf, color = ALL_STYLE_DICT[model_name][0], marker=ALL_STYLE_DICT[model_name][1], alpha=1, markersize=8, markevery=25)
            ax.set_title(task_to_plot)
    # Patches, Markers = get_model_patches(model_list)
    # _label_plot(fig, Patches, Markers, legend_loc=(1.2, 0.5))
    plt.show()
    
plot_single_seed_training('_ReLU128_14.6/single_holdouts/', 'Go', ALL_STYLE_DICT.keys(), 'correct', 2, smoothing=1)



def task_cmap(array): 
    all_task_dict = {}
    for task_colors in task_group_colors.values(): 
        all_task_dict.update(task_colors)
    color_list = []

    for index in array: 
        color_list.append(all_task_dict[task_list[index]])

    return color_list

def plot_trained_performance(self, model_list):
    barWidth = 0.1
    for i, model in enumerate(model_list):  
        perf_dict = self._get_performance(model, 5)
        values = list(perf_dict.values())
        len_values = len(task_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        plt.bar(r, values, width =barWidth, label = model_list[i])

    plt.ylim(0, 1.15)
    plt.title('Trained Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], task_list)
    plt.legend()
    plt.show()

def plot_model_response(model, task_type, trials = None, trial_num = 0, instruct = None):
    model.eval()
    with torch.no_grad(): 
        if trials == None: 
            task = construct_batch(task_type, 1)

        tar = task.targets
        ins = task.inputs

        if instruct is not None: task_info = [instruct]
        else: task_info = model.get_task_info(ins.shape[0], task_type)
        
        out, hid = model(task_info, ins)

        correct = isCorrect(out, torch.Tensor(tar), task.target_dirs)[trial_num]
        out = out.detach().cpu().numpy()[trial_num, :, :]
        hid = hid.detach().cpu().numpy()[trial_num, :, :]

        try: 
            task_info_embedding = model.langModel(task_info).unsqueeze(1).repeat(1, ins.shape[1], 1)
        except: 
            task_info_embedding = task_info.unsqueeze(1).repeat(1, ins.shape[1], 1)

        fix = ins[trial_num, :, 0:1]            
        mod1 = ins[trial_num, :, 1:1+Task.STIM_DIM]
        mod2 = ins[trial_num, :, 1+Task.STIM_DIM:1+(2*Task.STIM_DIM)]

        to_plot = [fix.T, mod1.squeeze().T, mod2.squeeze().T, task_info_embedding.T, tar[trial_num, :, :].T, out.squeeze().T]
        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 2, 5, 5])
        ylabels = ['fix.', 'mod. 1', 'mod. 2', 'Task Info', 'Target', 'Response']

        fig, axn = plt.subplots(6,1, sharex = True, gridspec_kw=gs_kw)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title(task_type +' trial info; correct: ' + str(correct))
            if i == 5: 
                ax.set_xlabel('time')

        plt.show()



# import pickle
# import itertools
# all_model_list = ['bertNet_layer_11', 'bertNet', 'gptNet_layer_11', 'gptNet', 'getNet_layer_11', 'sbertNet', 'sbertNet_layer_11', 'bowNet', 'simpleNet']
# seed = ['_seed0', '_seed1', '_seed2', '_seed3', '_seed4']

# all_models = list(itertools.product(all_model_list, seed))

# task = 'Anti RT Go'
# holdout_task = task.replace(' ', '_')
# model_string = 'gptNet_seed3' 
# for data_type in ['_training_correct', '_training_loss']:
#     try: 
#         df = pickle.load(open('_ReLU128_14.6/single_holdouts/'+holdout_task+'/'+model_string + data_type, 'rb'))
#         data_dict = dict.fromkeys(list(df.columns))
#         for tasks in list(df.columns): 
#             data_dict[tasks] = [x for x in df[tasks] if not np.isnan(x)]
#         pickle.dump(data_dict, open('_ReLU128_14.6/single_holdouts/'+holdout_task+'/'+model_string + data_type+'dict', 'wb'))
#     except: 
#         pass

# data_dict = pickle.load(open('_ReLU128_14.6/single_holdouts/MultiCOMP1/gptNet_layer_11_seed2_training_correctdict', 'rb'))
# data_dict['Anti Go']
