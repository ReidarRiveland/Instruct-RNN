from scipy.spatial.distance import jaccard, dice
from torch._C import device
from nlp_models import SBERT
from rnn_models import InstructNet, SimpleNet
from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_sim_scores
import numpy as np
import torch.optim as optim
from utils import sort_vocab, train_instruct_dict, task_swaps_map, task_list
from task import DM

from matplotlib.pyplot import axis, bar, get, ylim
from numpy.lib import utils
import torch
import torch.nn as nn


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



from sklearn import svm, metrics
from scipy.spatial.distance import dice
import torch

from task import Task


#sbertNet_layer_11
# model = InstructNet(SBERT(20, train_layers=[]), 128, 1)
# model.model_name += '_tuned'

# model.set_seed(0)
# swapped = 'Anti Go'
# task_file = task_swaps_map[swapped]
# model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
#lang_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
# reps, perf_array = get_task_reps(model, num_trials=200, epoch='stim_start', swapped_tasks=[swapped])

# reps[task_list.index(swapped), ...] = reps[-1, ...]
# perf_array[task_list.index(swapped), ...] = perf_array[-1]

def get_CCGP(reps, performance_array): 
    num_trials = reps.shape[1]
    dim = reps.shape[-1]
    all_decoding_score = np.zeros((16, 2))
    all_dice_score = np.zeros((16, 2))
    dichotomies = np.array([[[0, 1], [2, 3]], [[0,2], [1, 3]]])
    for i in range(4): 
        conditions=dichotomies+(4*i)
        for j in [0, 1]: 
            for k in [0, 1]: 

                print('\n')
                print('train condition ' +str(conditions[j][k]))
                test_condition = conditions[j][(k+1)%2]
                print('test condition' + str(test_condition))
                print('\n')

                classifier = svm.LinearSVC(max_iter=5000)
                classifier.classes_=[-1, 1]
                classifier.fit(reps[conditions[j][k], ...].reshape(-1, dim), np.array([0]*num_trials+[1]*num_trials))
                for index in [0, 1]: 
                    print('Task :' + str(test_condition[index]))
                    decoding_corrects = np.array([index]*num_trials) == classifier.predict(reps[test_condition[index], ...].reshape(-1, dim))
                    #print('Decoding Corrects :' + str(decoding_corrects))
                    #print('Perf :' + str(performance_array[test_condition[index]]))
                    decoding_score = np.mean(decoding_corrects)
                    #print(task_list[test_condition[index]] + ' ' + str(decoding_score))
                    jaccard_score = 1 - dice(performance_array[test_condition[index], ...].flatten(), decoding_corrects)
                    print('J-Score :' +str(jaccard_score))
                    all_decoding_score[test_condition[index], j] = decoding_score
                    all_dice_score[test_condition[index], j] = jaccard_score
            
                # print('Task :' + str(test_condition))
                # decoding_corrects = np.array([0]*num_trials+[1]*num_trials) == classifier.predict(reps[test_condition, ...].reshape(-1, dim))
                # print('Decoding Corrects :' + str(decoding_corrects))
                # print('Perf :' + str(perf_array[test_condition]))
                # jaccard_score = metrics.f1_score(performance_array[test_condition, ...].flatten(), decoding_corrects)
                # print('J-Score :' +str(jaccard_score))
                # all_decoding_score[test_condition[0], j] = np.mean(decoding_corrects[0:num_trials])
                # all_decoding_score[test_condition[1], j] = np.mean(decoding_corrects[num_trials:])
                # all_jaccard_score[test_condition[0], j] = jaccard_score
                # all_jaccard_score[test_condition[1], j] = jaccard_score
            
    return all_decoding_score, all_dice_score


def get_all_CCGP(model, task_rep_type, swap=False): 
    all_CCGP = np.empty((5, 16, 16, 2))
    all_dice = np.empty((5, 16, 16, 2))
    holdout_CCGP = np.empty((5, 16, 2))
    holdout_dice = np.empty((5, 16, 2))
    epoch = 'stim_start'
    for i in range(5):
        model.set_seed(i)
        for j, task in enumerate(task_list):
            print('\n') 
            print(task) 
            task_file = task_swaps_map[task]
            model.load_model('_ReLU128_5.7/swap_holdouts/'+task_file)
            if swap: 
                swapped_list = [task]
                swap_str = '_swap'
            else: 
                swapped_list = []
                swap_str = ''

            if task_rep_type == 'task': 
                reps, perf_array = get_task_reps(model, num_trials=256, epoch=epoch, stim_start_buffer=0, swapped_tasks=swapped_list)
            if task_rep_type == 'lang': 
                reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer', swapped_tasks=swapped_list)
                perf_array = np.zeros((16, 15))
            
            if swap:
                reps[task_list.index(task), ...] = reps[-1, ...]
                perf_array[task_list.index(task), ...] = perf_array[-1]

            decoding_score, dice_scores = get_CCGP(reps, perf_array)
            all_CCGP[i, j, ...] = decoding_score
            all_dice[i, j, ...] = dice_scores
            holdout_CCGP[i, j] = decoding_score[j, :]
            holdout_dice[i, j] = dice_scores[j, :]
    np.savez('_ReLU128_5.7/' + epoch + '_' + model.model_name + swap_str +'_CCGP_scores', all_CCGP=all_CCGP, all_dice = all_dice, holdout_CCGP= holdout_CCGP, holdout_dice= holdout_dice)
    return all_CCGP, all_dice, holdout_CCGP, holdout_dice


from trainer import ALL_MODEL_PARAMS, config_model_training
import torch

for swap_mode in [False]: 
    for model_params in ALL_MODEL_PARAMS.keys(): 
        model, _, _, _ = config_model_training(model_params)
        model.to(torch.device(0))
        print(model.model_name)
        get_all_CCGP(model, 'task', swap=swap_mode)



from plotting import MODEL_STYLE_DICT, task_list, Line2D, mpatches, all_models, plt

def plot_CCGP_scores(model_list, rep_type_file_str = '', save_file=None):
    keys_list = ['all_CCGP', 'holdout_CCGP']
    barWidth = 0.08
    Patches = []
    for i, model_name in enumerate(model_list):
        if '_tuned' in model_name: marker_shape = MODEL_STYLE_DICT[model_name][1]
        else: marker_shape='s'
        Patches.append(Line2D([0], [0], linestyle='None', marker=marker_shape, color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8))
        len_values = len(keys_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        if '_tuned' in model_name: 
            mark_size = 4
        else: 
            mark_size = 3
    
        for j, swap_mode in enumerate(['', '_swap']):
            values = np.empty((2,))
            spread_values = np.empty((len_values, 5))

            CCGP_score = np.load(open('_ReLU128_5.7/CCGP_measures/'+rep_type_file_str+model_name+swap_mode+'_CCGP_scores.npz', 'rb'))
            for k, key in enumerate(keys_list):  
                if k == 0 and swap_mode == '_swap': 
                    continue
                else:
                    values[k] = np.mean(np.nan_to_num(CCGP_score[key]))
                    if len(CCGP_score[key].shape)>3:
                        spread_values[k, :] = np.mean(np.nan_to_num(CCGP_score[key]), axis=(1,2,3))
                    else: 
                        spread_values[k, :] = np.mean(np.nan_to_num(CCGP_score[key]), axis=(1,2))

                    markers, caps, bars = plt.errorbar(r[k], values[k], yerr = np.std(spread_values[k, :]), elinewidth = 0.5, capsize=1.0, marker=marker_shape, linestyle="", mfc = [None, 'white'][j], alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
            [bar.set_alpha(0.2) for bar in bars]

    plt.ylim(0, 1.05)
    plt.title('CCGP Performance')
    plt.xlabel('Task Type', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth +0.15 for r in range(len_values)], keys_list)
    plt.yticks(np.linspace(0, 1, 11), size=8)

    plt.tight_layout()

    plt.legend(handles=Patches, fontsize=6, markerscale=0.5)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()


plot_CCGP_scores(all_models, 'stim_start')






# def plot_task_CCGP_scores(model_list, score_type,  rep_type_file_str = ''):
#     barWidth = 1/len(model_list)
#     Patches = []
#     for i, model_name in enumerate(model_list):
#         if '_tuned' in model_name: marker_shape = MODEL_STYLE_DICT[model_name][1]
#         else: marker_shape='s'
#         Patches.append(Line2D([0], [0], linestyle='None', marker=marker_shape, color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
#                 markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8))
#         len_values = len(task_list)
#         if i == 0:
#             r = np.arange(len_values)
#         else:
#             r = [x + barWidth for x in r]

#         if '_tuned' in model_name: 
#             mark_size = 4
#         else: 
#             mark_size = 3
#         for j, swap_mode in enumerate(['', '_swap']):
#             CCGP_score = np.load(open('_ReLU128_5.7/CCGP_measures/'+rep_type_file_str+model_name+swap_mode+'_CCGP_scores.npz', 'rb'))
# if 
#             else:
#                 values = np.mean(CCGP_score[score_type], axis=(0,1, -1)).flatten()
#                 std_values = np.std(np.mean(CCGP_score[score_type], axis=(1, -1)), axis=0).flatten()
#             markers, caps, bars = plt.errorbar(r, values, yerr = std_values, elinewidth = 0.5, capsize=1.0, marker=m

from plotting import MODEL_STYLE_DICT, task_list, Line2D, mpatches, all_models, plt

def plot_CCGP_scores(model_list, rep_type_file_str = '', save_file=None):
    keys_list = ['all_CCGP', 'holdout_CCGP']
    barWidth = 0.08
    Patches = []
    for i, model_name in enumerate(model_list):
        if '_tuned' in model_name: marker_shape = MODEL_STYLE_DICT[model_name][1]
        else: marker_shape='s'
        Patches.append(Line2D([0], [0], linestyle='None', marker=marker_shape, color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
                markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8))
        len_values = len(keys_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        if '_tuned' in model_name: 
            mark_size = 4
        else: 
            mark_size = 3
    
        for j, swap_mode in enumerate(['', '_swap']):
            values = np.empty((2,))
            spread_values = np.empty((len_values, 5))

            CCGP_score = np.load(open('_ReLU128_5.7/CCGP_measures/'+rep_type_file_str+model_name+swap_mode+'_CCGP_scores.npz', 'rb'))
            for k, key in enumerate(keys_list):  
                if k == 0 and swap_mode == '_swap': 
                    continue
                else:
                    values[k] = np.mean(np.nan_to_num(CCGP_score[key]))
                    if len(CCGP_score[key].shape)>3:
                        spread_values[k, :] = np.mean(np.nan_to_num(CCGP_score[key]), axis=(1,2,3))
                    else: 
                        spread_values[k, :] = np.mean(np.nan_to_num(CCGP_score[key]), axis=(1,2))

                    markers, caps, bars = plt.errorbar(r[k], values[k], yerr = np.std(spread_values[k, :]), elinewidth = 0.5, capsize=1.0, marker=marker_shape, linestyle="", mfc = [None, 'white'][j], alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
            [bar.set_alpha(0.2) for bar in bars]

    plt.ylim(0, 1.05)
    plt.title('CCGP Performance')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth +0.15 for r in range(len_values)], keys_list)
    plt.yticks(np.linspace(0, 1, 11), size=8)

    plt.tight_layout()

    plt.legend(handles=Patches, fontsize=6, markerscale=0.5)
    if save_file is not None: 
        plt.savefig('figs/'+save_file)
    plt.show()


plot_CCGP_scores(all_models, '')

plot_CCGP_scores(['sbertNet_tuned', 'sbertNet'], '')







# def plot_task_CCGP_scores(model_list, score_type,  rep_type_file_str = ''):
#     barWidth = 1/len(model_list)
#     Patches = []
#     for i, model_name in enumerate(model_list):
#         if '_tuned' in model_name: marker_shape = MODEL_STYLE_DICT[model_name][1]
#         else: marker_shape='s'
#         Patches.append(Line2D([0], [0], linestyle='None', marker=marker_shape, color=MODEL_STYLE_DICT[model_name][0], label=model_name, 
#                 markerfacecolor=MODEL_STYLE_DICT[model_name][0], markersize=8))
#         len_values = len(task_list)
#         if i == 0:
#             r = np.arange(len_values)
#         else:
#             r = [x + barWidth for x in r]

#         if '_tuned' in model_name: 
#             mark_size = 4
#         else: 
#             mark_size = 3
#         for j, swap_mode in enumerate(['', '_swap']):
#             CCGP_score = np.load(open('_ReLU128_5.7/CCGP_measures/'+rep_type_file_str+model_name+swap_mode+'_CCGP_scores.npz', 'rb'))
# if 
#             else:
#                 values = np.mean(CCGP_score[score_type], axis=(0,1, -1)).flatten()
#                 std_values = np.std(np.mean(CCGP_score[score_type], axis=(1, -1)), axis=0).flatten()
#             markers, caps, bars = plt.errorbar(r, values, yerr = std_values, elinewidth = 0.5, capsize=1.0, marker=marker_shape, linestyle="", mfc = [None, 'white'][j], alpha=0.8, color = MODEL_STYLE_DICT[model_name][0], markersize=mark_size)
#             [bar.set_alpha(0.2) for bar in bars]

#     for r in range(len_values)[::2]:
#         plt.axvspan(r-barWidth/2, r+(barWidth*len(model_list))-barWidth/2, facecolor='0.2', alpha=0.1)

#     plt.ylim(-0.05, 1.05)
#     plt.title('Trained Performance')
#     plt.xlabel('Task Type', fontweight='bold')
#     plt.title('Trained Performance')
#     plt.xlabel('Task Type', fontweight='bold')
#     plt.ylabel('Percentage Correct')
#     r = np.arange(len_values)
#     plt.xticks([r + barWidth*len(model_list)/2 for r in range(len_values)], task_list, size=5)
#     #plt.xticks([r + barWidth for r in range(len_values)], list(itertools.chain.from_iterable([tasks*2 for tasks in Task.TASK_GROUP_DICT.values()])), size=5)
#     plt.yticks(np.linspace(0, 1, 11), size=8)

#     plt.tight_layout()

#     plt.show()


plot_CCGP_scores(all_models, rep_type_file_str='')
plot_task_CCGP_scores(all_models, 'all_CCGP', rep_type_file_str='prep')
plot_task_CCGP_scores(all_models, 'holdout_CCGP', rep_type_file_str='prep')





CCGP_score = np.load(open('_ReLU128_5.7/CCGP_measures/sbertNet_tuned_CCGP_scores.npz', 'rb'))
CCGP_score['holdout_CCGP'].shape
