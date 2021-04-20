import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Plotting import plot_all_holdout_curves, plot_all_tasks_by_model, plot_avg_curves, plot_learning_curves
from LangModule import LangModule, swaps
from NLPmodels import gpt2, BERT, SBERT, BoW, SIFmodel, LangTransformer
from RNNs import instructNet, simpleNet
from jitRNNs import scriptSimpleNet 
import torch
import torch.nn as nn

from CogModule import CogModule, isCorrect
from Data import make_data
from Task import Task
task_list = Task.TASK_LIST

def train_holdout_swaps(model_dict, foldername, mode = ''):
    cog = CogModule(model_dict)
    if mode == 'swapped': 
        instruct_mode = 'instruct_swap'
    for swap in swaps:
        swapped_tasks = ''.join(swap).replace(' ', '_')
        cog.load_models(swapped_tasks, foldername)
        try: 
            cog.load_training_data(swapped_tasks, foldername, mode + 'holdout')
        except:
            pass
        task_dict = dict(zip(swap, [1/len(swap)]*len(swap)))
        print(task_dict)
        holdout_only = make_data(task_dict=task_dict, batch_size = 256, num_batches=120)
        cog.train(holdout_only, 2,  weight_decay=0.0, lr = 0.001, instruct_mode = instruct_mode)
        cog.save_training_data(swapped_tasks, foldername, mode + 'holdout')

def train_models(model_dict, foldername, epochs, init_lr, milestones, mode = '', tasks = task_list): 
    for holdout in tasks: 
        for model in model_dict.values(): 
            model.weights_init()
        cog = CogModule(model_dict)
        holdout_data = make_data(holdouts=[holdout], batch_size=64)
        cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
        cog.save_models(holdout, foldername)


def train_holdouts(model_dict, foldername, init_lr, tasks = task_list): 
    for holdout in tasks: 
        cog = CogModule(model_dict)
        cog.load_models(holdout, foldername)
        holdout_data = make_data(task_dict={holdout:1}, batch_size=64, num_batches=100)
        cog.train(holdout_data, 1, lr=init_lr, weight_decay=0.0)
        cog.save_training_data(holdout, foldername, 'holdout')

from collections import OrderedDict

for task in task_list: 
    for model_name in ['S-Bert_train.pt', 'Model1.pt']: 
        filename = 'ReLU128_/' + task+'/'+task+'_'+model_name
        filename = filename.replace(' ', '_')
        state_dict = torch.load(filename)
        new_state_dict = OrderedDict()
        if model_name == 'S-Bert_train.pt': 
            for key, value in state_dict.items(): 
                new_key = key.replace('rnn.rnn', 'rnn.recurrent_units')
                new_state_dict[new_key] = value
        else: 
            for key, value in state_dict.items(): 
                new_key = key.replace('rnn', 'recurrent_units')
                new_state_dict[new_key] = value
        torch.save(new_state_dict, filename)


task_list + ['Multitask']

epochs = 25
init_lr = 0.001
milestones = [10, 15, 20]

# epochs = 50
# init_lr = 0.001
# milestones = [10, 20, 25, 30, 35, 40]

seeds=5
foldername = '_ReLU128_12.4'
for i in [2]: 
    seed = '_seed'+str(i)
    for holdout in ['DMC']:
        model_dict = {}
        model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')
        cog = CogModule(model_dict)
        if holdout == 'Multitask':
            holdout_data = make_data(batch_size=128)
        else:
            holdout_data = make_data(holdouts=[holdout], batch_size=128)
        cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
        cog.save_models(holdout, foldername, seed)


i = 0
seed = '_seed'+str(i)
model_dict = {}
model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')
cog = CogModule(model_dict)

cog.plot_hid_traj(['Model1'+seed], ['RT Go'], 2)

import pickle

foldername = '_ReLU128_12.4'
seeds = 4

seed = '_seed'+str(2)
modelS_name = 'S-Bert train'+seed
model1_name = 'Model1'+seed
for holdout in ['DMC']:
    correct_dict = {key : np.zeros(100) for key in [modelS_name, model1_name]}
    loss_dict = correct_dict.copy()
    for i in range(5): 
        holdout_data = make_data(task_dict = {holdout:1}, num_batches=100, batch_size=256)
        model_dict = {}
        model_dict[modelS_name] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        model_dict[model1_name] = simpleNet(81, 128, 1, 'relu')
        cog = CogModule(model_dict)
        cog.load_models(holdout, foldername)
        cog.train(holdout_data, 1, lr=0.001)
        cog.sort_perf_by_task()
        for model_name in cog.model_dict.keys():
            correct_dict[model_name]+=np.round(np.array(cog.total_correct_dict[model_name])/5, 2)
            loss_dict[model_name]+= np.round(np.array(cog.total_loss_dict[model_name])/5, 2)
    holdout_name = holdout.replace(' ', '_')
    cog.total_correct_dict = correct_dict
    cog.total_loss_dict = loss_dict
    cog.sort_perf_by_task()
    cog.save_training_data(holdout_name, foldername, seed+'holdout')
        

foldername = '_ReLU128_12.4'

num_seeds = 5
model_list = ['S-Bert train', 'Model1']

def get_summary_perf(foldername, num_seeds, model_list, holdout):
    correct_data_dict = {model_name : np.zeros((100, num_seeds)) for model_name in model_list}
    correct_summary_dict = {}
    loss_data_dict = {model_name : np.zeros((100, num_seeds)) for model_name in model_list}
    loss_summary_dict = {}
    for i in range(num_seeds):
        seed = '_seed'+str(i)
        model_dict = {}
        for model_name in model_list: 
            model_dict[model_name+seed] = None
        
        cog = CogModule(model_dict)
        cog.load_training_data(holdout, foldername, seed+'holdout')

        for key in cog.model_dict.keys():
            correct_data_dict[key.replace(seed, '')][:, i] = cog.task_sorted_correct[key][holdout]
            loss_data_dict[key.replace(seed, '')][:, i] = cog.task_sorted_loss[key][holdout]

    for name in model_list:
        correct_summary_dict[name]=np.array([np.mean(correct_data_dict[name], axis=1), np.std(correct_data_dict[name], axis=1)])
        loss_summary_dict[name]=np.array([np.mean(loss_data_dict[name], axis=1), np.std(loss_data_dict[name], axis=1)])

    return correct_summary_dict, correct_data_dict

correct_summary_dict = get_summary_perf(foldername, num_seeds, model_list, 'Anti Go')

correct_summary_dict

from scipy.ndimage.filters import gaussian_filter1d

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from Plotting import label_plot

smoothing = 0.01
holdout_task = 'DM'
model_dict = {}
model_dict['S-Bert train'] = None
model_dict['Model1'] = None
cog = CogModule(model_dict)


fig, axn = plt.subplots(4,4, sharey = True, sharex=True)
plt.suptitle('Holdout Learning for All Tasks')
for i, ax in enumerate(axn.flat):
    ax.set_ylim(-0.05, 1.15)
    holdout_task = task_list[i]
    correct_summary_dict = get_summary_perf(foldername, 5, list(cog.model_dict.keys()), holdout_task)
    for model_name in model_dict.keys(): 
        smoothed_perf = gaussian_filter1d(correct_summary_dict[model_name][0], sigma=smoothing)
        ax.fill_between(np.linspace(0, 100, 100), np.min(np.array([np.ones(100), correct_summary_dict[model_name][0]+correct_summary_dict[model_name][1]]), axis=0), 
            correct_summary_dict[model_name][0]-correct_summary_dict[model_name][1], color =  cog.ALL_STYLE_DICT[model_name][0], alpha= 0.1)
        ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_name][0], marker=cog.ALL_STYLE_DICT[model_name][1], alpha=1, markersize=5, markevery=3)
    ax.set_title(holdout_task)

Patches, Markers = cog.get_model_patches()
label_plot(fig, Patches, Markers, legend_loc=(1.3, 0.5))
plt.show()



holdout = 'COMP2'
for seed_num in range(5): 
    seed = '_seed'+str(seed_num)
    model_dict = {}
    model_dict['S-Bert train'+seed] = None
    model_dict['Model1'+seed] = None
    cog = CogModule(model_dict)
    cog.load_training_data(holdout, foldername, seed+'holdout')
    cog.plot_learning_curve('correct', task_type=holdout)



seed_num = 2
smoothing = 0.01


seed = '_seed'+str(seed_num)
model_dict = {}
model_dict['S-Bert train'+seed] = None
model_dict['Model1'+seed] = None
cog = CogModule(model_dict)
fig, axn = plt.subplots(4,4, sharey = True, sharex=True)
plt.suptitle('Holdout Learning for All Tasks')
for i, ax in enumerate(axn.flat):
    ax.set_ylim(-0.05, 1.15)
    holdout_task = task_list[i]
    cog.load_training_data(holdout_task, foldername, seed+'holdout')
    for model_name in model_dict.keys(): 
        smoothed_perf = gaussian_filter1d(cog.task_sorted_correct[model_name][holdout_task][0:100], sigma=smoothing)
        ax.plot(smoothed_perf, color = cog.ALL_STYLE_DICT[model_name.replace(seed, '')][0], marker=cog.ALL_STYLE_DICT[model_name.replace(seed, '')][1], alpha=1, markersize=5, markevery=3)
    ax.set_title(holdout_task)
Patches, Markers = cog.get_model_patches()
label_plot(fig, Patches, Markers, legend_loc=(1.3, 0.5))
fig.show()

model_dict = {}
model_dict['S-Bert train'] = None
model_dict['Model1'] = None
plot_all_holdout_curves(model_dict, '_ReLU128_COMPstag_COMPboosted')