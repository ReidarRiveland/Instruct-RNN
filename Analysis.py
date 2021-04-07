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



epochs = 40
init_lr = 0.001
milestones = [15, 20, 25]

# epochs = 50
# init_lr = 0.001
# milestones = [10, 20, 25, 30, 35, 40]

# layer_list = ['layer.11', 'layer.10', 'layer.9', 'layer.8']

foldername = '_ReLU128_'
for holdout in ['COMP1', 'COMP2', 'DNMS']:
    model_dict = {}
    model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=[holdout], batch_size=64)
    cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
    cog.save_models(holdout, foldername)

foldername = '_ReLU128_'
model_dict = {}
model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
cog = CogModule(model_dict)

cog.plot_response('S-Bert train', 'DNMS')

cog.model_dict['S-Bert train'].langMod.plot_embedding(tasks = ['DMS', 'DNMS', 'DMC', 'DNMC'])

cog.plot_learning_curve('correct', smoothing=1)
cog._plot_trained_performance()


cog._plot_trained_performance()
cog.plot_learning_curve('correct')



holdout = 'DNMS'
from Task import construct_batch
from CogModule import isCorrect
from LangModule import get_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cog.load_models(holdout, foldername)
trials = construct_batch(holdout, 100)

tar = torch.Tensor(trials.targets).to(device)
mask = torch.Tensor(trials.masks).to(device)
ins = torch.Tensor(trials.inputs).to(device)
tar_dir = trials.target_dirs

out, _, instruct = cog._get_model_resp(cog.model_dict['S-Bert train'], 100, ins, holdout, None)

corrects = isCorrect(out, tar, tar_dir)
np.mean(corrects)

corrects1 = 

corrects == corrects1

instruct

instruct, _ = get_batch(100, None, task_type='DNMS')
instruct

for i in range()
cog.plot_response('S-Bert train', 'DNMS', task=trials, trial_num=9)

trials.plot_trial(4)


foldername = '_ReLU128_'
for holdout in ['Anti MultiDM']:
    model_dict = {}
    model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
    cog = CogModule(model_dict)
    cog.load_models(holdout, foldername)
    holdout_data = make_data(task_dict = {holdout:1}, num_batches=100, batch_size=256)
    cog.train(holdout_data, 1, lr=0.001)
    cog.sort_perf_by_task()
    cog.save_training_data(holdout, foldername, 'holdout')

cog.plot_learning_curve('correct', 'DMC', smoothing=0.01)

cog._plot_trained_performance()

plot_all_holdout_curves(model_dict, foldername, smoothing=1)
plot_avg_curves(model_dict, foldername, smoothing=0.1)