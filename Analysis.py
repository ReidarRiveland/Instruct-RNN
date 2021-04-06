import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Plotting import plot_all_holdout_curves, plot_all_tasks_by_model, plot_avg_curves, plot_learning_curves
from LangModule import LangModule, swaps
from NLPmodels import gpt2, BERT, SBERT, BoW, SIFmodel, LangTransformer
from RNNs import instructNet, simpleNet
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

filename

new_state_dict.keys()

new_keys

task = 'Go'
filename = 'ReLU128_/' + task+'/'+task+'_'+'S-Bert_train.pt'
state_dict = torch.load(filename.replace(' ', '_'))

state_dict.keys()

state_dict.keys()

model_dict['S-Bert train'].state_dict().keys()

from jitRNNs import scriptSimpleNet, scriptInstructNet

model_dict = 

modelS = instructNet(LangModule(SBERT(50)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])

modelS.langModel.model.state_dict().keys()

sNet = model_dict['S-Bert train']

from RNNs import count_parameters

for n, p in sNet.named_parameters(): 
    if p.requires_grad: 
        print(n)

count_parameters(sNet)

ins = torch.rand((128, 120, 81))
h0 = sNet.initHidden(128, 0.1)
sNet.rnn(ins, h0)[1].shape

cog = CogModule(model_dict)
holdout_data = make_data(num_batches=100, batch_size=128)
cog.train(holdout_data, 50, lr=0.001, milestones = [15, 20, 25])

epochs = 60
init_lr = 0.001
milestones = [30, 40]

foldername = 'SigModels128SBsigLangLR0.005_delay'
for holdout in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM', 'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'DMC', 'DNMC']: 
    model_dict = {}
    model_dict['Model1'] = simpleNet(81, 128, 1, 'sigmoid')
    model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20, output_nonlinearity=nn.Sigmoid())), 128, 1, 'sigmoid', tune_langModel=True)
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=[holdout], batch_size=128)
    cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0, langLR=0.005)
    cog.save_models(holdout, foldername)




epochs = 60
init_lr = 0.001
milestones = [30, 40, 50]

foldername = '_ReLU128_'
for holdout in task_list:
    model_dict = {}
    model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=[holdout], batch_size=128)
    cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0, langLR=0.0001, langWeightDecay=0.0)
    cog.save_models(holdout, foldername)

cog._get_performance(cog.model_dict['S-Bert train'])
cog.plot_learning_curve('loss')

cog._get_performance(cog.model_dict['S-Bert train'], num_batches=5)

cog.load_models('Anti MultiDM', foldername)

cog.plot_response('S-Bert train', 'RT Go')

cog.plot_learning_curve('correct', smoothing=0.001)
cog._plot_trained_performance()

correct_list = []


for n, p in model_dict['S-Bert train'].named_parameters(): 
    if p.requires_grad: 
        print(n)


cog._plot_trained_performance()
cog.plot_learning_curve('correct')

cog.model_dict['S-Bert train'].langMod.plot_embedding(tasks = ['Go', 'Anti Go', 'RT Go', 'Anti RT Go'])

foldername = '_ReLU128_dmStaggered'
for holdout in task_list:
    model_dict = {}
    #model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    model1_name = 'ReLU128_/'+holdout+'/'+holdout+'_Model1.pt'
    model1_name = model1_name.replace(' ', '_')
    Model1 = simpleNet(81, 128, 1, 'relu')
    Model1.load_state_dict(torch.load(model1_name))
    model_dict['Model1'] = Model1

    ModelS = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    ModelS_name = foldername +'/'+holdout+'/'+holdout+'_S-Bert_train.pt'
    ModelS_name = ModelS_name.replace(' ', '_')
    ModelS.load_state_dict(torch.load(ModelS_name))
    model_dict['S-Bert train'] = ModelS

    cog = CogModule(model_dict)
    #cog.load_models(holdout, 'ReLU128_')
    holdout_data = make_data(task_dict = {holdout:1}, num_batches=100, batch_size=128)
    cog.train(holdout_data, 1, lr=0.001, freeze_langModel=True)
    cog.sort_perf_by_task()
    cog.save_training_data(holdout, foldername, 'holdout')

plot_all_holdout_curves(model_dict, foldername, smoothing=0.001)

import torch.nn as nn

foldername = 'ReLU128Lang10'
model_dict = {}
model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
model_dict['S-Bert train'] = instructNet(LangModule(SBERT(10)), 128, 1, 'relu', tune_langModel=True)
cog = CogModule(model_dict)

task = 'Anti RT Go'

cog.load_models(task, foldername)

cog._plot_trained_performance()

model_dict['S-Bert train'] = 

cog.load_training_data(task, foldername, 'holdout')
cog.plot_learning_curve('correct', task, smoothing=0.1)


cog.plot_response('S-Bert train', 'MultiDM')



model_dict = {}
model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20, output_layers=2)), 128, 1, 'relu', tune_langModel=True)



holdout_data = make_data(task_dict = {'COMP2':1/3, 'MultiCOMP1':1/3, 'MultiCOMP2':1/3}, num_batches=250, batch_size=128)
cog.train(holdout_data, 5, lr=0.0001)

cog.plot_learning_curve('correct')

cog._get_performance(model_dict['Model1'])

task = 'COMP1'
cog.load_models(task, foldername)

model_dict['S-Bert train'].langMod.plot_embedding(tasks=['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], train_only=True, dim=3)

cog.load_training_data(task, foldername, task)
cog.plot_learning_curve('correct', smoothing=0.01)

cog.plot_response('S-Bert train', 'MultiCOMP2')
cog._plot_trained_performance()

plot_avg_curves(model_dict, foldername)


index_list = list(np.arange(500))
np.random.shuffle(index_list)
index_list
len(index_list)

