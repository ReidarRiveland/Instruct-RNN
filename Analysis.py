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
from Taskedit import Task
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



from jitRNNs import scriptSimpleNet


model_dict = {}

model_dict['Model1'] = scriptSimpleNet(81, 128, 1, 'relu')

sNet = model_dict['Model1']

ins = torch.rand((128, 120, 81))
h0 = sNet.initHidden(128, 0.1)
sNet.rnn(ins, h0)[1].shape

cog = CogModule(model_dict)
holdout_data = make_data(num_batches=500, batch_size=128)
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




epochs = 40
init_lr = 0.001
milestones = [15, 20, 25]

foldername = 'ReLU128_'
for holdout in task_list:
    model_dict = {}
    model_dict['S-Bert train'] = instructNet(LangModule(SBERT(50)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
    model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=[holdout], batch_size=128)
    cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
    cog.save_models(holdout, foldername)

cog._plot_trained_performance()
cog.plot_learning_curve('correct')

cog._get_performance(cog.model_dict['S-Bert train'], num_batches=5)

cog.load_models('COMP1', foldername)

cog.plot_response('S-Bert train', 'COMP2')

cog.plot_learning_curve('loss')
cog._plot_trained_performance()

correct_list = []


for n, p in model_dict['S-Bert train'].named_parameters(): 
    if p.requires_grad: 
        print(n)


cog._plot_trained_performance()
cog.plot_learning_curve('correct')

cog.model_dict['S-Bert train'].langMod.plot_embedding(tasks = ['Go', 'Anti Go', 'RT Go', 'Anti RT Go'])

foldername = 'ReLU128SBlayers2_delay'
for holdout in ['DM', 'Anti DM']:
    model_dict = {}
    model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
    model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20, output_layers=2)), 128, 1, 'relu', tune_langModel=True)
    cog = CogModule(model_dict)
    cog.load_models(holdout, foldername)
    holdout_data = make_data(task_dict = {holdout:1}, num_batches=100, batch_size=128)
    cog.train(holdout_data, 1, lr=0.001)
    cog.sort_perf_by_task()
    cog.save_training_data(holdout, foldername, 'holdout')



import torch.nn as nn

foldername = 'ReLU128Lang10'
model_dict = {}
model_dict['Model1'] = simpleNet(81, 128, 1, 'relu')
model_dict['S-Bert train'] = instructNet(LangModule(SBERT(10)), 128, 1, 'relu', tune_langModel=True)
cog = CogModule(model_dict)

task = 'MultiDM'

cog.load_models(task, foldername)

cog._plot_trained_performance()

cog.load_training_data(task, foldername, 'holdout')
cog.plot_learning_curve('correct', task, smoothing=1)


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

