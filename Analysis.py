import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Plotting import plot_all_holdout_curves, plot_all_tasks_by_model, plot_avg_curves, plot_learning_curves
from LangModule import LangModule, swaps
from NLPmodels import GPT, BERT, SBERT, BoW, SIFmodel, LangTransformer
from RNNs import instructNet, simpleNet
from jitRNNs import scriptSimpleNet 
import torch
import torch.nn as nn

from CogModule import CogModule, isCorrect
from Data import make_data
from Task import Task
task_list = Task.TASK_LIST




# epochs = 50
# init_lr = 0.001
# milestones = [10, 20, 25, 30, 35, 40]


###Model training loop
epochs = 25
init_lr = 0.001
milestones = [10, 15, 20]

seeds=5
foldername = '_ReLU128_12.4'
for i in range(5): 
    seed = '_seed'+str(i)
    for holdout in task_list + ['Multitask']:
        model_dict = {}
        model_dict['BERT train'+seed] = instructNet(LangModule(BERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        #model_dict['GPT train'+seed] = instructNet(LangModule(GPT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
        cog = CogModule(model_dict)
        if holdout == 'Multitask':
            holdout_data = make_data(batch_size=128)
        else:
            holdout_data = make_data(holdouts=[holdout], batch_size=128)
        cog.train(holdout_data, epochs, lr=init_lr, milestones = milestones, weight_decay=0.0)
        cog.save_models(holdout, foldername, seed)


###Holdout training loop 
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
        


