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
for i in range(seeds): 
    seed = '_seed'+str(i)
    for holdout in task_list+['Multitask']:
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



cog.model_dict.keys()
holdout
cog.save_models(holdout, foldername, seed)






foldername = '_ReLU128_COMPstag_COMPboosted'
model_dict = {}
model_dict['S-Bert train'] = None
model_dict['Model1'] = None
cog = CogModule(model_dict)

cog.load_models('DMS', foldername)
cog.load_training_data('DNMC', foldername,'')
cog.task_sorted_correct.keys()
cog.plot_learning_curve('correct')


cog.plot_response('S-Bert train', 'DNMC')

cog.model_dict['S-Bert train'].langMod.plot_embedding(tasks = ['DMS', 'DNMS', 'DMC', 'DNMC'])

cog.plot_learning_curve('correct', smoothing=1, task_type='DM')
cog._plot_trained_performance()


cog._plot_trained_performance()
cog.plot_learning_curve('correct')


foldername = '_ReLU128_COMPstag_COMPboosted'

foldername

holdout = 'MultiCOMP2'
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

modelS = cog.model_dict['S-Bert train']

out, _, instruct = cog._get_model_resp(cog.model_dict['S-Bert train'], 100, ins, 'MultiCOMP2', None)

out, _ = modelS(instruct, ins, modelS.initHidden(100, 0.1))

corrects = isCorrect(out, tar, tar_dir)
corrects
np.mean(corrects)

instruct, _ = get_batch(100, None, task_type='DNMS')
instruct

instruct = ['respond to the second direction when it has larger combined value over modalities than the first direction otherwise do not respond'] * 100

len(instruct)

cog.plot_response('S-Bert train', 'MultiCOMP2', task=trials, instruct = instruct, trial_num=2)

trials.plot_trial(4)

import pickle
foldername = '_ReLU128_4.9'
seeds = 4

for i in range(seeds): 
    seed = '_seed'+str(i)
    modelS_name = 'S-Bert train'+seed
    model1_name = 'Model1'+seed
    for holdout in task_list:
        correct_dict = {key : np.zeros(100) for key in [modelS_name, model1_name]}
        loss_dict = correct_dict.copy()
        holdout_data = make_data(task_dict = {holdout:1}, num_batches=100, batch_size=256)
        for i in range(5): 
            model_dict = {}
            model_dict[modelS_name] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
            model_dict[model1_name] = simpleNet(81, 128, 1, 'relu')
            cog = CogModule(model_dict)
            cog.load_models(holdout, foldername)
            cog.train(holdout_data, 1, lr=0.001)
            cog.sort_perf_by_task()
            for model_name in cog.model_dict.keys():
                correct_dict[model_name]+=np.array(cog.task_sorted_correct[model_name][holdout])/5
                loss_dict[model_name]+= np.array(cog.task_sorted_loss[model_name][holdout])/5
        holdout_name = holdout.replace(' ', '_')
        pickle.dump(correct_dict, open(foldername+'/'+holdout_name+'/'+seed+'avg_holdout_training_correct_dict', 'wb'))
        pickle.dump(loss_dict, open(foldername+'/'+holdout_name+'/'+seed+'avg_holdout_training_loss_dict', 'wb'))


pickle.load(open(foldername+'/'+holdout_task+'/'+seed+'avg_holdout_loss', 'wb'))

task = 'RT_Go'
seed_num = 5
for i in range(seed_num)

i=0
seed = '_seed'+str(i)
modelS_name = 'S-Bert train'+seed
model1_name = 'Model1'+seed
holdout_correct_dict = pickle.load(open(foldername+'/'+task+'/'+seed+'_training_correct_dict', 'rb'))
holdout_correct_dict['Model1']

holdout_correct_dict

list(holdout_correct_dict.values())[0]


correct_dict

seed_num = 0
seed = '_seed'+str(seed_num)
model_dict = {}
model_dict['S-Bert train'+seed] = None
model_dict['Model1'+seed] = None

cog = CogModule(model_dict)
cog.load_training_data('Go', foldername, seed+'avg_holdout')


cog.plot_learning_curve('correct', task_type='Go')

for model_name in cog.model_dict.keys():    
    smoothed_perf = cog.task_sorted_correct[model_name]['Go']

cog.task_sorted_correct

cog._plot_trained_performance()


seed_num = 0
seed = '_seed'+str(seed_num)
model_dict = {}
model_dict['S-Bert train'+seed] = None
model_dict['Model1'+seed] = None
plot_all_holdout_curves(model_dict, foldername, smoothing=0.01)


model_dict.keys()
for model_name in model_dict.keys(): 
    print(model_name.replace('_seed0','' ))

correct_dict = plot_avg_curves(model_dict, foldername, '_seed0', smoothing=0.01)

correct_dict['S-Bert train']/16

test_str = 'jfsadklfj_seed0'

test_str.index('_seed')

test_str[:test_str.index('_seed')]