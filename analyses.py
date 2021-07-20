from math import cos
from warnings import simplefilter

from pandas.core.indexing import convert_missing_indexer
from task import make_test_trials, Task

from model_analysis import get_hid_var_resp
from plotting import plot_hid_traj, plot_model_response, plot_rep_scatter, plot_neural_resp, MODEL_STYLE_DICT
from rnn_models import InstructNet, SimpleNet
from nlp_models import SBERT, BERT
from data import TaskDataSet
from utils import train_instruct_dict
import torch
from model_analysis import get_instruct_reps, get_hid_var_resp, get_task_reps, reduce_rep, get_hid_var_group_resp, get_model_performance
from utils import train_instruct_dict
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import itertools
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import colors, cm, markers 
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib.transforms as mtrans
import torch




model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
model.set_seed(0) 

model.load_model('_ReLU128_5.7/single_holdouts/Multitask')

instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')

import seaborn as sns
opp_task_list = Task.TASK_LIST.copy()
opp_task_list[1], opp_task_list[2] = opp_task_list[2], opp_task_list[1]
instruct_reps[[1,2], ...] = instruct_reps[[2,1], ...] 
corr = np.corrcoef(instruct_reps.reshape(-1, 768))
sns.heatmap(corr, xticklabels='', yticklabels='', cmap=sns.color_palette("rocket_r", as_cmap=True), vmin=-0.2, vmax=1)
plt.show()


reduced_instruct_reps, var_explained = reduce_rep(instruct_reps)


# model1 = SimpleNet(128, 1)
# model1.model_name+='_seed2'
# model1.load_model('_ReLU128_14.6/single_holdouts/Anti_Go')

#sbert_comp_hid_traj = get_hid_var_group_resp(model, 'COMP', 'diff_strength', num_trials=10, sigma_in=100)

sbert_dm_hid_traj = get_hid_var_group_resp(model, 'DM', 'diff_strength', num_trials=1, sigma_in=0.1)
#sbert_go_hid_traj = get_hid_var_group_resp(model, 'Go', 'direction', num_trials=6)


#model1_dm_hid_traj = get_hid_var_group_resp(model1, 'DM', 'diff_strength', num_trials=6)
#model1_go_hid_traj = get_hid_var_group_resp(model1, 'Go', 'direction', num_trials=6)

context_hids = np.empty((4, 15, 1, 120, 128))
for j, task in enumerate(['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']):
    contexts = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/context_vecs', 'rb'))
    contexts_correct = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/context_correct_data', 'rb'))
    for i in range(15): 
        trials, _  = make_test_trials(task, 'diff_strength', 1, num_trials=1)
        with torch.no_grad():
            context = torch.Tensor(contexts[i, :]).repeat(trials.inputs.shape[0], 1)
            proj = model.langModel.proj_out(context)
            _, rnn_hid = super(type(model), model).forward(proj, torch.Tensor(trials.inputs))
            context_hids[j, i, ...] = rnn_hid.numpy()


context_reps = np.empty((16, 768))
for i, task in enumerate(Task.TASK_LIST): 
    contexts = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_vecs', 'rb'))
    contexts_perf = pickle.load(open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_correct_data', 'rb'))
    context_reps[i, :] = np.mean(contexts[contexts_perf[:, -1] > 0.95], axis=0)




from sklearn.metrics.pairwise import cosine_similarity


import seaborn as sns
mean_instruct_reps = np.mean(instruct_reps, axis=1)
mean_instruct_reps.shape

opp_task_list = Task.TASK_LIST.copy()
opp_task_list[1], opp_task_list[2] = opp_task_list[2], opp_task_list[1]

mean_instruct_reps[[1,2], :] = mean_instruct_reps[[2,1], :] 
context_reps[[1,2], :] = context_reps[[2,1], :] 

corr = np.corrcoef(mean_instruct_reps, context_reps)
cos_sim = cosine_similarity(mean_instruct_reps, context_reps)
cos_sim.shape
corr.shape

sns.heatmap(cos_sim, xticklabels=opp_task_list*2, yticklabels=opp_task_list*2, cmap='magma_r', vmin=-0.1, vmax=1)
sns.heatmap(corr[:16, 16:], xticklabels=opp_task_list, yticklabels=opp_task_list, cmap='magma_r', vmin=-0.1, vmax=1)
sns.heatmap(cos_sim, xticklabels=opp_task_list, yticklabels=opp_task_list, cmap='magma_r', vmin=-0.1, vmax=1)

sns.heatmap(corr[:16, :16], xticklabels=opp_task_list, yticklabels=opp_task_list, cmap='magma_r', vmin=-0.2, vmax=1)



for i in range(4):
    plt.axhline(y = 4*i, xmin=i/4, xmax=(i+1)/4, color = 'k',linewidth = 3)
    plt.axhline(y = 4*(i+1), xmin=i/4, xmax=(i+1)/4, color = 'k',linewidth = 3)  
    plt.axvline(x = 4*i, ymin=1-i/4, ymax = 1-(i+1)/4, color = 'k',linewidth = 3)
    plt.axvline(x = 4*(i+1), ymin=1-i/4, ymax = 1-(i+1)/4, color = 'k',linewidth = 3)



plt.show()

# plot_hid_traj(sbert_comp_hid_traj, 'COMP', [0,1], [4], [1], subtitle='sbertNet_layer_11, COMP2 Heldout', annotate_tuples=[(1, 0, 1)])
# plot_hid_traj(sbert_comp_hid_traj, 'COMP', [0,1, 2, 3], [0], [0], subtitle='sbertNet_layer_11, COMP2 Heldout', annotate_tuples=[(1, 0, 0)])


# rnn_reps = get_task_reps(model)


# task_group_dict['COMP'].reverse()
# task_group_dict['COMP']
# plot_rep_scatter(reduced_instruct_reps, ['COMP1', 'COMP2'], annotate_tuples=[(1, 1), (1, 5)], annotate_args=[(-250, -100), (-200, -50)])



# len(train_instruct_dict['COMP2'][1].split())


# plot_hid_traj(sbert_dm_hid_traj, 'DM', [2], [0, 1, 2, 3], [0], subtitle='sbertNet_layer_11, Anti DM Heldout')
# plot_hid_traj(sbert_go_hid_traj, 'Go', [0], [0, 1, 2, 3], range(15), subtitle='sbertNet_layer_11, Anti Go Heldout')



#plot_hid_traj(model1_dm_hid_traj, 'DM', [0], [0, 1, 2, 3], [0], subtitle='simpleNet, Anti DM Heldout')
#plot_hid_traj(model1_go_hid_traj, 'Go', [0], [0, 1, 2, 3], [0], subtitle='simpleNet, Anti Go Heldout')


# sbert_dm_contexts = np.concatenate((np.expand_dims(context_hids, axis=0), sbert_dm_hid_traj.copy()))
# plot_hid_traj(sbert_dm_contexts, 'DM', [0, 1, 2, 3, 4],[0], [0], subtitle='sbertNet_layer_11, Anti Go Heldout; DM context', context_task = 'Anti DM')


#mean_instruct_rep = np.mean(instruct_reps_full[[4, 5, 6, 7],... ], axis=1)
# mean_instruct_rep.shape

# context_reps = np.mean(np.squeeze(np.array([[contexts] for contexts in context_dict.values()])), axis=1)
# context_reps.shape

# corr = np.nan_to_num(np.corrcoef(context_reps.astype(np.float128), mean_instruct_rep.astype(np.float128)))

# sns.heatmap(corr)
# plt.show()

# model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
# model.set_seed(0) 
# model.to(device)
# for task in ['DMS']: 
#     contexts = np.empty((15, 768))
#     streamer = TaskDataSet('_ReLU128_5.7/training_data', num_batches = 200, task_ratio_dict={task:1})
#     streamer.data_to_device(device)
#     for j in range(15): 
#         contexts[j, :]=train_context(model, streamer, 2, model_load_file='_ReLU128_5.7/single_holdouts/Multitask')
#     pickle.dump(contexts, open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_vecs', 'wb'))
#     pickle.dump(np.array(model._correct_data_dict[task]).reshape(15, -1), open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_correct_data', 'wb'))
#     pickle.dump(np.array(model._loss_data_dict[task]).reshape(15, -1), open('_ReLU128_5.7/single_holdouts/'+task.replace(' ', '_')+'/sbertNet_layer_11/context_loss_data', 'wb'))



rand_hids = np.empty((15, 1, 120, 128))
for i in range(15): 
    trials, _  = make_test_trials('DM', 'diff_strength', 1, num_trials=1, sigma_in=0.02)
    with torch.no_grad():
        context = torch.randn((1, 20))
        model.__hiddenInitValue__ = i/15
        _, rnn_hid = super(type(model), model).forward(context, torch.Tensor(trials.inputs))
        rand_hids[i, ...] = rnn_hid.numpy()

rand_hids.shape



plot_hid_traj(sbert_dm_hid_traj, 'DM', [0, 1, 2, 3], [0], [0], subtitle='sbertNet_layer_11, Anti Go Heldout; DM context')


sbert_dm_contexts = np.concatenate((sbert_dm_hid_traj.copy(), context_hids[1:2,...]))
plot_hid_traj(sbert_dm_contexts, 'DM', [0, 1, 2, 3, 4], [0], [0], subtitle='sbertNet_layer_11, Anti Go Heldout; DM context', context_task = 'DM')


from dPCA import dPCA

trials, var_of_insterest = make_test_trials('Go', 'direction', 0, num_trials=6)
var_of_insterest
hid_resp, mean_hid_resp = get_hid_var_resp(model, 'Anti Go', trials, num_repeats=10)

# # trial-average data
# R = mean(trialR,0)

# # center data
# R -= mean(R.reshape((N,-1)),1)[:,None,None]

reshape_mean_hid_resp = mean_hid_resp.T.swapaxes(-1, 1)
reshape_hid_resp = hid_resp.swapaxes(1, 2).swapaxes(-1, 1)

reshape_hid_resp.shape

np.expand_dims(reshape_mean_hid_resp, -1).shape
np.expand_dims(reshape_hid_resp, -1).shape


reshape_mean_hid_resp -= np.mean(mean_hid_resp.reshape((128, -1)), 1)[:, None, None]

dpca = dPCA.dPCA(labels='st',regularizer='auto')
dpca.protect = ['t']

Z = dpca.fit_transform(reshape_mean_hid_resp, reshape_hid_resp)


time = np.arange(120)

plt.figure(figsize=(16,7))
plt.subplot(131)

for s in range(6):
    plt.plot(time,Z['st'][0,s])

plt.title('1st mixing component')

plt.subplot(132)

for s in range(6):
    plt.plot(time,Z['t'][0,s])

plt.title('1st time component')
    
plt.subplot(133)
for s in range(6):
    plt.plot(time,Z['s'][0,s])

plt.title('1st Decision Variable component')
    

plt.figlegend(['delta'+ str(num) for num in np.round(var_of_insterest, 2)], loc=5)

plt.show()
