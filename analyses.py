from math import cos
from warnings import simplefilter

from pandas.core.indexing import convert_missing_indexer
from task import make_test_trials
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


model_list = list(MODEL_STYLE_DICT.keys())[0:3] + list(MODEL_STYLE_DICT.keys())[4:]
model_list


model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
model.set_seed(0) 

model.load_model('_ReLU128_5.7/single_holdouts/DMS')

get_model_performance(model, 3)


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


context_reps = np.empty((4, 15, 768))
for j, task in enumerate(['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']):
    contexts = pickle.load(open('_ReLU128_14.6/single_holdouts/'+task.replace(' ', '_')+'/context_vecs', 'rb'))
    context_reps[j, ...] = contexts

context_reps



from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(context_reps)
import seaborn as sns
mean_instruct_reps = np.mean(instruct_reps[[4, 5, 6, 7], ...], axis=1)
corr = np.corrcoef(np.mean(instruct_reps[[4, 5, 6, 7], ...], axis=1), context_reps[:, 1, :])
cos_sim = cosine_similarity(np.mean(instruct_reps[[4, 5, 6, 7], ...], axis=1), np.mean(context_reps[:, 2:, :], axis=1))
cos_sim.shape
corr.shape
sns.heatmap(cos_sim, xticklabels=['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], yticklabels=['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], cmap='magma_r')
plt.show()

# plot_hid_traj(sbert_comp_hid_traj, 'COMP', [0,1], [4], [1], subtitle='sbertNet_layer_11, COMP2 Heldout', annotate_tuples=[(1, 0, 1)])
# plot_hid_traj(sbert_comp_hid_traj, 'COMP', [0,1, 2, 3], [0], [0], subtitle='sbertNet_layer_11, COMP2 Heldout', annotate_tuples=[(1, 0, 0)])


# rnn_reps = get_task_reps(model)
instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict, depth='transformer')
reduced_instruct_reps, var_explained = reduce_rep(instruct_reps)

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





trials, var_of_insterest = make_test_trials('DM', 'diff_strength', 0, num_trials=1)
var_of_insterest
hid_resp, mean_hid_resp = get_hid_var_resp(model, 'DM', trials, num_repeats=3)

# # trial-average data
# R = mean(trialR,0)

# # center data
# R -= mean(R.reshape((N,-1)),1)[:,None,None]

reshape_mean_hid_resp = mean_hid_resp.T.swapaxes(-1, 1)
reshape_hid_resp = hid_resp.swapaxes(1, 2).swapaxes(-1, 1)

np.expand_dims(reshape_mean_hid_resp, -1).shape

#reshape_mean_hid_resp -= np.mean(mean_hid_resp.reshape((128, -1)), 1)[:, None, None]

dpca = dPCA.dPCA(labels='std',regularizer='auto')
dpca.protect = ['t']

Z = dpca.fit_transform(np.expand_dims(reshape_mean_hid_resp, -1), np.expand_dims(reshape_hid_resp, -1))


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
