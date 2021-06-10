from numpy.core.fromnumeric import size
from scipy.ndimage.measurements import label
from dPCA import dPCA

import numpy as np
import matplotlib.pyplot as plt

from cog import instructNet, simpleNet
from LangModule import LangModule
from NLPmodels import SBERT, BERT
from CogModule import CogModule
#from Plotting import make_test_trials, get_hid_var_resp

from scipy.stats import zscore

##Model training loop
epochs = 30
init_lr = 0.001
milestones = [5, 10, 15, 20]

seed = '_seed'+str(0)
model_dict = {}
model_dict['S-Bert train'+seed] = instructNet(LangModule(SBERT(20)), 128, 1, 'relu', tune_langModel=True, langLayerList=['layer.11'])
#model_dict['Model1'+seed] = simpleNet(81, 128, 1, 'relu')


cog = CogModule(model_dict)

from Data import data_streamer

cog.train(data_streamer(), 15, lr=init_lr, milestones=milestones)




cog.load_models('Anti DM', foldername)

cog._plot_trained_performance()



trials, var_of_insterest = make_test_trials('DM', 'diff_strength', 0, None, num_trials=6)
var_of_insterest
hid_resp, mean_hid_resp = get_hid_var_resp(model_dict['S-Bert train'+seed], 'DM', trials)

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


from LangModule import train_instruct_dict
indices, reps = model_dict['S-Bert train'+seed].langMod._get_instruct_rep(train_instruct_dict)

reps_reshaped = reps.reshape(16, -1, 20)

reps_reshaped.mean(1)

from CogModule import comp_one_hot


from Task import Task
task_list = Task.TASK_LIST

task_list.index('Go')

def get_lang_comp_rep(task_type):
    return reps_reshaped.mean(1)[task_list.index(task_type)]-np.matmul(comp_one_hot(task_type), reps_reshaped.mean(1))



import torch

fixed_point = torch.randn(128, requires_grad=True)

