# from collections import defaultdict
# from math import inf
# import pickle
# from models.model_trainer import config_model
# from re import I
# from matplotlib.cbook import flatten

# from matplotlib.pyplot import axis
# from numpy.core.fromnumeric import size, var
# from numpy.lib.function_base import append
# from numpy.ma import cos
# import transformers

# from utils.utils import train_instruct_dict
# from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_layer_sim_scores, get_hid_var_group_resp, get_hid_var_resp, get_all_CCGP
# import numpy as np
# from utils.utils import train_instruct_dict, task_swaps_map, all_models
# from task import DM
# from plotting import plot_RDM, plot_rep_scatter, plot_CCGP_scores, plot_model_response, plot_hid_traj_quiver, plot_dPCA, plot_neural_resp, plot_trained_performance, plot_tuning_curve
# import torch

# from task import Task, make_test_trials

# import torch
# from transformers import CLIPTokenizer, GPT2Tokenizer, GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# model.__dict__
# model.config.n_embd

# model.lm_head.state_dict()['weight'].shape

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss
# logits = outputs.logits


# cur_sentences = "Hello, my dog is"
# for i in range(20): 
#     inputs = tokenizer(cur_sentences, return_tensors="pt")
#     outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss
#     logits = outputs.logits
#     scores = torch.softmax(logits, dim=-1)
#     append_word = tokenizer.batch_decode(torch.max(scores, 2).indices[:, -1])
#     cur_sentences+=append_word[0]

# cur_sentences

# from PIL import Image
# import requests
# from transformers import CLIPTokenizer, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")



# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# instruct = tokenizer(['go in the direction of stimulus with greatest strength'], return_tensors='pt')

# model.get_text_features(**instruct).shape


# from transformers import CLIPTokenizer, CLIPTextModel

# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# model.config

# inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

# model

# outputs = model(**inputs, output_hidden_states=True)
# last_hidden_state = outputs.hidden_states

# outputs.hidden_states[3].shape
# last_hidden_state.shape
# pooled_output = outputs.pooler_output  # pooled (EOS token) states
# pooled_output.unsqueeze(0)[0].shape

# SIMPLENET
# [('Go', 0.9921875), ('Anti_Go', 0.04296875), ('RT_Go', 0.40234375), ('Anti_RT_Go', 0.05078125), 
# ('Go_Mod1', 0.2734375), ('Anti_Go_Mod1', 0.3046875), ('Go_Mod2', 0.46875), ('Anti_Go_Mod2', 0.16796875), 
# ('DelayGo', 0.1484375), ('Anti_DelayGo', 0.171875), 
# ('DM', 0.7734375), ('Anti_DM', 0.40234375), ('MultiDM', 0.4453125), ('Anti_MultiDM', 0.52734375), 
# ('RT_DM', 0.07421875), ('Anti_RT_DM', 0.4453125), 
# ('ConDM', 0.5859375), ('Anti_ConDM', 0.7109375), ('ConMultiDM', 0.73828125), ('Anti_ConMultiDM', 0.6953125), 
# ('DelayDM', 0.32421875), ('Anti_DelayDM', 0.1953125), ('DelayMultiDM', 0.30859375), ('Anti_DelayMultiDM', 0.16796875), 
# ('DM_Mod1', 0.37109375), ('Anti_DM_Mod1', 0.16796875), ('DM_Mod2', 0.36328125), ('Anti_DM_Mod2', 0.3359375), 
# ('COMP1', 0.39453125), ('COMP2', 0.453125), ('MultiCOMP1', 0.2109375), ('MultiCOMP2', 0.3828125), 
# ('DMS', 0.4375), ('DNMS', 0.359375), ('DMC', 0.21875), ('DNMC', 0.33984375)]
# 37%

# SBERTNET
# [('Go', 1.0), ('Anti_Go', 1.0), ('RT_Go', 0.98828125), ('Anti_RT_Go', 0.6484375), 
# ('Go_Mod1', 0.70703125), ('Anti_Go_Mod1', 0.86328125), ('Go_Mod2', 0.8515625), ('Anti_Go_Mod2', 0.96875), 
# ('DelayGo', 1.0), ('Anti_DelayGo', 1.0), 
# ('DM', 1.0), ('Anti_DM', 1.0), ('MultiDM', 0.953125), ('Anti_MultiDM', 0.8984375), 
# ('RT_DM', 0.96484375), ('Anti_RT_DM', 0.62890625), 
# ('ConDM', 0.9375), ('Anti_ConDM', 0.953125), ('ConMultiDM', 0.90234375), ('Anti_ConMultiDM', 0.953125), 
# ('DelayDM', 0.98828125), ('Anti_DelayDM', 0.8515625), ('DelayMultiDM', 0.91796875), ('Anti_DelayMultiDM', 0.58984375), 
# ('DM_Mod1', 0.8125), ('Anti_DM_Mod1', 0.73828125), ('DM_Mod2', 0.765625), ('Anti_DM_Mod2', 0.69921875), 
# ('COMP1', 0.72265625), ('COMP2', 0.6328125), ('MultiCOMP1', 0.4609375), ('MultiCOMP2', 0.4140625), 
# ('DMS', 0.61328125), ('DNMS', 0.859375), ('DMC', 0.578125), ('DNMC', 0.734375)]
#82%


from turtle import position
from instructRNN.data_loaders.dataset import TaskDataSet


from instructRNN.tasks.task_factory import TaskFactory
from instructRNN.models.full_models import *
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from instructRNN.plotting.plotting import plot_model_response
from instructRNN.tasks.tasks import SWAP_LIST, SWAPS_DICT, TASK_LIST 
from instructRNN.instructions.instruct_utils import get_instructions, train_instruct_dict
from instructRNN.tasks.task_criteria import isCorrect
import numpy as np
import torch


EXP_FILE = '6.7models/swap_holdouts'
sbertNet = CLIPNet(LM_out_dim=64, rnn_hidden_dim=256)
#sbertNet = SimpleNet(rnn_hidden_dim=256)

holdouts_file = 'Multitask'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

EXP_FILE = '6.7models/swap_holdouts'
comNet = ComNetPlus()
#sbertNet = SimpleNet(rnn_hidden_dim=256)

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0_CHECKPOINT')



for task in SWAPS_DICT[holdouts_file]:
    print(task)
    perf = task_eval(sbertNet, task, 256)
    print((task, perf))


def get_zero_shot_perf(model): 
    perf_array = np.empty(len(TASK_LIST))
    for label, tasks in list(SWAPS_DICT.items()):
        model.load_model(EXP_FILE+'/'+label+'/'+model.model_name, suffix='_seed0')
        for task in tasks: 
            # task_instructions = get_instructions(256, task, None)
            # data = TaskDataSet('5.25models/training_data', batch_len=256, num_batches =1, set_single_task=task, stream=False)
            # ins, tar, mask, tar_dirs, type = next(data.stream_batch())
            # out, hid = sbertNet(ins, task_instructions)
            # perf_array[TASK_LIST.index(task)] =  np.mean(isCorrect(out, tar, tar_dirs))
            print(task)
            perf = task_eval(model, task, 256) 
            perf_array[TASK_LIST.index(task)] = perf
    return perf_array

perf = get_zero_shot_perf(sbertNet)
perf
list(zip(TASK_LIST, perf))
np.mean(perf) 


from data_loaders.dataset import TaskDataSet
from task_factory import TaskFactory
TASK_LIST

data = TaskDataSet('6.5models/training_data', num_batches = 128, set_single_task='Anti_MultiDM', stream=False)
ins, tar, mask, tar_dirs, type = next(data.stream_batch())
task_instructions = get_instructions(128, 'Anti_MultiDM', None)

out, _ = sbertNet(ins, task_instructions)

isCorrect(out, tar, tar_dirs)

for index in range(5):
    TaskFactory.plot_trial(ins[index, ...], tar[index, ...], type)


from instructions.instruct_utils import train_instruct_dict
repeats = []
for instruct in train_instruct_dict['Anti_DM_Mod2']:
    perf = task_eval(sbertNet, 'Anti_DM_Mod2', 128, 
            instructions=[instruct]*128)
    repeats.append((instruct, perf))

TASK_LIST


task = 'Anti_Go_Mod2'
instructions = ['pick the opposite of the orientation in the second modality' ]*128
instructions[0] in train_instruct_dict[task]
task_eval(sbertNet, task, 128, instructions=instructions)


task_eval(sbertNet, 'Anti_Go_Mod2', 128)



get_instructions(128, 'DMC', None)

from plotting import plot_model_response

from tasks import AntiGoMod1
trials = AntiGoMod1(128)
task_instructions = get_instructions(128, 'Anti_Go_Mod1', None)
plot_model_response(sbertNet, trials, instructions=task_instructions)


task_instructions[2]





train_instruct_dict['Anti_DM_Mod2']

perf = get_model_performance(sbertNet)
list(zip(TASK_LIST, perf))
np.mean(repeats)




resp = get_task_reps(sbertNet)
reps_reduced, _ = reduce_rep(resp)


from model_analysis import get_layer_sim_scores, get_instruct_reps
from plotting import plot_RDM

reps = get_instruct_reps(sbertNet.langModel, depth='12')
reps.shape
np.max(reps[TASK_LIST.index('DM'), 0, :])
np.min(reps)


sim_scores = get_layer_sim_scores(sbertNet, rep_depth='full')
plot_RDM(sim_scores)



from models.full_models import SimpleNet
from model_analysis import get_DM_perf, get_noise_thresholdouts
import pickle
import numpy as np
EXP_FILE = '6.6models/noise_thresholding_model'

simpleNet = SimpleNet(rnn_hidden_dim=256)
simpleNet.load_model(EXP_FILE+'/'+simpleNet.model_name, suffix='_seed0')

task = 'DM'


#diff_strength = np.concatenate((np.linspace(-0.15, -0.05, num=7), np.linspace(0.05, 0.15, num=7)))
diff_strength = np.concatenate((np.linspace(-0.2, -0.1, num=7), np.linspace(0.1, 0.2, num=7)))

noises = np.linspace(0.05, 0.75, num=30)

correct_stats, pstim1_stats, trial = get_DM_perf(simpleNet, noises, diff_strength, task=task)

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

for x in range(15):
    plt.plot(noises, np.mean(correct_stats[:, :, x], axis=0))
plt.legend(labels=list(np.round(diff_strength, 2)))
plt.xlabel('Noise Level')
plt.ylabel('Correct Rate')
plt.show()

thresholds = get_noise_thresholdouts(correct_stats, diff_strength, noises, neg_cutoff=0.8)

pickle.dump(thresholds, open('6.7models/noise_thresholds/dm_noise_thresholds', 'wb'))


#THIS ISNT EXACTLY RIGHT BECAUSE YOU ARE COUNTING INCOHERENT ANSWERS AS ANSWER STIM2
for x in range(10):
    smoothed = gaussian_filter1d(np.mean(pstim1_stats[:, x, :], axis=0), 1)
    plt.plot(diff_strength, smoothed)

plt.legend(labels=list(np.round(noises, 2)[:10]))
plt.xlabel('Contrast')
plt.ylabel('p_stim1')
plt.show()


# simpleNetPlus_context = np.empty((16, 20))
# simpleNetPlus.eval()
# with torch.no_grad():
#     for i, task in enumerate(Task.TASK_LIST): 
#         rule = get_input_rule(1, task, None)
#         simpleNetPlus_context[i, :]=simpleNetPlus.rule_encoder(torch.matmul(rule, simpleNetPlus.rule_transform))

# simpleNetPlus_context = np.expand_dims(simpleNetPlus_context, 1)

# reps_reduced, _ = reduce_rep(simpleNetPlus_context)
# from plotting import plot_rep_scatter
# plot_rep_scatter(reps_reduced, Task.TASK_GROUP_DICT['Delay'], s=100)


import numpy as np
from instructions.instruct_utils import get_task_info
from task_criteria import isCorrect
from models.full_models import SBERTNet, SBERTNet_tuned, SimpleNet
from model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from plotting import plot_model_response
from tasks_utils import SWAPS_DICT 
from tasks import TASK_LIST, construct_trials
import torch
from tasks import Task
from task_factory import DMFactory
import task_factory
from models.full_models import SBERTNet
from instructions.instruct_utils import get_instructions
from tqdm import tqdm
from task_factory import TaskFactory

EXP_FILE = '6.3models/swap_holdouts'
sbertNet = SBERTNet()

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


CLUSTER_TASK_LIST = ['Go', 'RT_Go','DelayGo','Go_Mod1','Go_Mod2',
                    'Anti_Go',  'Anti_RT_Go', 'Anti_DelayGo', 'Anti_Go_Mod1',  'Anti_Go_Mod2',
                    'DM', 'RT_DM', 'MultiDM', 'DelayDM', 'DelayMultiDM', 'ConDM','ConMultiDM','DM_Mod1',  'DM_Mod2',
                    'Anti_DM', 'Anti_RT_DM', 'Anti_MultiDM',    'Anti_DelayDM',  'Anti_DelayMultiDM','Anti_ConDM', 'Anti_ConMultiDM', 'Anti_DM_Mod1', 'Anti_DM_Mod2',        
                    
                    #'RT_DM_Mod1', 'Anti_RT_DM_Mod1', 'RT_DM_Mod2', 'Anti_RT_DM_Mod2', 

                    'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 

                    #'COMP1_Mod1', 'COMP2_Mod1', 'COMP1_Mod2', 'COMP2_Mod2',

                    'DMS', 'DNMS', 'DMC', 'DNMC']


def get_hidden_reps(model, num_trials, tasks=CLUSTER_TASK_LIST, instruct_mode=None):
    hidden_reps = np.empty((num_trials, 120, 128, len(tasks)))
    with torch.no_grad():
        for i, task in enumerate(tasks): 
            ins, _, _, _, _ =  construct_trials(task, num_trials)

            task_info = get_task_info(num_trials, task, model.is_instruct, instruct_mode=instruct_mode)
            _, hid = model(torch.Tensor(ins).to(model.__device__), task_info)
            hidden_reps[..., i] = hid.cpu().numpy()
    return hidden_reps

hid_reps = get_hidden_reps(sbertNet, 256)

A = np.matrix(np.mean(np.var(hid_reps, axis=0), axis=0))

B = (A-A.min(axis=0))/(A.max(axis=0)-A.min(axis=0))
np.sum(B, axis=0)

#HOW TO SORT?

def NormalizeData(data):
    return (data[:,] - np.min(data, axis=1)) / (np.max(data, axis=1) - np.min(data, axis=1))

NormalizeData(task_var)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(B.T, yticklabels=CLUSTER_TASK_LIST)
plt.show()


from scipy.stats import ortho_group  # Requires version 0.18 of scipy

m = ortho_group.rvs(dim=64)

# import pickle 
# pickle.dump(m[:11, ], open('models/ortho_rule_vecs/ortho_comp_rules64', 'wb'))