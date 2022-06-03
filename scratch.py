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

#SIMPLENET
# [('Go', 1.0), ('Anti_Go', 0.1484375), ('RT_Go', 0.0), ('Anti_RT_Go', 0.1171875), 
# ('Go_Mod1', 0.359375), ('Anti_Go_Mod1', 0.12109375), ('Go_Mod2', 0.44140625), ('Anti_Go_Mod2', 0.14453125), 
# ('DelayGo', 0.72265625), ('Anti_DelayGo', 0.015625), 
# ('DM', 0.19140625), ('Anti_DM', 0.15234375), ('MultiDM', 0.20703125), ('Anti_MultiDM', 0.53515625), 
# ('RT_DM', 0.29296875), ('Anti_RT_DM', 0.34375), 
# ('DelayDM', 0.6171875), ('Anti_DelayDM', 0.11328125), ('DelayMultiDM', 0.41796875), ('Anti_DelayMultiDM', 0.19921875), 
# ('DM_Mod1', 0.2578125), ('Anti_DM_Mod1', 0.3515625), ('DM_Mod2', 0.2265625), ('Anti_DM_Mod2', 0.21484375), 
# ('COMP1', 0.35546875), ('COMP2', 0.5), ('MultiCOMP1', 0.34765625), ('MultiCOMP2', 0.61328125), 
# ('COMP1_Mod1', 0.4140625), ('COMP2_Mod1', 0.5078125), ('COMP1_Mod2', 0.39453125), ('COMP2_Mod2', 0.25390625), 
# ('DMS', 0.49609375), ('DNMS', 0.2734375), ('DMC', 0.4375), ('DNMC', 0.50390625)]

#SBERTNET
# [('Go', 1.0), ('Anti_Go', 0.99609375), ('RT_Go', 0.96484375), ('Anti_RT_Go', 0.48046875), 
# ('Go_Mod1', 0.99609375), ('Anti_Go_Mod1', 0.59375), ('Go_Mod2', 0.90234375), ('Anti_Go_Mod2', 0.78515625), 
# ('DelayGo', 0.99609375), ('Anti_DelayGo', 0.9921875), 
# ('DM', 0.96875), ('Anti_DM', 1.0), ('MultiDM', 0.984375), ('Anti_MultiDM', 0.98046875), 
# ('RT_DM', 0.83203125), ('Anti_RT_DM', 0.82421875), 
# ('DelayDM', 0.90625), ('Anti_DelayDM', 0.8125), ('DelayMultiDM', 0.796875), ('Anti_DelayMultiDM', 0.68359375), 
# ('DM_Mod1', 0.99609375), ('Anti_DM_Mod1', 0.890625), ('DM_Mod2', 0.59375), ('Anti_DM_Mod2', 0.32421875), 
# ('COMP1', 0.51171875), ('COMP2', 0.73046875), ('MultiCOMP1', 0.9453125), ('MultiCOMP2', 0.65234375), 
# ('COMP1_Mod1', 0.296875), ('COMP2_Mod1', 0.24609375), ('COMP1_Mod2', 0.2890625), ('COMP2_Mod2', 0.1015625), 
# ('DMS', 0.609375), ('DNMS', 0.64453125), ('DMC', 0.74609375), ('DNMC', 0.71484375)]

#SBERTNET_TUNED
# [('Go', 1.0), ('Anti_Go', 1.0), ('RT_Go', 0.99609375), ('Anti_RT_Go', 0.98046875), 
# ('Go_Mod1', 1.0), ('Anti_Go_Mod1', 0.5078125), ('Go_Mod2', 0.93359375), ('Anti_Go_Mod2', 0.9765625), 
# ('DelayGo', 1.0), ('Anti_DelayGo', 0.9453125), 
# ('DM', 0.97265625), ('Anti_DM', 1.0), ('MultiDM', 1.0), ('Anti_MultiDM', 1.0), 
# ('RT_DM', 0.828125), ('Anti_RT_DM', 0.7578125), 
# ('DelayDM', 0.87109375), ('Anti_DelayDM', 0.90625), ('DelayMultiDM', 0.84375), ('Anti_DelayMultiDM', 0.76171875), 
# ('DM_Mod1', 1.0), ('Anti_DM_Mod1', 0.875), ('DM_Mod2', 0.70703125), ('Anti_DM_Mod2', 0.3984375), 
# ('COMP1', 0.7109375), ('COMP2', 0.68359375), ('MultiCOMP1', 0.953125), ('MultiCOMP2', 0.95703125), 
# ('COMP1_Mod1', 0.34765625), ('COMP2_Mod1', 0.53125), ('COMP1_Mod2', 0.24609375), ('COMP2_Mod2', 0.07421875), 
# ('DMS', 0.6796875), ('DNMS', 0.7109375), ('DMC', 0.6328125), ('DNMC', 0.6640625)]
#85%

from turtle import position
from dataset import TaskDataSet

from task_factory import TaskFactory
from models.full_models import SBERTNet, SBERTNet_tuned, SimpleNetPlus, SimpleNet
from model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from plotting import plot_model_response
from tasks_utils import SWAP_LIST, SWAPS_DICT 
from instruct_utils import get_instructions
from task_criteria import isCorrect
import numpy as np
import torch
from tasks import TASK_LIST


EXP_FILE = '6.4models/swap_holdouts'
sbertNet = SBERTNet()

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


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


from dataset import TaskDataSet
from task_factory import TaskFactory
TASK_LIST

data = TaskDataSet('6.4models/training_data', num_batches = 10, set_single_task='Anti_DM_Mod2', stream=False)
ins, tar, mask, tar_dirs, type = next(data.stream_batch())

for index in range(5):
    TaskFactory.plot_trial(ins[index, ...], tar[index, ...], type)


from instruct_utils import train_instruct_dict

repeats = []
for instruct in train_instruct_dict['Anti_DM_Mod2']:
    perf = task_eval(sbertNet, 'Anti_DM_Mod1', 128, 
            instructions=[instruct]*128)
    repeats.append((instruct, perf))

repeats

get_instructions(128, 'DMC', None)

from plotting import plot_model_response

from tasks import AntiDMMod2
trials = AntiDMMod2(128)
plot_model_response(sbertNet, trials, instructions=task_instructions)

task_instructions = get_instructions(128, 'Anti_DM_Mod2', None)
task_instructions[2]

perf = task_eval(sbertNet, 'Anti_DM_Mod2', 128, instructions=task_instructions)

perf = get_model_performance(sbertNet)
list(zip(TASK_LIST, perf))
np.mean(repeats)


for task in SWAPS_DICT[holdouts_file]:
    print(task)
    perf = task_eval(sbertNet, task, 256)
    print((task, perf))







resp = get_task_reps(sbertNet)
reps_reduced, _ = reduce_rep(resp)


from model_analysis import get_layer_sim_scores, get_instruct_reps
import seaborn as sns
import matplotlib.pyplot as plt

#reps = get_instruct_reps(sbertNet.langModel, depth='full')

sim_scores = get_layer_sim_scores(sbertNet, rep_depth='full')

sim_scores.shape


def plot_RDM(sim_scores, rep_type, cmap=sns.color_palette("rocket_r", as_cmap=True), plot_title = 'RDM', save_file=None):
    # if rep_type == 'lang': label_buffer = 2
    # if rep_type == 'task': label_buffer = 8
    number_reps=sim_scores.shape[1]/len(TASK_LIST)

    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(10, 8))
    sns.heatmap(sim_scores, yticklabels = '', xticklabels= '', 
                        cmap=cmap, vmin=0, vmax=1, ax=axn, cbar_kws={'label': '1-r'})

    for i, task in enumerate(TASK_LIST):
        plt.text(-2, (number_reps/2+number_reps*i), task, va='center', ha='right', size=5)
        plt.text(number_reps/2+number_reps*i, number_reps*(len(TASK_LIST)), task, va='top', ha='center', rotation='vertical', size=5)
    plt.title(plot_title, fontweight='bold', fontsize=12)

    if save_file is not None: 
        plt.savefig(save_file, dpi=400)

    plt.show()
    

plot_RDM(sim_scores, 'lang')




from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt


for x in range(15):
    plt.plot(noises, np.mean(correct_stats0[:, :, x], axis=0))
plt.legend(labels=list(np.round(multi_diff_strength, 2)))
plt.xlabel('Noise Level')
plt.ylabel('Correct Rate')
plt.show()


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
from instruct_utils import get_task_info
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
from instruct_utils import get_instructions
from tqdm import tqdm
from task_factory import TaskFactory

EXP_FILE = '6.3models/swap_holdouts'
sbertNet = SBERTNet()

holdouts_file = 'swap1'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

def get_hidden_reps(model, num_trials, tasks=TASK_LIST, instruct_mode=None):
    hidden_reps = np.empty((num_trials, 120, 128, len(tasks)))
    with torch.no_grad():
        for i, task in enumerate(tasks): 
            ins, _, _, _, _ =  construct_trials(task, num_trials)

            task_info = get_task_info(num_trials, task, model.is_instruct, instruct_mode=instruct_mode)
            _, hid = model(torch.Tensor(ins).to(model.__device__), task_info)
            hidden_reps[..., i] = hid.cpu().numpy()
    return hidden_reps

hid_reps = get_hidden_reps(sbertNet, 256)

task_var = np.mean(np.var(hid_reps, axis=0), axis=0)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(task_var)
plt.show()
