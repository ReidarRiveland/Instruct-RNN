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

#SBERTNET
# [('Go', 1.0), ('Anti_Go', 1.0), ('RT_Go', 1.0), ('Anti_RT_Go', 0.3046875), 
# ('Go_Mod1', 0.9375), ('Anti_Go_Mod1', 0.8984375), ('Go_Mod2', 0.4765625), ('Anti_Go_Mod2', 0.6015625), 
# ('DelayGo', 0.9375), ('Anti_DelayGo', 0.953125), 
# ('DM', 1.0), ('Anti_DM', 1.0), ('MultiDM', 0.9375), ('Anti_MultiDM', 0.921875), 
# ('RT_DM', 1.0), ('Anti_RT_DM', 1.0), 
# ('ConDM', 0.9453125), ('Anti_ConDM', 0.921875), ('ConMultiDM', 0.8203125), ('Anti_ConMultiDM', 0.8515625), 
# ('DelayDM', 0.9609375), ('Anti_DelayDM', 0.9765625), ('DelayMultiDM', 0.7109375), ('Anti_DelayMultiDM', 0.5), 
# ('DM_Mod1', 1.0), ('Anti_DM_Mod1', 0.8984375), ('DM_Mod2', 0.9921875), ('Anti_DM_Mod2', 0.1953125), 
# ('RT_DM_Mod1', 0.984375), ('Anti_RT_DM_Mod1', 0.9765625), ('RT_DM_Mod2', 0.9609375), ('Anti_RT_DM_Mod2', 0.8046875), 
# ('COMP1', 0.4609375), ('COMP2', 0.5625), ('MultiCOMP1', 0.3671875), ('MultiCOMP2', 0.46875), 
# ('DMS', 0.3515625), ('DNMS', 0.6015625), ('DMC', 0.4609375), ('DNMC', 0.734375)]


#SBERTNET_TUNED
# [('Go', 1.0), ('Anti_Go', 1.0), ('RT_Go', 0.9765625), ('Anti_RT_Go', 0.4296875), 
# ('Go_Mod1', 0.7890625), ('Anti_Go_Mod1', 0.8125), ('Go_Mod2', 0.296875), ('Anti_Go_Mod2', 0.1015625), 
# ('DelayGo', 1.0), ('Anti_DelayGo', 0.90625), 
# ('DM', 1.0), ('Anti_DM', 1.0), ('MultiDM', 1.0), ('Anti_MultiDM', 0.9453125), 
# ('RT_DM', 1.0), ('Anti_RT_DM', 1.0), 
# ('ConDM', 0.9296875), ('Anti_ConDM', 0.9375), ('ConMultiDM', 0.78125), ('Anti_ConMultiDM', 0.828125), 
# ('DelayDM', 0.84375), ('Anti_DelayDM', 0.9453125), ('DelayMultiDM', 0.6875), ('Anti_DelayMultiDM', 0.4140625), 
# ('DM_Mod1', 1.0), ('Anti_DM_Mod1', 0.9140625), ('DM_Mod2', 0.9921875), ('Anti_DM_Mod2', 0.203125), 
# ('RT_DM_Mod1', 0.984375), ('Anti_RT_DM_Mod1', 0.9765625), ('RT_DM_Mod2', 0.9609375), ('Anti_RT_DM_Mod2', 0.8203125), 
# ('COMP1', 0.625), ('COMP2', 0.8046875), ('MultiCOMP1', 0.6484375), ('MultiCOMP2', 0.5), 
# ('DMS', 0.640625), ('DNMS', 0.84375), ('DMC', 0.2734375), ('DNMC', 0.6875)]


##WEIRD TASKS Go Mods, DELAY DM, Anti_DM_Mod2, DMC

import numpy as np
from instruct_utils import get_task_info
from task_criteria import isCorrect
from models.full_models import SBERTNet, SimpleNetPlus, SBERTNet_tuned, GPTNet
from model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from plotting import plot_model_response
from tasks_utils import SWAPS_DICT 
from tasks import TASK_LIST, construct_trials


EXP_FILE = '5.5models/swap_holdouts'
sbertNet = SBERTNet_tuned()

for n,p in sbertNet.named_parameters(): 
    if p.requires_grad: print(n)

holdouts_file = 'swap7'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/sbertNet_tuned', suffix='_seed0')


def get_zero_shot_perf(model): 
    perf_array = np.empty((40))
    for label, tasks in SWAPS_DICT.items():
        model.load_model(EXP_FILE+'/'+label+'/'+model.model_name, suffix='_seed0')
        for task in tasks: 
            print(task)
            perf = task_eval(model, task, 128) 
            perf_array[TASK_LIST.index(task)] = perf
    return perf_array

perf = get_zero_shot_perf(sbertNet)
list(zip(TASK_LIST, perf))
np.mean(perf) 


from dataset import TaskDataSet
from task_factory import TaskFactory

data = TaskDataSet(num_batches = 10, set_single_task='Anti_DM_Mod2', stream=False)
ins, tar, mask, tar_dirs, type = next(data.stream_batch())

for index in range(5):
    TaskFactory.plot_trial(ins[index, ...], tar[index, ...], type)


from instruct_utils import train_instruct_dict
repeats = []
for instruct in train_instruct_dict['Anti_DM_Mod2']:
    perf = task_eval(sbertNet, 'Anti_DM_Mod2', 128, 
            instructions=[instruct]*128)
    repeats.append(perf)

np.mean(repeats)

from plotting import plot_model_response

from tasks import AntiDMMod2
trials = AntiDMMod2(12)
plot_model_response(sbertNet, trials)



perf = task_eval(sbertNet, 'Anti_RT_Go', 128, 
        instructions=[instruct]*128)


np.mean(repeats)


perf = get_model_performance(sbertNet)
list(zip(TASK_LIST, perf))







resp = get_task_reps(sbertNet)
reps_reduced, _ = reduce_rep(resp)


from model_analysis import get_layer_sim_scores, get_instruct_reps
import seaborn as sns
import matplotlib.pyplot as plt

reps = get_instruct_reps(sbertNet.langModel, depth='full')

sim_scores = get_layer_sim_scores(sbertNet, rep_depth='task')

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




repeats = 1
p_right = np.empty((repeats, num_trials))
for i in range(repeats): 
    print(i)
    task_info = get_task_info(num_trials, 'DM', True)
    out, _ = sbertNet_tuned(torch.Tensor(trials.inputs).to(sbertNet_tuned.__device__), task_info)
    p_right[i, :] = isCorrect(out, torch.Tensor(trials.targets), trials.target_dirs)

    #p_right[i, :] = np.where(isCorrect(out, torch.Tensor(trials.targets), target_dirs), diff_strength > 0, diff_strength<=0)

np.std(trials.inputs[0, 60:65], axis=1)

np.mean(p_right)
abs(np.nansum(trials.conditions_arr[:, 0, 1, :]-trials.conditions_arr[:, 1, 1, :], axis=0)/0.3)>0.8












num_trials = 65
conditions_arr = np.full((2, 2, 2, num_trials), np.NaN)
intervals = np.empty((num_trials, 5), dtype=tuple)
for i in range(num_trials): 
    intervals[i, :] = ((0, 20), (20, 50), (50, 70), (70, 100), (100, 120))

directions = np.array([[np.pi/2] * num_trials, [3*np.pi/2] * num_trials])

#directions = np.array([[np.pi/2] * num_trials, [3*np.pi/2] * num_trials])
fixed_strengths = np.array([1]* num_trials)
diff_strength = np.linspace(-0.3, 0.3, num=num_trials)
strengths = np.array([fixed_strengths, fixed_strengths-diff_strength])

target_dirs = np.where([strengths[0, ...] > strengths[1, ...]], directions[0], directions[1]).squeeze()

diff_strength

trials.conditions_arr

np.nansum(trials.conditions_arr[:, 0, 1, :]-trials.conditions_arr[:, 1, 1, :], axis=0)/0.15

mod=1
conditions_arr[mod, :, 0, : ] = directions
conditions_arr[mod, :, 1, : ] = strengths
trials = Task(128, 'full', noise=0.2, conditions_factory = dm_factory, 
                                    intervals=intervals, target_dirs = target_dirs, conditions_arr=conditions_arr)
trials.task_type = 'DM'
np.linspace(-0.3, 0.3, num=num_trials)[32]
plot_trial(trials.inputs[35, ...], trials.targets[35, ...], 'DM')




import matplotlib.pyplot as plt
plt.plot(np.linspace(-0.3, 0.3, num=num_trials), np.mean(p_right, axis=0))
plt.show()

p_right[:, 30]

from plotting import plot_model_response
plot_model_response(sbertNet_tuned, trials, plotting_index=31)










simpleNetPlus_context = np.empty((16, 20))


simpleNetPlus.eval()
with torch.no_grad():
    for i, task in enumerate(Task.TASK_LIST): 
        rule = get_input_rule(1, task, None)
        simpleNetPlus_context[i, :]=simpleNetPlus.rule_encoder(torch.matmul(rule, simpleNetPlus.rule_transform))

simpleNetPlus_context = np.expand_dims(simpleNetPlus_context, 1)

reps_reduced, _ = reduce_rep(simpleNetPlus_context)
from plotting import plot_rep_scatter
plot_rep_scatter(reps_reduced, Task.TASK_GROUP_DICT['Delay'], s=100)



from perfDataFrame import HoldoutDataFrame

list(range(1))

EXP_FILE = '_ReLU128_4.11'

data = HoldoutDataFrame(EXP_FILE, 'aligned_holdouts', 'sbertNet_tuned', 'correct', seeds=range(1))
data.data
mean(data.get_k_shot(0)[0])


import pickle
data = pickle.load(open('5.5models/Multitask/gptNet/gptNet_seed0_CHECKPOINT_attrs', 'rb'))

import matplotlib.pyplot as plt

from tasks import TASK_LIST

for task in TASK_LIST[14:]: 
    plt.plot(data[task])
    plt.title(task)
    plt.show()

from tasks import TASK_LIST
DEFAULT_TASK_DICT = dict.fromkeys(TASK_LIST, 1/len(TASK_LIST)) 

numer = np.ones(len(TASK_LIST))
TASK_LIST[TASK_LIST.index('MultiCOMP1'):TASK_LIST.index('COMP2_Mod2')+1]
numer[TASK_LIST.index('MultiCOMP1'):TASK_LIST.index('COMP2_Mod2')+1]=2
HARD_TASK_DICT = dict(zip(TASK_LIST, (1/len(TASK_LIST))*numer))

HARD_TASK_DICT