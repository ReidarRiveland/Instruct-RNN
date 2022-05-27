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
# [('Go', 1.0), ('Anti_Go', 1.0), ('RT_Go', 0.70703125), ('Anti_RT_Go', 0.79296875), 
# ('Go_Mod1', 0.99609375), ('Anti_Go_Mod1', 0.35546875), ('Go_Mod2', 0.3203125), ('Anti_Go_Mod2', 0.0546875), 
# ('DelayGo', 0.9921875), ('Anti_DelayGo', 0.93359375), 
# ('DM', 1.0), ('Anti_DM', 1.0), ('MultiDM', 0.96484375), ('Anti_MultiDM', 0.9921875), 
# ('RT_DM', 0.921875), ('Anti_RT_DM', 0.90234375), 
# ('ConDM', 0.8515625), ('Anti_ConDM', 0.90625), ('ConMultiDM', 0.6875), ('Anti_ConMultiDM', 0.7734375), 
# ('DelayDM', 0.921875), ('Anti_DelayDM', 0.859375), ('DelayMultiDM', 0.94140625), ('Anti_DelayMultiDM', 0.578125), 
# ('DM_Mod1', 0.6015625), ('Anti_DM_Mod1', 0.31640625), ('DM_Mod2', 0.61328125), ('Anti_DM_Mod2', 0.12109375), 
# ('COMP1', 0.59375), ('COMP2', 0.84765625), ('MultiCOMP1', 0.6171875), ('MultiCOMP2', 0.7421875), 
# ('COMP1_Mod1', 0.25), ('COMP2_Mod1', 0.2578125), ('COMP1_Mod2', 0.26953125), ('COMP2_Mod2', 0.11328125), 
# ('DMS', 0.671875), ('DNMS', 0.6484375), ('DMC', 0.74609375), ('DNMC', 0.640625)]

#SBERTNET_TUNED
# [('Go', 1.0), ('Anti_Go', 1.0), ('RT_Go', 0.9921875), ('Anti_RT_Go', 0.984375), 
# ('Go_Mod1', 1.0), ('Anti_Go_Mod1', 0.5859375), ('Go_Mod2', 0.94140625), ('Anti_Go_Mod2', 0.1640625), 
# ('DelayGo', 1.0), ('Anti_DelayGo', 0.90234375), 
# ('DM', 1.0), ('Anti_DM', 1.0), ('MultiDM', 0.9921875), ('Anti_MultiDM', 0.97265625), 
# ('RT_DM', 0.94140625), ('Anti_RT_DM', 0.8671875), 
# ('ConDM', 0.859375), ('Anti_ConDM', 0.8515625), ('ConMultiDM', 0.60546875), ('Anti_ConMultiDM', 0.61328125), 
# ('DelayDM', 0.89453125), ('Anti_DelayDM', 0.91796875), ('DelayMultiDM', 0.98828125), ('Anti_DelayMultiDM', 0.55078125), 
# ('DM_Mod1', 0.9453125), ('Anti_DM_Mod1', 0.70703125), ('DM_Mod2', 0.67578125), ('Anti_DM_Mod2', 0.26171875), 
# ('COMP1', 0.734375), ('COMP2', 0.9296875), ('MultiCOMP1', 0.8828125), ('MultiCOMP2', 0.894531)
# ('COMP1_Mod1', 0.1640625), ('COMP2_Mod1', 0.40234375), ('COMP1_Mod2', 0.13671875), ('COMP2_Mod2', 0.09765625),
# ('DMS', 0.6953125), ('DNMS', 0.7109375), ('DMC', 0.65625), ('DNMC', 0.65625)]

from dataset import TaskDataSet

from task_factory import TaskFactory
from models.full_models import SBERTNet, SBERTNet_tuned
from model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from plotting import plot_model_response
from tasks_utils import SWAP_LIST, SWAPS_DICT 
from tasks import TASK_LIST, AntiGoMod1, construct_trials
from instruct_utils import get_instructions
from task_criteria import isCorrect
import numpy as np
import torch



EXP_FILE = '5.25models/swap_holdouts'
sbertNet = SBERTNet_tuned()

sbertNet.rnn_hidden_dim
for n,p in sbertNet.named_parameters(): 
    if p.requires_grad: print(n)

holdouts_file = 'swap7'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/sbertNet_tuned', suffix='_seed0')


def get_zero_shot_perf(model): 
    perf_array = np.empty((40))
    for label, tasks in list(SWAPS_DICT.items())[:-1]:
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

data = TaskDataSet('5.25models/training_data', num_batches = 10, set_single_task='DNMC', stream=False)
ins, tar, mask, tar_dirs, type = next(data.stream_batch())

for index in range(5):
    TaskFactory.plot_trial(ins[index, ...], tar[index, ...], type)


from instruct_utils import train_instruct_dict
repeats = []
for instruct in train_instruct_dict['DMC']:
    perf = task_eval(sbertNet, 'DMC', 128, 
            instructions=[instruct]*128)
    repeats.append((instruct, perf))

repeats

from plotting import plot_model_response

from tasks import AntiDMMod2
trials = AntiDMMod2(128)
plot_model_response(sbertNet, trials, instructions=task_instructions)

task_instructions = get_instructions(128, 'Anti_DM_Mod2', None)
task_instructions[2]

perf = task_eval(sbertNet, 'Anti_DM_Mod2', 128, instructions=task_instructions)

perf

np.mean(repeats)


for task in SWAPS_DICT[holdouts_file]:
    perf = task_eval(sbertNet, task, 256)
    print((task, perf))







resp = get_task_reps(sbertNet)
reps_reduced, _ = reduce_rep(resp)


from model_analysis import get_layer_sim_scores, get_instruct_reps
import seaborn as sns
import matplotlib.pyplot as plt

reps = get_instruct_reps(sbertNet.langModel, depth='full')

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




# repeats = 1
# p_right = np.empty((repeats, num_trials))
# for i in range(repeats): 
#     print(i)
#     task_info = get_task_info(num_trials, 'DM', True)
#     out, _ = sbertNet_tuned(torch.Tensor(trials.inputs).to(sbertNet_tuned.__device__), task_info)
#     p_right[i, :] = isCorrect(out, torch.Tensor(trials.targets), trials.target_dirs)

#     #p_right[i, :] = np.where(isCorrect(out, torch.Tensor(trials.targets), target_dirs), diff_strength > 0, diff_strength<=0)

# np.std(trials.inputs[0, 60:65], axis=1)

# np.mean(p_right)
# abs(np.nansum(trials.conditions_arr[:, 0, 1, :]-trials.conditions_arr[:, 1, 1, :], axis=0)/0.3)>0.8










import numpy as np
from instruct_utils import get_task_info
from task_criteria import isCorrect
from models.full_models import SBERTNet, SBERTNet_tuned
from model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from plotting import plot_model_response
from tasks_utils import SWAPS_DICT 
from tasks import TASK_LIST, construct_trials
import torch

from tasks import Task
from task_factory import DMFactory



from models.full_models import SBERTNet
from instruct_utils import get_instructions

sbertNet = SBERTNet(LM_out_dim=20, rnn_hidden_dim = 128) 

EXP_FILE = '_ReLU128_4.11/swap_holdouts'
holdouts_file = 'Multitask'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/sbertNet', suffix='_seed0')



noises = [0.1, 0.2, 0.3, 0.4, 0.5]
noises
num_repeats = 100
diff_strength = np.linspace(-0.4, 0.4, num=9)
num_trials = len(diff_strength)
pstim1_stats = np.empty((num_repeats, len(noises), num_trials), dtype=bool)
correct_stats = np.empty((num_repeats, len(noises), num_trials), dtype=bool)
for i in range(num_repeats): 
    print(i)
    trial_list = []
    for j, noise in enumerate(noises): 

        conditions_arr = np.full((2, 2, 2, num_trials), np.NaN)
        intervals = np.empty((num_trials, 5), dtype=tuple)
        for k in range(num_trials): 
            intervals[k, :] = ((0, 20), (20, 50), (50, 70), (70, 100), (100, 120))

        directions = np.array([[np.pi/2] * num_trials, [3*np.pi/2] * num_trials])

        fixed_strengths = np.array([1]* num_trials)
        strengths = np.array([fixed_strengths, fixed_strengths-diff_strength])

        target_dirs = np.where([strengths[0, ...] > strengths[1, ...]], directions[0], directions[1]).squeeze()

        conditions_arr[0, :, 0, :] = directions
        conditions_arr[0, :, 1, :] = strengths
        trial = Task(num_trials, noise, DMFactory, str_chooser = np.argmax, intervals=intervals, cond_arr=conditions_arr)
        
        task_instructions = ['respond to the stimulus with greatest strength']*num_trials

        out, hid = sbertNet(torch.Tensor(trial.inputs), task_instructions)
        correct_stats[i, j, :] =  isCorrect(out, torch.Tensor(trial.targets), target_dirs)
        pstim1_stats[i, j, :] =  np.where(isCorrect(out, torch.Tensor(trial.targets), target_dirs), diff_strength > 0, diff_strength<=0)


from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt
for x in range(len(diff_strength)):
    plt.plot(noises, np.mean(correct_stats[:, :, x], axis=0))

plt.legend(labels=list(np.round(diff_strength, 2)))
plt.xlabel('Noise Level')
plt.ylabel('Correct Rate')
plt.show()

for x in range(len(noises)):
    smoothed = gaussian_filter1d(np.mean(pstim1_stats[:, x, :], axis=0), 1)
    plt.plot(diff_strength, smoothed)

plt.legend(labels=list(np.round(noises, 2)))
plt.xlabel('Contrast')
plt.ylabel('p_stim1')
plt.show()











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

