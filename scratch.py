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
from task_factory import dm_factory, plot_trial
import numpy as np
from tasks import Task, MultiDM, ConDM
from utils.task_info_utils import get_task_info
from task_criteria import isCorrect
from models.full_models import SBERTNet, SimpleNetPlus, SBERTNet_tuned

low_high = (0,1)
np.random.uniform(*low_high)

from utils.task_info_utils import get_input_rule
from model_analysis import task_eval
import torch

EXP_FILE = '_ReLU128_4.11/swap_holdouts'
sbertNet_tuned = SBERTNet_tuned()
holdouts_file = 'Multitask'
sbertNet_tuned.load_model(EXP_FILE+'/'+holdouts_file+'/sbertNet_tuned', suffix='_seed0')

num_trials = 128
trials = ConDM(num_trials)
task_eval(sbertNet_tuned, 'DM', 128, noise=0.25)

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