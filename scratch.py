from collections import defaultdict
from math import inf
import pickle
from model_trainer import config_model
from re import I
from matplotlib.cbook import flatten

from matplotlib.pyplot import axis
from numpy.core.fromnumeric import size, var
from numpy.lib.function_base import append
from numpy.ma import cos
import transformers

from utils import train_instruct_dict
from model_analysis import get_instruct_reps, get_model_performance, get_task_reps, reduce_rep, get_layer_sim_scores, get_hid_var_group_resp, get_hid_var_resp, get_all_CCGP
import numpy as np
from utils import train_instruct_dict, task_swaps_map, all_models
from task import DM
from plotting import plot_RDM, plot_rep_scatter, plot_CCGP_scores, plot_model_response, plot_hid_traj_quiver, plot_dPCA, plot_neural_resp, plot_trained_performance, plot_tuning_curve
import torch

from task import Task, make_test_trials

import torch
from transformers import CLIPTokenizer, GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

model.__dict__
model.config.n_embd

model.lm_head.state_dict()['weight'].shape

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits


cur_sentences = "Hello, my dog is"
for i in range(20): 
    inputs = tokenizer(cur_sentences, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    scores = torch.softmax(logits, dim=-1)
    append_word = tokenizer.batch_decode(torch.max(scores, 2).indices[:, -1])
    cur_sentences+=append_word[0]

cur_sentences

from PIL import Image
import requests
from transformers import CLIPTokenizer, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

instruct = tokenizer(['go in the direction of stimulus with greatest strength'], return_tensors='pt')

model.get_text_features(**instruct).shape


from transformers import CLIPTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

model.config

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

model

outputs = model(**inputs, output_hidden_states=True)
last_hidden_state = outputs.hidden_states

outputs.hidden_states[3].shape
last_hidden_state.shape
pooled_output = outputs.pooler_output  # pooled (EOS token) states
pooled_output.unsqueeze(0)[0].shape


from models.full_models import CLIPNet, BERTNet_tuned, GPTNet
from utils import task_swaps_map
from model_analysis import get_model_performance

gptNet = GPTNet()
EXP_FILE = '13.4models/swap_holdouts'
holdouts = task_swaps_map['Go']
gptNet.load_model(EXP_FILE+'/'+holdouts+'/gptNet', suffix='_seed0_CHECKPOINT')
perf = get_model_performance(gptNet, 1)



from transformers import BertTokenizer, BertForPreTraining
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")
model.state_dict()['cls.seq_relationship.weight'].shape