import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Plotting import plot_all_holdout_curves, plot_all_tasks_by_model, plot_avg_curves, plot_learning_curves
from LangModule import LangModule, swaps
from NLPmodels import gpt2, BERT, SBERT, BoW, SIFmodel, LangTransformer
from RNNs import instructNet, simpleNet
import torch

from CogModule import CogModule, isCorrect
from Data import make_data
from Taskedit import Task
task_list = Task.TASK_LIST

foldername = 'swapModel3.3'

swaps= [['Go', 'Anti DM'], ['Anti RT Go', 'DMC']]

epochs = 40
lr = 0.001
milestones = [30, 35]

for swap in swaps:   
    swapped_tasks = ''.join(swap).replace(' ', '_')
    print(swapped_tasks)
    model_dict = {}
    model_dict['Model1'] = simpleNet(81, 128, 1, 'tanh')
    model_dict['Model1shuffled'] = simpleNet(81, 128, 1, 'tanh', instruct_mode='shuffled_one_hot')
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=swap)
    cog.train(holdout_data, epochs,  weight_decay=0.0, lr = lr)
    cog.save_models(swapped_tasks, foldername)




model_dict = {}
model_dict['Model1'] = simpleNet(81, 128, 1, 'sigmoid')
model_dict['S-Bert train'] = simpleNet(81, 128, 1, 'sigmoid', instruct_mode='shuffled_one_hot') 
cog = CogModule(model_dict)




foldername = 'swapModel3.3'

model_dict = {}
model_dict['Model1'] = None
model_dict['Model1shuffled'] = None
cog = CogModule(model_dict)

task = 'Anti RT GoDMC'
cog.load_training_data(task, foldername, 'holdout')
cog.plot_learning_curve('correct', task)

cog.load_training_data(task, foldername, 'holdoutAllLR0.005')
cog.plot_learning_curve('correct', task)


for swap in swaps:
    swapped_tasks = ''.join(swap).replace(' ', '_')
    model_dict = {}
    model_dict['Model1'] = simpleNet(81, 128, 1, 'tanh')
    model_dict['Model1shuffled'] = simpleNet(81, 128, 1, 'tanh', instruct_mode='shuffled_one_hot')
    cog = CogModule(model_dict)
    cog.load_models(swapped_tasks, foldername)
    try: 
        cog.load_training_data(swapped_tasks, foldername, mode + 'holdout')
    except:
        pass
    task_dict = dict(zip(swap, [1/len(swap)]*len(swap)))
    print(task_dict)
    holdout_only = make_data(task_dict=task_dict, num_batches = 100, batch_size=256)
    cog.train(holdout_only, 2,  weight_decay=0.0, lr = 0.001, instruct_mode = 'instruct_swap')
    cog.save_training_data(swapped_tasks, foldername, 'swap_holdout')


model_dict = {}
model_dict['Model1'] = simpleNet(81, 128, 1, 'tanh')
model_dict['Model1shuffled'] = simpleNet(81, 128, 1, 'tanh', instruct_mode='shuffled_one_hot')
cog = CogModule(model_dict)
cog.load_models('GoAnti_DM', foldername)
cog.load_training_data('GoAnti_DM', foldername, 'holdout')
cog._plot_trained_performance()

cog.plot_learning_curve('correct', 'Anti DM')


foldername = '22.9Models'



import torch.nn as nn
model_dict = {}
model_dict['S-Bert train'] = instructNet(LangModule(SBERT(20, nn.Sigmoid())), 128, 1, 'relu', tune_langModel=True)

model_dict['S-Bert train'] = None

plot_avg_curves(model_dict, foldername, smoothing=0.0001)

print(model_dict['S-Bert train'].langModel.state_dict()['model.0.bert.encoder.layer.11.attention.output.dense.weight'].grad)

cog = CogModule(model_dict)

cog.load_models('DM', foldername)

cog.plot_response('S-Bert train', 'DM')

cog._plot_trained_performance()

stock_params = SBERT(20).state_dict()
trained_params = model_dict['S-Bert train'].langModel.state_dict()

layer_list = ['layer.{layer_num}'.format(layer_num=layer) for layer in np.arange(12)]

with torch.no_grad():
    difference_dict = {}
    for n, p in model_dict['S-Bert train'].langModel.named_parameters(): 
        difference_dict[n] = torch.mean(torch.log(torch.abs(p - SBERT(20).state_dict()[n]))).cpu().numpy()

difference_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

len(list(model_dict['S-Bert train'].rnn.parameters()))





s_bert_train = instructNet(LangModule(SBERT(20)), 128, 1, 'sigmoid', tune_langModel=True)


for param in s_bert_train.langModel.parameters(): 
    print(param.requires_grad)

for param in s_bert_train.langModel.parameters():
    print(param.requires_grad)



cog = CogModule({'S-Bert train': s_bert_train})

data = make_data(num_batches=200)

s_bert_train.langModel.state_dict()['model.0.bert.encoder.layer.11.attention.self.query.weight'].requires_grad

s_bert_train.langModel.training

cog.train(data, 1)

ins = make_data(batch_size=1, num_batches=1)

resp = cog._get_model_resp(s_bert_train, 1, torch.Tensor(ins[0][0, :, :, :]).to(device), 'Go', None)

resp[1].shape

correct = isCorrect(resp[0], torch.Tensor(ins[1][0, :, :, :]), ins[3][0])

str(correct[0])

torch.Tensor(size=(128, 65, 120))

isCorrect


import torch.optim as optim


s_bert_train.langModel.train()

for i in range(1200): 
    lang_inputs = cog._get_lang_input(s_bert_train, 128, 'Go', None)
    opt = optim.Adam(s_bert_train.langModel.parameters())
    opt.zero_grad()
    out = s_bert_train.langModel(lang_inputs)
    loss = torch.mean(out-torch.zeros_like(out))
    loss.backward()
    opt.step()
    loss.item()

list(s_bert_train.langModel.parameters())

# opt.zero_grad()
# out, _ = self._get_model_resp(model, batch_len, ins, task_type, instruct_mode)
# loss = masked_MSE_Loss(out, tar, mask) 
# loss.backward()
# opt.step()