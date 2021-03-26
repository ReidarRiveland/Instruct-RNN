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




import torch.optim as optim
from CogModule import masked_MSE_Loss

s_bert_train = instructNet(LangModule(SBERT(20)), 128, 1, 'sigmoid', tune_langModel=True)

cog = CogModule({'S-Bert train': s_bert_train})

data = make_data(num_batches=200)

model = cog.model_dict['S-Bert train']


model.train
opt = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0)


ins_tensor = data[0]
tar_tensor = data[1]
mask_tensor = data[2]
tar_dir_vec = data[3]
task_type_vec = data[4]

batch_len = ins_tensor.shape[1]
batch_num = ins_tensor.shape[0]
correct_array = np.empty((batch_len, batch_num), dtype=bool)

epochs = 1
opt = optim.Adam(cog.model_dict['S-Bert train'].langModel.model.parameters(), lr=0.0001)

from LangModule import get_batch

sentence_batch = get_batch(128, None, 'Go')[0]

sentence_batch

model = SentenceTransformer('bert-base-nli-mean-tokens')


tokens = model.tokenize(sentence_batch)

out_dict = model(tokens)

out_dict['sentence_embedding'].shape
loss_list = []

for _ in range(1000): 
    lang_ins = cog._get_lang_input(cog.model_dict['S-Bert train'], 128, 'Go', None)

    tokens = cog.model_dict['S-Bert train'].langModel.model.tokenize(lang_ins)
    for keys, values in tokens.items(): 
        tokens[keys] = values.to(device)
    tokens['input_ids'].shape
    lang_out= cog.model_dict['S-Bert train'].langModel.model(tokens)['sentence_embedding']
    #sent_embedding = torch.tensor(lang_out, requires_grad=True).to(device)

    loss = torch.abs(torch.mean(lang_out-torch.zeros_like(lang_out)))

    loss.backward()
    opt.step()

    # for n, p in cog.model_dict['S-Bert train'].langModel.named_parameters():
    #     if 'layer.11' in n:
    #         print(p.grad)
    loss_list.append(loss_list)
    print(loss.item())

s_bert_train.langModel.state_dict().keys()


epochs = 1

opt = optim.Adam(cog.model_dict['S-Bert train'].parameters(), lr=0.001)

for i in range(epochs):
    print('epoch', i)
    index_list = list(np.arange(batch_num))
    np.random.shuffle(index_list)
    for j in range(batch_num): 
        index = index_list[j]
        task_type = task_type_vec[index]
        task_index = task_list.index(task_type)
        tar = torch.Tensor(tar_tensor[index, :, :, :]).to(device)
        mask = torch.Tensor(mask_tensor[index, :, :, :]).to(device)
        ins = torch.Tensor(ins_tensor[index, :, :, :]).to(device)
        tar_dir = tar_dir_vec[index]

        model.langModel.training
        for params in model.parameters(): 
            print(params.requires_grad)
        
        opt.zero_grad()
        out, _ = cog._get_model_resp(model, batch_len, ins, task_type, None)
        loss = masked_MSE_Loss(out, tar, mask)

        for n, p in model.named_parameters(): 
            if 'layer.11' in n:
                p.retain_grad()
        for n, p in model.named_parameters():
            if 'weight_ih' in n:
                p.retain_grad()
        loss.backward()
        for n, p in model.langModel.named_parameters(): 
            if 'layer.11' in n:
                print(p.grad)
        for n, p in model.named_parameters():
            if 'weight_ih' in n:
                print(p.grad)


        if model.isLang:
            torch.nn.utils.clip_grad_value_(model.rnn.rnn.parameters(), 0.5)
        else: 
            torch.nn.utils.clip_grad_value_(model.rnn.parameters(), 0.5)                    
        opt.step()

        frac_correct = np.mean(isCorrect(out, tar, tar_dir))
        cog.total_loss_dict[model_type].append(loss.item())
        cog.total_correct_dict[model_type].append(frac_correct)
        if j%50 == 0:
            print(task_type)
            print(j, ':', model_type, ":", "{:.2e}".format(loss.item()))
            self.sort_perf_by_task()
            print('Frac Correct ' + str(frac_correct) + '\n')
        cog.total_task_list.append(task_type) 
    cog.sort_perf_by_task()
    if scheduler: 
        opt_scheduler.step()               
