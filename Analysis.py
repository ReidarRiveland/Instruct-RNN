import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


from LangModule import LangModule, gpt2, BERT, SBERT, Pretrained_Embedder, BoW
from CogModule import CogModule, instructNet, simpleNet
from Data import make_data
from Task import Task

task_list = Task.TASK_LIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sort_by_model(dict): 
    model_sorted = defaultdict(list)
    for task in dict.keys(): 
        for model, avg in dict[task].items():
            model_sorted[model].append(avg)
    return model_sorted

def get_robustness(foldername): 
    robustness_dict = {}
    for holdout_task  in task_list:
        cog.load_models(holdout_task, foldername)
        pre_perf_dict = {}
        for model_name, model in cog.model_dict.items(): 
            pre_perf_dict[model_name] = cog.get_performance(model)
        holdout_only = make_data(BATCH_LEN = 256, task_dict={holdout_task: 1}, NUM_BATCHES=50)
        epochs = 1
        if holdout_task in ['MultiDM', 'Anti MultiDM', 'DMS', 'DNMS', 'DMC', 'DNMC']: 
            epochs = 3
        cog.train(holdout_only, epochs,  weight_decay=0.0, lr = 0.001, holdout_task=holdout_task)
        diff_perf_dict = {}
        for model_name, model in cog.model_dict.items(): 
            post_perf_dict = cog.get_performance(model)
            pre_perf = pre_perf_dict[model_name]
            diff_list = []
            for task in task_list: 
                diff_list.append(post_perf_dict[task] - pre_perf[task])
            diff_perf_dict[model_name]=np.mean(diff_list)
        robustness_dict[holdout_task]=diff_perf_dict
        model_sorted_robustness_dict = sort_by_model(robustness_dict)
    return model_sorted_robustness_dict

def plot_robustness(robustness_dict):
    barWidth = 0.1
    for i, model_name in enumerate(cog.model_dict.keys()):  
        keys = list(cog.model_dict.keys())
        values = robustness_dict[model_name]
        len_values = len(task_list)
        if i == 0:
            r = np.arange(len_values)
        else:
            r = [x + barWidth for x in r]
        plt.bar(r, values, width =barWidth, label = list(cog.model_dict.keys())[i])
    plt.title('Avg. Performance Loss')
    plt.ylim(-1.15, 0)
    plt.xlabel('Holdout Task', fontweight='bold')
    plt.ylabel('Percentage Correct')
    r = np.arange(len_values)
    plt.xticks([r + barWidth for r in range(len_values)], task_list)
    plt.legend()
    plt.show()

def train_holdouts(model_dict, foldername, mode = ''): 
    cog = CogModule(model_dict)
    for holdout_task  in task_list:
        cog.load_models(holdout_task, foldername)
        try: 
            cog.load_training_data(holdout_task, foldername, 'holdout'+mode)
        except:
            pass
        holdout_only = make_data(task_dict={holdout_task: 1}, BATCH_LEN=256, NUM_BATCHES=50)
        cog.train(holdout_only, 2,  weight_decay=0.0, lr = 0.001, holdout_task=holdout_task, instruct_mode = mode)
        cog.save_training_data(holdout_task, foldername, 'holdout'+mode)

def plot_avg_curves(model_dict, foldername, smoothing = 1): 
    fig, ax = plt.subplots(1,1)
    plt.suptitle('Avg. Performance over Tasks')
    ax.set_ylim(0, 1.15)
    cog = CogModule(model_dict)
    avg_perf_dict = {}
    for model_name in model_dict.keys(): 
        avg_perf_dict[model_name] = np.zeros(100)
    for holdout_task in task_list: 
        cog.reset_data()
        cog.load_training_data(holdout_task, foldername, 'holdout')
        for model_type in model_dict.keys():       
            train_data = cog.task_sorted_correct[model_type][holdout_task.replace(' ', '_')]
            avg_perf_dict[model_type]+=np.array(train_data)
            smoothed_perf = gaussian_filter1d(train_data, sigma=smoothing)
            ax.plot(smoothed_perf, color = cog.COLOR_DICT[model_type], alpha=0.1)
    to_legend = []
    for model_type in model_dict.keys(): 
        train_data = avg_perf_dict[model_type]/len(task_list)
        smoothed_perf = gaussian_filter1d(train_data, sigma=smoothing)
        line, = ax.plot(smoothed_perf, color = cog.COLOR_DICT[model_type], alpha=1)
        to_legend.append(line)
    fig.legend(to_legend, model_dict.keys())
    fig.text(0.5, 0.04, 'Batches', ha='center')
    fig.text(0.04, 0.5, 'Fraction Correct', va='center', rotation='vertical')
    fig.show()

def plot_all_holdout_curves(model_dict, foldername, smoothing=1):
    cog = CogModule(model_dict)
    fig, axn = plt.subplots(4,4, sharey = True)
    plt.suptitle('Holdout Learning for All Tasks')
    for i, ax in enumerate(axn.flat):
        ax.set_ylim(0, 1.15)
        holdout_task = task_list[i]
        cog.load_training_data(holdout_task, foldername, 'holdout')
        for model_name in model_dict.keys(): 
            smoothed_perf = gaussian_filter1d(cog.task_sorted_correct[model_name][holdout_task.replace(' ', '_')], sigma=smoothing)
            ax.plot(smoothed_perf, color = cog.COLOR_DICT[model_name])
        ax.set_title(holdout_task)
    fig.legend(cog.model_dict.keys())
    fig.text(0.5, 0.04, 'Batches', ha='center')
    fig.text(0.04, 0.5, 'Fraction Correct', va='center', rotation='vertical')
    fig.show()



foldername = '22.9Models'
#foldername = '15.9MultiCompModels'

train_sBertMod = LangModule(SBERT(20), foldername)

sBert_Mod = LangModule(SBERT(20), foldername)
sBert_Mod.loadLangModel()

bertMod = LangModule(BERT(20), foldername)
bertMod.loadLangModel()

gptMod = LangModule(gpt2(20), foldername)
gptMod.loadLangModel()

sifMod = LangModule(Pretrained_Embedder('SIF'), foldername)

bowMod = LangModule(BoW(), foldername)


model_dict = {}
model_dict['Model1'] = simpleNet(81, 128, 1)
model_dict['BERT_cat'] = instructNet(bertMod, 128, 1)
model_dict['GPT_cat'] = instructNet(gptMod, 128, 1)
model_dict['S-Bert_cat'] = instructNet(sBert_Mod, 128, 1)
model_dict['S-Bert train'] = instructNet(train_sBertMod, 128, 1, tune_langModel=True)
model_dict['BoW'] = instructNet(bowMod, 128, 1)
model_dict['SIF'] = instructNet(sifMod, 128, 1)


model_dict = {}
model_dict['Model1'] = None
model_dict['BERT_cat'] = None
model_dict['GPT_cat'] = None
model_dict['S-Bert_cat'] = None
model_dict['S-Bert train'] = None
model_dict['BoW'] = None
model_dict['SIF'] = None

plot_all_holdout_curves(model_dict, foldername)

holdout_task = 'DNMC'
cog = CogModule(model_dict)
cog.load_training_data(holdout_task, foldername, 'holdout')
cog.plot_learning_curve('correct', task_type=holdout_task, smoothing=1)
cog.task_sorted_correct




hold_only = make_data(task_dict={holdout_task: 1}, BATCH_LEN=256, NUM_BATCHES=50)
cog.train(hold_only, 2, holdout_task=holdout_task)

cog.plot_learning_curve('correct', task_type=holdout_task, smoothing=1)

train_holdouts(model_dict, foldername)

cog = CogModule(model_dict)
cog.load_training_data('Go', foldername, 'holdout')
cog.total_correct_dict


for swaps in [['Anti RT Go', 'DMC'], ['COMP2', 'RT Go']]:
    net = simpleNet(81, 128, 1)
    gpt_net = instructNet(gptMod, 128, 1)
    bert_net = instructNet(bertMod, 128, 1)
    sBert_net = instructNet(sBert_Mod, 128, 1)
    train_sBert_net = instructNet(train_sBertMod, 128, 1, tune_langModel=True)
    SIFnet = instructNet(sifMod, 128, 1)
    BoWnet = instructNet(bowMod, 128, 1)
    model_dict = {}
    model_dict['Model1'] = net
    model_dict['GPT_cat'] = gpt_net
    model_dict['BERT_cat'] = bert_net
    model_dict['S-Bert_cat'] = sBert_net
    model_dict['S-Bert train'] = train_sBert_net
    model_dict['SIF'] = SIFnet
    model_dict['BoW'] = BoWnet
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=swaps)
    cog.train(holdout_data, 15, lr=0.001)
    cog.train(holdout_data, 5, lr=0.0005)
    #cog.train(holdout_data, 5, lr=0.0001)
    cog.save_models(''.join(swaps).replace(' ', '_'), foldername)


swaps= [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['COMP2', 'RT Go']]

''.join(swaps[0]).replace(' ', '_')

net = simpleNet(81, 128, 1)
SIFnet = instructNet(sifMod, 128, 1)
BoWnet = instructNet(bowMod, 128, 1)


model_dict['Model1'] = net
model_dict['SIF'] = SIFnet
model_dict['BoW'] = BoWnet




cog.plot_trained_performance()

train_sBertMod.plot_embedding('PCA', dim=3, tasks = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], train_only=True)


hold_only = make_data(task_dict={holdout_task: 1}, BATCH_LEN=256, NUM_BATCHES=50)
cog.train(hold_only, 1, holdout_task=holdout_task)

cog.plot_learning_curve('correct', task_type=holdout_task, smoothing=1)
cog.plot_k_shot_learning([0, 3], holdout_task)

from Task import construct_batch
test = construct_batch('Go', 1)






array = np.arange(0, 16)
test = dict(zip(array, task_list))

dict((num, test[num]) for num in np.arange(0, 7))






cog.load_models('Go', foldername)

cog.plot_response(model_dict['S-Bert train'], 'DMS', 'respond in the opposite direction if the stimuli match')




cog.model_dict.keys()

trainMod= LangModule(cog.model_dict['S-Bert train'].langModel)
trainMod.plot_embedding('PCA', dim=3)

sBMod = LangModule(cog.model_dict['S-Bert'].langModel)
sBMod.plot_embedding('PCA', dim=3)


from LangModule import train_instruct_dict
from LangModule import get_batch

shuffledDM = get_batch(10, None, 'DM', instruct_mode='shuffled')[0]
shuffledGo = get_batch(10, None, 'Go', instruct_mode='shuffled')[0]


shuffledReps = []
shuffledGoReps = []

for instruct in shuffledGo: 
    rep = trainMod.langModel([instruct])
    shuffledReps.append(rep.squeeze().cpu().detach().numpy())

shuffledReps
shuffledReps[0]
sims = cosine_similarity(shuffledReps)

sns.heatmap(sims,  annot=True, vmin=0, vmax=1)
plt.title('Language Representation Similarity Scores (S-BERT train)')
plt.ylabel('Shuffled DM Instructions')
plt.xlabel('Shuffled Go Instructions')
plt.show()


indices, reps = trainMod._get_instruct_rep(train_instruct_dict)
rep_dict = defaultdict(list)
for index, rep in list(zip(indices, reps)): 
    rep_dict[task_list[index]].append(rep.cpu().detach().numpy())

sims = cosine_similarity(rep_dict['COMP1'], rep_dict['COMP2'])

sns.heatmap(sims,  annot=True, vmin=0, vmax=1)
plt.title('Language Representation Similarity Scores (S-BERT train)')
plt.ylabel('COMP1 Instructions')
plt.xlabel('COMP2 Instructions')
plt.show()

cosine_similarity([comp2_avg, multicomp2_avg])
cosine_similarity([comp2_avg, comp1_avg-dif])


holdout_task = 'MultiCOMP2'
cog.load_models(holdout_task, foldername)

hold_only = make_data(task_dict={holdout_task: 1}, BATCH_LEN=256, NUM_BATCHES=50)
cog.train(hold_only, 1, holdout_task=holdout_task, instruct_mode='shuffled')

cog.plot_learning_curve('correct', task_type=holdout_task, smoothing=1)
cog.plot_k_shot_learning([0, 3], holdout_task)




color_dict = {'Model1': 'blue', 'Model1 Masked': 'orange', 'GPT': 'green', 'BERT': 'red', 'S-BERT': 'purple', 'S-BERT train': 'brown'}

{'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}

cog.plot_trained_performance()



from LangModule import train_instruct_dict
import seaborn as sns
indices, reps = sBertMod._get_instruct_rep(train_instruct_dict)
rep_dict = defaultdict(list)
for index, rep in list(zip(indices, reps)): 
    rep_dict[task_list[index]].append(rep.cpu().numpy())

avg_rep_dict = {}
for task in task_list:
    avg_rep_dict[task] = np.mean(np.array(rep_dict[task]), axis=0)

avg_rep_dict

sims = cosine_similarity(np.array(list(avg_rep_dict.values())))



sns.heatmap(sims,  yticklabels = task_list, xticklabels= task_list,  annot=True)
plt.title('Language Representation Similarity Scores')
plt.show()

