import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Plotting import plot_all_holdout_curves, plot_all_tasks_by_model, plot_avg_curves, plot_learning_curves
from LangModule import LangModule, swaps
from NLPmodels import gpt2, BERT, SBERT, BoW, SIFmodel, LangTransformer, InferSent
from RNNs import instructNet, simpleNet
from CogModule import CogModule
from Data import make_data
from Task import Task
task_list = Task.TASK_LIST

def train_swaps(model_dict, foldername, mode = ''):
    cog = CogModule(model_dict)
    if mode == 'swapped': 
        instruct_mode = 'instruct_swap'
    for swap in swaps:
        swapped_tasks = ''.join(swap).replace(' ', '_')
        cog.load_models(swapped_tasks, foldername)
        try: 
            cog.load_training_data(swapped_tasks, foldername, mode + 'holdout')
        except:
            pass
        task_dict = dict(zip(swap, [1/len(swap)]*len(swap)))
        print(task_dict)
        holdout_only = make_data(task_dict=task_dict, BATCH_LEN=256, NUM_BATCHES=120)
        cog.train(holdout_only, 2,  weight_decay=0.0, lr = 0.001, instruct_mode = instruct_mode)
        cog.save_training_data(swapped_tasks, foldername, mode + 'holdout')


def train_holdouts(model_dict, foldername, mode = ''): 
    cog = CogModule(model_dict)
    if mode == '': instruct_mode = None 
    else: instruct_mode = mode
    for holdout_task  in task_list:
        cog.load_models(holdout_task, foldername)
        try: 
            cog.load_training_data(holdout_task, foldername, 'holdout'+mode)
        except:
            pass
        holdout_only = make_data(task_dict={holdout_task: 1}, BATCH_LEN=256, NUM_BATCHES=50)
        cog.train(holdout_only, 2,  weight_decay=0.0, lr = 0.001, holdout_task=holdout_task, instruct_mode = instruct_mode)
        cog.save_training_data(holdout_task, foldername, 'holdout'+mode)


#foldername = 'TempInstructModels'
#foldername = '15.9MultiCompModels'
#foldername = '22.9Models'

train_holdouts(model_dict, foldername, mode='comp')
#train_holdouts(model_dict, foldername)


plot_all_holdout_curves(model_dict, foldername)
model_dict = {}
#model_dict['Model1'] = None
#model_dict['BERT_cat'] = None
#model_dict['GPT_cat'] = None
model_dict['GPT train'] = None
model_dict['S-Bert'] = None
model_dict['S-Bert_cat'] = None
model_dict['S-Bert train'] = None
#model_dict['BoW'] = None
#model_dict['SIF'] = None
model_dict['Transformer']=None
model_dict['InferSent train'] = None
#model_dict['InferSent_cat'] = None
plot_all_holdout_curves(model_dict, foldername)

plot_avg_curves(model_dict, foldername)
plot_learning_curves(model_dict, ['RT Go', 'COMP2', 'MultiDM', 'DMC'], foldername, None)
plot_learning_curves(model_dict, ['Go', 'Anti DM', 'Anti RT Go', 'DMC'], foldername, 'swapped')



sBertMod = LangModule(SBERT(20), foldername=foldername)
sBertMod.train_classifier(64, 100, 10)
sBertMod.plot_loss('validation')
sBertMod.load_classifier_training_data()

bertMod = LangModule(BERT(20), foldername=foldername)
bertMod.train_classifier(64, 100, 10, lr=0.0001)
bertMod.plot_loss('validation')


gptMod = LangModule(gpt2(20), foldername=foldername)
gptMod.train_classifier(64, 100, 10, lr=0.0001)
gptMod.plot_loss('validation')
gptMod.plot_confusion_matrix()

cog.load_models('Anti RT Go', foldername)
cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], instruct_mode = None, epoch = 'prep', avg_rep = False, holdout = 'S-Bert (end-to-end)')
cog.plot_task_rep('GPT train', dim = 2, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], epoch = 'prep', avg_rep = False, holdout = 'GPT (end-to-end)')
cog.plot_task_rep('Model1', dim = 2, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], epoch = 'prep', avg_rep = False, holdout = 'One Hot Task Encoding')

cog.load_models('MultiDM', foldername)
cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['DM', 'Anti DM', 'Anti MultiDM', 'MultiDM'], epoch = 'prep', avg_rep = False, holdout = 'S-Bert (end-to-end)')
cog.plot_task_rep('GPT train', dim = 2, tasks = ['DM', 'Anti DM', 'Anti MultiDM', 'MultiDM'], epoch = 'prep', avg_rep = False, holdout = 'S-Bert (end-to-end)')

cog.plot_task_rep('Model1', dim = 2, tasks = ['DM', 'Anti DM', 'Anti MultiDM', 'MultiDM'], epoch = 'prep', avg_rep = False, holdout = 'One Hot Task Encoding')

cog.load_models('COMP2', foldername)
cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], epoch = 'prep', avg_rep = False,  holdout = 'S-Bert (end-to-end)')
cog.plot_task_rep('Model1', dim = 2, tasks = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], epoch = 'prep', avg_rep = False, holdout = 'One Hot Task Encoding')

cog.load_models('DMC', foldername)
cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['DMS', 'DNMS', 'DMC', 'DNMC'], epoch = 'prep', avg_rep = False,  holdout = 'S-Bert (end-to-end)')
cog.plot_task_rep('Model1', dim = 2, tasks = ['DMS', 'DNMS', 'DMC', 'DNMC'], epoch = 'prep', avg_rep = False, holdout = 'One Hot Task Encoding')

cog.plot_hid_traj(['S-Bert train'], tasks = ['Anti RT Go', 'Anti Go'], dim = 2)


cog.plot_task_rep('Model1', tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], epoch = 'prep', avg_rep = False)

cog.plot_task_rep('S-Bert train', dim = 3, epoch = 'prep', avg_rep = False)
cog.plot_task_rep('S-Bert train', dim = 3, epoch = 'prep', avg_rep = False)

cog.plot_task_rep('S-Bert train', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = '1', avg_rep = False)
cog.plot_task_rep('S-Bert train', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = 'prep', avg_rep = False)
cog.plot_task_rep('Model1', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = 'prep', avg_rep = False)
cog.plot_task_rep('Model1', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = 'prep', avg_rep = False)


cog.model_dict['S-Bert train'].langMod.plot_embedding(tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'])
cog.model_dict['GPT train'].langMod.plot_embedding(tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'])



cog.load_models('Anti DM', foldername)


for holdout_task in task_list:
    # model_dict = {}
    # model_dict['GPT train'] = instructNet(LangModule(gpt2(20), foldername), 128, 1, tune_langModel=True)
    # cog = CogModule(model_dict)
    # holdout_data = make_data(holdouts=holdout_task)
    # cog.train(holdout_data, 25, lr=0.001)
    # cog.save_models(holdout_task, foldername)
    # inferSentMod = LangModule(InferSent(20), foldername=foldername)
    # inferSentMod.loadLangModel()
    net = simpleNet(81, 128, 1)
    #gpt_net = instructNet(LangModule(gpt2(20)), 128, 1)
    # bert_net = instructNet(LangModule(BERT(20)), 128, 1)
    # sBertMod = LangModule(SBERT(20), foldername=foldername)
    # sBertMod.loadLangModel()
    #sBert_net = instructNet(sBertMod, 128, 1)
    train_sBert_net = instructNet(LangModule(SBERT(20)), 128, 1, tune_langModel=True)
    #train_gpt_net = instructNet(LangModule(gpt2(20)), 128, 1, tune_langModel=True, temp_instruct=True)
    #SIFnet = instructNet(LangModule(SIFmodel()), 128, 1)
    #BoWnet = instructNet(LangModule(BoW()), 128, 1)
    #trans_net = instructNet(TransMod, 128, 1, tune_langModel=True)
    #gptBig_net = instructNet(gptBig_Mod, 128, 1, tune_langModel=True)
    model_dict = {}
    model_dict['Model1'] = net
    # model_dict['GPT_cat'] = gpt_net
    # model_dict['BERT_cat'] = bert_net
    #model_dict['S-Bert_cat'] = sBert_net
    model_dict['S-Bert train'] = train_sBert_net
    # #model_dict['BERT train'] = instructNet(LangModule(BERT(20), foldername), 128, 1, tune_langModel=True)
    #model_dict['GPT train'] = train_gpt_net
    # model_dict['SIF'] = SIFnet
    # model_dict['BoW'] = BoWnet
    #model_dict['Transformer']=trans_net
    #model_dict['gptBig'] = gptBig_net
    #model_dict['SIF'] = instructNet(LangModule(SIFmodel()), 128, 1)
    # model_dict['InferSent train'] = instructNet(LangModule(InferSent(20)), 128, 1, tune_langModel=True)
    # model_dict['InferSent_cat'] = instructNet(inferSentMod, 128, 1)
    cog = CogModule(model_dict)
    holdout_data = make_data(holdouts=holdout_task)
    cog.train(holdout_data, 25, lr=0.001)
    cog.save_models(holdout_task, foldername)

cog._plot_trained_performance()

model_dict.keys()

cog = CogModule(model_dict)
cog.model_dict.keys()
cog.load_models('Anti DM', foldername)

from LangModule import train_instruct_dict
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


indices, reps = cog.model_dict['S-Bert train'].langMod._get_instruct_rep(train_instruct_dict)
indices, reps = cog.model_dict['S-Bert_cat'].langMod._get_instruct_rep(train_instruct_dict)
indices, reps = LangModule(SBERT(20))._get_instruct_rep(train_instruct_dict)

rep_dict = defaultdict(list)
for index, rep in list(zip(indices, reps)): 
    rep_dict[task_list[index]].append(rep)

sims = cosine_similarity(rep_dict['COMP1'], rep_dict['COMP2'])

sims = cosine_similarity(np.array([np.mean(np.array(rep_dict['COMP1']), 0), np.mean(np.array(rep_dict['COMP2']), 0)]))

sns.heatmap(sims, annot=True, vmin=0, vmax=1)
plt.title('S-BERT (end-to-end)')
plt.ylabel('COMP1 Instructions')
plt.xlabel('COMP2 Instructions')
plt.show()

shuffled_dict = {}

for task, instructs in train_instruct_dict.items(): 
    instruction_list = []
    for instruct in instructs: 
        instruct = instruct.split()
        shuffled = np.random.permutation(instruct)
        instruct = ' '.join(list(shuffled))
        instruction_list.append(instruct)
    shuffled_dict[task] = instruction_list

shuffled_dict['Go']


indices, reps = cog.model_dict['S-Bert train'].langMod._get_instruct_rep(train_instruct_dict)
shuffled_rep_dict = defaultdict(list)
for index, rep in list(zip(indices, reps)): 
    shuffled_rep_dict[task_list[index]].append(rep)
shuffled_sims = cosine_similarity(rep_dict['DM'], rep_dict['DM'])

sns.heatmap(shuffled_sims,  annot=True, vmin=0, vmax=1)
plt.title('Language Representation Similarity Scores (S-BERT train)')
plt.ylabel('COMP1 Instructions')
plt.xlabel('COMP2 Instructions')
plt.show()











from Task import construct_batch
from LangModule import get_batch
import torch

task = construct_batch('Go', 128)
noise = torch.Tensor(np.stack([[task.make_noise(20) for i in range(120)] for i in range(128)]).squeeze()).to(device)
noise.shape

torch.rand((128, 120, 20)).shape



batch = get_batch(128, None, task_type='Go')[0]
lang_rep = cog.model_dict['S-Bert train'].langMod.langModel(batch)

lang_rep.shape[0]

lang_rep.unsqueeze(1).repeat(1, 5, 1)
noise[:, 0:5, : ] = lang_rep.unsqueeze(1).repeat(1, 5, 1)

noise[:, 5, :]

noise[:, 1, :]


