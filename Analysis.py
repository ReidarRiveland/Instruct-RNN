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
        holdout_only = make_data(task_dict={holdout_task: 1}, BATCH_LEN=256, NUM_BATCHES=100)
        cog.train(holdout_only, 2,  weight_decay=0.0, lr = 0.001, holdout_task=holdout_task, instruct_mode = instruct_mode)
        cog.save_training_data(holdout_task, foldername, 'holdout'+mode)



#train_swaps(model_dict, foldername, mode='swapped')
#foldername = 'TempInstructModels'
#foldername = '15.9MultiCompModels'
foldername = '22.9Models'


# model_dict.keys()
# train_holdouts(model_dict, foldername, mode='shuffled')

# model_dict = {}
# model_dict['Model1'] = None
# #model_dict['BERT_cat'] = None
# # model_dict['GPT_cat'] = None
# model_dict['GPT train'] = None
# #model_dict['S-Bert'] = None
# #model_dict['S-Bert_cat'] = None
# model_dict['S-Bert train'] = None
# # model_dict['BoW'] = None
# # model_dict['SIF'] = None
# #model_dict['S-Bert'] = None
# #model_dict['Transformer']=None
# #model_dict['InferSent train'] = None
# #model_dict['InferSent_cat'] = None

# #plot_all_holdout_curves(model_dict, foldername)


# # plot_all_holdout_curves(model_dict, foldername)

# plot_avg_curves(model_dict, foldername)


# plot_learning_curves(model_dict, ['COMP2', 'MultiDM', 'DMC'], foldername, None)



# plot_learning_curves(model_dict, ['RT Go', 'COMP2', 'MultiDM', 'DMC'], foldername, 'swapped')
# plot_learning_curves(model_dict, ['Go', 'Anti DM', 'Anti RT Go', 'DMC'], foldername, 'swapped')

# cog.plot_response('Model1', 'Go', instruct_mode = 'instruct_swap')



# sBertMod = LangModule(SBERT(20), foldername=foldername)
# sBertMod.train_classifier(64, 100, 10)
# sBertMod.plot_loss('validation')
# sBertMod.load_classifier_training_data()

# bertMod = LangModule(BERT(20), foldername=foldername)
# bertMod.train_classifier(64, 100, 10, lr=0.0001)
# bertMod.plot_loss('validation')


# gptMod = LangModule(gpt2(20), foldername=foldername)
# gptMod.train_classifier(64, 100, 10, lr=0.0001)
# gptMod.plot_loss('validation')
# gptMod.plot_confusion_matrix()



# cog.load_models('Anti RT Go', foldername)
# cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], instruct_mode = None, epoch = 'prep', avg_rep = False, Title = 'S-Bert (end-to-end)')
# cog.plot_task_rep('Model1', dim = 2, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], holdout_task = 'Anti RT Go', epoch = 'prep', avg_rep = False, Title = 'Compositional One Hot Task Encoding')
# plot_learning_curves(model_dict, ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'], foldername, 'comp')

# cog.plot_hid_traj(['Model1', 'S-Bert train'], ['Go'], dim = 3)


# holdout = 'MultiDM'
# cog.load_models(holdout, foldername)
# ax1= cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['DM', 'Anti DM', 'Anti MultiDM', 'MultiDM'], epoch = 'prep', avg_rep = False, Title = 'S-Bert (end-to-end)')
# ax2= cog.plot_task_rep('Model1', dim = 2, tasks = ['DM', 'Anti DM', 'Anti MultiDM', 'MultiDM'], epoch = 'prep', holdout_task = holdout, avg_rep = False, Title = 'Compositional One Hot Task Encoding')
# plot_learning_curves(model_dict, ['DM', 'Anti DM', 'Anti MultiDM', 'MultiDM'], foldername, 'comp')


# holdout = 'COMP2'
# cog.load_models(holdout, foldername)
# ax1 = cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], epoch = 'prep', avg_rep = False, Title = 'S-Bert')
# ax2 = cog.plot_task_rep('Model1', dim = 2, tasks = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], holdout_task = holdout, epoch = 'prep', avg_rep = False, Title = 'One Hot Task Encoding')
# plot_learning_curves(model_dict, ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], foldername, 'comp')

# holdout = 'DMC'
# cog.load_models(holdout, foldername)
# cog.plot_task_rep('S-Bert train', dim = 2, tasks = ['DMS', 'DNMS', 'DMC', 'DNMC'], epoch = 'prep', avg_rep = False,  Title = 'S-Bert (end-to-end)')
# cog.plot_task_rep('Model1', dim = 2, tasks = ['DMS', 'DNMS', 'DMC', 'DNMC'],  instruct_mode ='comp', holdout_task = holdout, epoch = 'prep', avg_rep = False, Title = 'One Hot Task Encoding')
# plot_learning_curves(model_dict, ['DMS', 'DNMS', 'DMC', 'DNMC'], foldername, 'comp')




# import matplotlib.pyplot as plt


# def plot_side_by_side(ax_list, title, ax_titles): 
#     fig, axn = plt.subplots(1, len(ax_list), figsize = (12, 5))
#     plt.suptitle(r'$\textbf{PCA of Preparatory Sensory-Motor Activity (COMP2 Holdout)}$', fontsize=14, fontweight='bold')

#     for i, ax in enumerate(ax_list):
#         ax = axn.flat[i]
#         ax.set_title(ax_titles[i]) 
#         scatter = ax_list[i]
#         ax.scatter(scatter[0], scatter[1], c=scatter[2], cmap=scatter[3], s=scatter[4])
#         ax.set_ylabel('PC1')
#         ax.set_xlabel('PC2')
#         ax.xaxis.set_major_locator(plt.MaxNLocator(3))
#         ax.yaxis.set_major_locator(plt.MaxNLocator(3))        

#     plt.legend(handles = scatter[-1], loc='lower right')
#     plt.show()


# plot_side_by_side([ax1[1], ax2[1]], 'PCA of Preparatory Sensory-Motor Activity (COMP2 Holdout)', ['S-Bert (end-to-end)','One Hot Task Encoding'])



# test = (1, 2, 3, 4)
# test.append(3)

# cog.load_models('Multitask', foldername)
# cog._plot_trained_performance(instruct_mode = 'comp')



# cog.plot_hid_traj(['S-Bert train'], tasks = ['COMP1', 'COMP2'], dim = 3)


# cog.plot_task_rep('Model1', tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'], epoch = 'prep', avg_rep = False)

# cog.plot_task_rep('S-Bert train', dim = 3, epoch = 'prep', avg_rep = False)
# cog.plot_task_rep('S-Bert train', dim = 3, epoch = 'prep', avg_rep = False)

# cog.plot_task_rep('S-Bert train', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = '1', avg_rep = False)
# cog.plot_task_rep('S-Bert train', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = 'prep', avg_rep = False)
# cog.plot_task_rep('Model1', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = 'prep', avg_rep = False)
# cog.plot_task_rep('Model1', dim = 3, tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], epoch = 'prep', avg_rep = False)


# cog.model_dict['S-Bert train'].langMod.plot_embedding(tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'])
# cog.model_dict['GPT train'].langMod.plot_embedding(tasks = ['Go', 'Anti Go', 'Anti RT Go', 'RT Go'])



# cog.load_models('Anti DM', foldername)


# for holdout_task in ['Go']:
#     # model_dict = {}
#     # model_dict['GPT train'] = instructNet(LangModule(gpt2(20), foldername), 128, 1, tune_langModel=True)
#     # cog = CogModule(model_dict)
#     # holdout_data = make_data(holdouts=holdout_task)
#     # cog.train(holdout_data, 25, lr=0.001)
#     # cog.save_models(holdout_task, foldername)
#     # inferSentMod = LangModule(InferSent(20), foldername=foldername)
#     # inferSentMod.loadLangModel()
#     net = simpleNet(81, 128, 1)
#     #gpt_net = instructNet(LangModule(gpt2(20)), 128, 1)
#     #bert_train_net = instructNet(LangModule(BERT(20)), 128, 1, tune_langModel=True)
#     # sBertMod = LangModule(SBERT(20), foldername=foldername)
#     # sBertMod.loadLangModel()
#     #sBert_net = instructNet(sBertMod, 128, 1)
#     train_sBert_net = instructNet(LangModule(SBERT(20)), 128, 1, tune_langModel=True)
#     #train_gpt_net = instructNet(LangModule(gpt2(20)), 128, 1, tune_langModel=True)
#     #SIFnet = instructNet(LangModule(SIFmodel()), 128, 1)
#     #BoWnet = instructNet(LangModule(BoW()), 128, 1)
#     #trans_net = instructNet(TransMod, 128, 1, tune_langModel=True)
#     #gptBig_net = instructNet(gptBig_Mod, 128, 1, tune_langModel=True)
#     model_dict = {}
#     model_dict['Model1'] = net
#     # model_dict['GPT_cat'] = gpt_net
#     # model_dict['BERT_cat'] = bert_net
#     #model_dict['S-Bert_cat'] = sBert_net
#     model_dict['S-Bert train'] = train_sBert_net
#     #model_dict['S-Bert'] = instructNet(LangModule(SBERT(20)), 128, 1)
#     #model_dict['BERT train'] = instructNet(LangModule(BERT(20), foldername), 128, 1, tune_langModel=True)
#     #model_dict['GPT train'] = train_gpt_net
#     # model_dict['SIF'] = SIFnet
#     # model_dict['BoW'] = BoWnet
#     #model_dict['Transformer']=trans_net
#     #model_dict['gptBig'] = gptBig_net
#     #model_dict['SIF'] = instructNet(LangModule(SIFmodel()), 128, 1)
#     # model_dict['InferSent train'] = instructNet(LangModule(InferSent(20)), 128, 1, tune_langModel=True)
#     # model_dict['InferSent_cat'] = instructNet(inferSentMod, 128, 1)
#     cog = CogModule(model_dict)
#     holdout_data = make_data(holdouts=holdout_task)
#     cog.train(holdout_data, 25, lr=0.001)
#     cog.save_models(holdout_task, foldername)


# holdout_data = make_data(NUM_BATCHES = 250, task_dict={'MultiDM':1})
# cog.train(holdout_data, 25, lr=0.001)



net = simpleNet(81, 128, 1)
train_sBert_net = instructNet(LangModule(SBERT(20)), 128, 1, tune_langModel=True)
model_dict = {}
model_dict['Model1'] = net
model_dict['S-Bert train'] = train_sBert_net
cog = CogModule(model_dict)

from Task import construct_batch, Go
import torch
from sklearn.preprocessing import normalize
from LangModule import get_batch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Rectangle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cog.load_models('Anti RT Go', foldername)

modelS = cog.model_dict['S-Bert train']

model1 = cog.model_dict['Model1']


def get_task_var_resp(model, task_type, task_variable, mod, instruct_mode, num_trials=100): 
    assert task_variable in ['direction', 'strength']
    intervals = np.empty((num_trials, 5), dtype=tuple)
    if task_variable == 'direction': 
        directions = np.linspace(0, 2*np.pi, num=num_trials)
        strengths = [1]* num_trials
    elif task_variable == 'strength': 
        directions = np.array([np.pi+1] * num_trials)
        strengths = np.linspace(0.3, 1.8, num=num_trials)
    elif task_variable == 'diff_strength': 
        directions = np.array([np.pi+1] * num_trials)
        strengths = np.linspace(-0.3, 0.3, num=num_trials)

    

    if task_type in ['Go', 'Anti Go', 'RT Go', 'Anti RT Go']:
        stim_mod_arr = np.empty((2, num_trials), dtype=list)
        for i in range(num_trials): 
            intervals[i, :] = ((0, 20), (20, 60), (60, 80), (80, 100), (100, 120))
            strength_dir = [(strengths[i], directions[i])]
            stim_mod_arr[mod, i] = strength_dir
            stim_mod_arr[((mod+1)%2), i] = None
        trials = Go(task_type, num_trials, intervals=intervals, stim_mod_arr=stim_mod_arr, directions=directions)

    # if task_type in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']:
    #     stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)
    #     for i in range(num_trials): 
    #         intervals[i, :] = ((0, 20), (20, 60), (60, 80), (80, 100), (100, 120))
    #         strength_dir = [(strengths[i], directions[i])]
    #         stim_mod_arr[mod, i] = strength_dir
    #         stim_mod_arr[((mod+1)%2), i] = None
    #     trials = DM(task_type, num_trials, intervals=intervals, stim_mod_arr=stim_mod_arr, directions=directions)


    tar_dirs = trials.target_dirs
    tar = trials.targets
    if model.isLang:
        instructions = get_batch(num_trials, model.langModel.tokenizer, task_type=task_type)[0]


    stim_info = np.empty((2, num_trials))
    stim_info.shape
    for i in range(num_trials): 
        temp_stim_info = trials.stim_mod_arr[:, i]
        mod_index = np.where(temp_stim_info != None)[0][0]
        stim_info[0, i] = mod_index
        stim_info[1, i] = temp_stim_info[mod_index][0][task_variable == 'direction']


    h0 = model.initHidden(num_trials, 0.1).to(device)
    model.eval()
    out, hid = cog._get_model_resp(model, num_trials, torch.Tensor(trials.inputs).to(device), task_type, instruct_mode, None)

    hid = hid.detach().cpu().numpy()
    
    return hid

def plot_neural_resp(models, task_type, unit, task_variable, mod, instruct_mode=None):
    assert task_variable in ['direction', 'strength']

    for model in models:

        hid = get_task_var_resp(model, task_type, task_variable, mod, instruct_mode)


        if task_variable == 'direction':
            labels = ["0", "$2\pi$"]
            cmap = plt.get_cmap('twilight') 
        elif task_variable == 'strength':
            labels = ["0.3", "1.8"]
            cmap = plt.get_cmap('plasma') 

        cNorm  = colors.Normalize(vmin=0, vmax=100)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        fig, axn = plt.subplots()
        for i in [x*4 for x in range(25)]:
            plot = plt.plot(hid[i, :, unit], c = scalarMap.to_rgba(i))
        plt.vlines(20, -1.5, 1.15, colors='k', linestyles='dashed')
        plt.vlines(100, -1.5, 1.15, colors='k', linestyles='dashed')

        plt.xticks([20, 100], labels=['Stim. Onset', 'Reponse'])

        axn.set_ylim(-1, 1.15)
        cbar = plt.colorbar(scalarMap, orientation='vertical', label = task_variable, ticks = [0, 100])
        plt.title(task_type + ' response for Unit' + str(unit))

        cbar.set_ticklabels(labels)

        plt.show()



for task in ['Go', 'Anti Go', 'RT Go', 'Anti RT Go']:
    plot_neural_resp([modelS], task, 22, 'direction', 1, instruct_mode=None)





print(j)
fig, axn = plt.subplots()
plt.plot(np.linspace(0, 2*np.pi, num=num_trials), hid[:, 60, j], alpha = normed[i], c = cmap(normed[i]))
axn.set_ylim(-1, 1)
plt.show()



torch.cat((torch.empty(100, 120, 81), torch.zeros(100, 120, 30)), dim=2).shape


# cog = CogModule(model_dict)
# cog.model_dict.keys()
# cog.load_models('Anti DM', foldername)

# from LangModule import train_instruct_dict
# from collections import defaultdict
# import seaborn as sns
# import matplotlib.pyplot as plt


# indices, reps = cog.model_dict['S-Bert train'].langMod._get_instruct_rep(train_instruct_dict)
# indices, reps = cog.model_dict['S-Bert_cat'].langMod._get_instruct_rep(train_instruct_dict)
# indices, reps = LangModule(SBERT(20))._get_instruct_rep(train_instruct_dict)

# rep_dict = defaultdict(list)
# for index, rep in list(zip(indices, reps)): 
#     rep_dict[task_list[index]].append(rep)

# sims = cosine_similarity(rep_dict['COMP1'], rep_dict['COMP2'])

# sims = cosine_similarity(np.array([np.mean(np.array(rep_dict['COMP1']), 0), np.mean(np.array(rep_dict['COMP2']), 0)]))

# sns.heatmap(sims, annot=True, vmin=0, vmax=1)
# plt.title('S-BERT (end-to-end)')
# plt.ylabel('COMP1 Instructions')
# plt.xlabel('COMP2 Instructions')
# plt.show()

# shuffled_dict = {}

# for task, instructs in train_instruct_dict.items(): 
#     instruction_list = []
#     for instruct in instructs: 
#         instruct = instruct.split()
#         shuffled = np.random.permutation(instruct)
#         instruct = ' '.join(list(shuffled))
#         instruction_list.append(instruct)
#     shuffled_dict[task] = instruction_list

# shuffled_dict['Go']


# indices, reps = cog.model_dict['S-Bert train'].langMod._get_instruct_rep(train_instruct_dict)
# shuffled_rep_dict = defaultdict(list)
# for index, rep in list(zip(indices, reps)): 
#     shuffled_rep_dict[task_list[index]].append(rep)
# shuffled_sims = cosine_similarity(rep_dict['DM'], rep_dict['DM'])

# sns.heatmap(shuffled_sims,  annot=True, vmin=0, vmax=1)
# plt.title('Language Representation Similarity Scores (S-BERT train)')
# plt.ylabel('COMP1 Instructions')
# plt.xlabel('COMP2 Instructions')
# plt.show()



# cog.load_models('Anti DM', foldername)


# from Task import construct_batch
# from CogModule import mask_input_rule, isCorrect
# import torch
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import matplotlib


# dim=2
# epoch = 'prep'
# avg_rep = False
# instruct_mode = None
# num_trials = 50
# model = cog.model_dict['Model1']
# holdout_task= None

# tasks = ['Go', 'Anti Go']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# task_reps = []
# correct_list  = []
# for task in task_list: 
#     trials = construct_batch(task, num_trials)
#     tar_dirs = trials.target_dirs
#     tar = trials.targets
#     ins = mask_input_rule(torch.Tensor(trials.inputs), num_trials, 120).to(device)
#     h0 = torch.FloatTensor(num_trials, 128).uniform_(-1, 1).unsqueeze(0).to(device)
#     out, hid = model(ins, h0)

#     correct_list += list(isCorrect(out, tar, tar_dirs))
#     hid = hid.detach().cpu().numpy()
#     epoch_state = []

#     for i in range(num_trials): 
#         if epoch.isnumeric(): 
#             epoch_index = int(epoch)
#             epoch_state.append(hid[i, epoch_index, :])
#         if epoch == 'stim': 
#             epoch_index = np.where(tar[i, :, 0] == 0.85)[0][-1]
#             epoch_state.append(hid[i, epoch_index, :])
#         if epoch == 'response':
#             epoch_state.append(hid[i, -1, :])
#         if epoch == 'input':
#             epoch_state.append(hid[i, 0, :])
#         if epoch == 'prep': 
#             epoch_index = np.where(ins[i, :, 18:]>0.25)[0][0]-1
#             epoch_state.append(hid[i, epoch_index, :])
    
#     if avg_rep: 
#         epoch_state = [np.mean(np.stack(epoch_state), axis=0)]
    
#     task_reps += epoch_state

# embedded = PCA(n_components=dim).fit_transform(task_reps)
# cmap = matplotlib.cm.get_cmap('tab20')

# if avg_rep: 
#     to_plot = np.stack([embedded[task_list.index(task), :] for task in tasks])
#     task_indices = np.array([task_list.index(task) for task in tasks]).astype(int)
#     marker_size = 100
# else: 
#     to_plot = np.stack([embedded[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials, :] for task in tasks]).reshape(len(tasks)*num_trials, dim)
#     correct = np.stack([correct_list[task_list.index(task)*num_trials: task_list.index(task)*num_trials+num_trials] for task in tasks]).flatten()
#     task_indices = np.array([[task_list.index(task)]*num_trials for task in tasks]).astype(int).flatten()
#     marker_size = 25
# tasks
# len(task_indices)
# dots = cmap(task_indices)
# correct = np.where(correct<1, 0.25, correct)
# dots[:, 3] = correct

# plt.scatter(to_plot[:, 0], to_plot[:, 1], c=dots, s=25)
# plt.xlabel("PC 1", fontsize = 18)
# plt.ylabel("PC 2", fontsize = 18)
# plt.show()

