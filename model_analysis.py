import torch
import numpy as np
from torch.nn.modules import transformer

from task import Task
from utils import isCorrect

task_list = Task.TASK_LIST

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_model_performance(model, num_batches): 
    model.eval()
    batch_len = 128
    with torch.no_grad():
        perf_dict = dict.fromkeys(task_list)
        for task in task_list:
            print(task)
            mean_list = [] 
            for _ in range(num_batches): 
                ins, targets, _, target_dirs, _ = next(TaskDataSet(num_batches=1, task_ratio_dict={task:1}).stream_batch())
                task_info = model.get_task_info(batch_len, task)
                out, _ = model(task_info, ins.to(model.__device__))
                mean_list.append(np.mean(isCorrect(out, targets, target_dirs)))
            perf_dict[task] = np.mean(mean_list)
    return perf_dict 

#get_model_performance(model, 5)



def get_instruct_reps(langModel, instruct_dict, depth='full'):
    langModel.eval()
    if depth=='transformer': 
        assert hasattr(langModel, 'transformer'), 'language model must be transformer to evaluate a that depth'
        rep_dim = 768
    else: rep_dim = langModel.out_dim 
    instruct_reps = torch.empty(len(instruct_dict.keys()), len(list(instruct_dict.values())[0]), rep_dim)
    with torch.no_grad():      
        for instructions in instruct_dict.values():
            if depth == 'full': 
                out_rep = langModel(list(instructions))
            elif depth == 'transformer': 
                out_rep = langModel.forward_transformer(list(instructions))
            instruct_reps[0, :, :] = out_rep
    return instruct_reps.cpu().numpy().astype(np.float64)

def get_task_reps(model, epoch='prep', num_trials =100):
    assert epoch in ['stim', 'prep'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
    model.eval()
    with torch.no_grad(): 
        task_reps = np.empty((len(task_list), 100, model.hid_dim))
        for i, task in enumerate(task_list): 
            ins, targets, _, _, _ =  next(TaskDataSet(num_batches=1, batch_len=num_trials, task_ratio_dict={task:1}).stream_batch())

            task_info = model.get_task_info(num_trials, task)
            _, hid = model(task_info, ins.to(model.__device__))

            hid = hid.cpu().numpy()

            for j in range(num_trials): 
                if epoch.isnumeric(): epoch_index = int(epoch)
                if epoch == 'stim': epoch_index = np.where(targets.numpy()[j, :, 0] == 0.85)[0][-1]
                if epoch == 'prep': epoch_index = np.where(ins.numpy()[j, :, 1:]>0.25)[0][0]-1
                task_reps[i, j, :] = hid[j, epoch_index, :]
    return task_reps.astype(np.float64)


def get_hid_var_resp(model, task, trials, num_repeats = 10): 
    model.eval()
    with torch.no_grad(): 
        num_trials = trials.inputs.shape[0]
        total_neuron_response = np.empty((num_repeats, num_trials, 120, 128))
        for i in range(num_repeats): 
            task_info = model.get_task_info(num_trials, task)
            _, hid = model(task_info, torch.Tensor(trials.inputs).to(model.__device__))
            hid = hid.cpu().numpy()
            total_neuron_response[i, :, :, :] = hid
        mean_neural_response = np.mean(total_neuron_response, axis=0)
    return total_neuron_response, mean_neural_response

def reduce_rep(reps, dim=2, reduction_method='PCA'): 
    if reduction_method == 'PCA': 
        embedder = PCA(n_components=dim)
    elif reduction_method == 'tSNE': 
        embedder = TSNE(n_components=2)

    embedded = embedder.fit_transform(reps.reshape(16*reps.shape[1], -1))

    if reduction_method == 'PCA': 
        explained_variance = embedder.explained_variance_ratio_
    else: 
        explained_variance = None

    return embedded.reshape(16, reps.shape[1], dim), explained_variance




import matplotlib.pyplot as plt

def task_cmap(array): 
    all_task_dict = {}
    for task_colors in task_group_colors.values(): 
        all_task_dict.update(task_colors)
    color_list = []

    for index in array: 
        color_list.append(all_task_dict[task_list[index]])

    return color_list

from collections import defaultdict
import itertools
import matplotlib.patches as mpatches


task_group_colors = defaultdict(dict)

task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

task_group_dict = {'Go': ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'],
                'DM': ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], 
                'COMP': ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'],
                'Delay': ['DMS', 'DNMS', 'DMC', 'DNMC']}

#tasks_to_plot = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2']


def plot_rep_reduced(reps_reduced, tasks_to_plot): 
    colors_to_plot = list(itertools.chain.from_iterable([[task_colors[task]]*reps_reduced.shape[1] for task in tasks_to_plot]))
    task_indices = [Task.TASK_LIST.index(task) for task in tasks_to_plot]
    reps_to_plot = reps_reduced[task_indices, ...]
    flattened_reduced = reps_to_plot.reshape(-1, reps_to_plot.shape[-1])
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(flattened_reduced[:, 0], flattened_reduced[:, 1], c = colors_to_plot, s=35)

    plt.xlabel("PC 1", fontsize = 18)
    plt.ylabel("PC 2", fontsize = 18)
    Patches = [mpatches.Patch(color=task_colors[task], label=task) for task in tasks_to_plot]
    plt.legend(handles=Patches, loc=7)
    plt.show()


from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

from rnn_models import InstructNet
from nlp_models import SBERT
from data import TaskDataSet

sim_scores = np.corrcoef(reps)
model1_sim_scores = np.corrcoef(reps_model1)
lang_rep_scores = np.corrcoef(lang_reps)

opp_task_list = ['Go', 'Anti Go', 'RT Go',  'Anti RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM', 'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'DMC', 'DNMC']


map = sns.heatmap(model1_sim_scores,yticklabels = task_list, xticklabels= task_list, vmin=0, vmax=1)
map = sns.heatmap(sim_scores,yticklabels = task_list, xticklabels= task_list, vmin=0, vmax=1)
map = sns.heatmap(lang_rep_scores,yticklabels = task_list, xticklabels= task_list, vmin=0, vmax=1, cmap='GnBu')

for i in range(4):
    plt.axhline(y = 4*i, xmin=i/4, xmax=(i+1)/4, color = 'k',linewidth = 3)
    plt.axhline(y = 4*(i+1), xmin=i/4, xmax=(i+1)/4, color = 'k',linewidth = 3)  
    plt.axvline(x = 4*i, ymin=1-i/4, ymax = 1-(i+1)/4, color = 'k',linewidth = 3)
    plt.axvline(x = 4*(i+1), ymin=1-i/4, ymax = 1-(i+1)/4, color = 'k',linewidth = 3)

plt.show()

model = InstructNet(SBERT(20, train_layers=['11']), 128, 1)
model.model_name+='_seed2'

model.load_model_weights('_ReLU128_14.6/single_holdouts/DNMC')
model.to(torch.device(0))


reps = get_task_reps(model)
reps_reduced, var_explained = reduce_rep(reps, reduction_method='PCA')


# from utils import train_instruct_dict
# instruct_reps = get_instruct_reps(model.langModel, train_instruct_dict)


plot_rep_reduced(reps_reduced, task_group_dict['Delay'])

np.mean(reps, axis=1)


# cmap = matplotlib.cm.get_cmap('tab20')
# Patches = []
# if dim ==3: 
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = [to_plot[:, 0], to_plot[:, 1], to_plot[:,2], cmap(task_indices), cmap, marker_size]
#     ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:,2], c = cmap(task_indices), cmap=cmap, s=marker_size)
#     ax.set_xlabel('PC 1')
#     ax.set_ylabel('PC 2')
#     ax.set_zlabel('PC 3')



# for 


# #plt.suptitle(r"$\textbf{PCA Embedding for Task Representation$", fontsize=18)
# plt.title(Title)
# digits = np.arange(len(tasks))
# plt.tight_layout()
# Patches = [mpatches.Patch(color=cmap(d), label=task_list[d]) for d in set(task_indices)]
# scatter.append(Patches)
# plt.legend(handles=Patches)
# #plt.show()
# return explained_variance, scatter

