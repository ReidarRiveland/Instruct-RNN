import numpy as np
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.models.full_models import *
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval

from instructRNN.tasks.tasks import *
import torch
from instructRNN.models.full_models import SBERTNet
from instructRNN.instructions.instruct_utils import get_instructions
from instructRNN.plotting.plotting import *

EXP_FILE = '7.20models/multitask_holdouts'
holdouts_file = 'Multitask'

clipNet = CLIPNet_lin()
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')
clipNet.to(device)
perf_array = get_model_performance(clipNet, instruct_mode='validation', num_repeats=5)

instructs = ['concentrate only on the second modality and respond to the first direction if spans a greater length of time than the second stimulus otherwise do not respond',
									'focus exclusively on the stimuli in the second modality and respond to the initial direction if it is displayed for a greater length of time than the latter stimulus otherwise do not respond',
									'only consider stimuli which appear in the second modality and select the first orientation if it appears for a greater span of time than the second orientation otherwise do not respond',
									'concentrate only on stimuli which appear in the second modality and choose the initial direction if it lasts for a greater period of time than the second direction otherwise do not respond',
									'attend exclusively to the second modality and respond to the first stimulus if it is presented for a greater span of time than the final stimulus otherwise do not respond']

task_eval(clipNet, 'Dur1Mod2', batch_size=50, instructions=[instructs[3]]*50)

task_eval(gptNet, 'Dur1Mod2', batch_size=50, instructions=[instructs[3]]*50)



list(zip(TASK_LIST, perf_array))


gptNet = GPTNetXL_lin()
gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed0')
gptNet.to(device)
perf_array = get_model_performance(gptNet, instruct_mode='validation')

np.mean(perf_array)

list(zip(TASK_LIST, perf_array))




holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed4')

def get_hidden_reps(model, num_trials, tasks=TASK_LIST, instruct_mode=None):
    hidden_reps = np.empty((num_trials, 150, 256, len(tasks)))
    with torch.no_grad():
        for i, task in enumerate(tasks): 
            trial = construct_trials(task, None)
            ins = trial(num_trials, max_var=True).inputs
            task_info = get_task_info(num_trials, task, model.info_type, instruct_mode=instruct_mode)
            _, hid = model(torch.Tensor(ins).to(model.__device__), task_info)
            hidden_reps[..., i] = hid.cpu().numpy()
    return hidden_reps

from sklearn.preprocessing import normalize


def plot_task_var_heatmap(task_var, cluster_labels, cmap = sns.color_palette("rocket", as_cmap=True)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    label_list = [task for task in TASK_LIST if 'Con' not in task]
    res = sns.heatmap(task_var.T, xticklabels = cluster_labels, yticklabels=label_list, vmin=0, cmap=cmap)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 6)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 4, rotation=0)

    plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from collections import defaultdict


def get_optim_clusters(task_var):
    score_list = []
    for i in range(3,25):
        km = KMeans(n_clusters=i, random_state=12)
        labels = km.fit_predict(task_var)
        score = silhouette_score(task_var, labels)
        score_list.append(score)
    return list(range(3, 50))[np.argmax(np.array(score_list))]

def cluster_units(task_var):
    n_clusters = get_optim_clusters(task_var)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(task_var)
    return labels

def plot_clustering(task_var):
    labels = cluster_units(task_var)
    tSNE = TSNE(n_components=2)
    fitted = tSNE.fit_transform(task_var)
    import matplotlib.pyplot as plt
    plt.scatter(fitted[:, 0], fitted[:, 1], cmap = plt.cm.tab10, c = labels)
    plt.show()

def get_norm_task_var(hid_reps): 
    task_var = np.mean(np.var(hid_reps[:, 30:, :,:], axis=1), axis=1)
    task_var = np.delete(task_var, np.where(np.sum(task_var, axis=0)<0.05)[0], axis=1)
    return normalize(task_var, axis=1, norm='max').T

def sort_units(norm_task_var): 
    labels = cluster_units(norm_task_var)
    cluster_labels, sorted_indices = list(zip(*sorted(zip(labels, range(256)))))
    cluster_dict = defaultdict(list)
    for key, value in zip(cluster_labels, sorted_indices): 
        cluster_dict[key].append(value)

    return cluster_dict, cluster_labels, sorted_indices

hid_reps = get_hidden_reps(sbertNet, 100, tasks= [task for task in TASK_LIST if 'Con' not in task])
#task_hid_reps = get_task_reps(sbertNet, num_trials = 100, epoch=None, tasks= [task for task in TASK_LIST if 'Con' not in task], max_var=True)

norm_task_var = get_norm_task_var(task_hid_reps)
norm_task_var.shape

plot_clustering(norm_task_var)
cluster_dict, cluster_labels, sorted_indices = sort_units(norm_task_var)
plot_task_var_heatmap(norm_task_var[sorted_indices, :], cluster_labels)



# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' --exclude '*_opt*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*gptNetXL_FOR_TUNING*' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin /home/riveland/Instruct-RNN/7.20models/multitask_holdouts/Multitask/gptNet_lin


#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models

rename 'seed5' seed0 *; rename 'seed6' seed1 *; rename 'seed7' seed2 *; rename 'seed8' seed3 *; rename 'seed9' seed4 *