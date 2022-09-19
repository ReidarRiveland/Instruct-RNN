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

EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned()

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
    task_var = np.mean(np.var(hid_reps[:, 30:, :,:], axis=0), axis=0)
    task_var = np.delete(task_var, np.where(np.sum(task_var, axis=1)<0.05)[0], axis=0)
    return normalize(task_var, axis=1, norm='l2')

def sort_units(norm_task_var): 
    labels = cluster_units(norm_task_var)
    cluster_labels, sorted_indices = list(zip(*sorted(zip(labels, range(256)))))
    cluster_dict = defaultdict(list)
    for key, value in zip(cluster_labels, sorted_indices): 
        cluster_dict[key].append(value)

    return cluster_dict, cluster_labels, sorted_indices

hid_reps = get_hidden_reps(sbertNet, 100, tasks= [task for task in TASK_LIST if 'Con' not in task])

norm_task_var = get_norm_task_var(hid_reps)
plot_clustering(norm_task_var)
cluster_dict, cluster_labels, sorted_indices = sort_units(norm_task_var)
plot_task_var_heatmap(norm_task_var[sorted_indices, :], cluster_labels)

cluster_dict[2]

def inactiv_test(model, cluster_dict):
    perf_array = np.empty((len(TASK_LIST), len(cluster_dict)))
    base_perf = get_model_performance(model, batch_len=32)
    for cluster_num, units in cluster_dict.items(): 
        model.set_inactiv_units(units)
        inactiv_perf = get_model_performance(model, batch_len=32)
        perf_array[:, cluster_num]=inactiv_perf-base_perf
    return perf_array


sbertNet.set_inactiv_units(None)
base_perf = get_model_performance(sbertNet, batch_len = 32)

sbertNet.set_inactiv_units(cluster_dict[5])
inactiv_perf = get_model_performance(sbertNet, batch_len = 32)

base_perf

diff = inactiv_perf-base_perf

sns.heatmap(diff[None, ...], xticklabels=TASK_LIST)
plt.show()



cluster_dict[0]
print(list(zip(TASK_LIST, perf)))



from pathlib import Path
def make_batch_slurm(filename,
                     scriptpath,
                     job_name='model_training',
                     partition='shared-cpu',
                     time='01:00:00',
                     condapath=Path('~/miniconda3/'),
                     envname='instruct-rnn',
                     logpath=Path('~/worker-logs/'),
                     cores_per_job=4,
                     memory='16GB',
                     array_size='1-100',
                     f_args=[]):
    fw = open(filename, 'wt')
    fw.write('#!/bin/sh\n')
    fw.write(f'#SBATCH --job-name={job_name}\n')
    fw.write(f'#SBATCH --time={time}\n')
    fw.write(f'#SBATCH --partition={partition}\n')
    fw.write(f'#SBATCH --array={array_size}\n')
    fw.write(f'#SBATCH --output={logpath.joinpath(job_name)}.%a.out\n')
    fw.write('#SBATCH --ntasks=1\n')
    fw.write(f'#SBATCH --cpus-per-task={cores_per_job}\n')
    fw.write(f'#SBATCH --mem={memory}\n')
    fw.write('\n')
    fw.write(f'source {condapath}/bin/activate\n')
    fw.write(f'conda activate {envname}\n')
    fw.write(f'python {scriptpath} {" ".join(f_args)}\n')
    fw.close()
    return

make_batch_slurm('make_dataset_test.sbatch', 'Instruct-RNN/dataset.py', partition='debug-cpu')

# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/reidar/Projects/Instruct-RNN/7.20models

# rsync -a  -P --include '*gptNetXL_FOR_TUNING*' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ /home/riveland/Instruct-RNN/7.20models

#pscp  riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
# rsync -a  -P --include 'simpleNet_seed0.pt' --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/7.20models/swap_holdouts/ C:\Users\reida\OneDrive\Desktop\Projects\Instruct-RNN\7.20models
