from turtle import position
from instructRNN.data_loaders.dataset import TaskDataSet

from instructRNN.tasks.task_factory import DELTA_T, TaskFactory
from instructRNN.models.full_models import *
from instructRNN.analysis.model_analysis import *
from instructRNN.tasks.tasks import *
from instructRNN.instructions.instruct_utils import get_instructions, train_instruct_dict
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.plotting.plotting import *
import numpy as np
import torch




EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)


holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')
plot_scatter(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1',  'GoMod2', 'AntiGoMod2'], dims=3, pcs=[1, 2, 3], num_trials=50, rep_depth='12')



# list(SWAPS_DICT.items())[0]

# from instructRNN.trainers.model_trainer import *
# test_model('7.20models', 'gptNetXL', '6', list(SWAPS_DICT.items())[0], stream_data=True)


# plot_all_holdout_curves('7.20models', 'swap', ['sbertNet_lin_tuned', 'simpleNet'], seeds=range(5))



# plot_k_shot_learning('7.20models', 'swap', ['simpleNet', 'bowNet', 'clipNet', 'clipNet_tuned','bertNet', 'bertNet_tuned', 'sbertNet', 'sbertNet_tuned', 'sbertNet_lin', 'sbertNet_lin_tuned'], seeds=range(4))

data = HoldoutDataFrame('7.20models', 'swap', 'gptNetXL', seeds=range(5))

np.nanmean(data.get_k_shot(0))



# EXP_FILE = '7.20models/swap_holdouts'
# #sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
# sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)

# holdouts_file = 'swap0'
# sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

# #plot_scatter(sbertNet, ['MultiDM', 'AntiMultiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2])

# plot_hid_traj(sbertNet, ['MultiDM', 'AntiMultiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], pcs=[0, 1, 2], s=7)

# instruct_reps = get_instruct_reps(sbertNet.langModel)

# avg_reps = np.mean(instruct_reps, axis=1)

# avg_reps.shape

# u, s, v = np.linalg.svd(avg_reps)

# s.shape
# plt.bar(range(50), s)
# plt.show()


# repeats = []
# for instruct in train_instruct_dict['Dur1Mod1']:
#     perf = task_eval(sbertNet, 'AntiDur2', 128, 
#             instructions=[instruct]*128)
#     repeats.append((instruct, perf))

# repeats



# def get_zero_shot_perf(model): 
#     perf_array = np.full(len(TASK_LIST), np.NaN)
#     for label, tasks in list(SWAPS_DICT.items()):
#         model.load_model('7.20models/swap_holdouts/'+label+'/'+model.model_name, suffix='_seed0')
#         for task in tasks: 
#             print(task)
#             perf = task_eval(model, task, 256) 
#             perf_array[TASK_LIST.index(task)] = perf
#     return perf_array

# sbertNet.to(torch.device(0))
# perf = get_zero_shot_perf(sbertNet)
# perf
# list(zip(TASK_LIST, perf))
# list(zip(TASK_LIST, data.get_k_shot(0)[0]))
# np.nanmean(perf) 




# EXP_FILE = '7.16models/multitask_holdouts/Multitask'
# simpleNet = GPTNet()
# simpleNet.load_model(EXP_FILE+'/'+simpleNet.model_name, suffix='_seed0')

# simpleNet.state_dict().keys()

# task_eval(simpleNet, 'AntiDM', 128)

# diff_strength = np.concatenate((np.linspace(-0.2, -0.05, num=7), np.linspace(0.05, 0.2, num=7)))
# #@diff_strength = np.concatenate((np.linspace(-0.2, -0.1, num=7), np.linspace(0.1, 0.2, num=7)))
# #noises = np.linspace(0.15, 0.75, num=20)
# noises = np.linspace(0.15, 0.75, num=20)


# correct_stats, pstim1_stats, trial = get_DM_perf(simpleNet, noises, diff_strength, task='AntiDM')
# trial.task_type = 'DM'
# plot_model_response(simpleNet, trial, plotting_index=5)

# from scipy.ndimage.filters import gaussian_filter1d
# import matplotlib.pyplot as plt

# for x in range(15):
#     plt.plot(noises, np.mean(correct_stats[:, :, x], axis=0))
# plt.legend(labels=list(np.round(diff_strength, 2)))
# plt.xlabel('Noise Level')
# plt.ylabel('Correct Rate')
# plt.show()
# thresholds = get_noise_thresholdouts(correct_stats, diff_strength, noises, neg_cutoff=0.85)


# path = '/home/reidar/Projects/Instruct-RNN/instructRNN/tasks/noise_thresholds'
# pickle.dump(thresholds, open(path+'/anti_dm_noise_thresholds', 'wb'))


# #THIS ISNT EXACTLY RIGHT BECAUSE YOU ARE COUNTING INCOHERENT ANSWERS AS ANSWER STIM2
# for x in range(10):
#     smoothed = gaussian_filter1d(np.mean(pstim1_stats[:, x, :], axis=0), 1)
#     plt.plot(diff_strength, smoothed)

# plt.legend(labels=list(np.round(noises, 2)[:10]))
# plt.xlabel('Contrast')
# plt.ylabel('p_stim1')
# plt.show()


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

holdouts_file = 'swap2'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

task_eval(sbertNet, 'AntiDM', 128)

trials = GoMod1(100, max_var=True)
plot_tuning_curve(sbertNet, ['Go'], 0, [50]*4, np.linspace(0, 2*np.pi, num=100))

sim_scores = get_layer_sim_scores(sbertNet, rep_depth='task')
plot_RDM(sim_scores,  cmap=sns.color_palette("Blues", as_cmap=True))


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
