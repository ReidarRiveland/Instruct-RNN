# from instructRNN.tasks.tasks import *
# trials = MultiDur1(100)
# trials.plot_trial(0)
# trials.factory.intervals[:, [1,3], 0, 0]-trials.factory.intervals[:, [1,3], 1, 0]



# import numpy as np
# from instructRNN.tasks.tasks import *
# from instructRNN.analysis.model_analysis import *
# from instructRNN.models.full_models import BoWNet


# _intervals = np.array([(0, 30), (30, 60), (60, 90), (90, 130), (130, TRIAL_LEN)])
# intervals = np.repeat(_intervals[:,:, None],  100, axis=-1)

#np.repeat(np.repeat(_intervals[..., None], 100, axis=-1)[None, ...], 2, axis=0)

# EXP_FILE = '7.3models'
# sbertNet = BoWNet()

# holdouts_file = 'multitask_holdouts/Multitask'
# sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

# noises = np.linspace(0.2, 0.6, 10)
# contrasts = np.concatenate((np.linspace(-0.1, -0.01, 5), np.linspace(0.01, 0.1, 5)))

# correct, _, _ = get_DM_perf(sbertNet, noises, contrasts)

# from matplotlib import pyplot as plt

# from scipy.ndimage.filters import gaussian_filter1d
# import matplotlib.pyplot as plt

# for x in range(len(contrasts)):
#     plt.plot(noises, np.mean(correct[:, :, x], axis=0))
# plt.legend(labels=list(np.round(contrasts, 2)))
# plt.xlabel('Noise Level')
# plt.ylabel('Correct Rate')
# plt.show()




# #THIS ISNT EXACTLY RIGHT BECAUSE YOU ARE COUNTING INCOHERENT ANSWERS AS ANSWER STIM2
# for x in range(10):
#     smoothed = gaussian_filter1d(np.mean(pstim1_stats[:, x, :], axis=0), 1)
#     plt.plot(diff_strength, smoothed)
# plt.legend(labels=list(np.round(noises, 2)[:10]))
# plt.xlabel('Contrast')
# plt.ylabel('p_stim1')
# plt.show()

from turtle import position
from instructRNN.data_loaders.dataset import TaskDataSet


from instructRNN.tasks.task_factory import DELTA_T, TaskFactory
from instructRNN.models.full_models import *
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval
from instructRNN.tasks.tasks import SWAP_LIST, SWAPS_DICT, TASK_LIST 
from instructRNN.instructions.instruct_utils import get_instructions, train_instruct_dict
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.plotting.plotting import *
import numpy as np
import torch

plot_all_holdout_curves('7.11models', 'swap', ['sbertNet_lin_tuned', 'bowNet'])
data = HoldoutDataFrame('7.11models', 'swap', 'bowNet')



EXP_FILE = '7.11models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap4'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

#sbertNet = SimpleNet(rnn_hidden_dim=256)

plot_model_response(sbertNet, Dur2Mod2(100), 0)


test = np.random.randn(2,2)

test[1]

task = 'Anti_Go_Mod2'
instructions = ['pick the opposite of the orientation in the second modality' ]*128
instructions[0] in train_instruct_dict[task]
task_eval(sbertNet, 'AntiMultiCOMP1', 128)



repeats = []
for instruct in train_instruct_dict['AntiMultiCOMP1']:
    perf = task_eval(sbertNet, 'AntiMultiCOMP1', 128, 
            instructions=[instruct]*128)
    repeats.append((instruct, perf))

repeats



def get_zero_shot_perf(model): 
    perf_array = np.empty(len(TASK_LIST))
    for label, tasks in list(SWAPS_DICT.items()):
        model.load_model(EXP_FILE+'/'+label+'/'+model.model_name, suffix='_seed0')
        for task in tasks: 
            print(task)
            perf = task_eval(model, task, 256) 
            perf_array[TASK_LIST.index(task)] = perf
    return perf_array

sbertNet.to(torch.device(0))
perf = get_zero_shot_perf(sbertNet)
perf
list(zip(TASK_LIST, perf))
np.mean(perf) 



resp = get_task_reps(sbertNet)
reps_reduced, _ = reduce_rep(resp)


reps = get_instruct_reps(sbertNet.langModel, depth='12')
reps.shape
np.max(reps[TASK_LIST.index('DM'), 0, :])
np.min(reps)


sim_scores = get_layer_sim_scores(sbertNet, rep_depth='full')
plot_RDM(sim_scores)



from models.full_models import SimpleNet
from model_analysis import get_DM_perf, get_noise_thresholdouts
import pickle
import numpy as np
EXP_FILE = '6.6models/noise_thresholding_model'

simpleNet = SimpleNet(rnn_hidden_dim=256)
simpleNet.load_model(EXP_FILE+'/'+simpleNet.model_name, suffix='_seed0')

task = 'DM'


#diff_strength = np.concatenate((np.linspace(-0.15, -0.05, num=7), np.linspace(0.05, 0.15, num=7)))
diff_strength = np.concatenate((np.linspace(-0.2, -0.1, num=7), np.linspace(0.1, 0.2, num=7)))

noises = np.linspace(0.05, 0.75, num=30)

correct_stats, pstim1_stats, trial = get_DM_perf(simpleNet, noises, diff_strength, task=task)

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

for x in range(15):
    plt.plot(noises, np.mean(correct_stats[:, :, x], axis=0))
plt.legend(labels=list(np.round(diff_strength, 2)))
plt.xlabel('Noise Level')
plt.ylabel('Correct Rate')
plt.show()

thresholds = get_noise_thresholdouts(correct_stats, diff_strength, noises, neg_cutoff=0.8)

pickle.dump(thresholds, open('6.7models/noise_thresholds/dm_noise_thresholds', 'wb'))


#THIS ISNT EXACTLY RIGHT BECAUSE YOU ARE COUNTING INCOHERENT ANSWERS AS ANSWER STIM2
for x in range(10):
    smoothed = gaussian_filter1d(np.mean(pstim1_stats[:, x, :], axis=0), 1)
    plt.plot(diff_strength, smoothed)

plt.legend(labels=list(np.round(noises, 2)[:10]))
plt.xlabel('Contrast')
plt.ylabel('p_stim1')
plt.show()


# simpleNetPlus_context = np.empty((16, 20))
# simpleNetPlus.eval()
# with torch.no_grad():
#     for i, task in enumerate(Task.TASK_LIST): 
#         rule = get_input_rule(1, task, None)
#         simpleNetPlus_context[i, :]=simpleNetPlus.rule_encoder(torch.matmul(rule, simpleNetPlus.rule_transform))

# simpleNetPlus_context = np.expand_dims(simpleNetPlus_context, 1)

# reps_reduced, _ = reduce_rep(simpleNetPlus_context)
# from plotting import plot_rep_scatter
# plot_rep_scatter(reps_reduced, Task.TASK_GROUP_DICT['Delay'], s=100)


import numpy as np
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.models.full_models import SBERTNet_tuned, SimpleNet, CLIPNet
from instructRNN.analysis.model_analysis import get_model_performance, get_task_reps, reduce_rep, task_eval

from instructRNN.tasks.tasks import *
import torch
from instructRNN.models.full_models import SBERTNet
from instructRNN.instructions.instruct_utils import get_instructions

EXP_FILE = '6.7models/swap_holdouts'
sbertNet = SBERTNet(LM_output_nonlinearity='relu')

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


from instructRNN.plotting.plotting import plot_model_response
from instructRNN.tasks.tasks import DMS


task_instructions = get_instructions(128, 'Anti_Go_Mod1', None)
plot_model_response(sbertNet, AntiDM(100))


CLUSTER_TASK_LIST = ['Go', 'RTGo', 'GoMod1','GoMod2',
                    'AntiGo',  'AntiRTGo', 'AntiGoMod1',  'AntiGoMod2',
                    'DM',  'MultiDM',  
                    #'ConDM','ConMultiDM', 'AntiConDM', 'AntiConMultiDM',
                    'DMMod1', 'DMMod2',
                    'AntiDM', 'AntiMultiDM',  'AntiDMMod1', 'AntiDMMod2',        
                    'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 
                    'DMS', 'DNMS', 'DMC', 'DNMC']


def get_hidden_reps(model, num_trials, tasks=CLUSTER_TASK_LIST, instruct_mode=None):
    hidden_reps = np.empty((num_trials, 150, 256, len(tasks)))
    with torch.no_grad():
        for i, task in enumerate(tasks): 
            print(task)
            trial = construct_trials(task, None)
            _intervals = np.array([(0, 30), (30, 60), (60, 90), (90, 120), (120, 150)])
            intervals = _intervals[:,:,None].repeat(num_trials, -1)
            ins = trial(num_trials, max_var=True, intervals=intervals).inputs

            task_info = get_task_info(num_trials, task, model.info_type, instruct_mode=instruct_mode)
            _, hid = model(torch.Tensor(ins).to(model.__device__), task_info)
            hidden_reps[..., i] = hid.cpu().numpy()
    return hidden_reps

from sklearn.preprocessing import normalize
def get_norm_task_var(hid_reps): 
    task_var = np.mean(np.var(hid_reps[:, 30:,:,:], axis=0), axis=0)
    task_var = np.delete(task_var, np.where(np.sum(task_var, axis=1)<0.001)[0], axis=0)
    return normalize(task_var, axis=1)
    

def plot_task_var_heatmap(task_var):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(task_var.T, yticklabels=CLUSTER_TASK_LIST, vmin=0)
    plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def get_optim_clusters(norm_task_var):
    score_list = []
    for i in range(3,50):
        km = KMeans(n_clusters=i, random_state=12)
        labels = km.fit_predict(norm_task_var)
        score = silhouette_score(norm_task_var, labels)
        score_list.append(score)
    return list(range(3, 50))[np.argmax(np.array(score_list))]


def cluster_units(n_clusters, task_var):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(task_var)
    tSNE = TSNE(n_components=2)
    fitted = tSNE.fit_transform(task_var)
    return labels, fitted


def plot_clustering(n_clusters, task_var):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(task_var)
    tSNE = TSNE(n_components=2)
    fitted = tSNE.fit_transform(task_var)
    import matplotlib.pyplot as plt
    plt.scatter(fitted[:, 0], fitted[:, 1], cmap = plt.cm.tab10, c = labels)
    plt.show()

hid_reps = get_hidden_reps(sbertNet, 100)
norm_task_var = get_norm_task_var(hid_reps)
optim_clusters = get_optim_clusters(norm_task_var)
labels, _ = cluster_units(optim_clusters, norm_task_var)

labels.max()
sorted_indices = list(zip(*sorted(zip(labels, range(256)))))[1]
sorted_array = norm_task_var[sorted_indices, :]

plot_task_var_heatmap(sorted_array)




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

# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login2.baobab.hpc.unige.ch:/home/riveland/Instruct-RNN/6.20models/ /home/reidar/Projects/Instruct-RNN/6.20models
# rsync -a  -P --exclude '*.pt*' --exclude '*_attrs*' --exclude '*.npy*' riveland@login1.yggdrasil.hpc.unige.ch:/home/riveland/Instruct-RNN/7.1models/ /home/reidar/Projects/Instruct-RNN/7.1models