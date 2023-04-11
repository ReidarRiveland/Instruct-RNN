import torch
import numpy as np
from instructRNN.models.full_models import SimpleNetPlus
from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, construct_trials
from instructRNN.plotting.plotting import plot_scatter
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import one_hot_input_rule
from instructRNN.analysis.model_analysis import get_rule_embedder_reps
from instructRNN.analysis.model_eval import task_eval, task_eval_info_embedded

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

EXP_FILE = '3.16models/swap_holdouts'
simpleNet = SimpleNetPlus(rule_encoder_hidden=128)
holdouts_file = 9
simpleNet.load_model(EXP_FILE+'/swap'+str(holdouts_file)+'/'+simpleNet.model_name, suffix='_seed0')

_rule_encoding_set = get_rule_embedder_reps(simpleNet, depth=-1)
rule_encoding_set = np.delete(_rule_encoding_set, np.array([TASK_LIST.index(holdout_task) for holdout_task in SWAP_LIST[holdouts_file]]), axis=0)
in_tasks = [task for task in TASK_LIST if task not in SWAP_LIST[holdouts_file]]

#plot_scatter(simpleNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='full', num_trials=100)

from sklearn.decomposition import PCA
embedder = PCA()
embedder.fit(rule_encoding_set.reshape(-1, 128))
componenets = embedder.components_[:20, :]

rule_encoding_set = rule_encoding_set.mean(1)

def softmax(x, beta=1):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def recall_mhopf(pattern, all_encodings, beta=10.0): 
    dotted = np.matmul(pattern, all_encodings.T)
    softmaxed = softmax(dotted, beta=beta)
    return np.matmul(softmaxed, all_encodings)

def recall_mhopf_w_inhibition(pattern, last_pattern, all_encodings, beta=10.0): 
    dotted = np.matmul(pattern, all_encodings.T)
    last_dotted = np.matmul(last_pattern, all_encodings.T)

    softmaxed = softmax(np.maximum(dotted-last_dotted, 0), beta=beta)
    return np.matmul(softmaxed, all_encodings)

def make_periodic_beta(max_beta, phase, num_cycles, num_points):
    return max_beta/2*np.cos(phase*np.linspace(0, 2*num_cycles*np.pi, num_points+1))+max_beta/2

def get_PCA_memory_trace(max_beta, phase, num_cycles, num_points, init_task, use_inhibition=True): 
    betas = make_periodic_beta(max_beta, phase, num_cycles, num_points)

    dir_memory_trace = np.empty((128, len(betas)))
    task_memory_trace = np.empty((128, len(betas)))

    dir_trace = np.empty((128, len(betas)))
    task_trace = np.empty((128, len(betas)))

    task_recalled = []
    dir_recalled = []

    dir_last_recalled = np.zeros(128)
    task_last_recalled = np.zeros(128)
    
    task_trace_pattern = _rule_encoding_set[TASK_LIST.index(init_task), :, :].mean(0) + np.random.normal(scale = 0.1, size=128)
    dir_trace_pattern = componenets[np.random.randint(0, 20), :]

    for i, beta in enumerate(betas):
        dir_trace_pattern = recall_mhopf_w_inhibition(dir_trace_pattern, dir_last_recalled, componenets, beta=beta)+np.random.normal(scale = 0.1, size=128)
        task_trace_pattern = recall_mhopf_w_inhibition(task_trace_pattern, task_last_recalled, rule_encoding_set, beta=beta)+np.random.normal(scale = 0.1, size=128)

        dir_memory_trace[:, i] = dir_trace_pattern
        task_memory_trace[:, i] = task_trace_pattern

        if beta == max_beta: 
            dir_recalled.append(dir_trace_pattern)
            task_recalled.append(task_trace_pattern)
        if beta == 0:
            dir_last_recalled = dir_trace_pattern
            task_last_recalled = task_trace_pattern


    return task_memory_trace, dir_memory_trace, np.array(task_recalled), np.array(dir_recalled), betas


def plot_memory_trace(trace, betas): 
    sims = cosine_similarity(trace.T, _rule_encoding_set.mean(1))
    fig, axn = plt.subplots(2,1, sharex=True, figsize =(8, 4))
    res = sns.heatmap(sims.T, xticklabels=[], yticklabels=in_tasks, ax=axn[0], cbar=False)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 3)

    axn[1].plot(betas, linewidth=0.9)

    plt.show()


SWAP_LIST[9]

max_beta = 100
task = 'Dur1'
task_trace, dir_trace, task_recalls, dir_recalls, betas = get_PCA_memory_trace(max_beta, 1.0, 100, 10_000, task, use_inhibition=True)

plot_memory_trace(task_trace+dir_trace, betas)

num_trials=50
task_perf = []

comp_recalls = (task_recalls+dir_recalls)
for i in range(comp_recalls.shape[0]):
    info_embedded = simpleNet.rule_encoder.rule_layer2(torch.tensor(comp_recalls[i, :]).float())[None, :].repeat(num_trials, 1)
    task_perf.append(task_eval_info_embedded(simpleNet, task, num_trials, info_embedded))

plt.plot(task_perf)
plt.show()










def get_memory_trace(max_beta, phase, num_cycles, num_points, init_task, use_inhibition=True): 
    betas = make_periodic_beta(max_beta, phase, num_cycles, num_points)
    memory_trace = np.empty((128, len(betas)))
    recalled = []
    last_recalled = np.zeros(128)
    trace_pattern = _rule_encoding_set[TASK_LIST.index(init_task), :] + np.random.normal(scale = 0.1, size=128)

    for i, beta in enumerate(betas):
        if use_inhibition:
            trace_pattern = recall_mhopf_w_inhibition(trace_pattern, last_recalled, rule_encoding_set, beta=beta)+np.random.normal(scale = 0.1, size=128)
        else: 
            trace_pattern = recall_mhopf(trace_pattern, rule_encoding_set, beta=beta)+np.random.normal(scale = 0.1, size=128)

        memory_trace[:, i] = trace_pattern

        if beta == max_beta: 
            recalled.append(trace_pattern)
        if beta == 0:
            last_recalled = trace_pattern


    return memory_trace, np.array(recalled), betas

def plot_memory_trace(trace, betas): 
    sims = cosine_similarity(trace.T, rule_encoding_set)
    fig, axn = plt.subplots(2,1, sharex=True, figsize =(8, 4))
    res = sns.heatmap(sims.T, xticklabels=[], yticklabels=in_tasks, ax=axn[0], cbar=False)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 3)

    axn[1].plot(betas, linewidth=0.9)

    plt.show()

def test_recall(model, task, beta = 10.0, noise = 0.1):
    init_pattern = _rule_encoding_set[TASK_LIST.index(task), :] + np.random.normal(scale = noise, size=128)
    test_recalled = recall_mhopf(init_pattern, rule_encoding_set, beta=10.0)
    info_embedded = model.rule_encoder.rule_layer2(torch.tensor(test_recalled).float())[None, :].repeat(50, 1)
    return task_eval_info_embedded(model, task, 50, info_embedded=info_embedded)

def get_recalled_tasks(recalls): 
    recalled_sims = cosine_similarity(recalls, rule_encoding_set)
    recalled_tasks = [in_tasks[ind] for ind in np.argmax(recalled_sims, axis=1)]
    return recalled_tasks

def get_recalled_performance(recalls): 
    task_perf = []
    recalled_tasks = get_recalled_tasks(recalls)
    for recalled_task, recalled_rep in zip(recalled_tasks, recalls): 
        info_embedded = simpleNet.rule_encoder.rule_layer2(torch.tensor(recalled_rep).float())[None, :].repeat(50, 1)
        task_perf.append(task_eval_info_embedded(simpleNet, recalled_task, 50, info_embedded))

    return recalled_tasks, task_perf

def get_compositional_mem_perf(recalls, task, num_trials=1): 
    task_perf = []
    for i in range(2, recalls.shape[0]): 
        comp_rep = recalls[i-2, :]+(recalls[i-1, :]-recalls[i, :])
        info_embedded = simpleNet.rule_encoder.rule_layer2(torch.tensor(comp_rep).float())[None, :].repeat(num_trials, 1)
        task_perf.append(task_eval_info_embedded(simpleNet, task, num_trials, info_embedded))
    return task_perf

def update_value_weights(W_v, embedding, reward, alpha=0.01): 
    rpe = reward-np.dot(W_v.T, embedding)
    return W_v + (alpha*abs(rpe))*(rpe*embedding)

def weibel(x): 
    lam=0.5
    k=8
    return 1-np.exp(-(x/lam)**k)

def value_sim(recalls, task, init_embedding): 
    task_perf = []
    W_v = np.maximum(np.random.normal(scale=0.1, size=128), 0)
    i = 2
    comp_rep = init_embedding

    for _ in range(2000): 
        if i == recalls.shape[0]: 
            print('broke')
            break

        for _ in range(5):
            info_embedded = simpleNet.rule_encoder.rule_layer2(torch.tensor(comp_rep).float())[None, :]
            reward = task_eval_info_embedded(simpleNet, task, 1, info_embedded)
            task_perf.append(reward)

        W_v = update_value_weights(W_v, comp_rep, np.mean(task_perf[-5:]), alpha=0.05-((0.05/recalls.shape[0])*i))
        value = np.dot(W_v.T, comp_rep)
        print(value)
        stay = np.random.binomial(1, weibel(np.maximum(value, 0)))

        if not stay:
            print('SWITCHED')
            comp_rep = recalls[i-2, :]+(recalls[i-1, :]-recalls[i, :])
            i+=1

    return np.array(task_perf), W_v

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



max_beta = 100
trace, recalls, betas = get_memory_trace(max_beta, 1.0, 1000, 100_000, 'AntiGo', use_inhibition=True)
#plot_memory_trace(trace, betas)
#recalled_task, perf = get_recalled_performance(recalls)
#comp_perf = get_compositional_mem_perf(recalls, 'AntiGo', num_trials=50)
# plt.plot(comp_perf)
# plt.ylim(0, 1.0)
# plt.show()



x = np.linspace(0, 5, 100)
plt.plot(x, weibel(x))
plt.show()



perf, W_v = value_sim(recalls, 'AntiGo', _rule_encoding_set[TASK_LIST.index('AntiGo')])

from scipy.ndimage.filters import gaussian_filter1d



plt.plot(moving_average(perf[3000:], 20))
plt.plot(gaussian_filter1d(moving_average(perf[3000:], 20), sigma=500.0), color='red')
plt.show()


not np.random.binomial(1, 0.99)

