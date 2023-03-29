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
holdouts_file = 0
simpleNet.load_model(EXP_FILE+'/swap'+str(holdouts_file)+'/'+simpleNet.model_name, suffix='_seed0')

_rule_encoding_set = get_rule_embedder_reps(simpleNet, depth=-1).mean(1)
rule_encoding_set = np.delete(_rule_encoding_set, np.array([TASK_LIST.index(holdout_task) for holdout_task in SWAP_LIST[holdouts_file]]), axis=0)
in_tasks = [task for task in TASK_LIST if task not in SWAP_LIST[holdouts_file]]

#plot_scatter(simpleNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], dims=3, rep_depth='full', num_trials=100)

def softmax(x, beta=1):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

def recall_mhopf(pattern, all_encodings, beta=10.0): 
    dotted = np.matmul(pattern, all_encodings.T)
    softmaxed = softmax(dotted, beta=beta)
    return np.matmul(softmaxed, all_encodings)

def recall_mhopf_w_inhibition(pattern, last_pattern, all_encodings, beta=10.0): 
    dotted = np.matmul(pattern, all_encodings.T)
    last_dotted = np.matmul(last_pattern, all_encodings.T)

    softmaxed = softmax(np.maximum(dotted-last_dotted, 0), beta=beta)
    return np.matmul(softmaxed, all_encodings)

def test_recall(model, task, beta = 10.0, noise = 0.1):
    init_pattern = _rule_encoding_set[TASK_LIST.index(task), :] + np.random.normal(scale = noise, size=128)
    test_recalled = recall_mhopf(init_pattern, rule_encoding_set, beta=10.0)
    info_embedded = model.rule_encoder.rule_layer2(torch.tensor(test_recalled).float())[None, :].repeat(50, 1)
    return task_eval_info_embedded(model, task, 50, info_embedded=info_embedded)

def make_periodic_beta(max_beta, phase, num_cycles, num_points):
    return max_beta/2*np.cos(phase*np.linspace(0, 2*num_cycles*np.pi, num_points+1))+max_beta/2

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



SWAP_LIST[0]

max_beta = 100
trace, recalls, betas = get_memory_trace(max_beta, 1.0, 100, 10000, 'RTGo', use_inhibition=False)
recalls.shape
plot_memory_trace(trace, betas)
recalled_tasks = get_recalled_tasks(recalls)
len(set(recalled_tasks))

perf = get_compositional_mem_perf(recalls, 'RTGo', 50)
plt.plot(perf)
plt.show()


get_recalled_performance(recalls)

