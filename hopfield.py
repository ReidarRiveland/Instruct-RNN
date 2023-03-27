import torch
import numpy as np
from instructRNN.models.full_models import SimpleNetPlus
from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, construct_trials
from instructRNN.plotting.plotting import plot_scatter
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import one_hot_input_rule
from instructRNN.analysis.model_analysis import get_rule_embedder_reps

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

EXP_FILE = '3.16models/swap_holdouts'
simpleNet = SimpleNetPlus(rule_encoder_hidden=128)
holdouts_file = 0
simpleNet.load_model(EXP_FILE+'/swap'+str(holdouts_file)+'/'+simpleNet.model_name, suffix='_seed0')

_rule_encoding_set = get_rule_embedder_reps(simpleNet, depth='full')[:,0,:]
rule_encoding_set = np.delete(_rule_encoding_set, np.array([TASK_LIST.index(holdout_task) for holdout_task in SWAP_LIST[holdouts_file]]), axis=0)
in_tasks = [task for task in TASK_LIST if task not in SWAP_LIST[holdouts_file]]

plot_scatter(simpleNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], dims=3, rep_depth='full', num_trials=100)


def softmax(x, beta=1):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

def recall_mhopf(pattern, all_encodings, beta=10.0): 
    dotted = np.matmul(pattern, all_encodings.T)
    softmaxed = softmax(dotted, beta=beta)
    return np.matmul(softmaxed, all_encodings)


SWAP_LIST[0]

holdout_task = 'RTGo'
test_pattern = _rule_encoding_set[TASK_LIST.index(holdout_task)] #+ np.random.normal(scale = 0.1, size=64)
recalled_pattern = recall_mhopf(test_pattern, rule_encoding_set, beta=50.0)

in_tasks[np.argmax(cosine_similarity(test_pattern[None, :], rule_encoding_set))]


periodic = np.cos(np.linspace(0, 6*np.pi, 1000))+1

tasks_explored = []
for i in periodic: 
    recalled_pattern = recall_mhopf(recalled_pattern, rule_encoding_set, beta=i)+np.random.normal(scale = 0.1, size=64)
    tasks_explored.append(in_tasks[np.argmax(np.dot(recalled_pattern, rule_encoding_set.T))])

tasks_explored

perf_list = []
ins, targets, _, target_dirs, _ = construct_trials('Go', 50)
for beta in np.linspace(0.1, 10.0, 100):
    recalled_pattern = recall_mhopf(recalled_pattern, rule_encoding_set, beta=beta)
    context = torch.tensor(recalled_pattern)[None, :].repeat(50, 1)
    out, _, _ = simpleNet(torch.Tensor(ins), context = context)
    perf_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))

perf_list

plt.plot(perf_list)
plt.show()