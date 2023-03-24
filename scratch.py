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


TASK_LIST
SWAP_LIST[3]

EXP_FILE = '3.16models/swap_holdouts'
simpleNet = SimpleNetPlus(rule_encoder_hidden=128)
holdouts_file = 'swap3'
simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')

#plot_scatter(simpleNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], dims=3)
# transform1 = (rule_encoding[TASK_LIST.index('AntiGoMod1')] - rule_encoding[TASK_LIST.index('GoMod1')])
# transform2 = (rule_encoding[TASK_LIST.index('AntiGoMod2')] - rule_encoding[TASK_LIST.index('Go')])
# plot_scatter(simpleNet, ['AntiGo', 'AntiGoMod1', 'Go', 'GoMod1'], dims=3, rep_depth='full', transform = transform[None, ], transform_task='AntiGo', num_trials=100)


rule_encoding_set = get_rule_embedder_reps(simpleNet, depth='full').mean(1)

def softmax(x, beta=1):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

def sim_of_rule_encodings(rule_encoding, all_encodings):
    return (np.dot(rule_encoding, all_encodings.T)/ 
            (np.linalg.norm(rule_encoding)*np.linalg.norm(all_encodings, axis=1)))

def draw_next_rep(rule_encoding, all_encodings, beta, holdout_task): 
    similarities = sim_of_rule_encodings(rule_encoding, all_encodings)
    zeroed_sims = np.where(np.isclose(similarities, 1.0), np.zeros_like(similarities), similarities)
    zeroed_sims[TASK_LIST.index(holdout_task)] = -1e5
    softmaxed = softmax(zeroed_sims, beta=beta)
    next_rep_index = np.random.choice(range(all_encodings.shape[0]), p=softmaxed)
    return all_encodings[next_rep_index, :], TASK_LIST[next_rep_index]

def get_reps_recalled(init_task, rule_encodings, holdout_task, beta, n_recalls):
    candidate_array = np.empty((n_recalls, 64))
    task_recalled = []
    candidate_rep, _ = draw_next_rep(rule_encodings[TASK_LIST.index(init_task), :], rule_encodings, beta, holdout_task)

    for i in range(n_recalls):
        encoding_t, task_t = draw_next_rep(candidate_rep, rule_encodings, beta, holdout_task)
        encoding_t_1, task_t_1 = draw_next_rep(encoding_t, rule_encodings, beta, holdout_task)
        encoding_t_2, task_t_2 = draw_next_rep(encoding_t_1, rule_encodings, beta, holdout_task)
        candidate_rep = encoding_t+(encoding_t_1-encoding_t_2)
        candidate_array[i, :] = candidate_rep
        task_recalled.append((task_t, task_t_1, task_t_2))
    return candidate_array, task_recalled

def get_recall_perf(task, rule_encodings, beta=1.0, n_recalls=100):
    ins, targets, _, target_dirs, _ = construct_trials(task, 25)
    simpleNet.to('cpu')

    recall_array, task_recalled = get_reps_recalled(task, rule_encodings, task, beta, n_recalls)
    perf_list = []
    for context in recall_array:
        context = torch.tensor(context)[None, :].repeat(25, 1)
        out, _ = simpleNet(torch.Tensor(ins), task_rule=None, context=context)
        perf_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))

    return perf_list, task_recalled


tries, tasks = get_recall_perf('AntiRTGo', rule_encoding_set, beta=10.0, n_recalls=1000)
np.max(tries)
plt.plot(tries[:50])
plt.show()

prop_rep = (rule_encoding_set[TASK_LIST.index('RTGoMod2'), :]+
            (rule_encoding_set[TASK_LIST.index('GoMod1'), :]-rule_encoding_set[TASK_LIST.index('RTGoMod1'), :]))

ins, targets, _, target_dirs, _ = construct_trials('GoMod2', 100)
context = torch.tensor(prop_rep)[None, :].repeat(100, 1)
out, _ = simpleNet(torch.Tensor(ins), context = context)
np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

plot_scatter(simpleNet, ['GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], dims=3, rep_depth='full', 
        transform = context[None, ...], transform_task='GoMod1', num_trials=100)
