import enum
from matplotlib.pyplot import axis
import torch
import numpy as np
from torch._C import device
from torch.nn.modules import transformer

from task import Task, construct_batch, make_test_trials
from data import TaskDataSet
from utils import isCorrect, train_instruct_dict
from data import TaskDataSet

task_list = Task.TASK_LIST
task_group_dict = Task.TASK_GROUP_DICT


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_model_performance(model, num_batches): 
    model.eval()
    batch_len = 128
    with torch.no_grad():
        perf_dict = dict.fromkeys(task_list)
        for task in ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2']:
            print(task)
            mean_list = [] 
            for _ in range(num_batches): 
                #ins, targets, _, target_dirs, _ = next(TaskDataSet(num_batches=1, task_ratio_dict={task:1}).stream_batch())
                ins, targets, _, target_dirs, _ = construct_batch(task, batch_len)
                task_info = model.get_task_info(batch_len, task)
                out, _ = model(task_info, torch.Tensor(ins).to(model.__device__))
                mean_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
            perf_dict[task] = np.mean(mean_list)
    return perf_dict 

def get_instruct_reps(langModel, instruct_dict, depth='full'):
    assert depth in ['full', 'transformer']
    langModel.eval()
    if depth=='transformer': 
        assert hasattr(langModel, 'transformer'), 'language model must be transformer to evaluate a that depth'
        rep_dim = 768
    else: rep_dim = langModel.out_dim 
    instruct_reps = torch.empty(len(instruct_dict.keys()), len(list(instruct_dict.values())[0]), rep_dim)
    with torch.no_grad():      
        for i, instructions in enumerate(instruct_dict.values()):
            if depth == 'full': 
                out_rep = langModel(list(instructions))
            elif depth == 'transformer': 
                out_rep = langModel.forward_transformer(list(instructions))
            instruct_reps[i, :, :] = out_rep
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


def get_hid_var_resp(model, task, trials, num_repeats = 10, task_info=None): 
    model.eval()
    with torch.no_grad(): 
        num_trials = trials.inputs.shape[0]
        total_neuron_response = np.empty((num_repeats, num_trials, 120, 128))
        for i in range(num_repeats): 
            if task_info is None or 'simpleNet' in model.model_name: task_info = model.get_task_info(num_trials, task)
            _, hid = model(task_info, torch.Tensor(trials.inputs).to(model.__device__))
            hid = hid.cpu().numpy()
            total_neuron_response[i, :, :, :] = hid
        mean_neural_response = np.mean(total_neuron_response, axis=0)
    return total_neuron_response, mean_neural_response

def get_hid_var_group_resp(model, task_group, var_of_insterest, mod=0, num_trials=1, sigma_in = 0.05): 
    if task_group == 'Go': assert var_of_insterest in ['direction', 'strength']
    if task_group == 'DM': assert var_of_insterest in ['diff_direction', 'diff_strength']
    task_group_hid_traj = np.empty((4, 15, num_trials, 120, 128))
    for i, task in enumerate(task_group_dict[task_group]): 
        trials, vars = make_test_trials(task, var_of_insterest, mod, num_trials=num_trials, sigma_in=sigma_in)
        print(task)
        for j, instruct in enumerate(train_instruct_dict[task]): 
            print(instruct)
            _, hid_mean = get_hid_var_resp(model, task, trials, num_repeats=3, task_info=[instruct]*num_trials)
            task_group_hid_traj[i, j,  ...] = hid_mean
    return task_group_hid_traj

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

