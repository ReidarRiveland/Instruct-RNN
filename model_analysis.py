import torch
import numpy as np

from task import Task, construct_batch
from utils import isCorrect

task_list = Task.TASK_LIST

def get_model_performance(model, num_batches): 
    model.eval()
    batch_len = 128
    with torch.no_grad():
        perf_dict = dict.fromkeys(task_list)
        for task_type in task_list:
            print(task_type)
            for _ in range(num_batches): 
                mean_list = [] 
                ins, targets, _, target_dirs, _ = construct_batch(task_type, batch_len)
                task_info = model.get_task_info(batch_len, task_type)
                out, _ = model(task_info, torch.Tensor(ins))
                mean_list.append(np.mean(isCorrect(out, torch.Tensor(targets), target_dirs)))
            perf_dict[task_type] = np.mean(mean_list)
    return perf_dict 

def get_instruct_reps(langModel, instruct_dict, depth='full'):
    langModel.eval()
    with torch.no_grad(): 
        for instructions in instruct_dict.items():
            if depth == 'full': 
                out_rep = langModel(instructions)
            elif depth == 'transformer': 
                out_rep = langModel.transformer_forward(instructions)
    return out_rep.cpu().detach().numpy()

def get_task_reps(model, epoch, num_trials =100):
    assert epoch in ['stim', 'prep'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
    
    task_reps = np.array(100, model.hid_dim, len(Task.TASK_LIST))
    for i, task in enumerate(task_list): 
        trials = construct_batch(task, num_trials)
        tar = trials.targets
        ins = trials.inputs

        task_info = model.get_task_info(num_trials)
        _, hid = model(task_info, ins)

        hid = hid.detach().cpu().numpy()

        for j in range(num_trials): 
            if epoch.isnumeric(): epoch_index = int(epoch)
            if epoch == 'stim': epoch_index = np.where(tar[j, :, 0] == 0.85)[0][-1]
            if epoch == 'prep': epoch_index = np.where(ins[j, :, 1:]>0.25)[0][0]-1
            task_reps[j, :, i] = hid[j, epoch_index, :]
    return task_reps

def get_hid_var_resp(model, task_type, trials, num_repeats = 10, instruct_mode=None): 
    with torch.no_grad(): 
        num_trials = trials.inputs.shape[0]
        total_neuron_response = np.empty((num_repeats, num_trials, 120, 128))
        for i in range(num_repeats): 
            task_info = model.get_task_info(task_type, num_trials)
            _, hid = model(task_info, torch.Tensor(trials.inputs))
            hid = hid.detach().cpu().numpy()
            total_neuron_response[i, :, :, :] = hid
        mean_neural_response = np.mean(total_neuron_response, axis=0)
    return total_neuron_response, mean_neural_response

