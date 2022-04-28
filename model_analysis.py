from matplotlib.pyplot import axis
import torch
import numpy as np

from utils.utils import task_swaps_map
from utils.task_info_utils import train_instruct_dict, get_task_info, get_instruction_dict

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

from task import Task, construct_batch, make_test_trials, isCorrect

task_list = Task.TASK_LIST
swapped_task_list = Task.SWAPPED_TASK_LIST
task_group_dict = Task.TASK_GROUP_DICT

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def task_eval(model, task, batch_size): 
    ins, targets, _, target_dirs, _ = construct_batch(task, batch_size)
    task_info = get_task_info(batch_size, task, model.is_instruct)
    out, _ = model(torch.Tensor(ins).to(model.__device__), task_info)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def get_model_performance(model, num_batches): 
    model.eval()
    perf_array = np.empty(len(task_list))
    with torch.no_grad():
        for i, task in enumerate(Task.TASK_LIST):
            print(task)
            mean_list = [] 
            for _ in range(num_batches): 
                frac_correct = task_eval(model, task, 128)
                mean_list.append(frac_correct)
            perf_array[i] = np.mean(mean_list)
    return perf_array

def get_multitask_val_performance(model, foldername, seeds=np.array(range(5))): 
    performance = np.empty((len(seeds), len(Task.TASK_LIST)))
    model.instruct_model = 'validation'
    for seed in seeds:
        model.set_seed(seed)
        model.load_model(foldername+'/Multitask')
        perf = get_model_performance(model, 3)
        performance[seed, :] = perf
    return performance



def get_instruct_reps(langModel, depth='full', instruct_mode=None):
    if depth.isnumeric(): 
        rep_dim = 768
    else: 
        rep_dim = langModel.LM_out_dim 

    instruct_dict = get_instruction_dict(instruct_mode)
    instruct_reps = torch.empty(len(instruct_dict.keys()), len(list(instruct_dict.values())[0]), rep_dim)
    
    with torch.no_grad():      
        for i, task in enumerate(list(instruct_dict.keys())):
            
            instructions = instruct_dict[task]    
            if depth == 'full': 
                out_rep = langModel(list(instructions))
            elif depth.isnumeric(): 
                out_rep = torch.mean(langModel.forward_transformer(list(instructions))[1][int(depth)], dim=1)
            instruct_reps[i, :, :] = out_rep

    return instruct_reps.cpu().numpy().astype(np.float64)

def get_task_reps(model, epoch='stim_start', stim_start_buffer=0, num_trials =100, instruct_mode=None, contexts=None):
    assert epoch in ['stim', 'prep', 'stim_start'] or epoch.isnumeric(), "entered invalid epoch: %r" %epoch
    model.eval()
    with torch.no_grad(): 
        task_reps = np.empty((len(task_list), num_trials, model.rnn_hidden_dim))

        for i, task in enumerate(Task.TASK_LIST): 
            ins, targets, _, _, _ =  construct_batch(task, num_trials)

            if contexts is not None: 
                _, hid = model(torch.Tensor(ins).to(model.__device__), context=contexts[i, ...])
            else: 
                task_info = get_task_info(num_trials, task, model.is_instruct, instruct_mode=instruct_mode)
                _, hid = model(torch.Tensor(ins).to(model.__device__), task_info)

            hid = hid.cpu().numpy()
            for j in range(num_trials): 
                if epoch.isnumeric(): epoch_index = int(epoch)
                if epoch == 'stim': epoch_index = np.where(targets[j, :, 0] == 0.85)[0][-1]
                if epoch == 'stim_start': epoch_index = np.where(ins[j, :, 1:]>0.25)[0][0]+stim_start_buffer
                if epoch == 'prep': epoch_index = np.where(ins[j, :, 1:]>0.25)[0][0]-1
                task_reps[i, j, :] = hid[j, epoch_index, :]

    return task_reps.astype(np.float64)


def reduce_rep(reps, dim=2, reduction_method='PCA'): 
    if reduction_method == 'PCA': 
        embedder = PCA(n_components=dim)
    elif reduction_method == 'tSNE': 
        embedder = TSNE(n_components=2)

    embedded = embedder.fit_transform(reps.reshape(reps.shape[0]*reps.shape[1], -1))

    if reduction_method == 'PCA': 
        explained_variance = embedder.explained_variance_ratio_
    else: 
        explained_variance = None

    return embedded.reshape(reps.shape[0], reps.shape[1], dim), explained_variance

def get_layer_sim_scores(model, rep_depth='12'): 
    if rep_depth.isnumeric(): 
        rep_dim = model.langModel.LM_intermediate_lang_dim
    if rep_depth =='full': 
        rep_dim = 20
    
    if rep_depth == 'task': 
        rep_dim = 128
    
    if rep_depth == 'task': 
        reps, _ = get_task_reps(model)
    if rep_depth == 'full' or rep_depth.isnumeric(): 
        reps = get_instruct_reps(model.langModel, depth=rep_depth)

    sim_scores = 1-cosine_similarity(reps.reshape(-1, rep_dim))
    
    return sim_scores

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

def get_hid_var_group_resp(model, task_group, var_of_insterest, swapped_tasks = [], mod=0, num_trials=1, sigma_in = 0.05): 
    if task_group == 'Go': assert var_of_insterest in ['direction', 'strength']
    if task_group == 'DM': assert var_of_insterest in ['diff_direction', 'diff_strength']
    task_group_hid_traj = np.empty((4+len(swapped_tasks), 15, num_trials, 120, 128))
    for i, task in enumerate(task_group_dict[task_group]+swapped_tasks): 
        trials, vars = make_test_trials(task, var_of_insterest, mod, num_trials=num_trials, sigma_in=sigma_in)
        if i>=4: task_instructs = Task.SWAPPED_TASK_LIST[task_list.index(task)]
        else: task_instructs = task
        for j, instruct in enumerate(train_instruct_dict[task_instructs]): 
            _, hid_mean = get_hid_var_resp(model, task, trials, num_repeats=3, task_info=[instruct]*num_trials)
            task_group_hid_traj[i, j,  ...] = hid_mean
    return task_group_hid_traj

def get_CCGP(reps): 
    num_trials = reps.shape[1]
    dim = reps.shape[-1]
    all_decoding_score = np.zeros((16, 2))
    dichotomies = np.array([[[0, 1], [2, 3]], [[0,2], [1, 3]]])
    for i in range(4): 
        conditions=dichotomies+(4*i)
        for j in [0, 1]: 
            for k in [0, 1]: 

                print('\n')
                print('train condition ' +str(conditions[j][k]))
                test_condition = conditions[j][(k+1)%2]
                print('test condition' + str(test_condition))
                print('\n')

                classifier = svm.LinearSVC(max_iter=5000)
                classifier.classes_=[-1, 1]
                classifier.fit(reps[conditions[j][k], ...].reshape(-1, dim), np.array([0]*num_trials+[1]*num_trials))
                for index in [0, 1]: 
                    print('Task :' + str(test_condition[index]))
                    decoding_corrects = np.array([index]*num_trials) == classifier.predict(reps[test_condition[index], ...].reshape(-1, dim))
                    decoding_score = np.mean(decoding_corrects)
                    all_decoding_score[test_condition[index], j] = decoding_score
            

    return all_decoding_score


def get_all_CCGP(model, task_rep_type, foldername, swap=False): 
    if not swap: 
        tasks_to_compute = task_list +['Multitask']
    else: 
        tasks_to_compute = task_list
    all_CCGP = np.empty((5, len(tasks_to_compute), 16, 2))
    holdout_CCGP = np.empty((5, 16, 2))
    epoch = 'stim_start'
    for i in range(5):
        model.set_seed(i)
        for j, task in enumerate(tasks_to_compute):
            print('\n') 
            print(task) 
            task_file = task_swaps_map[task]
            model.load_model(foldername+'/swap_holdouts/'+task_file)
            if swap: 
                swapped_list = [task]
                swap_str = '_swap'
            else: 
                swapped_list = []
                swap_str = ''

            if task_rep_type == 'task': 
                reps, _ = get_task_reps(model, num_trials=128, epoch=epoch, stim_start_buffer=0, swapped_tasks=swapped_list)
            if task_rep_type == 'lang': 
                if model.langModel.embedder_name == 'bow': depth = 'full'
                else: depth = 'transformer'
                reps = get_instruct_reps(model.langModel, train_instruct_dict, depth=depth, swapped_tasks=swapped_list)
            
            if swap:
                reps[task_list.index(task), ...] = reps[-1, ...]

            decoding_score = get_CCGP(reps)
            all_CCGP[i, j, ...] = decoding_score
            if j != 16: 
                holdout_CCGP[i, j] = decoding_score[j, :]
    np.savez(foldername+'/CCGP_measures/' +task_rep_type+'_'+ epoch + '_' + model.model_name + swap_str +'_CCGP_scores', all_CCGP=all_CCGP, holdout_CCGP= holdout_CCGP)
    return all_CCGP, holdout_CCGP

if __name__ == "__main__":

    from model_trainer import config_model
    import pickle
    from utils.utils import all_models
    model_file = '_ReLU128_4.11'

    ###GET ALL MODEL CCGPs###
    for swap_bool in [False]: 
        for model_name in ['gptNet_tuned']:
            print(model_name)
            model = config_model(model_name)
            model.to(torch.device(0))
            get_all_CCGP(model, 'task', model_file, swap=swap_bool)




