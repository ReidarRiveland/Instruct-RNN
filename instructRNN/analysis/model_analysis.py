import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import _draw_ortho_dirs, DMFactory, TRIAL_LEN, _get_default_intervals, max_var_dir
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import train_instruct_dict, get_task_info, get_instruction_dict

import sklearn.svm as svm

def task_eval(model, task, batch_size, noise=None, instructions = None): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, noise)
    if instructions is None: 
        task_info = get_task_info(batch_size, task, model.info_type)
    else: 
        task_info = instructions
    out, _ = model(torch.Tensor(ins).to(model.__device__), task_info)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def get_model_performance(model, num_repeats = 1, batch_len=128): 
    model.eval()
    perf_array = np.empty(len(TASK_LIST))
    with torch.no_grad():
        for i, task in enumerate(TASK_LIST):
            print(task)
            mean_list = [] 
            for _ in range(num_repeats): 
                frac_correct = task_eval(model, task, batch_len)
                mean_list.append(frac_correct)
            perf_array[i] = np.mean(mean_list)
    return perf_array

def get_multitask_val_performance(model, foldername, seeds=np.array(range(5))): 
    performance = np.empty((len(seeds), len(TASK_LIST)))
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
    instruct_reps = torch.empty(len(TASK_LIST), len(list(instruct_dict.values())[0]), rep_dim)
    
    with torch.no_grad():      
        for i, task in enumerate(TASK_LIST):
            
            instructions = instruct_dict[task]    
            if depth == 'full': 
                out_rep = langModel(list(instructions))
            elif depth.isnumeric(): 
                out_rep = torch.mean(langModel.forward_transformer(list(instructions))[1][int(depth)], dim=1)
            instruct_reps[i, :, :] = out_rep

    return instruct_reps.cpu().numpy().astype(np.float64)

def get_rule_embedder_reps(model):
    reps = np.empty((len(TASK_LIST), model.rule_dim))
    with torch.no_grad():
        for i, task in enumerate(TASK_LIST): 
            info = get_task_info(1, task, False)
            reps[i, :] = model.rule_encoder(info).cpu().numpy()
    return reps

def get_task_reps(model, epoch='stim_start', stim_start_buffer=0, num_trials =100, tasks=TASK_LIST, 
                    instruct_mode=None, contexts=None, default_intervals=False, max_var=False, main_var=False, num_repeats=1):
    model.eval()
    if epoch is None: 
        task_reps = np.empty((num_repeats, len(tasks), num_trials, TRIAL_LEN, model.rnn_hidden_dim))
    else: 
        task_reps = np.empty((num_repeats, len(tasks), num_trials, model.rnn_hidden_dim))

    with torch.no_grad(): 
        for k in range(num_repeats):
            for i, task in enumerate(tasks): 
                if default_intervals and 'Dur' not in task:
                    intervals = _get_default_intervals(num_trials)
                    ins, targets, _, _, _ =  construct_trials(task, num_trials, max_var=max_var, main_var = main_var, intervals=intervals)
                else: 
                    ins, targets, _, _, _ =  construct_trials(task, num_trials, max_var=max_var)

                if contexts is not None: 
                    _, hid = model(torch.Tensor(ins).to(model.__device__), context=contexts[i, ...])
                else: 
                    task_info = get_task_info(num_trials, task, model.info_type, instruct_mode=instruct_mode)
                    _, hid = model(torch.Tensor(ins).to(model.__device__), task_info)

                hid = hid.cpu().numpy()
                if epoch is None: 
                    task_reps[k, i, ...] = hid
                else: 
                    for j in range(num_trials): 
                        if epoch.isnumeric(): epoch_index = int(epoch)
                        if epoch == 'stim': epoch_index = np.where(targets[j, :, 0] == 0.85)[0][-1]
                        if epoch == 'stim_start': epoch_index = np.where(ins[j, :, 1:]>0.25)[0][0]+stim_start_buffer
                        if epoch == 'prep': epoch_index = np.where(ins[j, :, 1:]>0.25)[0][0]-1
                        task_reps[k, i, j, :] = hid[j, epoch_index, :]

    return np.mean(task_reps, axis=0).astype(np.float64)



def reduce_rep(reps, pcs=[0, 1], reduction_method='PCA'): 
    if reduction_method == 'PCA': 
        embedder = PCA()
    elif reduction_method == 'tSNE': 
        embedder = TSNE()

    _embedded = embedder.fit_transform(reps.reshape(-1, reps.shape[-1]))
    embedded = _embedded.reshape(reps.shape)

    if reduction_method == 'PCA': 
        explained_variance = embedder.explained_variance_ratio_
    else: 
        explained_variance = None

    return embedded[..., pcs], explained_variance

def get_layer_sim_scores(model, rep_depth='12'): 
    if rep_depth.isnumeric(): 
        rep_dim = model.langModel.LM_intermediate_lang_dim
    if rep_depth =='full': 
        rep_dim = model.langModel.LM_out_dim
    
    if rep_depth == 'task': 
        rep_dim = model.rnn_hidden_dim

    if rep_depth == 'rule_encoder': 
        rep_dim = model.rule_dim
    
    if rep_depth == 'task': 
        reps = get_task_reps(model, num_trials=32)
    if rep_depth == 'full' or rep_depth.isnumeric(): 
        reps = get_instruct_reps(model.langModel, depth=rep_depth)
    if rep_depth == 'rule_encoder': 
        reps = get_rule_embedder_reps(model)

    sim_scores = 1-cosine_similarity(reps.reshape(-1, rep_dim))
    
    return sim_scores


def get_DM_perf(model, noises, diff_strength, num_repeats=100, mod=0, task='DM'):
    num_trials = len(diff_strength)
    pstim1_stats = np.empty((num_repeats, len(noises), num_trials), dtype=bool)
    correct_stats = np.empty((num_repeats, len(noises), num_trials), dtype=bool)
    for i in tqdm(range(num_repeats)): 
        for j, noise in enumerate(noises): 
            intervals = _get_default_intervals(num_trials)
            multi = 'Multi' in task
            dir_arr = max_var_dir(num_trials, None, multi, 2)

            if 'Multi' in task:
                mod_base_strs = np.array([1-diff_strength/2, 1+diff_strength/2])
                _coh = np.empty((2, num_trials))
                
                for k in range(num_trials):
                    redraw = True
                    while redraw: 
                        coh = np.random.choice([-0.05, -0.1, 0.1, 0.05], size=2, replace=False)
                        if coh[0] != -1*coh[1] and ((coh[0] <0) ^ (coh[1] < 0)): 
                            redraw = False
                    coh[:, k] = coh

                diff_strength = np.array([mod_base_strs + _coh, mod_base_strs-_coh]).T
            else: 
                coh = np.array([diff_strength/2, -diff_strength/2])

            if task == 'DM':
                trial = Task(num_trials, noise, DMFactory, str_chooser = np.argmax, 
                                                intervals=intervals, coh_arr = coh, dir_arr=dir_arr)
            elif task =='AntiDM':
                trial = Task(num_trials, noise, DMFactory, str_chooser = np.argmin, 
                                                intervals=intervals, coh_arr = coh, dir_arr=dir_arr)
            elif task =='MultiDM':
                trial = Task(num_trials, noise, DMFactory, str_chooser = np.argmax, multi=True, 
                                                intervals=intervals, coh_arr = diff_strength, dir_arr=dir_arr)
            elif task =='AntiMultiDM':
                trial = Task(num_trials, noise, DMFactory, str_chooser = np.argmin, multi=True, 
                                                intervals=intervals, coh_arr = diff_strength, dir_arr=dir_arr)

            task_instructions = get_task_info(num_trials, task, model.info_type)

            out, hid = model(torch.Tensor(trial.inputs), task_instructions)
            correct_stats[i, j, :] =  isCorrect(out, torch.Tensor(trial.targets), trial.target_dirs)
            pstim1_stats[i, j, :] =  np.where(isCorrect(out, torch.Tensor(trial.targets), trial.target_dirs), diff_strength > 0, diff_strength<=0)
    
    return correct_stats, pstim1_stats, trial

def get_noise_thresholdouts(correct_stats, diff_strength, noises, pos_cutoff=0.95, neg_cutoff=0.75): 
    pos_coords = np.where(np.mean(correct_stats, axis=0) > pos_cutoff)
    neg_coords = np.where(np.mean(correct_stats, axis=0) < neg_cutoff)
    pos_thresholds = np.array((noises[pos_coords[0]], diff_strength[pos_coords[1]]))
    neg_thresholds = np.array((noises[neg_coords[0]], diff_strength[neg_coords[1]]))
    return pos_thresholds, neg_thresholds


def get_CCGP(reps): 
    num_trials = reps.shape[1]
    dim = reps.shape[-1]
    all_decoding_score = np.zeros((len(TASK_LIST), 2))
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

# def get_CCGP(model, dichotomy, num_trials = 100): 
#     rep_dict = {}
#     all_tasks = dichotomy[0]+dichotomy[1]
#     reps = get_task_reps(model, tasks=all_tasks, num_trials=num_trials)
#     training_pairs = list(itertools.product(DICH_DICT['dich0'][0], DICH_DICT['dich0'][1]))

#     for pair in training_pairs: 
#         classifier = svm.LinearSVC(max_iter=5000)
#         classifier.classes_=[-1, 1]
#         training_reps = reps[[all_tasks.index(pair[0]), all_tasks.index(pair[1])]]
#         classifier.fit(training_reps, np.array([0]*num_trials+[1]*num_trials))
#         for test_pair in training_pairs.remove(pair): 
#             decoding_corrects = np.array([index]*num_trials) == classifier.predict(reps[test_condition[index], ...].reshape(-1, dim))
#             decoding_score = np.mean(decoding_corrects)
#             all_decoding_score[test_condition[index], j] = decoding_score

#     return all_decoding_score


# def get_all_CCGP(model, task_rep_type, foldername, swap=False): 
#     if not swap: 
#         tasks_to_compute = task_list +['Multitask']
#     else: 
#         tasks_to_compute = task_list
#     all_CCGP = np.empty((5, len(tasks_to_compute), 16, 2))
#     holdout_CCGP = np.empty((5, 16, 2))
#     epoch = 'stim_start'
#     for i in range(5):
#         model.set_seed(i)
#         for j, task in enumerate(tasks_to_compute):
#             print('\n') 
#             print(task) 
#             task_file = task_swaps_map[task]
#             model.load_model(foldername+'/swap_holdouts/'+task_file)
#             if swap: 
#                 swapped_list = [task]
#                 swap_str = '_swap'
#             else: 
#                 swapped_list = []
#                 swap_str = ''

#             if task_rep_type == 'task': 
#                 reps, _ = get_task_reps(model, num_trials=128, epoch=epoch, stim_start_buffer=0, swapped_tasks=swapped_list)
#             if task_rep_type == 'lang': 
#                 if model.langModel.embedder_name == 'bow': depth = 'full'
#                 else: depth = 'transformer'
#                 reps = get_instruct_reps(model.langModel, train_instruct_dict, depth=depth, swapped_tasks=swapped_list)
            
#             if swap:
#                 reps[task_list.index(task), ...] = reps[-1, ...]

#             decoding_score = get_CCGP(reps)
#             all_CCGP[i, j, ...] = decoding_score
#             if j != 16: 
#                 holdout_CCGP[i, j] = decoding_score[j, :]
#     np.savez(foldername+'/CCGP_measures/' +task_rep_type+'_'+ epoch + '_' + model.model_name + swap_str +'_CCGP_scores', all_CCGP=all_CCGP, holdout_CCGP= holdout_CCGP)
#     return all_CCGP, holdout_CCGP

# if __name__ == "__main__":

#     from model_trainer import config_model
#     import pickle
#     from utils.utils import all_models
#     model_file = '_ReLU128_4.11'

#     ###GET ALL MODEL CCGPs###
#     for swap_bool in [False]: 
#         for model_name in ['gptNet_tuned']:
#             print(model_name)
#             model = config_model(model_name)
#             model.to(torch.device(0))
#             get_all_CCGP(model, 'task', model_file, swap=swap_bool)




