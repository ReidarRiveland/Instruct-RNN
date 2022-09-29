import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import DMFactory, TRIAL_LEN, _get_default_intervals, max_var_dir
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import get_task_info, get_instruction_dict, sort_vocab

import sklearn.svm as svm
from instructRNN.models.full_models import make_default_model

from instructRNN.tasks.tasks import DICH_DICT, TASK_LIST, SWAPS_DICT

import numpy as np
import itertools
import sklearn.svm as svm
import os

if torch.cuda.is_available:
    device = torch.device(0)
    print(torch.cuda.get_device_name(device), flush=True)
else: 
    device = torch.device('cpu')
    

def task_eval(model, task, batch_size, noise=None, instruct_mode = None, instructions = None): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, noise)
    if instructions is None: 
        task_info = get_task_info(batch_size, task, model.info_type, instruct_mode=instruct_mode)
    else: 
        task_info = instructions
    out, _ = model(torch.Tensor(ins).to(model.__device__), task_info)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def get_model_performance(model, num_repeats = 1, batch_len=128, instruct_mode=None): 
    model.eval()
    perf_array = np.empty(len(TASK_LIST))
    with torch.no_grad():
        for i, task in enumerate(TASK_LIST):
            print(task)
            mean_list = [] 
            for _ in range(num_repeats): 
                frac_correct = task_eval(model, task, batch_len, instruct_mode=instruct_mode)
                mean_list.append(frac_correct)
            perf_array[i] = np.mean(mean_list)
    return perf_array

def eval_model_0_shot(model, folder_name, exp_type, seed, instruct_mode=None): 
    if exp_type == 'swap': 
        exp_dict = SWAPS_DICT
    perf_array = np.full(len(TASK_LIST), np.NaN)
    with torch.no_grad():
        for holdout_label, tasks in exp_dict.keys(): 
            model.load_model(folder_name+'/'+exp_type+'/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
            for task in tasks: 
                perf_array[TASK_LIST.index(task)] = task_eval(model, task, 64, instruct_mode=instruct_mode)
    return perf_array

def get_instruct_reps(langModel, depth='full', instruct_mode=None):
    if depth.isnumeric(): 
        rep_dim = langModel.LM_intermediate_lang_dim
    elif depth == 'bow': 
        rep_dim = len(sort_vocab())
    else: 
        rep_dim = langModel.LM_out_dim 

    instruct_dict = get_instruction_dict(instruct_mode)
    instruct_reps = torch.empty(len(TASK_LIST), len(list(instruct_dict.values())[0]), rep_dim)

    with torch.no_grad():      
        for i, task in enumerate(TASK_LIST):
            instructions = instruct_dict[task]    
            if depth == 'full':   
                out_rep = langModel(list(instructions))
            elif depth == 'bow': 
                out_rep_list = []
                for instruct in list(instructions):
                    out_rep_list.append(langModel._make_freq_tensor(instruct))
                out_rep = torch.stack(out_rep_list)
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
        instruct_mode=None, contexts=None, default_intervals=False, max_var = False, main_var=False, num_repeats=1):
    model.eval()
    model.to(device)
    if epoch is None: 
        task_reps = np.empty((num_repeats, len(tasks), num_trials, TRIAL_LEN, model.rnn_hidden_dim))
    else: 
        task_reps = np.empty((num_repeats, len(tasks), num_trials, model.rnn_hidden_dim))

    with torch.no_grad(): 
        for k in range(num_repeats):
            for i, task in enumerate(tasks): 
                if default_intervals and 'Dur' not in task:
                    intervals = _get_default_intervals(num_trials)
                    ins, targets, _, _, _ =  construct_trials(task, num_trials, max_var = max_var, main_var = main_var, intervals=intervals, noise=0.0)
                else: 
                    ins, targets, _, _, _ =  construct_trials(task, num_trials, max_var = max_var, main_var=main_var, noise=0.0)

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
    dims = max(pcs)+1
    if reduction_method == 'PCA': 
        embedder = PCA(n_components=dims)
    elif reduction_method == 'tSNE': 
        embedder = TSNE()

    _embedded = embedder.fit_transform(reps.reshape(-1, reps.shape[-1]))
    embedded = _embedded.reshape(reps.shape[0], reps.shape[1], dims)

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



def get_reps_from_tasks(reps, tasks): 
    indices = [TASK_LIST.index(task) for task in tasks]
    return reps[indices, ...]


def get_dich_CCGP(reps, dich, holdouts_involved=[]):
    dim = reps.shape[-1]
    num_trials = reps.shape[1]

    all_dich_pairs = list(itertools.product(dich[0], dich[1]))
    decoding_score_arr = np.empty((len(all_dich_pairs), len(all_dich_pairs)-1))
    holdouts_score_list = []

    for i, train_pair in enumerate(all_dich_pairs):
        test_pairs = all_dich_pairs.copy()
        test_pairs.remove(train_pair)
        train_reps = get_reps_from_tasks(reps,train_pair).reshape(-1, dim)

        classifier = svm.LinearSVC(max_iter=100_000)
        classifier.classes_=[-1, 1]
        classifier.fit(train_reps, np.array([0]*num_trials+[1]*num_trials))

        for j, test_pair in enumerate(test_pairs): 
            decoding_corrects0 = np.array([0]*num_trials) == classifier.predict(reps[TASK_LIST.index(test_pair[0]), ...].reshape(-1, dim))
            decoding_corrects1 = np.array([1]*num_trials) == classifier.predict(reps[TASK_LIST.index(test_pair[1]), ...].reshape(-1, dim))
            decoding_score = np.array([decoding_corrects0, decoding_corrects1]).mean()
            decoding_score_arr[i, j] = decoding_score

            in_test_pair = any([holdout_involved in test_pair for holdout_involved in holdouts_involved])
            in_train_pair = any([holdout_involved in train_pair for holdout_involved in holdouts_involved])
            if in_test_pair or in_train_pair: 
                holdouts_score_list.append(decoding_score)

    return decoding_score_arr, np.mean(holdouts_score_list)

def get_multitask_CCGP(exp_folder, model_name, seed, save=False, layer='task'):
    multi_CCGP = np.full((len(TASK_LIST), len(DICH_DICT)), np.NAN)
    model = make_default_model(model_name)
    model.load_model(exp_folder+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))

    if layer == 'task':
        reps = get_task_reps(model, num_trials = 100,  main_var=True)
    else: 
        reps = get_instruct_reps(model.langModel, depth=layer)
    
    for i, dich in enumerate(DICH_DICT.values()):
        decoding_score_arr, _ = get_dich_CCGP(reps, dich)
        decoding_score = np.mean(decoding_score_arr)
        for task in dich[0]+dich[1]: 
            multi_CCGP[TASK_LIST.index(task), i] = decoding_score
    

    task_holdout_scores = np.nanmean(multi_CCGP, axis=1)
    dich_holdout_scores = np.nanmean(multi_CCGP, axis=0)

    if save:
        file_path = exp_folder+'/CCGP_scores/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+'layer'+layer+'_task_multi_seed'+str(seed), task_holdout_scores)
        np.save(file_path+'/'+'layer'+layer+'_dich_multi_seed'+str(seed), dich_holdout_scores)



def update_holdout_CCGP(reps, holdouts, holdout_CCGP_array): 
    for i, values in enumerate(DICH_DICT.items()): 
        _, dich = values
        holdouts_involved = [holdout for holdout in holdouts if holdout in dich[0]+dich[1]]
        if len(holdouts_involved)>0:
            _, holdout_score = get_dich_CCGP(reps, dich, holdouts_involved=holdouts_involved)
            indices = [TASK_LIST.index(task) for task in holdouts_involved]
            holdout_CCGP_array[indices, i] = holdout_score
        else: 
            continue

def get_holdout_CCGP(exp_folder, model_name, seed, save=False, layer='task'): 
    holdout_CCGP = np.full((len(TASK_LIST), len(DICH_DICT)), np.NAN)
    if 'swap_holdouts' in exp_folder: 
        exp_dict = SWAPS_DICT

    model = make_default_model(model_name)

    for holdout_file, holdouts in exp_dict.items():
        print('processing '+ holdout_file)
        model.load_model(exp_folder+'/'+holdout_file+'/'+model.model_name, suffix='_seed'+str(seed))

        if layer == 'task':
            reps = get_task_reps(model, num_trials = 100,  main_var=True)
        else: 
            reps = get_instruct_reps(model.langModel, depth=layer)

        update_holdout_CCGP(reps, holdouts, holdout_CCGP)

    task_holdout_scores = np.nanmean(holdout_CCGP, axis=1)
    dich_holdout_scores = np.nanmean(holdout_CCGP, axis=0)

    if save:
        file_path = exp_folder+'/CCGP_scores/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+'layer'+layer+'_task_holdout_seed'+str(seed), task_holdout_scores)
        np.save(file_path+'/'+'layer'+layer+'_dich_holdout_seed'+str(seed), dich_holdout_scores)

    return task_holdout_scores, dich_holdout_scores
