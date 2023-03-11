import torch
import numpy as np
import numpy.linalg as LA

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import chain
import sklearn.svm as svm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import DMFactory, TRIAL_LEN, _get_default_intervals, max_var_dir
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import get_task_info, get_instruction_dict, sort_vocab
import instructRNN.models.full_models as full_models
from instructRNN.tasks.tasks import DICH_DICT, TASK_LIST, SWAPS_DICT
from instructRNN.data_loaders.perfDataFrame import *
import os

if torch.cuda.is_available:
    device = torch.device(0)
else: 
    device = torch.device('cpu')

def get_reps_from_tasks(reps, tasks): 
    indices = [TASK_LIST.index(task) for task in tasks]
    return reps[indices, ...]

def task_eval(model, task, batch_size, instruct_mode = None, instructions = None, comp_task=None, context = None, **trial_kwargs): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, **trial_kwargs)
    if instruct_mode =='': 
        instructions = get_task_info(batch_size, task, model.info_type, instruct_mode=instruct_mode)

    out, _ = model(torch.Tensor(ins).to(model.__device__), instructions, context = context, comp_task=comp_task)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def get_model_performance(model, num_repeats = 1, batch_len=128, instruct_mode=None, context = None, use_comp=False, **trial_kwargs): 
    model.eval()
    perf_array = np.empty(len(TASK_LIST))
    with torch.no_grad():
        for i, task in enumerate(TASK_LIST):
            print(task)
            if use_comp:
                comp_task = task
            else: 
                comp_task = None 

            mean_list = [] 
            for _ in range(num_repeats): 
                frac_correct = task_eval(model, task, batch_len,  instruct_mode = instruct_mode, comp_task=comp_task, context = context, **trial_kwargs)
                mean_list.append(frac_correct)
            perf_array[i] = np.mean(mean_list)
    return perf_array

def get_val_perf(foldername, model_name, seed, num_repeats = 5, batch_len=100, save=False): 
    model = full_models.make_default_model(model_name)
    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    perf_array = get_model_performance(model, num_repeats=num_repeats, batch_len=batch_len, instruct_mode='validation')
    if save:
        file_path = foldername+'/val_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+model_name+'_val_perf_seed'+str(seed), perf_array)

    return perf_array

def get_multi_comp_perf(foldername, model_name, seed, num_repeats = 5, batch_len=100, save=False): 
    model = full_models.make_default_model(model_name)
    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    perf_array = get_model_performance(model, num_repeats=num_repeats, batch_len=batch_len, instruct_mode='combined', use_comp=True)
    if save:
        file_path = foldername+'/multi_comp_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+model_name+'_multi_comp_perf_seed'+str(seed), perf_array)

    return perf_array

def eval_model_0_shot(model_name, folder_name, exp_type, seed, instruct_mode=None, use_comp = False): 
    if 'swap' in exp_type: 
        exp_dict = SWAPS_DICT
    perf_array = np.full(len(TASK_LIST), np.NaN)
    model = full_models.make_default_model(model_name)
    with torch.no_grad():
        for holdout_label, tasks in exp_dict.items(): 
            print(holdout_label)
            model.load_model(folder_name+'/'+exp_type+'_holdouts/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
            for task in tasks: 
                if use_comp: 
                    comp_task = task 
                else: 
                    comp_task = None
                perf_array[TASK_LIST.index(task)] = task_eval(model, task, 64, instruct_mode=instruct_mode, comp_task=comp_task)
    return perf_array

def eval_model_exemplar(model_name, foldername, exp_type, seed, exemplar_num, **trial_kwargs):
    if 'swap' in exp_type: 
        exp_dict = SWAPS_DICT

    perf_array = np.full(len(TASK_LIST), np.NaN)
    model = full_models.make_default_model(model_name)

    if 'simpleNet' in model_name:
        context_dim = 64
    else: 
        context_dim = 64

    with torch.no_grad():
        for holdout_label, tasks in exp_dict.items(): 
            model_folder = foldername+'/'+exp_type+'_holdouts/'+holdout_label+'/'+model.model_name
            print(holdout_label)
            model.load_model(model_folder, suffix='_seed'+str(seed))
            for task in tasks: 
                contexts = pickle.load(open(model_folder+'/contexts/seed'+str(seed)+'_'+task+'exemplar'+str(exemplar_num)+'_context_vecs'+str(context_dim), 'rb'))
                perf_array[TASK_LIST.index(task)] = task_eval(model, task, 50, context=torch.tensor(contexts).repeat(5, 1))

    return perf_array

def get_instruct_reps(langModel, depth='full', instruct_mode=None):
    langModel.eval()
    langModel.to(device)
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
                out_rep = langModel._reducer(langModel.forward_transformer(list(instructions))[1][int(depth)])
            instruct_reps[i, :, :] = out_rep
    return instruct_reps.cpu().numpy().astype(np.float64)

def get_rule_embedder_reps(model):
    reps = np.empty((len(TASK_LIST), model.rule_dim))
    with torch.no_grad():
        for i, task in enumerate(TASK_LIST): 
            task_rule = get_task_info(1, task, False)
            rule_transformed = torch.matmul(task_rule.to(model.__device__), model.rule_transform.float())
            info= model.rule_encoder(rule_transformed)
            reps[i, :] = model.rule_encoder(info).cpu().numpy()
    return reps

def get_task_reps(model, epoch='stim_start', stim_start_buffer=0, num_trials =100, tasks=TASK_LIST, instruct_mode=None, 
                    contexts=None, default_intervals=False, num_repeats=1, use_comp=False, **trial_kwargs):
    model.eval()
    model.to(device)
    if epoch is None or epoch=='stim_after': 
        task_reps = np.full((num_repeats, len(tasks), num_trials, TRIAL_LEN, model.rnn_hidden_dim), np.nan)
    else: 
        task_reps = np.full((num_repeats, len(tasks), num_trials, model.rnn_hidden_dim), np.nan)

    with torch.no_grad(): 
        for k in range(num_repeats):
            for i, task in enumerate(tasks): 
                if use_comp: 
                    comp_task = task
                else: 
                    comp_task = None

                if default_intervals and 'Dur' not in task:
                    intervals = _get_default_intervals(num_trials)
                    ins, targets, _, target_dirs, _ =  construct_trials(task, num_trials, intervals=intervals, **trial_kwargs)
                else: 
                    ins, targets, _, target_dirs, _ =  construct_trials(task, num_trials, **trial_kwargs)

                if contexts is not None: 
                    out, hid = model(torch.Tensor(ins).to(model.__device__), context=contexts[i, ...])
                else: 
                    task_info = get_task_info(num_trials, task, model.info_type, instruct_mode=instruct_mode)
                    out, hid = model(torch.Tensor(ins).to(model.__device__), task_info, comp_task=comp_task)

                hid = hid.cpu().numpy()
                if epoch is None: 
                    task_reps[k, i, ...] = hid
                else: 
                    for j in range(num_trials): 
                        if epoch.isnumeric(): epoch_index = int(epoch)
                        if epoch == 'stim': epoch_index = np.where(targets[j, :, 0] == 0.85)[0][-1]
                        if epoch == 'stim_start' or epoch =='stim_after': epoch_index = np.where(ins[j, :, 1:]>0.6)[0][0]+stim_start_buffer
                        if epoch == 'prep': epoch_index = np.where(ins[j, :, 1:]>0.25)[0][0]-1
                        
                        if epoch =='stim_after':
                            task_reps[k, i, j, epoch_index:, :] = hid[j, epoch_index:, :]
                        else:
                            task_reps[k, i, j, :] = hid[j, epoch_index, :]


    return np.nanmean(task_reps, axis=0).astype(np.float64)


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

def get_layer_sim_scores(model, rep_depth='12', dist = 'pearson'): 
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

    if dist == 'pearson': 
        sim_scores = 1-cosine_similarity(reps.reshape(-1, rep_dim))
    else: 
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


def get_dich_CCGP(reps, dich, holdouts_involved=[], use_mean = False, max_iter=1_000_000):
    dim = reps.shape[-1]
    num_trials = reps.shape[1]
    labels = np.array([0]*num_trials+[1]*num_trials)
    decoding_score_arr = np.full((len(dich), len(dich)-1), np.nan)
    holdouts_score_list = []

    for i, train_pair in enumerate(dich):
        test_pairs = dich.copy()
        test_pairs.remove(train_pair)
        train_reps = get_reps_from_tasks(reps,train_pair).reshape(-1, dim)

        if use_mean:
            train_reps = np.squeeze(get_reps_from_tasks(reps,train_pair).mean(axis=1))
            labels = np.array([0, 1])
        else: 
            train_reps = get_reps_from_tasks(reps,train_pair).reshape(-1, dim)

        classifier = svm.LinearSVC(max_iter=max_iter, random_state = 0, tol=1e-5)
        classifier.classes_=[0, 1]
        classifier.fit(train_reps, labels)

        for j, test_pair in enumerate(test_pairs): 
            if use_mean: 
                test_reps = np.squeeze(get_reps_from_tasks(reps,test_pair).mean(axis=1))
            else: 
                test_reps = get_reps_from_tasks(reps,test_pair).reshape(-1, dim)

            decoding_score = classifier.score(test_reps, labels)
            decoding_score_arr[i, j] = decoding_score

            in_test_pair = any([holdout_involved in test_pair for holdout_involved in holdouts_involved])
            in_train_pair = any([holdout_involved in train_pair for holdout_involved in holdouts_involved])

            if in_test_pair or in_train_pair: 
                holdouts_score_list.append(decoding_score)

    return decoding_score_arr, np.mean(holdouts_score_list)


def get_multitask_CCGP(exp_folder, model_name, seed, save=False, layer='task', max_iter=1_000_000):
    multi_CCGP = np.full((len(TASK_LIST), len(DICH_DICT)), np.NAN)
    model = full_models.make_default_model(model_name)
    model.load_model(exp_folder+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))

    if layer == 'task':
        reps = get_task_reps(model, num_trials = 100)
    elif layer =='full' and model_name =='simpleNetPlus':
        reps = get_rule_embedder_reps(model)[:, None, :]
    else: 
        reps = get_instruct_reps(model.langModel, depth=layer)
    
    for i, dich in enumerate(DICH_DICT.values()):
        decoding_score_arr, _ = get_dich_CCGP(reps, dich, max_iter=max_iter)
        decoding_score = np.mean(decoding_score_arr)
        for task in list(chain(*dich)): 
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
        np.save(file_path+'/'+'layer'+layer+'_array_multi_seed'+str(seed), multi_CCGP)


def update_holdout_CCGP(reps, holdouts, holdout_CCGP_array, use_mean, max_iter=1_000_000): 
    for i, items in enumerate(DICH_DICT.items()): 
        _, dich = items
        all_dich_tasks = list(chain.from_iterable(dich))
        holdouts_involved = [holdout for holdout in holdouts if holdout in all_dich_tasks]
        
        if len(holdouts_involved)>0:
            _, holdout_score = get_dich_CCGP(reps, dich, holdouts_involved=holdouts, use_mean=use_mean)
            indices = [TASK_LIST.index(task) for task in holdouts_involved]
            holdout_CCGP_array[indices, i] = holdout_score
        else: 
            continue

def get_holdout_CCGP(exp_folder, model_name, seed, epoch = 'stim_start', save=False, layer='task', use_mean=False, instruct_mode='combined', max_iter=10_000_000): 
    holdout_CCGP = np.full((len(TASK_LIST), len(DICH_DICT)), np.NAN)
    if 'swap_holdouts' in exp_folder: 
        exp_dict = SWAPS_DICT

    model = full_models.make_default_model(model_name)

    for holdout_file, holdouts in exp_dict.items():
        print('processing '+ holdout_file)
        model.load_model(exp_folder+'/'+holdout_file+'/'+model.model_name, suffix='_seed'+str(seed))

        if layer == 'task':
            reps = get_task_reps(model, num_trials = 250, instruct_mode=instruct_mode, epoch=epoch, use_comp=False)
        elif layer =='full' and model_name =='simpleNetPlus':
            reps = get_rule_embedder_reps(model)[:, None, :]
        else: 
            reps = get_instruct_reps(model.langModel, depth=layer, instruct_mode=instruct_mode)

        if instruct_mode == 'swap_combined':
            swapped_reps = get_task_reps(model, num_trials = 250, instruct_mode='swap_combined', tasks = holdouts, epoch=epoch, use_comp=False)
            for i, holdout in enumerate(holdouts): 
                reps[TASK_LIST.index(holdout), ...] = swapped_reps[i, ...]

        update_holdout_CCGP(reps, holdouts, holdout_CCGP, use_mean=use_mean, max_iter=max_iter)

    task_holdout_scores = np.nanmean(holdout_CCGP, axis=1)
    dich_holdout_scores = np.nanmean(holdout_CCGP, axis=0)

    if save:
        file_path = exp_folder+'/CCGP_scores/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+'layer'+layer+'_task_holdout_seed'+str(seed)+instruct_mode, task_holdout_scores)
        np.save(file_path+'/'+'layer'+layer+'_dich_holdout_seed'+str(seed)+instruct_mode, dich_holdout_scores)
        np.save(file_path+'/'+'layer'+layer+'_array_holdout_seed'+str(seed)+instruct_mode, holdout_CCGP)


    return task_holdout_scores, dich_holdout_scores, holdout_CCGP



def get_perf_ccgp_corr(folder, exp_type, model_list):
    ccgp_scores = np.empty((len(model_list), len(TASK_LIST)))
    perf_scores = np.empty((len(model_list), len(TASK_LIST)))
    for i, model_name in enumerate(model_list): 
        task_ccgp = PerfDataFrame(folder, exp_type, model_name, mode='ccgp')
        data = PerfDataFrame(folder, exp_type, model_name, mode='combined')
        perf_mean, _ = data.avg_seeds(k_shot=0)
        ccgp_mean, _ = task_ccgp.avg_seeds(k_shot=0)
        ccgp_scores[i, :] = ccgp_mean
        perf_scores[i, :] = perf_mean
    corr, p_val = pearsonr(ccgp_scores.flatten(), perf_scores.flatten())

    return corr, p_val, ccgp_scores, perf_scores


def get_norm_task_var(hid_reps): 
    task_var = np.nanmean(np.nanvar(hid_reps[:, :, :,:], axis=1), axis=1)
    task_var = np.delete(task_var, np.where(np.sum(task_var, axis=0)<0.001)[0], axis=1)
    normalized = np.divide(np.subtract(task_var, np.min(task_var, axis=0)[None, :]), (np.max(task_var, axis=0)-np.min(task_var, axis=0)[None, :])).T

    return normalized

def get_optim_clusters(task_var):
    score_list = []
    for i in range(3,25):
        km = KMeans(n_clusters=i, random_state=42)
        labels = km.fit_predict(task_var)
        score = silhouette_score(task_var, labels)
        score_list.append(score)
    return list(range(3, 50))[np.argmax(np.array(score_list))]

def cluster_units(task_var):
    n_clusters = get_optim_clusters(task_var)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(task_var)
    return labels

def sort_units(norm_task_var): 
    labels = cluster_units(norm_task_var)
    cluster_labels, sorted_indices = list(zip(*sorted(zip(labels, range(256)))))
    cluster_dict = defaultdict(list)
    for key, value in zip(cluster_labels, sorted_indices): 
        cluster_dict[key].append(value)
    return cluster_dict, cluster_labels, sorted_indices

def get_cluster_info(load_folder, model_name, seed):
    model = full_models.make_default_model(model_name)
    model.load_model(load_folder+'/'+model.model_name, suffix='_seed'+str(seed))
    task_hid_reps = get_task_reps(model, num_trials = 100, epoch=None, tasks= TASK_LIST, max_var=True)
    norm_task_var = get_norm_task_var(task_hid_reps)
    cluster_dict, cluster_labels, sorted_indices = sort_units(norm_task_var)
    return norm_task_var, cluster_dict, cluster_labels, sorted_indices

def get_model_clusters(foldername, model_name, seed, num_repeats=10, save=False):
    if 'swap' in foldername: 
        exp_dict = SWAPS_DICT

    num_cluster_array = np.full((len(exp_dict), num_repeats), np.nan)
    model = full_models.make_default_model(model_name)

    for i, holdout_file in enumerate(exp_dict.keys()):
        for j in range(num_repeats):
            print('processing '+ holdout_file)
            model.load_model(foldername+'/'+holdout_file+'/'+model.model_name, suffix='_seed'+str(seed))
            task_hid_reps = get_task_reps(model, num_trials = 100, epoch='stim_after', max_var=True)
            task_var = get_norm_task_var(task_hid_reps)
            optim_clusters = get_optim_clusters(task_var)
            num_cluster_array[i, j] = optim_clusters


    if save:
        file_path = foldername+'/cluster_measures/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/optim_clusters_seed'+str(seed), num_cluster_array)

