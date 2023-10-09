from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import DMFactory, TRIAL_LEN, _get_default_intervals, max_var_dir
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import get_task_info, get_instruction_dict, sort_vocab
import instructRNN.models.full_models as full_models
from instructRNN.tasks.tasks import DICH_DICT, TASK_LIST, SWAPS_DICT
from instructRNN.data_loaders.perfDataFrame import *

from tqdm import tqdm
from itertools import permutations
import os

if torch.cuda.is_available:
    device = torch.device(0)
else: 
    device = torch.device('cpu')

def task_eval(model, task, batch_size, instruct_mode = None, **trial_kwargs): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, **trial_kwargs)
    info = get_task_info(batch_size, task, model.info_type, instruct_mode=instruct_mode)
    out, _ = model(torch.Tensor(ins).to(model.__device__), info)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def task_eval_info_embedded(model, task, batch_size, info_embedded, **trial_kwargs): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, **trial_kwargs)
    out, _ = model(torch.Tensor(ins).to(model.__device__), info_embedded=info_embedded)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def task_eval_compositional(model, task, batch_size, reference_tasks=None, **trial_kwargs): 
    if reference_tasks is None: 
        reference_tasks = construct_trials(task).comp_ref_tasks
    info_embedded = model.get_comp_task_rep(reference_tasks, batch_size)
    return task_eval_info_embedded(model, task, batch_size, info_embedded, **trial_kwargs)

def task_eval_compositional_all_combos(model, task, batch_size, **trial_kwargs):
    all_task_combos = sorted(permutations(TASK_LIST, 3))
    perf_array = np.full(117_600, np.NaN)

    for i, reference_tasks in tqdm(enumerate(all_task_combos), desc='combos tested'): 
        perf_array[i] = task_eval_compositional(model, task, batch_size, reference_tasks = reference_tasks)
    
    return perf_array

def _get_model_performance(_task_eval_func, model, num_repeats, batch_len, **eval_kwargs): 
    model.eval()
    perf_array = np.empty(len(TASK_LIST))
    with torch.no_grad():
        for i, task in enumerate(TASK_LIST):
            mean_list = [] 
            for _ in range(num_repeats): 
                frac_correct = _task_eval_func(model, task, batch_len, **eval_kwargs)
                mean_list.append(frac_correct)
            perf_array[i] = np.mean(mean_list)
    return perf_array

def eval_model_perf(model, num_repeats=1, batch_len=128, instruct_mode=None): 
    return _get_model_performance(task_eval, model, num_repeats, batch_len, instruct_mode=instruct_mode)

def eval_model_compositional_perf(model, num_repeats=1, batch_len=128):
    return _get_model_performance(task_eval_compositional, model, num_repeats, batch_len)

def get_val_perf(foldername, model_name, seed, num_repeats = 5, batch_len=128, save=False): 
    model = full_models.make_default_model(model_name)
    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    perf_array = eval_model_perf(model, num_repeats=num_repeats, batch_len=batch_len, instruct_mode='validation')
    if save:
        file_path = foldername+'/val_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+model_name+'_val_perf_seed'+str(seed), perf_array)

    return perf_array

def get_holdout_all_comp_perf(foldername, model_name, labeled_holdouts, seed, num_repeats = 1, batch_len=50, save=False): 
    if 'swap' in foldername: 
        exp_dict = SWAPS_DICT

    holdout_label, tasks = labeled_holdouts
    model = full_models.make_default_model(model_name)
    model.load_model(foldername+'/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
    model.to(device)
    file_path = foldername+'/all_comp_scores/'+model_name

    with torch.no_grad():
        print('processing '+holdout_label)
        for task in tasks: 
            if os.path.exists(file_path+'/'+model_name+'_'+task+'_holdout_comp_scores_seed'+str(seed)+'.npy'):
                print('Already processed task '+task)
                continue
            else:
                print('processing task '+task)
                task_perf = eval_model_compositional_perf(model)

                if save:
                    if os.path.exists(file_path):
                        pass
                    else: os.makedirs(file_path)
                    np.save(file_path+'/'+model_name+'_'+task+'_holdout_comp_scores_seed'+str(seed), task_perf)


def get_multi_all_comp_perf(foldername, model_name, seed, num_repeats = 1, batch_len=50, save=False): 
    perf_array = np.full((len(TASK_LIST), 117_600), np.NaN)
    model = full_models.make_default_model(model_name)

    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    model.to(device)
    file_path = foldername+'/all_comp_scores/'+model_name

    for task in TASK_LIST:
        if os.path.exists(file_path+'/'+model_name+'_'+task+'_multi_comp_scores_seed'+str(seed)+'.npy'):
            print('Already processed task '+task)
            continue
        else:
            print('processing task '+task)
            task_perf = task_eval_compositional_all_combos(model, task, batch_len)

            if save:
                if os.path.exists(file_path):
                    pass
                else: os.makedirs(file_path)
                np.save(file_path+'/'+model_name+'_'+task+'_multi_comp_scores_seed'+str(seed), task_perf)

def get_multi_comp_perf(foldername, model_name, seed, num_repeats = 1, batch_len=50, save=False): 
    model = full_models.make_default_model(model_name)
    file_path = foldername+'/multi_comp_perf/'+model_name

    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    model.to(device)
    file_path = foldername+'/multi_comp_perf/'+model_name
    task_perf = eval_model_compositional_perf(model)

    if save:
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+model_name+'_multi_comp_perf_seed'+str(seed), task_perf)
    return task_perf


def _get_model_0_shot(_task_eval_func, model_name, folder_name, exp_type, seed, batch_size, **eval_kwargs): 
    if 'swap' in exp_type: 
        exp_dict = SWAPS_DICT
    perf_array = np.full(len(TASK_LIST), np.NaN)
    model = full_models.make_default_model(model_name)
    with torch.no_grad():
        for holdout_label, tasks in exp_dict.items(): 
            model.load_model(folder_name+'/'+exp_type+'_holdouts/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
            for task in tasks: 
                perf_array[TASK_LIST.index(task)] = _task_eval_func(model, task, batch_size, **eval_kwargs)
    return perf_array

def eval_model_0_shot(model_name, folder_name, exp_type, seed, batch_size = 128, instruct_mode=None):
    return _get_model_0_shot(task_eval, model_name, folder_name, exp_type, seed, batch_size = batch_size, instruct_mode=None)

def eval_model_compositional_0_shot(model_name, folder_name, exp_type, seed, batch_size = 128):
    return _get_model_0_shot(task_eval_compositional, model_name, folder_name, exp_type, seed, batch_size = batch_size)

