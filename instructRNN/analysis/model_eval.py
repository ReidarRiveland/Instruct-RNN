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

def task_eval(model, task, batch_size, instruct_mode = None, **trial_kwargs): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, **trial_kwargs)
    info = get_task_info(batch_size, task, model.info_type, instruct_mode=instruct_mode)
    out, _ = model(torch.Tensor(ins).to(model.__device__), info)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def task_eval_info_embedded(model, task, batch_size, info_embedded, **trial_kwargs): 
    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size, **trial_kwargs)
    out, _ = model(torch.Tensor(ins).to(model.__device__), info_embedded=info_embedded)
    return np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))

def task_eval_compositional(model, task, batch_size, **trial_kwargs): 
    info_embedded = model.get_comp_task_rep(task, batch_size)
    return task_eval_info_embedded(model, task, batch_size, info_embedded, **trial_kwargs)

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

def eval_model_info_embedded_perf(model, num_repeats=1, batch_len=128, info_embedded=None):
    assert info_embedded is not None, 'must enter a value for info embedded'
    return _get_model_performance(task_eval_info_embedded, model, num_repeats, batch_len, info_embedded=info_embedded)

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

def get_multi_comp_perf(foldername, model_name, seed, num_repeats = 5, batch_len=128, save=False): 
    model = full_models.make_default_model(model_name)
    model.load_model(foldername+'/Multitask/'+model.model_name, suffix='_seed'+str(seed))
    perf_array = eval_model_compositional_perf(model, num_repeats=num_repeats, batch_len=batch_len)
    if save:
        file_path = foldername+'/multi_comp_perf/'+model_name
        if os.path.exists(file_path):
            pass
        else: os.makedirs(file_path)
        np.save(file_path+'/'+model_name+'_multi_comp_perf_seed'+str(seed), perf_array)

    return perf_array

def _get_model_0_shot(_task_eval_func, model_name, folder_name, exp_type, seed, batch_size, **eval_kwargs): 
    if 'swap' in exp_type: 
        exp_dict = SWAPS_DICT
    perf_array = np.full(len(TASK_LIST), np.NaN)
    model = full_models.make_default_model(model_name)
    with torch.no_grad():
        for holdout_label, tasks in exp_dict.items(): 
            model.load_model(folder_name+'/'+exp_type+'_holdouts/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
            perf_array[TASK_LIST.index(task)] = _task_eval_func(model, task, batch_size, **eval_kwargs)
    return perf_array

def eval_model_0_shot(model_name, folder_name, exp_type, seed, batch_size = 128, instruct_mode=None):
    return _get_model_0_shot(task_eval, model_name, folder_name, exp_type, seed, batch_size = batch_size, instruct_mode=None)

def eval_model_compositional_0_shot(model_name, folder_name, exp_type, seed, batch_size = 128):
    return _get_model_0_shot(task_eval_compositional, model_name, folder_name, exp_type, seed, batch_size = batch_size)