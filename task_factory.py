import stat
from numpy.core.fromnumeric import argmax
from numpy.lib.twodim_base import tri
import numpy as np

1+ np.nan

TASK_LIST = ['Go', 'Anti Go', 'RT Go', 'Anti RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM', 'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'DMC', 'DNMC']
SWAPPED_TASK_LIST = ['Anti DM', 'MultiCOMP1', 'DNMC', 'DMC', 'MultiCOMP2', 'Go', 'DNMS', 'COMP1', 'Anti MultiDM', 'DMS', 'Anti Go', 'DM', 'COMP2', 'MultiDM', 'Anti RT Go', 'RT Go']
TASK_GROUP_DICT = {'Go': ['Go', 'Anti Go', 'RT Go', 'Anti RT Go'],
            'DM': ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], 
            'COMP': ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'],
            'Delay': ['DMS', 'DNMS', 'DMC', 'DNMC']}
STIM_DIM = 32
TUNING_DIRS = [((2*np.pi*i)/32) for i in range(STIM_DIM)]
TRIAL_LEN = int(120)
INPUT_DIM = 1 + STIM_DIM*2
OUTPUT_DIM = STIM_DIM + 1
DELTA_T = 20

def add_noise(array, sigma):
    noise = np.sqrt(2/DELTA_T)*(sigma) * np.random.normal(size=array.shape)
    return array+noise

def _draw_ortho_dirs(): 
    dir1 = np.random.uniform(0, 2*np.pi)
    dir2 = (dir1+np.pi+np.random.uniform(-np.pi*0.2, np.pi*0.2))%(2*np.pi)
    return (dir1, dir2)

def get_swap(task): 
    return SWAPPED_TASK_LIST[TASK_LIST.index(task)]

def make_intervals(num_trials): 
    intervals = np.empty((num_trials, 5), dtype=tuple)
    for i in range(num_trials):
        T_go = (int(TRIAL_LEN - np.floor(np.random.uniform(300, 600)/DELTA_T)), TRIAL_LEN)
        T_stim2 = (int(T_go[0]-np.floor(np.random.uniform(300, 600)/DELTA_T)), T_go[0])
        T_delay = (int(T_stim2[0]-np.floor(np.random.uniform(300, 600)/DELTA_T)), T_stim2[0])
        T_stim1 = (int(T_delay[0]-np.floor(np.random.uniform(300, 600)/DELTA_T)), T_delay[0])
        T_fix = (0, T_stim1[0])
        intervals[i] = (T_fix, T_stim1, T_delay, T_stim2, T_go)
    return intervals

def _expand_along_intervals(intervals: np.ndarray, activity_vecs: np.ndarray) -> np.ndarray:
    '''
    Exapnds activities in fill_vecs by time intervals given by intervals
    Parameters: 
        intervals[num_trials, 5]: ndarray of tuples containing the start and end times for given stimulus in a trial 
        activity_vecs[5, num_trials, activity_dim]: ndarray containing vectors of activity for a number of trials 
            to be expanded in the time intervals defined by the intervals parameter 
    Returns: 
        batch_array[num_trials, TRIAL_LEN, activity_dim]: ndarray with activity vectors expanded across time intervals 
            for a given number of trials
    '''
    num_trials = intervals.shape[0]
    activity_fills = np.swapaxes(activity_vecs, 1, 2)
    intervals = np.expand_dims(intervals.T, axis= 1)
    zipped_fill = np.reshape(np.concatenate((activity_fills, intervals), axis = 1), (-1, num_trials))

    def _filler(zipped_filler): 
        filler_array = zipped_filler.reshape(5, -1)
        trial_array = np.empty((filler_array.shape[1]-1, TRIAL_LEN))
        for i in range(filler_array.shape[0]): 
            start, stop, filler_vec = filler_array[i, -1][0], filler_array[i, -1][1], filler_array[i, :-1]
            if start == stop:
                continue
            trial_array[:, start:stop] = np.repeat(np.expand_dims(filler_vec, axis=1), stop-start, axis=1)
        return trial_array

    batch_array = np.apply_along_axis(_filler, 0, zipped_fill)
    return batch_array.T

def _make_activity_vectors(mod_dir_str_conditions: np.ndarray) -> np.ndarray:
    '''
    Constructs multi-dim arrays representing hills of activity across modalities and trials for a given stimulus 
    Modality dim may be 1 to create target activities
    Parameters: 
        mod_dir_str_conditions[mod, dir_str, num_trials]:  array contatining stimulus conditions for modalities and trials 
    Returns: 
        np.ndarray[num_trials, STIM_DIM*mod]: array representing hills of activity for both modalities in a given stimulus 
    '''
    mod_dim = mod_dir_str_conditions.shape[0]
    num_trials = mod_dir_str_conditions.shape[-1]
    centered_dir = np.repeat(np.array([[0.8*np.exp(-0.5*(((8*abs(np.pi-i))/np.pi)**2)) for i in TUNING_DIRS]]), num_trials*2, axis=0)
    roll = np.nan_to_num(np.floor((mod_dir_str_conditions[: , 0, :]/(2*np.pi))*STIM_DIM)- np.floor(STIM_DIM/2)).astype(int)
    rolled = np.array(list(map(np.roll, centered_dir, np.expand_dims(roll.flatten(), axis=1)))) * np.expand_dims(np.nan_to_num(mod_dir_str_conditions[:, 1, :]).flatten() , axis=1)
    if mod_dim>1: 
        rolled_reshaped = np.concatenate((rolled.reshape(mod_dim, num_trials, STIM_DIM)[0, :, :], rolled.reshape(mod_dim, num_trials, STIM_DIM)[1, :, :]), axis=1)
    else:
        rolled_reshaped = rolled.reshape(num_trials, -1)

    return rolled_reshaped

def make_trial_inputs(fill_style: str, conditions_arr: np.ndarray, intervals, sigma_in) -> np.ndarray:
    '''
    Creates stimulus activity arrays for given trials of a particular task type 
    Parameters: 
        task_type: string identifying the task type
        conditions_arr[mods, stim, dir_str, num_trials]: array defining the stimulus conditions for a batch of task trials 
    Returns: 
        ndarray[num_trials, TRIAL_LEN, INPUT_DIM]: array conditing stimulus inputs for a batch of task trials 
    '''
    num_trials = conditions_arr.shape[-1]
    fix = np.ones((num_trials,1)) 
    no_fix = np.zeros((num_trials,1))
    null_stim = np.zeros((num_trials, STIM_DIM*2))


    stim1 = _make_activity_vectors(conditions_arr[:, 0, :, :])
    stim2 = _make_activity_vectors(conditions_arr[:, 1, :, :])

    if fill_style=='full': 
            input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1+stim2), 1), np.concatenate((fix, stim1+stim2), 1),  
                                np.concatenate((fix, stim1+stim2), 1), np.concatenate((no_fix, null_stim), 1)])

    elif fill_style =='RT':
        input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1), 1), 
                    np.concatenate((fix, stim1), 1),  np.concatenate((fix, stim1), 1), np.concatenate((no_fix, stim1+stim2), 1)])
        
    elif fill_style == 'delay':
        input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1), 1), np.concatenate((fix, null_stim), 1),  
                            np.concatenate((fix, stim2), 1), np.concatenate((no_fix, null_stim), 1)])
    else: 
        input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1), 1), np.concatenate((fix, null_stim), 1),  
                            np.concatenate((fix, stim2), 1), np.concatenate((no_fix, null_stim), 1)])
    
    return add_noise(_expand_along_intervals(intervals, input_activity_vecs), sigma_in)

def make_loss_mask(intervals) -> np.ndarray: 
    '''
    Defines loss mask used during task training where there is a zero weight 
    grace period in the first 5 time steps of the response stimulus followed by 
    higher weigthing 
    Parameters:
        None 
    Returns: 
        ndarray[num_trials, TRIAL_LEN, OUTPUT_DIM]: ndarray of weights for loss function
    '''
    def __make_loss_mask__(intervals): 
        ones = np.ones((1, OUTPUT_DIM))
        zeros = np.zeros((1, OUTPUT_DIM))
        go_weights = np.full((1, OUTPUT_DIM), 5)
        go_weights[:,0] = 10

        pre_go = intervals[-1][0]
        zero_per = int(np.floor(100/DELTA_T))

        pre_go_mask = ones.repeat(pre_go, axis = 0)
        zero_mask = zeros.repeat(zero_per, axis=0)
        go_mask = go_weights.repeat((TRIAL_LEN-(pre_go+zero_per)), axis = 0)
        return np.concatenate((pre_go_mask, zero_mask, go_mask), 0)
    return np.array(list(map(__make_loss_mask__, intervals)))
    

def make_trial_targets(target_dirs: np.array, intervals) -> np.ndarray: 
    '''
    Makes target output activities for a batch of trials
    Parameters:
        target_dirs[num_trials]
    Returns: 
        ndarray[num_trials, TRIAL_LEN, OUTPUT_DIM]: ndarray of target activities for a batch of trials
    '''
    num_trials = target_dirs.shape[0]
    fix = np.full((num_trials,1), 0.85)
    go = np.full((num_trials,1), 0.05)
    strengths = np.where(np.isnan(target_dirs), np.zeros_like(target_dirs), np.ones_like(target_dirs))
    target_conditions = np.expand_dims(np.stack((target_dirs, strengths), axis=1).T, axis=0)
    target_activities = _make_activity_vectors(target_conditions)
    resp = np.concatenate((go, target_activities+0.05), 1)
    no_resp = np.concatenate((fix, np.full((num_trials, STIM_DIM), 0.05)), 1)
    trial_target = _expand_along_intervals(intervals, (no_resp, no_resp, no_resp, no_resp, resp))
    return trial_target

def make_dual_stim_conditions(num_trials, multi=False):        
    #mod, stim, dir_strengths, num_trials
    conditions_arr = np.empty((2, 2, 2, num_trials))

    for i in range(num_trials):
        directions = _draw_ortho_dirs()

        if multi:    
            base_strength = np.random.uniform(0.8, 1.2, size=2)

            redraw = True
            while redraw: 
                coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2], size=2, replace=False)
                if coh[0] != -1*coh[1]: 
                    redraw = False
            
            strengths = np.array([base_strength + coh, base_strength - coh]).T
            conditions_arr[:, :, 0, i] = np.array([directions, directions])
            conditions_arr[:, :, 1, i] = strengths

        else:
            mod = np.random.choice([0, 1])
            base_strength = np.random.uniform(0.8, 1.2)
            coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2])

            strengths = np.array([base_strength+coh, base_strength-coh]).T
            
            conditions_arr[mod, :, 0, i] = np.array(directions)
            conditions_arr[mod, :, 1, i] = strengths
            conditions_arr[((mod+1)%2), :, :, i] = np.NaN
    return conditions_arr

import seaborn as sns
import matplotlib.pyplot as plt

def plot_trial(ins, tars, task_type):
    ins = ins.T
    tars = tars.T
    fix = np.expand_dims(ins[0, :], 0)
    mod1 = ins[1:STIM_DIM, :]
    mod2 = ins[1+STIM_DIM:1+(2*STIM_DIM), :]

    to_plot = (fix, mod1, mod2, tars)

    gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 5])

    fig, axn = plt.subplots(4,1, sharex = True, gridspec_kw=gs_kw)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ylabels = ('fix.', 'mod. 1', 'mod. 2', 'Target')
    for i, ax in enumerate(axn.flat):
        sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1.5, cbar_ax=None if i else cbar_ax)

        ax.set_ylabel(ylabels[i])
        if i == 0: 
            ax.set_title('%r Trial Info' %task_type)
        if i == 3: 
            ax.set_xlabel('time')
    plt.show()

