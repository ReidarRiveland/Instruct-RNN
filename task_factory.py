import stat
from numpy.core.fromnumeric import argmax
from numpy.lib.twodim_base import tri
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


STIM_DIM = 32
TUNING_DIRS = [((2*np.pi*i)/32) for i in range(STIM_DIM)]
TRIAL_LEN = int(120)
INPUT_DIM = 1 + STIM_DIM*2
OUTPUT_DIM = STIM_DIM + 1
DELTA_T = 20

def choose_pro(x): 
    return x

def choose_anti(x): 
    return (x + np.pi)%(2*np.pi)

def _add_noise(array, sigma):
    noise = sigma * np.random.normal(size=array.shape)
    return array+noise

class TaskFactory(): 
    def __init__(self, num_trials,  fill_type, noise, intervals):
        self.num_trials = num_trials
        self.fill_type = fill_type
        self.noise = noise
        
        if intervals is not None: 
            self.intervals = intervals
        else: 
            self.intervals = self.make_intervals
                
    def _draw_ortho_dirs(self): 
        dir1 = np.random.uniform(0, 2*np.pi)
        dir2 = (dir1+np.pi+np.random.uniform(-np.pi*0.2, np.pi*0.2))%(2*np.pi)
        return (dir1, dir2)

    def make_intervals(self): 
        intervals = np.empty((self.num_trials, 5), dtype=tuple)
        for i in range(self.num_trials):
            T_go = (int(TRIAL_LEN - np.floor(np.random.uniform(300, 600)/DELTA_T)), TRIAL_LEN)
            T_stim2 = (int(T_go[0]-np.floor(np.random.uniform(300, 600)/DELTA_T)), T_go[0])
            T_delay = (int(T_stim2[0]-np.floor(np.random.uniform(300, 600)/DELTA_T)), T_stim2[0])
            T_stim1 = (int(T_delay[0]-np.floor(np.random.uniform(300, 600)/DELTA_T)), T_delay[0])
            T_fix = (0, T_stim1[0])
            intervals[i] = (T_fix, T_stim1, T_delay, T_stim2, T_go)
        return intervals

    def _expand_along_intervals(self, intervals: np.ndarray, activity_vecs: np.ndarray) -> np.ndarray:
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

    def _make_activity_vectors(self, mod_dir_str_conditions: np.ndarray) -> np.ndarray:
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

    def make_trial_inputs(self, fill_style: str, conditions_arr: np.ndarray, intervals, sigma_in) -> np.ndarray:
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


        stim1 = self._make_activity_vectors(self.cond_arr[:, 0, :, :])
        stim2 = self._make_activity_vectors(self.cond_arr[:, 1, :, :])

        if self.fill_style=='full': 
                input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1+stim2), 1), np.concatenate((fix, stim1+stim2), 1),  
                                    np.concatenate((fix, stim1+stim2), 1), np.concatenate((no_fix, null_stim), 1)])

        elif fill_style =='RT':
            input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, null_stim), 1), 
                        np.concatenate((fix, null_stim), 1),  np.concatenate((fix, null_stim), 1), np.concatenate((no_fix, stim1+stim2), 1)])
            
        elif fill_style == 'delay':
            input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1), 1), np.concatenate((fix, null_stim), 1),  
                                np.concatenate((fix, stim2), 1), np.concatenate((no_fix, null_stim), 1)])

        return _add_noise(self._expand_along_intervals(intervals, input_activity_vecs), self.noise)

    def make_loss_mask(self, intervals) -> np.ndarray: 
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
        

    def make_trial_targets(self, target_dirs: np.array, intervals) -> np.ndarray: 
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
        target_activities = self._make_activity_vectors(target_conditions)
        resp = np.concatenate((go, target_activities+0.05), 1)
        no_resp = np.concatenate((fix, np.full((num_trials, STIM_DIM), 0.05)), 1)
        trial_target = self._expand_along_intervals(intervals, (no_resp, no_resp, no_resp, no_resp, resp))
        return trial_target

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


class GoFactory(TaskFactory): 
    def __init__(self, num_trials,  noise, dir_chooser,
                            fill_type= 'full', mod=None, multi=False, 
                            intervals= None, conditions_arr=None):
        super().__init__(num_trials, fill_type, noise, intervals)
        self.dir_chooser = dir_chooser
        self.mod = mod
        self.multi = multi

        if conditions_arr is None: 
            self.cond_arr = self._make_cond_arr()
        else: 
            self.cond_arr = conditions_arr

        self.target_dirs = self._set_target_dirs()

    def _make_cond_arr(self):        
        #mod, stim, dir_strengths, num_trials
        conditions_arr = np.full((2, 2, 2, self.num_trials), np.NaN)
        for i in range(self.num_trials):
            if self.multi:    
                directions = self._draw_ortho_dirs()
                base_strength = np.random.uniform(1.0, 1.2, size=2)
                conditions_arr[0, 0, :, i] = [directions[0], base_strength[0]]
                conditions_arr[1, 0, :, i] = [directions[1], base_strength[1]]

            else:
                direction = np.random.uniform(0, 2*np.pi)
                base_strength = np.random.uniform(1.0, 1.2)    
                tmp_mod = np.random.choice([0, 1])
                conditions_arr[tmp_mod, 0, :, i] = [direction, base_strength]
                #conditions_arr[((tmp_mod+1)%2), 0, :, i] = [np.nan, np.nan]
            if (conditions_arr[:, :, 1, :]>2).any(): 
                raise ValueError
        return conditions_arr
        
    def _set_target_dirs(self): 
        if self.mod is None: 
            dirs = np.nansum(self.cond_arr[:, 0, 0, :], axis=0)
        else: 
            dirs = self.cond_arr[self.mod, 0, 0, :]

        return self.dir_chooser(dirs)

def _set_dm_target_dirs(cond_arr, str_chooser, mod, conf_threshold=None, noise=None): 
    if np.isnan(cond_arr).any(): 
        directions = np.nansum(cond_arr[:, :, 0, :], axis=0)
    else: 
        directions = cond_arr[0, :, 0, :]

    if mod is None: 
        strs = np.nansum(cond_arr[:, :, 1, :], axis=0)
    else: 
        strs = cond_arr[mod, :, 1, :]

    chosen_str = str_chooser(strs, axis=0)
    target_dirs = np.where(chosen_str, directions[1, :], directions[0, :])
    if conf_threshold is not None: 
        d_prime = abs(np.nansum(cond_arr[:, 0, 1, :]-cond_arr[:, 1, 1, :], axis=0))/noise
        target_dirs = np.where(d_prime>conf_threshold, target_dirs, np.NaN)
    return target_dirs

def _set_order_target_dirs(cond_arr, respond_stim):
    target_dirs = np.nansum(cond_arr[:, respond_stim-1, 0, :], axis=0)
    return target_dirs

def _dual_stim_factory(num_trials, multi):        
    #mod, stim, dir_strengths, num_trials
    conditions_arr = np.full((2, 2, 2, num_trials), np.NaN)

    for i in range(num_trials):
        directions = _draw_ortho_dirs()

        if multi:    
            base_strength = np.random.uniform(0.8, 1.2, size=2)

            redraw = True
            while redraw: 
                coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2], size=2, replace=False)
                if coh[0] != -1*coh[1] and (coh[0] <0 or coh[1] < 0): 
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

def order_factory(num_trials, respond_stim, multi=False):
    conditions_arr = _dual_stim_factory(num_trials, multi=multi)
    target_dirs = _set_order_target_dirs(conditions_arr, respond_stim)
    return conditions_arr, target_dirs

def dm_factory(num_trials, str_chooser, mod, multi=False):
    conditions_arr = _dual_stim_factory(num_trials, multi=multi)
    target_dirs = _set_dm_target_dirs(conditions_arr, str_chooser, mod)
    return conditions_arr, target_dirs


def con_dm_factory(num_trials, str_chooser, mod, noise, conf_threshold = 0.8, multi=False):
    conditions_arr = _dual_stim_factory(num_trials, multi=multi)
    target_dirs = _set_dm_target_dirs(conditions_arr, str_chooser, mod, conf_threshold=conf_threshold, noise=noise)
    return conditions_arr, target_dirs


def _set_comp_strs(pos_str, neg_str, req_resp, resp_stim):
    if (resp_stim==1 and req_resp) or (resp_stim==2 and not req_resp): 
        strs = np.array([pos_str, neg_str])
    elif (resp_stim==2 and req_resp) or (resp_stim==1 and not req_resp): 
        strs = np.array([neg_str, pos_str])
    return strs

def comp_factory(num_trials, resp_stim, mod, multi=False):
    #mod, stim, dir_strengths, num_trials
    conditions_arr = np.full((2, 2, 2, num_trials), np.NaN)
    target_dirs = np.empty(num_trials)
    
    if num_trials >1: 
        requires_response_list = list(np.random.permutation([True]*int(num_trials/2) + [False] * int(num_trials/2)))
    else: 
        requires_response_list = [np.random.choice([True, False])]

    for i in range(num_trials): 
        directions = _draw_ortho_dirs()
        requires_response = requires_response_list.pop()

        if requires_response and resp_stim==1:
            target_dirs[i] = directions[0]
        elif requires_response and resp_stim==2: 
            target_dirs[i] = directions[1]
        else: 
            target_dirs[i] = None
  
        if multi: 
            base_strength = np.random.uniform(0.8, 1.2, size=2)
            redraw = True
            while redraw: 
                coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2], size=2, replace=False)
                if coh[0] != -1*coh[1] and (coh[0] <0 or coh[1] < 0): 
                    redraw = False
            
            tmp_strengths = np.array([base_strength + coh, base_strength - coh]).T
            if mod == None: 
                candidate_strs = np.sum(tmp_strengths, axis=0)
            else: 
                candidate_strs = tmp_strengths[mod, :]

            positive_index = argmax(candidate_strs)
            positive_strength = tmp_strengths[:, positive_index]
            negative_strength = tmp_strengths[:, (positive_index+1)%2]
            strs = _set_comp_strs(positive_strength, negative_strength, requires_response, resp_stim)
            conditions_arr[:, :, 0, i] = np.array([directions, directions])
            conditions_arr[:, :, 1, i] = strs.T

        else: 
            base_strength = np.random.uniform(0.8, 1.)
            coh = np.random.choice([0.1, 0.15, 0.2])
            positive_strength = base_strength + coh
            negative_strength = base_strength - coh
            strs = _set_comp_strs(positive_strength, negative_strength, requires_response, resp_stim)
            mod = np.random.choice([0, 1])
            conditions_arr[mod, :, 0, i] = np.array([directions])
            conditions_arr[mod, :, 1, i] = strs
            conditions_arr[((mod+1)%2), :, :, i] = np.NaN
    
    return conditions_arr, target_dirs

def matching_factory(num_trials: int, matching_task:bool, match_type: str):
    #mod, stim, dir_strengths, num_trials
    conditions_arr = np.full((2, 2, 2, num_trials), np.NaN)
    target_dirs = np.empty(num_trials)

    if num_trials>1: 
        match_trial_list = list(np.random.permutation([True]*int(num_trials/2) + [False] * int(num_trials/2)))
    else: 
        match_trial_list = [np.random.choice([True, False])]

    for i in range(num_trials): 
        match_trial = match_trial_list.pop()

        direction1 = np.random.uniform(0, 2*np.pi)
        if direction1 < np.pi: 
            cat_range = (0, np.pi)
        else: cat_range = (np.pi, 2*np.pi)
        
        if match_trial and match_type=='stim': direction2 = direction1
        elif match_trial and match_type=='cat': direction2 = np.random.uniform(cat_range[0], cat_range[1])
        elif not match_trial and match_type=='cat': direction2 = (np.random.uniform(cat_range[0]+(np.pi/3), cat_range[1]-(np.pi/3)) + np.pi)%(2*np.pi)
        else: direction2 = (direction1 + np.random.uniform(np.pi/3, (2*np.pi - np.pi/3)))%(2*np.pi)
        
        mod = np.random.choice([0, 1])
        conditions_arr[mod, :, 0, i] = np.array([direction1, direction2])
        conditions_arr[mod, :, 1, i] = np.array([1, 1])
        conditions_arr[((mod+1)%2), :, :, i] = np.NaN

        if match_trial and matching_task:
            target_dirs[i] = direction1
        elif not match_trial and not matching_task:
            target_dirs[i] = direction2
        else: target_dirs[i] = None
    return conditions_arr, target_dirs

