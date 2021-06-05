import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def add_noise(array):
    noise = np.sqrt(2/Task.DELTA_T)*(Task.SIGMA_IN) * np.random.normal(size=array.shape)
    return array+noise

def _draw_ortho_dirs(): 
    dir1 = np.random.uniform(0, 2*np.pi)
    dir2 = (dir1+np.pi+np.random.uniform(-np.pi*0.2, np.pi*0.2))%(2*np.pi)
    return (dir1, dir2)


class Task():
    TASK_LIST = ['Go', 'RT Go', 'Anti Go', 'Anti RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM', 'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'DMC', 'DNMC']
    SHUFFLED_TASK_LIST = ['MultiCOMP2', 'RT Go', 'COMP2', 'DNMS', 'Anti Go', 'COMP1', 'DNMC', 'Anti DM', 'MultiCOMP1', 'Anti MultiDM', 'MultiDM', 'DM', 'Anti RT Go', 'DMC', 'DMS', 'Go']
    STIM_DIM = 32
    TUNING_DIRS = [((2*np.pi*i)/32) for i in range(STIM_DIM)]
    TRIAL_LEN = int(120)
    INPUT_DIM = 1 + STIM_DIM*2
    OUTPUT_DIM = STIM_DIM + 1
    SIGMA_IN = 0.05
    DELTA_T = 20

    def __init__(self, num_trials, intervals): 
        self.num_trials = num_trials
        self.intervals = np.empty((num_trials, 5), dtype=tuple)
        self.null_stim = np.zeros((self.num_trials, self.STIM_DIM*2))
        if type(intervals) == type(None): 
            for i in range(num_trials):
                T_go = (int(self.TRIAL_LEN - np.floor(np.random.uniform(300, 600)/self.DELTA_T)), self.TRIAL_LEN)
                T_stim2 = (int(T_go[0]-np.floor(np.random.uniform(300, 600)/self.DELTA_T)), T_go[0])
                T_delay = (int(T_stim2[0]-np.floor(np.random.uniform(300, 600)/self.DELTA_T)), T_stim2[0])
                T_stim1 = (int(T_delay[0]-np.floor(np.random.uniform(300, 600)/self.DELTA_T)), T_delay[0])
                T_fix = (0, T_stim1[0])
                self.intervals[i] = (T_fix, T_stim1, T_delay, T_stim2, T_go)
        else: 
            self.intervals = intervals

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

        activity_fills = np.swapaxes(activity_vecs, 1, 2)
        intervals = np.expand_dims(intervals.T, axis= 1)
        zipped_fill = np.reshape(np.concatenate((activity_fills, intervals), axis = 1), (-1, self.num_trials))

        def _filler(zipped_filler): 
            filler_array = zipped_filler.reshape(5, -1)
            trial_array = np.empty((filler_array.shape[1]-1, Task.TRIAL_LEN))
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
        centered_dir = np.repeat(np.array([[0.8*np.exp(-0.5*(((8*abs(np.pi-i))/np.pi)**2)) for i in Task.TUNING_DIRS]]), num_trials*2, axis=0)
        roll = np.nan_to_num(np.floor((mod_dir_str_conditions[: , 0, :]/(2*np.pi))*Task.STIM_DIM)- np.floor(Task.STIM_DIM/2)).astype(int)
        rolled = np.array(list(map(np.roll, centered_dir, np.expand_dims(roll.flatten(), axis=1)))) * np.expand_dims(np.nan_to_num(mod_dir_str_conditions[:, 1, :]).flatten() , axis=1)
        if mod_dim>1: 
            rolled_reshaped = np.concatenate((rolled.reshape(mod_dim, num_trials, Task.STIM_DIM)[0, :, :], rolled.reshape(mod_dim, num_trials, Task.STIM_DIM)[1, :, :]), axis=1)
        else:
            rolled_reshaped = rolled.reshape(num_trials, -1)

        return rolled_reshaped


    def _get_trial_inputs(self, task_type: str, conditions_arr: np.ndarray) -> np.ndarray:
        '''
        Creates stimulus activity arrays for given trials of a particular task type 

        Parameters: 
            task_type: string identifying the task type

            conditions_arr[mods, stim, dir_str, num_trials]: array defining the stimulus conditions for a batch of task trials 

        Returns: 
            ndarray[num_trials, TRIAL_LEN, INPUT_DIM]: array conditing stimulus inputs for a batch of task trials 
        '''
        fix = np.ones((self.num_trials,1)) 
        no_fix = np.zeros((self.num_trials,1))

        stim1 = self._make_activity_vectors(conditions_arr[:, 0, :, :])
        stim2 = self._make_activity_vectors(conditions_arr[:, 1, :, :])

        if 'Go' in task_type: 
            if 'RT' in task_type: 
                input_activity_vecs = np.array([np.concatenate((fix, self.null_stim), 1), np.concatenate((fix, self.null_stim), 1), 
                                np.concatenate((fix, self.null_stim), 1),  np.concatenate((fix, self.null_stim), 1), np.concatenate((no_fix, stim1), 1)])
            else: 
                input_activity_vecs = np.array([np.concatenate((fix, self.null_stim), 1), np.concatenate((fix, stim1), 1), 
                                np.concatenate((fix, stim1), 1),  np.concatenate((fix, stim1), 1), np.concatenate((no_fix, stim1), 1)])
        elif task_type in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']: 
            input_activity_vecs = np.array([np.concatenate((fix, self.null_stim), 1), np.concatenate((fix, stim1+stim2), 1), np.concatenate((fix, stim1+stim2), 1),  
                                np.concatenate((fix, stim1+stim2), 1), np.concatenate((no_fix, self.null_stim), 1)])
        elif 'COMP' in task_type:
            input_activity_vecs = np.array([np.concatenate((fix, self.null_stim), 1), np.concatenate((fix, stim1), 1), np.concatenate((fix, stim1+stim2), 1),  
                                np.concatenate((fix, stim2), 1), np.concatenate((no_fix, self.null_stim), 1)])
        else: 
            input_activity_vecs = np.array([np.concatenate((fix, self.null_stim), 1), np.concatenate((fix, stim1), 1), np.concatenate((fix, self.null_stim), 1),  
                                np.concatenate((fix, stim2), 1), np.concatenate((no_fix, self.null_stim), 1)])
        
        return add_noise(self._expand_along_intervals(self.intervals, input_activity_vecs)).astype(np.float16)
        
    def _get_loss_mask(self) -> np.ndarray: 
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
            ones = np.ones((1, Task.OUTPUT_DIM))
            zeros = np.zeros((1, Task.OUTPUT_DIM))
            go_weights = np.full((1, Task.OUTPUT_DIM), 5)
            go_weights[:,0] = 10

            pre_go = intervals[-1][0]
            zero_per = int(np.floor(100/Task.DELTA_T))

            pre_go_mask = ones.repeat(pre_go, axis = 0)
            zero_mask = zeros.repeat(zero_per, axis=0)
            go_mask = go_weights.repeat((self.TRIAL_LEN-(pre_go+zero_per)), axis = 0)
            return np.concatenate((pre_go_mask, zero_mask, go_mask), 0)
        return np.array(list(map(__make_loss_mask__, self.intervals))).astype(np.float16)
    

    def _get_trial_targets(self, target_dirs: np.array) -> np.ndarray: 
        '''
        Makes target output activities for a batch of trials

        Parameters:
            target_dirs[num_trials]

        Returns: 
            ndarray[num_trials, TRIAL_LEN, OUTPUT_DIM]: ndarray of target activities for a batch of trials
        '''
        fix = np.full((self.num_trials,1), 0.85)
        go = np.full((self.num_trials,1), 0.05)
        strengths = np.where(np.isnan(target_dirs), np.zeros_like(target_dirs), np.ones_like(target_dirs))
        target_conditions = np.expand_dims(np.stack((target_dirs, strengths), axis=1).T, axis=0)
        target_activities = self._make_activity_vectors(target_conditions)
        resp = np.concatenate((go, target_activities+0.05), 1)
        no_resp = np.concatenate((fix, np.full((self.num_trials, self.STIM_DIM), 0.05)), 1)
        trial_target = self._expand_along_intervals(self.intervals, (no_resp, no_resp, no_resp, no_resp, resp))
        return trial_target.astype(np.float16)

    def _plot_trial(self, ins, tars, task_type):
        fix = np.expand_dims(ins[0, :], 0)
        mod1 = ins[1:self.STIM_DIM, :]
        mod2 = ins[1+self.STIM_DIM:1+(2*self.STIM_DIM), :]

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


class Go(Task): 
    def __init__(self, task_type, num_trials, intervals=None, conditions_arr =None):
        super().__init__(num_trials, intervals)
        assert task_type in ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type

        #mod, stim, dir_strengths, num_trials
        self.conditions_arr = np.empty((2, 2, 2, num_trials), dtype=np.float32)

        self.conditions_arr[:, 1, :, :] = np.NaN
        if type(intervals) == type(None): 
            for i in range(num_trials): 
                direction = np.random.uniform(0, 2*np.pi)
                base_strength = np.random.uniform(1.0, 1.2)    
                mod = np.random.choice([0, 1])
                self.conditions_arr[mod, 0, :, i] = [direction, base_strength]
                self.conditions_arr[((mod+1)%2), 0, :, i] = np.NaN
        else: 
            self.conditions_arr = conditions_arr

        if 'Anti' in self.task_type: 
            self.target_dirs = np.nansum((self.conditions_arr[:, 0, 0, :] + np.pi)%(2*np.pi), axis=0)
        else:
            self.target_dirs = np.nansum(self.conditions_arr[:, 0, 0, :], axis=0)

        self.inputs = self._get_trial_inputs(self.task_type, self.conditions_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index, :, :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

class Comp(Task):
    def __init__(self, task_type, num_trials, intervals=None): 
        super().__init__(num_trials, intervals)
        assert task_type in ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], "entered invalid task type: %r" %task_type
        self.task_type = task_type

        #mod, stim, dir_strengths, num_trials
        self.conditions_arr = np.empty((2, 2, 2, num_trials))
        self.target_dirs = np.empty(num_trials)
        
        if num_trials >1: 
            requires_response_list = list(np.random.permutation([True]*int(num_trials/2) + [False] * int(num_trials/2)))
        else: 
            requires_response_list = [np.random.choice([True, False])]

        for i in range(num_trials): 
            directions = _draw_ortho_dirs()
            requires_response = requires_response_list.pop()

            if requires_response and 'COMP1' in self.task_type:
                self.target_dirs[i] = directions[0]
            elif requires_response and 'COMP2' in self.task_type: 
                self.target_dirs[i] = directions[1]
            else: 
                self.target_dirs[i] = None

            
            if 'Multi' in self.task_type: 
                base_strength = np.random.uniform(1.3, 1.5, size=2)
                coh = np.random.choice([0.05, 0.1, 0.15], size=2)

                positive_strength = base_strength + coh
                negative_strength = base_strength - coh
            else: 
                base_strength = np.random.uniform(1.3, 1.5)
                coh = np.random.choice([0.1, 0.15, 0.2])
                positive_strength = base_strength + coh
                negative_strength = base_strength - coh

            if ('COMP1' in self.task_type and requires_response) or ('COMP2' in self.task_type and not requires_response): 
                strengths = np.array([positive_strength, negative_strength])
            elif ('COMP2' in self.task_type and requires_response) or ('COMP1' in self.task_type and not requires_response): 
                strengths = np.array([negative_strength, positive_strength])
            
            if 'Multi' in self.task_type: 
                self.conditions_arr[:, :, 0, i] = np.array([directions, directions])
                self.conditions_arr[:, :, 1, i] = strengths

            else: 
                mod = np.random.choice([0, 1])
                self.conditions_arr[mod, :, 0, i] = np.array([directions])
                self.conditions_arr[mod, :, 1, i] = strengths
                self.conditions_arr[((mod+1)%2), :, :, i] = np.NaN
                

        self.inputs = self._get_trial_inputs(self.task_type, self.conditions_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index, :, :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

class Delay(Task): 
    def __init__(self, task_type, num_trials, intervals=None): 
        super().__init__(num_trials, intervals)
        assert task_type in ['DMS', 'DNMS', 'DMC', 'DNMC'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)

        self.target_dirs = np.empty(num_trials)

        #mod, stim, dir_strengths, num_trials
        self.conditions_arr = np.empty((2, 2, 2, num_trials))        

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
            
            if match_trial and task_type in ['DMS', 'DNMS']: direction2 = direction1
            elif match_trial and task_type in ['DMC', 'DNMC']: direction2 = np.random.uniform(cat_range[0], cat_range[1])
            elif not match_trial and task_type in ['DMC', 'DNMC']: direction2 = (np.random.uniform(cat_range[0]+(np.pi/3), cat_range[1]-(np.pi/3)) + np.pi)%(2*np.pi)
            else: direction2 = (direction1 + np.random.uniform(np.pi/2, (2*np.pi - np.pi/2)))%(2*np.pi)
            
            mod = np.random.choice([0, 1])
            self.conditions_arr[mod, :, 0, i] = np.array([direction1, direction2])
            self.conditions_arr[mod, :, 1, i] = np.array([1, 1])
            self.conditions_arr[((mod+1)%2), :, :, i] = np.NaN


            if match_trial and task_type in ['DMS', 'DMC']:
                self.target_dirs[i] = direction1
            elif not match_trial and task_type in ['DNMS', 'DNMC']:
                self.target_dirs[i] = direction2
            else: self.target_dirs[i] = None


        self.inputs = self._get_trial_inputs(self.task_type, self.conditions_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index,: , :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

class DM(Task): 
    def __init__(self, task_type, num_trials, intervals=None, conditions_arr =None):
        super().__init__(num_trials, intervals)
        assert task_type in ['DM', 'MultiDM', 'Anti DM', 'Anti MultiDM'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.mods = np.empty(num_trials, dtype=tuple)
        self.target_dirs = np.empty(num_trials)
        
        if conditions_arr is None: 
            #mod, stim, dir_strengths, num_trials
            self.conditions_arr = np.empty((2, 2, 2, num_trials))

            for i in range(num_trials):

                directions = _draw_ortho_dirs()

                if 'Multi' in self.task_type:    
                    base_strength = np.random.uniform(0.8, 1.2, size=2)

                    redraw = True
                    while redraw: 
                        coh = np.random.choice([-0.15, -0.1, -0.05, 0.05, 0.1, 0.15], size=2, replace=False)
                        if abs(coh[0]) != abs(coh[1]): 
                            redraw = False
                    
                    strengths = np.array([base_strength + coh, base_strength - coh]).T
                    self.conditions_arr[:, :, 0, i] = np.array([directions, directions])
                    self.conditions_arr[:, :, 1, i] = strengths

                else:
                    mod = np.random.choice([0, 1])
                    base_strength = np.random.uniform(0.8, 1.2)
                    coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2])

                    strengths = np.array([base_strength+coh, base_strength-coh]).T
                    
                    self.conditions_arr[mod, :, 0, i] = np.array(directions)
                    self.conditions_arr[mod, :, 1, i] = strengths
                    self.conditions_arr[((mod+1)%2), :, :, i] = np.NaN

                    
        else:
            self.conditions_arr= conditions_arr


        if 'Anti' in self.task_type:
            dir_chooser = np.argmin
        else: 
            dir_chooser = np.argmax 

        if 'Multi' in self.task_type: 
            directions = self.conditions_arr[0, :, 0, :]
        else: 
            directions = np.nansum(self.conditions_arr[:, :, 0, :], axis=0)


        chosen_dir = dir_chooser(np.nansum(self.conditions_arr[:, :, 1, :], axis=0), axis=0)
        self.target_dirs = np.where(chosen_dir, directions[1, :], directions[0, :])


        self.inputs = self._get_trial_inputs(self.task_type, self.conditions_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index, :,:].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

def construct_batch(task_type, num):
    assert task_type in Task.TASK_LIST, "entered invalid task type"
    if task_type == 'Go':
        trial = Go('Go', num)
    if task_type == 'RT Go':
        trial = Go('RT Go', num)
    if task_type == 'Anti Go':
        trial = Go('Anti Go', num)
    if task_type == 'Anti RT Go':
        trial = Go('Anti RT Go', num)
    if task_type == 'DM':
        trial = DM('DM', num)
    if task_type == 'MultiDM': 
        trial = DM('MultiDM', num)
    if task_type == 'Anti DM':
        trial = DM('Anti DM', num)
    if task_type == 'Anti MultiDM': 
        trial = DM('Anti MultiDM', num)
    if task_type == 'COMP1': 
        trial = Comp('COMP1', num)
    if task_type == 'COMP2': 
        trial = Comp('COMP2', num)
    if task_type == 'MultiCOMP1': 
        trial = Comp('MultiCOMP1', num)
    if task_type == 'MultiCOMP2': 
        trial = Comp('MultiCOMP2', num)
    if task_type == 'DMS': 
        trial = Delay('DMS', num)
    if task_type == 'DNMS': 
        trial = Delay('DNMS', num)
    if task_type == 'DMC': 
        trial = Delay('DMC', num)
    if task_type == 'DNMC': 
        trial = Delay('DNMC', num)
    return trial.inputs, trial.targets, trial.masks, trial.target_dirs, Task.TASK_LIST.index(task_type)


