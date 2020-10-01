import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Task():
    #TASK_LIST = ['Go', 'RT Go', 'Anti Go', 'Anti RT Go']
    TASK_LIST = ['Go', 'RT Go', 'Anti Go', 'Anti RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM', 'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'DMC', 'DNMC']

    STIM_DIM = 32
    TUNING_DIRS = [((2*np.pi*i)/32) for i in range(STIM_DIM)]
    TRIAL_LEN = int(120)
    INPUT_DIM = 1 + len(TASK_LIST) + STIM_DIM*2
    OUTPUT_DIM = STIM_DIM + 1
    SIGMA_IN = 0.01
    DELTA_T = 20
    DM_DELAY = False

    def __init__(self, num_trials): 
        self.num_trials = num_trials
        self.intervals = np.empty((num_trials, 5), dtype=tuple)
        self.null_stim = np.zeros((self.num_trials, self.STIM_DIM*2))
        self.null_input_vec = np.full((self.num_trials, self.INPUT_DIM), None)
        for i in range(num_trials):
            T_go = (int(self.TRIAL_LEN - np.floor(np.random.uniform(300, 600)/self.DELTA_T)), self.TRIAL_LEN)
            T_stim2 = (int(T_go[0]-np.floor(np.random.uniform(300, 600)/self.DELTA_T)), T_go[0])
            T_delay = (int(T_stim2[0]-np.floor(np.random.uniform(300, 600)/self.DELTA_T)), T_stim2[0])
            T_stim1 = (int(T_delay[0]-np.floor(np.random.uniform(300, 600)/self.DELTA_T)), T_delay[0])
            T_fix = (0, T_stim1[0])

            self.intervals[i] = (T_fix, T_stim1, T_delay, T_stim2, T_go)

    def make_noise(self, dim):
        noise = np.sqrt(2/self.DELTA_T)*(self.SIGMA_IN) * np.random.normal(size=dim)
        return np.expand_dims(noise, 0)

    def _rule_one_hot(self, task_type):
        index = self.TASK_LIST.index(task_type)
        one_hot = np.zeros(len(self.TASK_LIST))
        one_hot[index] = 1
        return np.expand_dims(one_hot, 0)

    def _draw_ortho_dirs(self): 
        dir1 = np.random.uniform(0, 2*np.pi)
        dir2 = (dir1+np.pi+np.random.uniform(-np.pi*0.2, np.pi*0.2))%(2*np.pi)
        return (dir1, dir2)

    def _fill_pref_dir(self, mod_params): 
        stim = np.zeros(self.STIM_DIM)    
        if not (mod_params is None): 
            centered_dir = np.array([0.8*np.exp(-0.5*(((8*abs(np.pi-i))/np.pi)**2)) for i in self.TUNING_DIRS])
            for params in mod_params:
                roll = int(np.floor((params[1]/(2*np.pi))*self.STIM_DIM)- np.floor(self.STIM_DIM/2))
                dir_vec = np.roll(centered_dir, roll)
                dir_vec = params[0] *dir_vec
                stim = stim + dir_vec
        return stim

    def _make_input_stim(self, stim_mod1, stim_mod2):
        input_stim_mod1 = np.array(list(map(self._fill_pref_dir, stim_mod1)))
        input_stim_mod2 = np.array(list(map(self._fill_pref_dir, stim_mod2)))
        return np.concatenate((input_stim_mod1, input_stim_mod2), 1)

    def _make_input_vecs(self, task_type, input_stim):
        fix = np.ones((self.num_trials,1))
        no_fix = np.zeros((self.num_trials,1))

        rule_vec = np.repeat(self._rule_one_hot(task_type), self.num_trials, axis=0)

        noise = np.repeat(self.make_noise(self.INPUT_DIM), self.num_trials, axis = 0)

        fix_no_stim = np.concatenate((fix, rule_vec, self.null_stim), 1) + noise
        fix_stim = np.concatenate((fix, rule_vec, input_stim), 1) + noise
        go_stim = np.concatenate((no_fix, rule_vec, input_stim), 1) + noise
        go_no_stim = np.concatenate((no_fix, rule_vec, self.null_stim), 1) + noise

        return fix_no_stim, fix_stim, go_stim, go_no_stim

    def _make_target_vecs(self, target_dirs):
        fix = np.full((self.num_trials,1), 0.85)
        go = np.full((self.num_trials,1), 0.05)

        strengths_dir = []
        for direc in target_dirs: 
            if not np.isnan(direc):
                strengths_dir.append([(1, direc)])
            else: 
                strengths_dir.append(None)

        target_stims = np.array(list(map(self._fill_pref_dir, strengths_dir)))
        resp = np.concatenate((go, target_stims+0.05), 1)
        no_resp = np.concatenate((fix, np.full((self.num_trials, self.STIM_DIM), 0.05)), 1)

        return no_resp, resp

    def _filler(self, zipped_filler): 
        filler_array = zipped_filler.reshape(5, -1)
        trial_array = np.empty((filler_array.shape[1]-1, self.TRIAL_LEN))
        for i in range(filler_array.shape[0]): 
            start, stop, filler_vec = filler_array[i, -1][0], filler_array[i, -1][1], filler_array[i, :-1]
            if start == stop:
                continue
            trial_array[:, start:stop] = np.repeat(np.expand_dims(filler_vec, axis=1), stop-start, axis=1)
        return trial_array

    def _fill_trials(self, intervals, fill_vecs): 
        trial_fills = np.swapaxes(np.array(fill_vecs), 1, 2)
        intervals = np.expand_dims(intervals.T, axis= 1)
        zipped_filler = np.reshape(np.concatenate((trial_fills, intervals), axis = 1), (-1, self.num_trials))
        batch_array = np.apply_along_axis(self._filler, 0, zipped_filler)
        return batch_array.T

    def _plot_trial(self, ins, tars, task_type):
        fix = np.expand_dims(ins[0, :], 0)
        num_rules = len(self.TASK_LIST)
        rule_vec = ins[1:num_rules+1, :]
        mod1 = ins[1+num_rules:1+num_rules+self.STIM_DIM, :]
        mod2 = ins[1+num_rules+self.STIM_DIM:1+num_rules+(2*self.STIM_DIM), :]

        to_plot = (fix, rule_vec, mod1, mod2, tars)

        fig, axn = plt.subplots(5,1, sharex = True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        ylabels = ('fix', 'rule_vec', 'mod1', 'mod2', 'Target')
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title('%r Trial Info' %task_type)
            if i == 4: 
                ax.set_xlabel('time (DELTA_T=%r ms)'%self.DELTA_T)
        plt.show()

    def _get_trial_inputs(self, task_type, stim_mod_arr):
        if len(stim_mod_arr.shape) == 2: 
            stim_mod_arr = np.array([stim_mod_arr]*2)
        stim1 = self._make_input_stim(stim_mod_arr[0, 0, :], stim_mod_arr[0, 1, :])
        stim2 = self._make_input_stim(stim_mod_arr[1, 0, :], stim_mod_arr[1, 1, :])
        epoch_vecs1 = self._make_input_vecs(task_type, stim1)
        epoch_vecs2 = self._make_input_vecs(task_type, stim2)
        if task_type in ['DM', 'MultiDM', 'Anti DM', 'Anti MultiDM']:
            if self.DM_DELAY:
                input_vecs = (epoch_vecs1[0], epoch_vecs1[1], epoch_vecs1[1],  epoch_vecs2[1], epoch_vecs2[3])            
            else: 
                input_vecs = (epoch_vecs1[0], epoch_vecs1[1]+epoch_vecs2[1], epoch_vecs1[1]+epoch_vecs2[1], epoch_vecs1[1]+epoch_vecs2[1], epoch_vecs2[3])            
        elif 'Go' in task_type: 
            input_vecs = (epoch_vecs1[0], epoch_vecs1[1], epoch_vecs1[1],  epoch_vecs2[1], epoch_vecs2[3])
        else:
            input_vecs = (epoch_vecs1[0], epoch_vecs1[1], epoch_vecs1[0],  epoch_vecs2[1], epoch_vecs2[3])
        return self._fill_trials(self.intervals, input_vecs)
        
    def _make_loss_mask(self, intervals): 
        ones = np.ones((1, self.OUTPUT_DIM))
        zeros = np.zeros((1, self.OUTPUT_DIM))
        go_weights = np.full((1, self.OUTPUT_DIM), 5)
        go_weights[:,0] = 10

        pre_go = intervals[-1][0]
        zero_per = int(np.floor(100/self.DELTA_T))

        pre_go_mask = ones.repeat(pre_go, axis = 0)
        zero_mask = zeros.repeat(zero_per, axis=0)
        go_mask = go_weights.repeat((self.TRIAL_LEN-(pre_go+zero_per)), axis = 0)
        return np.concatenate((pre_go_mask, zero_mask, go_mask), 0)
        
    def _get_loss_mask(self): 
        return np.array(list(map(self._make_loss_mask, self.intervals)))

    def _get_trial_targets(self, target_dirs): 
        no_resp, resp = self._make_target_vecs(target_dirs)
        trial_target = self._fill_trials(self.intervals, (no_resp, no_resp, no_resp, no_resp, resp))
        return trial_target

class Go(Task): 
    def __init__(self, task_type, num_trials):
        super().__init__(num_trials)
        assert task_type in ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, num_trials), dtype=list)
        self.directions = np.empty(num_trials)

        for i in range(num_trials):
            direction = np.random.uniform(0, 2*np.pi)
            self.directions[i] = direction
            base_strength = np.random.uniform(0.6, 1.4)
            strength_dir = [(base_strength, direction)]
            
            mod = np.random.choice([0, 1])
            self.stim_mod_arr[mod, i] = strength_dir
            self.stim_mod_arr[((mod+1)%2), i] = None

        if 'Anti' in self.task_type: 
            self.target_dirs = (self.directions + np.pi)%(2*np.pi)
        else:
            self.target_dirs = self.directions

        self.inputs = self._get_trial_inputs(self.task_type, self.stim_mod_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index, :, :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

class Comp(Task):
    def __init__(self, task_type, num_trials): 
        super().__init__(num_trials)
        assert task_type in ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], "entered invalid task type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)
        self.target_dirs = np.empty(num_trials)
        self.directions = []

        for i in range(num_trials): 
            direction1 = np.random.uniform(0, 2*np.pi)
            direction2 = np.random.uniform(0, 2*np.pi)
            requires_response = np.random.choice([True, False])
            self.directions.append((direction1, direction2))

            if requires_response and 'COMP1' in self.task_type:
                self.target_dirs[i] = direction1
            elif requires_response and 'COMP2' in self.task_type: 
                self.target_dirs[i] = direction2
            else: 
                self.target_dirs[i] = None

            
            if 'Multi' in self.task_type: 
                positive_strength = np.random.uniform(2, 2.25)
                negative_strength = np.random.uniform(1.7, 1.95)

                positive_split = np.random.uniform(0.6, 1.2)
                negative_split = np.random.uniform(0.5, 1.0)
            else: 
                positive_strength = np.random.uniform(1.05, 1.25)
                negative_strength = np.random.uniform(0.75, 0.95 )

            if ('COMP1' in self.task_type and requires_response) or ('COMP2' in self.task_type and not requires_response): 
                if 'Multi' in self.task_type: 
                    strengths = np.array([np.random.permutation([positive_split, positive_strength-positive_split]),
                                                np.random.permutation([negative_split, negative_strength-negative_split])])
                else:
                    strengths = np.array([positive_strength, negative_strength])
            elif ('COMP2' in self.task_type and requires_response) or ('COMP1' in self.task_type and not requires_response): 
                if 'Multi' in self.task_type: 
                    strengths = np.array([np.random.permutation([negative_split, negative_strength-negative_split]),
                                                np.random.permutation([positive_split, positive_strength-positive_split])])
                else: 
                    strengths = np.array([negative_strength, positive_strength])
            
            if 'Multi' in self.task_type: 
                self.stim_mod_arr[0, 0, i] = [(strengths[0, 0], direction1)]
                self.stim_mod_arr[1, 0, i] = [(strengths[1, 0], direction2)]
                self.stim_mod_arr[0, 1, i] = [(strengths[0, 1], direction1)]
                self.stim_mod_arr[1, 1, i] = [(strengths[1, 1], direction2)]
            else: 
                mod = np.random.choice([0, 1])
                self.stim_mod_arr[0, mod, i] = [(strengths[0], direction1)]
                self.stim_mod_arr[1, mod, i] = [(strengths[1], direction2)]
                self.stim_mod_arr[0, ((mod+1)%2), i] = None
                self.stim_mod_arr[1, ((mod+1)%2), i] = None

        self.inputs = self._get_trial_inputs(self.task_type, self.stim_mod_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index, :, :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)


class Delay(Task): 
    def __init__(self, task_type, num_trials): 
        super().__init__(num_trials)
        assert task_type in ['DMS', 'DNMS', 'DMC', 'DNMC'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)
        self.target_dirs = np.empty(num_trials)
        self.directions = []
        for i in range(num_trials): 
            match_trial = np.random.choice([True, False])

            direction1 = np.random.uniform(0, 2*np.pi)
            if direction1 < np.pi: 
                cat_range = (0, np.pi)
            else: cat_range = (np.pi, 2*np.pi)
            
            if match_trial and task_type in ['DMS', 'DNMS', 'COMP1', 'COMP2']: direction2 = direction1
            elif match_trial and task_type in ['DMC', 'DNMC']: direction2 = np.random.uniform(cat_range[0], cat_range[1])
            elif not match_trial and task_type in ['DMC', 'DNMC']: direction2 = (np.random.uniform(cat_range[0]+(np.pi/5), cat_range[1]-(np.pi/5)) + np.pi)%(2*np.pi)
            else: direction2 = (direction1 + np.random.uniform(np.pi/4, (2*np.pi - np.pi/4)))%(2*np.pi)

            self.directions.append((direction1, direction2))

            base_strength = np.random.uniform(0.8, 1.2)
            coh = np.random.choice([-0.25, -0.15, -0.1, 0.1, 0.15, 0.25])
            strengths = np.array([base_strength+coh, base_strength-coh])
            max_strength = np.argmax(strengths)

            strength_dir1 = [(strengths[0], direction1)]
            strength_dir2 = [(strengths[1], direction2)]
            
            mod = np.random.choice([0, 1])
            self.stim_mod_arr[0, mod, i] = strength_dir1
            self.stim_mod_arr[1, mod, i] = strength_dir2
            self.stim_mod_arr[0, ((mod+1)%2), i] = None
            self.stim_mod_arr[1, ((mod+1)%2), i] = None

            if match_trial and task_type in ['DMS', 'DMC']:
                self.target_dirs[i] = direction2
            elif not match_trial and task_type in ['DNMS', 'DNMC']:
                self.target_dirs[i] = direction1
            elif task_type == 'COMP1': 
                if max_strength == 0: 
                    self.target_dirs[i] = direction1
                else: 
                    self.target_dirs[i] = None
            elif task_type == 'COMP2': 
                if max_strength == 1: 
                    self.target_dirs[i] = direction2
                else: 
                    self.target_dirs[i] = None
            else: self.target_dirs[i] = None

        self.inputs = self._get_trial_inputs(self.task_type, self.stim_mod_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index,: , :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

class DM(Task): 
    def __init__(self, task_type, num_trials):
        super().__init__(num_trials)
        assert task_type in ['DM', 'MultiDM', 'Anti DM', 'Anti MultiDM'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.mods = np.empty(num_trials, dtype=tuple)
        self.stim_mod_arr =  np.empty((2, 2, num_trials), dtype=tuple)
        self.target_dirs = np.empty(num_trials)

        for i in range(num_trials):
            directions = self._draw_ortho_dirs()
            if self.task_type == 'MultiDM' or task_type == 'Anti MultiDM': 
                base_strength = np.random.uniform(0.8, 1.2, size=2)
                coh = np.random.choice([-0.25, -0.15, -0.1, 0.1, 0.15, 0.25], size=2, replace=False)
                strengths = np.array([base_strength + coh, base_strength - coh])
                self.stim_mod_arr[0, 0, i] = [(strengths[0, 0], directions[0])]
                self.stim_mod_arr[1, 0, i] = [(strengths[1, 0], directions[1])]
                self.stim_mod_arr[0, 1, i] = [(strengths[0, 1], directions[0])]
                self.stim_mod_arr[1, 1, i] = [(strengths[1, 1], directions[1])]
                if task_type=='MultiDM': 
                    self.target_dirs[i] = directions[np.argmax(np.sum(strengths, axis=1))]
                else: 
                    self.target_dirs[i] = directions[np.argmin(np.sum(strengths, axis=1))]
            else:
                base_strength = np.random.uniform(0.8, 1.2)
                coh = np.random.choice([-0.25, -0.15, -0.1, 0.1, 0.15, 0.25])
                strengths = np.array([base_strength+coh, base_strength-coh])
                if task_type == 'DM': 
                    self.target_dirs[i] = directions[np.argmax(strengths)]
                else: 
                    self.target_dirs[i] = directions[np.argmin(strengths)]
                mod = np.random.choice([0, 1])
                self.stim_mod_arr[0, mod, i] = [(strengths[0], directions[0])]
                self.stim_mod_arr[1, mod, i] = [(strengths[1], directions[1])]
                self.stim_mod_arr[0, ((mod+1)%2), i] = None
                self.stim_mod_arr[1, ((mod+1)%2), i] = None

        self.inputs = self._get_trial_inputs(self.task_type, self.stim_mod_arr)
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
    return trial 


