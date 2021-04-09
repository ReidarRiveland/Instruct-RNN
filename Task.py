import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Task():
    TASK_LIST = ['Go', 'RT Go', 'Anti Go', 'Anti RT Go', 'DM', 'Anti DM', 'MultiDM', 'Anti MultiDM', 'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'DMS', 'DNMS', 'DMC', 'DNMC']
    SHUFFLED_TASK_LIST = ['MultiCOMP2', 'RT Go', 'COMP2', 'DNMS', 'Anti Go', 'COMP1', 'DNMC', 'Anti DM', 'MultiCOMP1', 'Anti MultiDM', 'MultiDM', 'DM', 'Anti RT Go', 'DMC', 'DMS', 'Go']
    STIM_DIM = 32
    TUNING_DIRS = [((2*np.pi*i)/32) for i in range(STIM_DIM)]
    TRIAL_LEN = int(120)
    INPUT_DIM = 1 + len(TASK_LIST) + STIM_DIM*2
    OUTPUT_DIM = STIM_DIM + 1
    SIGMA_IN = 0.05
    DELTA_T = 20
    DM_DELAY = 'no_delay'

    def __init__(self, num_trials, intervals): 
        self.num_trials = num_trials
        self.intervals = np.empty((num_trials, 5), dtype=tuple)
        self.null_stim = np.zeros((self.num_trials, self.STIM_DIM*2))
        self.null_input_vec = np.full((self.num_trials, self.INPUT_DIM), None)
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

    def add_noise(self, array):
        noise = np.sqrt(2/Task.DELTA_T)*(Task.SIGMA_IN) * np.random.normal(size=array.shape)
        return array+noise

    @staticmethod
    def _rule_one_hot(task_type, shuffled=False):
        if shuffled: index = Task.SHUFFLED_TASK_LIST.index(task_type) 
        else: index = Task.TASK_LIST.index(task_type)
        one_hot = np.zeros(len(Task.TASK_LIST))
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
        trial_fills = np.swapaxes(fill_vecs, 1, 2)
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

        to_plot = (fix, mod1, mod2, tars)

        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 5])

        fig, axn = plt.subplots(4,1, sharex = True, gridspec_kw=gs_kw)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        ylabels = ('fix.', 'mod. 1', 'mod. 2', 'Target')
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)

            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title('%r Trial Info' %task_type)
            if i == 3: 
                ax.set_xlabel('time')
        plt.show()


    def _get_trial_inputs(self, task_type, stim_mod_arr):
        fix = np.ones((self.num_trials,1)) 
        no_fix = np.zeros((self.num_trials,1))
        rule_vec = np.repeat(self._rule_one_hot(task_type), self.num_trials, axis=0)

        if len(stim_mod_arr.shape) == 2: 
            stim_mod_arr = np.array([stim_mod_arr]*2)
        stim1 = self._make_input_stim(stim_mod_arr[0, 0, :], stim_mod_arr[0, 1, :])
        stim2 = self._make_input_stim(stim_mod_arr[1, 0, :], stim_mod_arr[1, 1, :])

        if 'Go' in task_type: 
            if 'RT' in task_type: 
                input_vecs = np.array([np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((fix, rule_vec, self.null_stim), 1), 
                                np.concatenate((fix, rule_vec, self.null_stim), 1),  np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((no_fix, rule_vec, stim1), 1)])
            else: 
                input_vecs = (np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((fix, rule_vec, stim1), 1), 
                                np.concatenate((fix, rule_vec, stim1), 1),  np.concatenate((fix, rule_vec, stim1), 1), np.concatenate((no_fix, rule_vec, stim1), 1))
        elif self.DM_DELAY is not 'full_delay' and task_type in ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM']: 
            if self.DM_DELAY == 'no_delay': 
                input_vecs = (np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((fix, rule_vec, stim1+stim2), 1), np.concatenate((fix, rule_vec, stim1+stim2), 1),  
                                    np.concatenate((fix, rule_vec, stim1+stim2), 1), np.concatenate((no_fix, rule_vec, self.null_stim), 1))
            elif self.DM_DELAY == 'staggered': 
                input_vecs = (np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((fix, rule_vec, stim1), 1), np.concatenate((fix, rule_vec, stim1+stim2), 1),  
                                    np.concatenate((fix, rule_vec, stim2), 1), np.concatenate((no_fix, rule_vec, self.null_stim), 1)) 
        elif 'COMP' in task_type:
            input_vecs = (np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((fix, rule_vec, stim1), 1), np.concatenate((fix, rule_vec, stim1+stim2), 1),  
                                np.concatenate((fix, rule_vec, stim2), 1), np.concatenate((no_fix, rule_vec, self.null_stim), 1))
        else: 
            input_vecs = (np.concatenate((fix, rule_vec, self.null_stim), 1), np.concatenate((fix, rule_vec, stim1), 1), np.concatenate((fix, rule_vec, self.null_stim), 1),  
                                np.concatenate((fix, rule_vec, stim2), 1), np.concatenate((no_fix, rule_vec, self.null_stim), 1))
        
        return self.add_noise(self._fill_trials(self.intervals, input_vecs))

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
    def __init__(self, task_type, num_trials, intervals=None, stim_mod_arr =None, directions=None):
        super().__init__(num_trials, intervals)
        assert task_type in ['Go', 'RT Go', 'Anti Go', 'Anti RT Go'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, num_trials), dtype=list)
        self.directions = np.empty(num_trials)
        if type(intervals) == type(None): 
            for i in range(num_trials):
                direction = np.random.uniform(0, 2*np.pi)
                self.directions[i] = direction
                base_strength = np.random.uniform(1.0, 1.2)
                strength_dir = [(base_strength, direction)]
                
                mod = np.random.choice([0, 1])
                self.stim_mod_arr[mod, i] = strength_dir
                self.stim_mod_arr[((mod+1)%2), i] = None
        else: 
            self.stim_mod_arr = stim_mod_arr
            self.directions = directions

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
    def __init__(self, task_type, num_trials, intervals=None): 
        super().__init__(num_trials, intervals)
        assert task_type in ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'], "entered invalid task type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)
        self.target_dirs = np.empty(num_trials)
        self.directions = []
        if num_trials >1: 
            requires_response_list = list(np.random.permutation([True]*int(num_trials/2) + [False] * int(num_trials/2)))
        else: 
            requires_response_list = [np.random.choice([True, False])]
        for i in range(num_trials): 
            direction1, direction2 = self._draw_ortho_dirs()
            self.directions.append((direction1, direction2))
            requires_response = requires_response_list.pop()

            if requires_response and 'COMP1' in self.task_type:
                self.target_dirs[i] = direction1
            elif requires_response and 'COMP2' in self.task_type: 
                self.target_dirs[i] = direction2
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
    def __init__(self, task_type, num_trials, intervals=None): 
        super().__init__(num_trials, intervals)
        assert task_type in ['DMS', 'DNMS', 'DMC', 'DNMC'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.stim_mod_arr = np.empty((2, 2, num_trials), dtype=tuple)
        self.target_dirs = np.empty(num_trials)
        self.directions = []
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

            self.directions.append((direction1, direction2))

            # base_strength = np.random.uniform(1.3, 1.5)
            # coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2])
            # strengths = np.array([base_strength+coh, base_strength-coh])
            # max_strength = np.argmax(strengths)

            # strength_dir1 = [(strengths[0], direction1)]
            # strength_dir2 = [(strengths[1], direction2)]

            base_strength = np.random.uniform(0.8, 1.2)
            coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2])

            strength_dir1 = [(base_strength+coh, direction1)]
            strength_dir2 = [(base_strength-coh, direction2)]
            
            mod = np.random.choice([0, 1])
            self.stim_mod_arr[0, mod, i] = strength_dir1
            self.stim_mod_arr[1, mod, i] = strength_dir2
            self.stim_mod_arr[0, ((mod+1)%2), i] = None
            self.stim_mod_arr[1, ((mod+1)%2), i] = None

            if match_trial and task_type in ['DMS', 'DMC']:
                self.target_dirs[i] = direction1
            elif not match_trial and task_type in ['DNMS', 'DNMC']:
                self.target_dirs[i] = direction2
            else: self.target_dirs[i] = None


        self.inputs = self._get_trial_inputs(self.task_type, self.stim_mod_arr)
        self.targets = self._get_trial_targets(self.target_dirs)
        self.masks = self._get_loss_mask()

    def plot_trial(self, trial_index):
        trial_ins = self.inputs[trial_index,: , :].T
        trial_tars = self.targets[trial_index, :, :].T
        self._plot_trial(trial_ins, trial_tars, self.task_type)

class DM(Task): 
    def __init__(self, task_type, num_trials, intervals=None):
        super().__init__(num_trials, intervals)
        assert task_type in ['DM', 'MultiDM', 'Anti DM', 'Anti MultiDM'], "entered invalid task_type: %r" %task_type
        self.task_type = task_type
        self.mods = np.empty(num_trials, dtype=tuple)
        self.stim_mod_arr =  np.empty((2, 2, num_trials), dtype=tuple)
        self.target_dirs = np.empty(num_trials)
        self.coh_list = []
        for i in range(num_trials):
            directions = self._draw_ortho_dirs()
            if self.task_type == 'MultiDM' or task_type == 'Anti MultiDM': 
                base_strength = np.random.uniform(0.8, 1.2, size=2)
                
                redraw = True
                while redraw: 
                    coh = np.random.choice([-0.15, -0.1, -0.05, 0.05, 0.1, 0.15], size=2, replace=False)
                    if abs(coh[0]) != abs(coh[1]): 
                        redraw = False


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
                coh = np.random.choice([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2])
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
            self.coh_list.append(coh)

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


# trials = Comp('MultiCOMP1', 50)
# trials.plot_trial(0)

# sub = np.subtract(np.array(trials.directions)[:, 0], np.array(trials.directions)[:, 1])
# sub


# np.mean(np.isnan(trials.target_dirs))

# np.mean(np.array(trials.coh_list) > 0)


# trials.stim_mod_arr[0, 0, 0]
# trials.stim_mod_arr[0, 1, 0]

# coh_list = []
# for i in range(100): 
#     dir1_str = trials.stim_mod_arr[1, 0, i][0][0]+trials.stim_mod_arr[1, 1, i][0][0]
#     dir2_str = trials.stim_mod_arr[0, 0, i][0][0]+trials.stim_mod_arr[0, 1, i][0][0]
#     dir1_str-dir2_str
#     coh_list.append(dir1_str-dir2_str > 0)
#     print(abs(dir1_str-dir2_str)<0.001)

# np.mean(coh_list)

# base_strength = np.random.uniform(0.8, 1.2, size=2)
# coh = np.random.choice([0.15, 0.2, 0.25], size=2)
# coh
# positive_strength = base_strength + coh
# positive_strength
# base_strength
# negative_strength = base_strength - coh

# strengths = np.array([positive_strength, negative_strength])

# strengths[0,1]

# self.stim_mod_arr[0, 0, i] = [(strengths[0, 0], direction1)]
# self.stim_mod_arr[1, 0, i] = [(strengths[1, 0], direction2)]
# self.stim_mod_arr[0, 1, i] = [(strengths[0, 1], direction1)]
# self.stim_mod_arr[1, 1, i] = [(strengths[1, 1], direction2)]

# num_trials = 50

# requires_response_list = list(np.random.permutation([True]*int(num_trials/2) + [False] * int(num_trials/2)))

# len(requires_response_list)
# requires_response_list.pop()
# requires_response_list

# strengths