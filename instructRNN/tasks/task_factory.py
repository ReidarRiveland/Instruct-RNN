from importlib_metadata import requires
import numpy as np
import pickle
import pathlib

from torch import permute

location = str(pathlib.Path(__file__).parent.absolute())

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


arr = np.array([0, 1, 2])
isinstance(arr, np.ndarray)


def _add_noise(array, noise):
    if isinstance(noise, np.ndarray): 
        noise_arr = np.empty_like(array)
        for i, sd in enumerate(noise):
            noise_arr[i, ...] = sd * np.random.normal(size=array.shape[1:])
    else: 
        noise_arr = noise * np.random.normal(size=array.shape)
    noise_arr[:, :, 0] = 0 
    return array+noise_arr

def _draw_ortho_dirs(num=1, dir0=None): 
    if dir0 is None: 
        dir0 = np.random.uniform(0, 2*np.pi, num)
    dir1 = (dir0+np.pi)%(2*np.pi)
    return np.array((dir0, dir1))

def _permute_mod(dir_arr):
    num_trials = dir_arr.shape[-1]
    permuted = np.array([np.random.permutation(dir_arr[:,:,idx]) for idx in range(num_trials)])
    return np.moveaxis(permuted, 0, -1)

def _draw_multi_contrast(num_trials, draw_vals=[-0.25, 0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25]): 
    coh_arr = np.array(2, num_trials)
    for i in num_trials: 
        redraw = True
        while redraw: 
            coh = np.random.choice(draw_vals, size=2, replace=False)
            if coh[0] != -1*coh[1] and ((coh[0] <0) ^ (coh[1] < 0)): 
                redraw = False
        coh_arr[:, i] = coh
    return coh_arr

def _draw_requires_resp(num_trials): 
    if num_trials >1: 
        requires_response_list = list(np.random.permutation([True]*np.floor(int(num_trials/2)) + [False] * np.ceil(int(num_trials/2))))
    else: 
        requires_response_list = [np.random.choice([True, False])]
    return requires_response_list

def _draw_confidence_threshold(requires_resp_list, pos_thresholds, neg_thresholds): 
    contrasts = []
    noises = []
    for requires_response in requires_resp_list:
        if requires_response: 
            noise, contrast = pos_thresholds[:, np.random.randint(0, pos_thresholds.shape[-1])]
        else: 
            noise, contrast = neg_thresholds[:, np.random.randint(0, neg_thresholds.shape[-1])]
        noises.append(noise)
        contrasts.append(contrast)
    return np.array(noises), np.array(contrasts)

def max_var_dir(num_trials, mod, multi, num_stims, shuffle=True): 
    mod_dirs = _max_var_dir(num_trials, num_stims, shuffle)
    if mod is not None: 
        _mod_dirs = _max_var_dir(num_trials, num_stims, shuffle)
    elif multi: 
        _mod_dirs = mod_dirs
    else: 
        _mod_dirs = np.full_like(mod_dirs, np.NaN)
    dirs = np.random.permutation(np.array([mod_dirs, _mod_dirs]))
    return _permute_mod(dirs)

def _max_var_dir(num_trials, num_stims, shuffle): 
    dirs0 = np.linspace(0, 2*np.pi, num=num_trials)
    if num_stims==1: 
        dirs1 = np.full_like(dirs0, fill_value=np.NaN)
    else: 
        dirs1 = _draw_ortho_dirs(dir0=dirs0)[1,:]
    
    dirs = np.array([dirs0, dirs1])

    if shuffle: 
        dirs = np.random.permutation(dirs.T).T

    return dirs

def max_var_coh(num_trials, max_contrast=0.3, min_contrast=0.05, shuffle=True): 
    coh = np.concatenate((np.linspace(-max_contrast, -min_contrast, num=int(num_trials/2)), 
                np.linspace(min_contrast, max_contrast, num=int(num_trials/2))))

    if shuffle: 
        coh0=np.random.permutation(coh)
        coh1=np.random.permutation(coh)

    coh = np.random.permutation(np.array((coh0, coh1)))
    return coh


class TaskFactory(): 
    def __init__(self, num_trials, timing, noise, intervals):
        self.num_trials = num_trials
        self.timing = timing
        self.noise = noise
        
        if intervals is not None: 
            self.intervals = intervals
        else: 
            self.intervals = self.make_intervals()
                
    def make_intervals(self): 
        intervals = np.empty((self.num_trials, 5), dtype=tuple)
        for i in range(self.num_trials):
            T_go = (int(TRIAL_LEN - np.floor(np.random.uniform(300, 400)/DELTA_T)), TRIAL_LEN)
            T_stim2 = (int(T_go[0]-np.floor(np.random.uniform(500, 800)/DELTA_T)), T_go[0])
            T_delay = (int(T_stim2[0]-np.floor(np.random.uniform(200, 300)/DELTA_T)), T_stim2[0])
            T_stim1 = (int(T_delay[0]-np.floor(np.random.uniform(500, 800)/DELTA_T)), T_delay[0])
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
        centered_dir = np.repeat(np.array([[0.8*np.exp(-0.5*(((12*abs(np.pi-i))/np.pi)**2)) for i in TUNING_DIRS]]), num_trials*2, axis=0)
        roll = np.nan_to_num(np.floor((mod_dir_str_conditions[: , 0, :]/(2*np.pi))*STIM_DIM)- np.floor(STIM_DIM/2)).astype(int)
        rolled = np.array(list(map(np.roll, centered_dir, np.expand_dims(roll.flatten(), axis=1)))) * np.expand_dims(np.nan_to_num(mod_dir_str_conditions[:, 1, :]).flatten() , axis=1)
        if mod_dim>1: 
            rolled_reshaped = np.concatenate((rolled.reshape(mod_dim, num_trials, STIM_DIM)[0, :, :], rolled.reshape(mod_dim, num_trials, STIM_DIM)[1, :, :]), axis=1)
        else:
            rolled_reshaped = rolled.reshape(num_trials, -1)

        return rolled_reshaped

    def make_trial_inputs(self) -> np.ndarray:
        '''
        Creates stimulus activity arrays for given trials of a particular task type 
        Parameters: 
            task_type: string identifying the task type
            conditions_arr[mods, stim, dir_str, num_trials]: array defining the stimulus conditions for a batch of task trials 
        Returns: 
            ndarray[num_trials, TRIAL_LEN, INPUT_DIM]: array conditing stimulus inputs for a batch of task trials 
        '''
        num_trials = self.cond_arr.shape[-1]
        fix = np.ones((num_trials,1)) 
        no_fix = np.zeros((num_trials,1))
        null_stim = np.zeros((num_trials, STIM_DIM*2))


        stim1 = self._make_activity_vectors(self.cond_arr[:, 0, :, :])
        stim2 = self._make_activity_vectors(self.cond_arr[:, 1, :, :])

        if self.timing=='full': 
                input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1+stim2), 1), np.concatenate((fix, stim1+stim2), 1),  
                                    np.concatenate((fix, stim1+stim2), 1), np.concatenate((no_fix, null_stim), 1)])

        elif self.timing =='RT':
            input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, null_stim), 1), 
                        np.concatenate((fix, null_stim), 1),  np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1+stim2), 1)])
            
        elif self.timing == 'delay':
            input_activity_vecs = np.array([np.concatenate((fix, null_stim), 1), np.concatenate((fix, stim1), 1), np.concatenate((fix, null_stim), 1),  
                                np.concatenate((fix, stim2), 1), np.concatenate((no_fix, null_stim), 1)])

        return _add_noise(self._expand_along_intervals(self.intervals, input_activity_vecs), self.noise)

    def make_loss_mask(self) -> np.ndarray: 
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
        return np.array(list(map(__make_loss_mask__, self.intervals)))
        

    def make_trial_targets(self) -> np.ndarray: 
        '''
        Makes target output activities for a batch of trials
        Parameters:
            target_dirs[num_trials]
        Returns: 
            ndarray[num_trials, TRIAL_LEN, OUTPUT_DIM]: ndarray of target activities for a batch of trials
        '''
        num_trials = self.target_dirs.shape[0]
        fix = np.full((num_trials,1), 0.85)
        go = np.full((num_trials,1), 0.05)
        strengths = np.where(np.isnan(self.target_dirs), np.zeros_like(self.target_dirs), np.ones_like(self.target_dirs))
        target_conditions = np.expand_dims(np.stack((self.target_dirs, strengths), axis=1).T, axis=0)
        target_activities = self._make_activity_vectors(target_conditions)
        resp = np.concatenate((go, target_activities+0.05), 1)
        no_resp = np.concatenate((fix, np.full((num_trials, STIM_DIM), 0.05)), 1)
        trial_target = self._expand_along_intervals(self.intervals, (no_resp, no_resp, no_resp, no_resp, resp))
        return trial_target
    
class GoFactory(TaskFactory): 
    def __init__(self, num_trials,  noise, dir_chooser,
                            timing= 'full', mod=None, multi=False, 
                            cond_arr=None, dir_arr=None,
                            intervals= None,  max_var=False):
        super().__init__(num_trials, timing, noise, intervals)
        self.cond_arr = cond_arr
        self.timing = timing
        self.dir_chooser = dir_chooser
        self.mod = mod
        self.multi = multi

        if max_var: 
            dir_arr = max_var_dir(self.num_trials, self.mod, self.multi, 1)

        if self.cond_arr is None: 
            self.cond_arr = self._make_cond_arr(dir_arr)

        self.target_dirs = self._set_target_dirs()

    def _make_cond_arr(self, dir_arr):        
        #mod, stim, dir_strengths, num_trials
        conditions_arr = np.full((2, 2, 2, self.num_trials), np.NaN)

        if dir_arr is not None: 
            dirs = dir_arr
        if self.multi: 
            dirs0 = np.random.uniform(0, 2*np.pi, self.num_trials)
            dirs1 = dirs0+np.random.uniform(np.pi/4, 3*np.pi/4, self.num_trials)
            _dirs = np.array([dirs0, dirs1])
            nan_stim = np.full_like(_dirs, np.NaN)
            dirs = np.swapaxes(np.array([_dirs, nan_stim]), 0, 1)
        else:
            dirs0 = np.random.uniform(0, 2*np.pi, self.num_trials)
            nan_dirs = np.full_like(dirs0, np.NaN)
            _dirs = np.array([dirs0, nan_dirs])
            nan_mod = np.full_like(_dirs, np.NaN)
            dirs = np.swapaxes(np.array([_dirs, nan_mod]), 0, 1)
            dirs = _permute_mod(dirs)
        
        strs= np.where(np.isnan(dirs), np.full_like(dirs, np.NaN), 
                    np.random.uniform(0.8, 1.2, size=(2, self.num_trials)))

        conditions_arr[:, :, 0, :] = dirs
        conditions_arr[:, :, 1, :] = strs

        return conditions_arr
        
    def _set_target_dirs(self): 
        if self.mod is None: 
            dirs = np.nansum(self.cond_arr[:, 0, 0, :], axis=0)
        else: 
            dirs = self.cond_arr[self.mod, 0, 0, :]

        return self.dir_chooser(dirs)

class DMFactory(TaskFactory):
    def __init__(self, num_trials,  noise, str_chooser,
                        timing= 'full', mod=None, multi=False, 
                        dir_arr = None, coh_arr = None, max_var=False,
                        intervals= None, cond_arr=None):
        super().__init__(num_trials, timing, noise, intervals)
        self.cond_arr = cond_arr
        self.timing = timing
        self.str_chooser = str_chooser
        self.mod = mod
        self.multi = multi

        if max_var: 
            dir_arr = max_var_dir(self.num_trials, self.mod, self.multi, 2)
            coh_arr = max_var_coh(self.num_trials)

        if self.cond_arr is None: 
            self.cond_arr = self._make_cond_arr(dir_arr, coh_arr)

        self.target_dirs = self._set_target_dirs()

    def _make_cond_arr(self, dir_arr, coh_arr):
        conditions_arr = np.full((2, 2, 2, self.num_trials), np.NaN)

        if coh_arr is not None: 
            coh = coh_arr
            dirs = dir_arr
            base_strs = np.random.uniform(0.8, 1.2, size=(2, self.num_trials))

        elif self.mod is not None: 
            dirs0 = _draw_ortho_dirs(self.num_trials)
            dirs1 = _draw_ortho_dirs(self.num_trials)
            dirs = np.array([dirs0, dirs1])
            base_strs = np.random.uniform(0.8, 1.2, size=(2, self.num_trials))
            coh = np.random.choice([-0.175, -0.15, -0.1, 0.1, 0.15, 0.175], size=(2, self.num_trials))

        elif not self.multi: 
            dirs0 =  _draw_ortho_dirs(self.num_trials)
            nan_dirs = np.full_like(dirs0, np.NaN)
            dirs = np.random.permutation(np.array([dirs0, nan_dirs]))
            dirs = _permute_mod(dirs)
            base_strs = np.random.uniform(0.8, 1.2, size=(2, self.num_trials))
            coh = np.random.choice([-0.175, -0.15, -0.1, 0.1, 0.15, 0.175], size=(2, self.num_trials))

        else: 
            dirs0 = _draw_ortho_dirs(self.num_trials)
            dirs1 = dirs0
            dirs = np.array([dirs0, dirs1])
            mod_coh = np.random.choice([0.2, 0.175, 0.15, 0.125, -0.125, -0.15, -0.175, -0.2], self.num_trials)
            mod_base_strength = np.random.uniform(0.8, 1.2, self.num_trials)
            base_strs = np.array([mod_base_strength+mod_coh, mod_base_strength-mod_coh]) 
            coh = _draw_multi_contrast(self.num_trials)

        _strs= np.array([[base_strs[0]+coh[0], base_strs[0]-coh[0]],
                        [base_strs[1]+coh[1], base_strs[1]-coh[1]]])

        strs = np.where(np.isnan(dirs), np.full_like(dirs, np.NaN), _permute_mod(_strs))
        conditions_arr[:, :, 0, :] = dirs
        conditions_arr[:, :, 1, :] = strs

        return conditions_arr

    def _set_target_dirs(self): 
        if np.isnan(self.cond_arr).any(): 
            directions = np.nansum(self.cond_arr[:, :, 0, :], axis=0)
        elif self.mod is not None: 
            directions = self.cond_arr[self.mod, :, 0, :]
        else: 
            directions = self.cond_arr[0, :, 0, :]

        if self.mod is None: 
            strs = np.nansum(self.cond_arr[:, :, 1, :], axis=0)
        else: 
            strs = self.cond_arr[self.mod, :, 1, :]

        chosen_str = self.str_chooser(strs, axis=0)
        target_dirs = np.where(chosen_str, directions[1, :], directions[0, :])
        return target_dirs

class ConDMFactory(TaskFactory): 
    def __init__(self, num_trials,  noise, str_chooser, threshold_folder,
                    timing= 'full', mod=None, multi=False,  
                    dir_arr = None, coh_arr = None, noises=None,                      
                    max_var=False, intervals= None, cond_arr=None):
        super().__init__(num_trials, timing, noise, intervals)
        self.threshold_folder = threshold_folder
        self.multi = multi
        self.cond_arr = cond_arr
        self.timing = timing
        self.str_chooser = str_chooser
        self.mod = mod
        self.pos_thresholds, self.neg_thresholds = pickle.load(open(location+'/noise_thresholds/'+self.threshold_folder, 'rb'))

        if max_var: 
            dir_arr = max_var_dir(self.num_trials, self.mod, self.multi, 2)

        if self.cond_arr is None: 
            self.cond_arr = self._make_cond_arr(dir_arr, coh_arr, noises)
            
        self.target_dirs = self._set_target_dirs()

    def _make_cond_arr(self, dir_arr, coh_arr, noises):
        conditions_arr = np.full((2, 2, 2, self.num_trials), np.NaN)
        self.requires_response_list = _draw_requires_resp(self.num_trials)
        
        self.intervals = np.array([((0, 20), (20, 50), (50, 70), (70, 100), (100, 120))]*self.num_trials)
        dirs0 = _draw_ortho_dirs(self.num_trials)
        noises, contrasts = _draw_confidence_threshold(self.requires_response_list, self.pos_thresholds, self.neg_thresholds)
        self.noise = noises

        if dir_arr is not None: 
            coh = coh_arr
            dirs = dir_arr
            base_strs = np.random.uniform(0.8, 1.2, size=(2, self.num_trials))

        elif self.multi:    
            dirs1 = dirs0
            dirs = np.array([dirs0, dirs1])
            mod_coh = contrasts/2
            base_strs = np.array([1-mod_coh, 1+mod_coh])
            coh = _draw_multi_contrast(self.num_trials, draw_vals=[-0.05, -0.1, 0.1, 0.05])
        else:
            nan_dirs = np.full_like(dirs0, np.NaN)
            dirs = np.random.permutation(np.array([dirs0, nan_dirs]))
            dirs = _permute_mod(dirs)
            base_strs = np.full((2, self.num_trials), 1)
            coh = np.array([1+contrasts/2, 1-contrasts/2], 
                                    [1+contrasts/2, 1-contrasts/2])

        _strs= np.array([[base_strs[0]+coh[0], base_strs[0]-coh[0]],
                        [base_strs[1]+coh[1], base_strs[1]-coh[1]]])

        strs = np.where(np.isnan(dirs), np.full_like(dirs, np.NaN), _permute_mod(_strs))
        conditions_arr[:, :, 0, :] = dirs
        conditions_arr[:, :, 1, :] = strs

        return conditions_arr

    def _set_target_dirs(self):        
        if np.isnan(self.cond_arr).any(): 
            directions = np.nansum(self.cond_arr[:, :, 0, :], axis=0)
        else: 
            directions = self.cond_arr[0, :, 0, :]

        strs = np.nansum(self.cond_arr[:, :, 1, :], axis=0)
        chosen_str = self.str_chooser(strs, axis=0)
        tmp_target_dirs = np.where(chosen_str, directions[1, :], directions[0, :])
        target_dirs = np.where(self.requires_response_list, tmp_target_dirs, np.full_like(tmp_target_dirs, np.NaN))            

        return target_dirs
        
class COMPFactory(TaskFactory):
    def __init__(self, num_trials,  noise, resp_stim,
                            timing= 'delay', mod=None, multi=False, 
                            intervals= None, cond_arr=None, target_dirs=None):
        super().__init__(num_trials, timing, noise, intervals)
        self.cond_arr = cond_arr
        self.target_dirs = target_dirs
        self.timing = timing
        self.resp_stim = resp_stim
        self.mod = mod
        self.multi = multi
        if self.cond_arr is None and self.target_dirs is None: 
            self.cond_arr, self.target_dirs = self._make_cond_tar_dirs()
    
    def _make_cond_tar_dirs(self): 
        conditions_arr = np.full((2, 2, 2, self.num_trials), np.NaN)
        target_dirs = np.empty(self.num_trials)
        requires_response_list = _draw_requires_resp()
        dirs0 = _draw_ortho_dirs()


        if self.multi:        
            dirs0 = _draw_ortho_dirs()
            directions2 = directions1

            mod_coh = np.random.choice([0.2, 0.175, 0.15, 0.125, -0.125, -0.15, -0.175, -0.2])

            base_strength = np.random.uniform(0.8, 1.2)
            mod_base_strs = np.array([base_strength-mod_coh, base_strength+mod_coh]) 

            redraw = True
            while redraw: 
                coh = np.random.choice([-0.25, 0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25], size=2, replace=False)
                if coh[0] != -1*coh[1] and ((coh[0] <0) ^ (coh[1] < 0)): 
                    redraw = False

            mod_swap = np.random.choice([0,1])
            _mod_swap = (mod_swap+1)%2
            tmp_strengths = np.array([[mod_base_strs[mod_swap] - coh[mod_swap], mod_base_strs[mod_swap]+ coh[mod_swap]],
                                [mod_base_strs[_mod_swap] - coh[_mod_swap], mod_base_strs[_mod_swap] + coh[_mod_swap]] ])

            candidate_strs = np.sum(tmp_strengths, axis=0)

            positive_index = np.argmax(candidate_strs)
            positive_strength = tmp_strengths[:, positive_index]
            negative_strength = tmp_strengths[:, (positive_index+1)%2]
            strs = self._set_comp_strs(positive_strength, negative_strength, requires_response, self.resp_stim)
            directions = np.array([directions1, directions2])
            conditions_arr[:, :, 0, i] = directions
            conditions_arr[:, :, 1, i] = strs.T

            if requires_response and self.resp_stim==1:
                target_dirs[i] = directions[0, 0]
            elif requires_response and self.resp_stim==2: 
                target_dirs[i] = directions[0, 1]
            else: 
                target_dirs[i] = None

        else:  
            base_strength = np.random.uniform(0.8, 1.2)
            coh = np.random.choice([0.1, 0.15, 0.2])
            positive_strength = base_strength + coh
            negative_strength = base_strength - coh
            strs = self._set_comp_strs(positive_strength, negative_strength, requires_response, self.resp_stim)
            mod = np.random.choice([0, 1])
            directions = _draw_ortho_dirs()
            conditions_arr[mod, :, 0, i] = np.array([directions])
            conditions_arr[mod, :, 1, i] = strs
            conditions_arr[((mod+1)%2), :, :, i] = np.NaN

            if requires_response and self.resp_stim==1:
                target_dirs[i] = directions[0]
            elif requires_response and self.resp_stim==2: 
                target_dirs[i] = directions[1]
            else: 
                target_dirs[i] = None
    
    return conditions_arr, target_dirs
    
    def _set_comp_strs(self, pos_str, neg_str, req_resp, resp_stim):
        if (resp_stim==1 and req_resp) or (resp_stim==2 and not req_resp): 
            strs = np.array([pos_str, neg_str])
        elif (resp_stim==2 and req_resp) or (resp_stim==1 and not req_resp): 
            strs = np.array([neg_str, pos_str])
        return strs

class MatchingFactory(TaskFactory):
    def __init__(self, num_trials,  noise, matching_task, match_type,
                            timing= 'delay', intervals= None, cond_arr=None, target_dirs = None):
        super().__init__(num_trials, timing, noise, intervals)
        self.timing = timing
        self.cond_arr = cond_arr
        self.target_dirs = target_dirs

        self.matching_task = matching_task
        self.match_type = match_type
        if self.cond_arr is None and self.target_dirs is None: 
            self.cond_arr, self.target_dirs = self._make_cond_tar_dirs()

    def _make_cond_tar_dirs(self):
        #mod, stim, dir_strengths, num_trials
        conditions_arr = np.full((2, 2, 2, self.num_trials), np.NaN)
        target_dirs = np.empty(self.num_trials)
        cat_ranges = np.array([[0, np.pi], [np.pi, 2*np.pi]])

        if self.num_trials>1: 
            match_trial_list = list(np.random.permutation([True]*int(self.num_trials/2) + [False] * int(self.num_trials/2)))
        else: 
            match_trial_list = [np.random.choice([True, False])]

        for i in range(self.num_trials): 
            match_trial = match_trial_list.pop()

            direction1 = np.random.uniform(0, 2*np.pi)
            if direction1 < np.pi: 
                range_index=0
            else: range_index=1
            
            if match_trial and self.match_type=='stim': 
                direction2 = direction1
            elif match_trial and self.match_type=='cat': 
                direction2 = np.random.uniform(cat_ranges[range_index, 0], cat_ranges[range_index, 1])
            elif not match_trial and self.match_type=='cat': 
                opp_range_index = (range_index+1)%2
                direction2 = np.random.uniform(cat_ranges[opp_range_index, 0]+np.pi/5, cat_ranges[opp_range_index, 1]-+np.pi/5)
            else: direction2 = (direction1+np.pi+np.random.uniform(-np.pi*0.5, np.pi*0.5))%(2*np.pi)
            
            mod = np.random.choice([0, 1])
            conditions_arr[mod, :, 0, i] = np.array([direction1, direction2])
            conditions_arr[mod, :, 1, i] = np.random.uniform(0.8, 1.2, size=2)

            
            if match_trial and self.matching_task:
                target_dirs[i] = direction1
            elif not match_trial and not self.matching_task:
                target_dirs[i] = direction2
            else: target_dirs[i] = None
        return conditions_arr, target_dirs

