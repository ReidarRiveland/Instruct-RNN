from lib2to3.pytree import Base
import numpy as np
import torch
import task_factory as task_factory

class Task(): 
    def __init__(self, num_trials, conditions_factory, fill_type, sigma_in = 0.05, conditions_arr=None, intervals = None, **conditions_kwargs):
        if conditions_arr is None: 
            self.conditions_arr = conditions_factory(num_trials, **conditions_kwargs)
        else:
            self.conditions_arr= conditions_arr

        if intervals is None: 
            self.intervals = task_factory.make_intervals(num_trials)
        else: 
            self.intervals = intervals

        self.make_target_dirs()

        self.inputs = task_factory.make_trial_inputs(fill_type, self.conditions_arr, self.intervals, sigma_in)
        self.targets = task_factory.make_trial_targets(self.target_dirs, self.intervals)
        self.masks = task_factory.make_loss_mask(self.intervals)

class BaseDM(Task):
    def __init__(self, num_trials, dir_chooser, fill_style, **kw_args):
        self.dir_chooser = dir_chooser
        super().__init__(num_trials, task_factory.make_dual_stim_conditions, fill_style, **kw_args)

    def make_target_dirs(self): 
        if np.isnan(self.conditions_arr).any(): 
            directions = np.nansum(self.conditions_arr[:, :, 0, :], axis=0)
        else: 
            directions = self.conditions_arr[0, :, 0, :]

        chosen_dir = self.dir_chooser(np.nansum(self.conditions_arr[:, :, 1, :], axis=0), axis=0)
        self.target_dirs = np.where(chosen_dir, directions[1, :], directions[0, :])

class DM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmax, 'full')
        self.task_type = 'DM'

class AntiDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmin, 'full')
        self.task_type = 'Anti DM'

class MultiDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmax, 'full', multi=True)
        self.task_type = 'MultiDM'

class AntiMultiDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmin, 'full', multi=True)
        self.task_type = 'Anti DM'

class DelayDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmax, 'delay')
        self.task_type = 'DM'

class DelayAntiDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmin, 'delay')
        self.task_type = 'Anti DM'

class DelayMultiDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmax, 'delay', multi=True)
        self.task_type = 'MultiDM'

class DelayAntiMultiDM(BaseDM):
    def __init__(self, num_trials):
        super().__init__(num_trials, np.argmin, 'delay', multi=True)
        self.task_type = 'Anti DM'





trials = DelayMultiDM(128)

#mod, stim, dir_strengths, num_trials

index = 1
task_factory.plot_trial(trials.inputs[index], trials.targets[index], trials.task_type)



def construct_batch(task_type, num, return_tensor=False):
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

    if return_tensor: 
        return (torch.tensor(trial.inputs), 
                torch.tensor(trial.targets), 
                torch.tensor(trial.masks), 
                torch.tensor(trial.target_dirs), 
                task_type)
    else:
        return (trial.inputs.astype(np.float32), 
                trial.targets.astype(np.float32), 
                trial.masks.astype(int), 
                trial.target_dirs.astype(np.float32), 
                task_type)

def build_training_data(foldername):
    for task in Task.TASK_LIST: 
        print(task)
        task_file = task.replace(' ', '_')
        input_data, target_data, masks_data, target_dirs, trial_indices = construct_batch(task, 8000)
        np.save(foldername+'/training_data/' + task_file+'/input_data', input_data)
        np.save(foldername+'/training_data/' + task_file+'/target_data', target_data)
        np.save(foldername+'/training_data/' + task_file+'/masks_data', masks_data)
        np.save(foldername+'/training_data/' + task_file+'/target_dirs', target_dirs)
        np.save(foldername+'/training_data/' + task_file+'/type_indices', trial_indices)

def make_test_trials(task_type, task_variable, mod, num_trials=100, sigma_in = 0.05): 
    assert task_variable in ['direction', 'strength', 'diff_direction', 'diff_strength']

    conditions_arr = np.empty((2, 2, 2, num_trials))
    intervals = np.empty((num_trials, 5), dtype=tuple)
    for i in range(num_trials): intervals[i, :] = ((0, 20), (20, 50), (50, 70), (70, 100), (100, 120))

    if task_variable == 'direction': 
        directions = np.linspace(0, 2*np.pi, num=num_trials)
        strengths = np.array([1]* num_trials)
        var_of_interest = directions

    elif task_variable == 'strength': 
        directions = np.array([np.pi] * num_trials)
        strengths = np.linspace(0.3, 1.8, num=num_trials)
        var_of_interest = strengths

    elif task_variable == 'diff_strength': 
        directions = np.array([[np.pi/2] * num_trials, [3*np.pi/2] * num_trials])

        #directions = np.array([[np.pi/2] * num_trials, [3*np.pi/2] * num_trials])
        fixed_strengths = np.array([1]* num_trials)
        diff_strength = np.linspace(-0.5, 0.5, num=num_trials)
        strengths = np.array([fixed_strengths, fixed_strengths-diff_strength])
        var_of_interest = diff_strength

    elif task_variable == 'diff_direction': 
        fixed_direction = np.array([np.pi/2] * num_trials)
        diff_directions = np.linspace(0, 2*np.pi, num=num_trials)
        directions = np.array([fixed_direction, fixed_direction - diff_directions])
        strengths = np.array([[1.5] * num_trials, [1.5] * num_trials])
        var_of_interest = diff_directions

    if task_type in Task.TASK_GROUP_DICT['Go']:
        conditions_arr[mod, 0, 0, :] = directions
        conditions_arr[mod, 0, 1, :] = strengths
        conditions_arr[mod, 1, 0, :] = np.NaN
        conditions_arr[((mod+1)%2), :, :, :] = np.NaN
        trials = Go(task_type, num_trials, intervals=intervals, conditions_arr=conditions_arr, sigma_in=sigma_in)

    if task_type in Task.TASK_GROUP_DICT['DM'] or task_type in Task.TASK_GROUP_DICT['COMP']:
        assert task_variable not in ['directions', 'strengths']
        conditions_arr[mod, :, 0, : ] = directions
        conditions_arr[mod, :, 1, : ] = strengths

        if 'Multi' in task_type: 
            conditions_arr[((mod+1)%2), :, :, :] = conditions_arr[mod, :, :, : ]
        else: 
            conditions_arr[((mod+1)%2), :, :, : ] = np.NaN

        if 'DM' in task_type: 
            trials = DM(task_type, num_trials, intervals=intervals, conditions_arr=conditions_arr, sigma_in=sigma_in)
        if 'COMP' in task_type: 
            trials = Comp(task_type, num_trials, intervals=intervals, conditions_arr=conditions_arr, sigma_in=sigma_in)

    if task_type in Task.TASK_GROUP_DICT['Delay']: 
        conditions_arr[mod, :, 0, : ] = directions
        conditions_arr[mod, :, 1, : ] = strengths
        conditions_arr[((mod+1)%2), :, :, : ] = np.NaN
        trials = Delay(task_type, num_trials, intervals=intervals, conditions_arr=conditions_arr, sigma_in=sigma_in)

    return trials, var_of_interest

def gpu_to_np(t):
    """removes tensor from gpu and converts to np.array""" 
    if t.get_device() >= 0: 
        t = t.detach().to('cpu').numpy()
    elif t.get_device() == -1: 
        t = t.detach().numpy()
    return t

def popvec(act_vec):
    """Population vector decoder that reads the orientation of activity in vector of activities
    Args:      
        act_vec (np.array): output tensor of neural network model; shape: (features, 1)
    Returns:
        float: decoded orientation of activity (in radians)
    """
    act_sum = np.sum(act_vec, axis=1)
    temp_cos = np.sum(np.multiply(act_vec, np.cos(Task.TUNING_DIRS)), axis=1)/act_sum
    temp_sin = np.sum(np.multiply(act_vec, np.sin(Task.TUNING_DIRS)), axis=1)/act_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)
 
def get_dist(angle1, angle2):
    """Returns the true distance between two angles mod 2pi
    Args:      
        angle1, angle2 (float): angles in radians
    Returns:
        float: distance between given angles mod 2pi
    """
    dist = angle1-angle2
    return np.minimum(abs(dist),2*np.pi-abs(dist))

def isCorrect(nn_out, nn_target, target_dirs): 
    """Determines whether a given neural network response is correct, computed by batch
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_size, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_size, seq_len, features)
        target_dirs (np.array): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        np.array: weighted loss of neural network response; shape: (batch)
    """

    batch_size = nn_out.shape[0]
    if type(nn_out) == torch.Tensor: 
        nn_out = gpu_to_np(nn_out)
    if type(nn_target) == torch.Tensor: 
        nn_target = gpu_to_np(nn_target)

    criterion = (2*np.pi)/10

    #checks response maintains fixataion
    isFixed = np.all(np.where(nn_target[:, :, 0] == 0.85, nn_out[:, :, 0] > 0.5, True), axis=1)

    #checks for repressed responses
    isRepressed = np.all(nn_out[:, 114:119, :].reshape(batch_size, -1) < 0.15, axis = 1)

    #checks is responses are in the correct direction 
    is_response = np.max(nn_out[:, -1, 1:]) > 0.6
    loc = popvec(nn_out[:, -1, 1:])
    dist = get_dist(loc, np.nan_to_num(target_dirs))        
    isDir = np.logical_and(dist < criterion, is_response)

    #checks if responses were correctly repressed or produced 
    correct_reponse = np.where(np.isnan(target_dirs), isRepressed, isDir)

    #checks if correct responses also maintained fixation 
    is_correct = np.logical_and(correct_reponse, isFixed)
    return is_correct
