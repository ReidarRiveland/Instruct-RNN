from lib2to3.pytree import Base
import numpy as np
from sklearn import multiclass
import torch
import task_factory

TASK_LIST = ['Go', 'Anti_Go', 'RT_Go', 'Anti_RT_Go', 
            'Go_Mod1', 'Anti_Go_Mod1', 'Go_Mod2', 'Anti_Go_Mod2',
            'Order1', 'Order2',
            'DM', 'Anti_DM', 'RT_DM', 'Anti_RT_DM', 
            'MultiDM', 'Anti_MultiDM', 
            'DelayDM', 'Anti_DelayDM', 'DelayMultiDM', 'Anti_DelayMultiDM',
            'DM_Mod1', 'Anti_DM_Mod1', 'DM_Mod2', 'Anti_DM_Mod2',
            'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 
            'COMP1_Mod1', 'COMP2_Mod1', 'COMP1_Mod2', 'COMP2_Mod2',
            'DMS', 'DNMS', 'DMC', 'DNMC']

SWAPPED_TASK_LIST = ['Anti DM', 'MultiCOMP1', 'DNMC', 'DMC', 'MultiCOMP2', 'Go', 'DNMS', 'COMP1', 'Anti MultiDM', 'DMS', 'Anti Go', 'DM', 'COMP2', 'MultiDM', 'Anti RT Go', 'RT Go']
TASK_GROUP_DICT = {'Go': ['Go', 'Anti Go', 'RT Go', 'Anti RT Go'],
            'DM': ['DM', 'Anti DM', 'MultiDM', 'Anti MultiDM'], 
            'COMP': ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'],
            'Delay': ['DMS', 'DNMS', 'DMC', 'DNMC']}

class Task(): 
    def __init__(self, num_trials,  fill_type, noise, conditions_factory,  conditions_arr=None, intervals = None, **conditions_kwargs):
        if conditions_arr is None: 
            self.conditions_arr, self.target_dirs = conditions_factory(num_trials, **conditions_kwargs)
        else:
            self.conditions_arr= conditions_arr

        if intervals is None: 
            self.intervals = task_factory.make_intervals(num_trials)
        else: 
            self.intervals = intervals

        self.inputs = task_factory.make_trial_inputs(fill_type, self.conditions_arr, self.intervals, noise)
        self.targets = task_factory.make_trial_targets(self.target_dirs, self.intervals)
        self.masks = task_factory.make_loss_mask(self.intervals)

class Go(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.go_factory, 
                        mod = None, 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'Go'

class AntiGo(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.go_factory, 
                        mod = None, 
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_Go'

class RTGo(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'RT', noise,
                        task_factory.go_factory, 
                        mod = None, 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'RT_Go'

class AntiRTGo(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'RT', noise,
                        task_factory.go_factory, 
                        mod = None, 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'Anti_RT_Go'

class GoMod1(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.go_factory, 
                        mod = 0, 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'Go_Mod1'

class GoMod2(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.go_factory, 
                        mod = 1, 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'Go_Mod2'

class AntiGoMod1(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.go_factory, 
                        mod = 0, 
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_Go_Mod1'

class AntiGoMod2(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.go_factory, 
                        mod = 1, 
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_Go_Mod2'

class Order1(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.order_factory, 
                        respond_stim = 1, 
                        )
        self.task_type = 'Order1'

class Order2(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.order_factory, 
                        respond_stim = 2, 
                        )
        self.task_type = 'Order2'

class DM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax
                        )
        self.task_type = 'DM'

class AntiDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmin
                        )
        self.task_type = 'Anti_DM'


class RTDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'RT', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax
                        )
        self.task_type = 'RT_DM'

class AntiRTDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'RT', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmin
                        )
        self.task_type = 'Anti_RT_DM'

class MultiDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax,
                        multi=True
                        )
        self.task_type = 'MultiDM'

class AntiMultiDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax,
                        multi=True
                        )
        self.task_type = 'Anti_MultiDM'

class DelayDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax
                        )
        self.task_type = 'DelayDM'

class DelayAntiDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmin
                        )
        self.task_type = 'Anti_DelayDM'

class DelayMultiDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax,
                        multi=True
                        )
        self.task_type = 'DelayMultiDM'

class DelayAntiMultiDM(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.dm_factory, 
                        mod = None, 
                        str_chooser = np.argmax,
                        multi=True
                        )
        self.task_type = 'Anti_DelayMultiDM'

class DMMod1(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = 0, 
                        str_chooser = np.argmax, 
                        multi=True
                        )
        self.task_type = 'DM_Mod1'

class DMMod2(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = 1, 
                        str_chooser = np.argmax,
                        multi=True
                        )
        self.task_type = 'DM_Mod2'
        
class AntiDMMod1(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = 0, 
                        str_chooser = np.argmin,
                        multi=True
                        )
        self.task_type = 'Anti_DM_Mod1'

class AntiDMMod2(Task):
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'full', noise,
                        task_factory.dm_factory, 
                        mod = 1, 
                        str_chooser = np.argmin, 
                        multi=True
                        )
        self.task_type = 'Anti_DM_Mod2'

class COMP1(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = None, 
                        resp_stim = 1
                        )
        self.task_type = 'COMP1'

class COMP2(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = None, 
                        resp_stim = 2
                        )
        self.task_type = 'COMP2'

class MultiCOMP1(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = None, 
                        resp_stim = 1,
                        multi=True
                        )
        self.task_type = 'MultiCOMP1'

class MultiCOMP2(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = None, 
                        resp_stim = 2, 
                        multi = True
                        )
        self.task_type = 'MultiCOMP2'

class COMP1Mod1(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = 0, 
                        resp_stim = 1, 
                        multi=True
                        )
        self.task_type = 'COMP1_Mod1'

class COMP1Mod2(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = 1, 
                        resp_stim = 1,
                        multi=True
                        )
        self.task_type = 'COMP1_Mod2'

class COMP2Mod1(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = 0, 
                        resp_stim = 2,
                        multi=True
                        )
        self.task_type = 'COMP2_Mod1'

class COMP2Mod2(Task): 
    def __init__(self, num_trials, noise=0.05): 
        super().__init__(num_trials, 'delay', noise,
                        task_factory.comp_factory, 
                        mod = 1, 
                        resp_stim = 2,
                        multi=True
                        )
        self.task_type = 'COMP2_Mod2'

class DMS(Task):
    def __init__(self, num_trials, noise=0.05):
        super().__init__(num_trials, 'delay',noise,
                        task_factory.matching_factory,                        
                        matching_task = True, match_type = 'stim')
        self.task_type = 'DMS'

class DNMS(Task):
    def __init__(self, num_trials, noise=0.05):
        super().__init__(num_trials, 'delay',noise,
                        task_factory.matching_factory,                        
                        matching_task = False, match_type = 'stim')
        self.task_type = 'DNMS'

class DMC(Task):
    def __init__(self, num_trials, noise=0.05):
        super().__init__(num_trials, 'delay',noise,
                        task_factory.matching_factory,                        
                        matching_task = True, match_type = 'cat')
        self.task_type = 'DMC'

class DNMC(Task):
    def __init__(self, num_trials, noise=0.05):
        super().__init__(num_trials, 'delay',noise,
                        task_factory.matching_factory,                        
                        matching_task = False, match_type = 'cat')
        self.task_type = 'DNMC'
 

def construct_trials(task_type, num_trials, noise = 0.05, return_tensor=False):
    assert task_type in TASK_LIST, "entered invalid task type"
    if task_type == 'Go':
        trial = Go(num_trials, noise=noise)
    if task_type == 'RT_Go':
        trial = RTGo(num_trials, noise=noise)
    if task_type == 'Anti_Go':
        trial = AntiGo(num_trials, noise=noise)
    if task_type == 'Anti_RT_Go':
        trial = AntiRTGo(num_trials, noise=noise)
    if task_type == 'Go_Mod1':
        trial = GoMod1(num_trials, noise=noise)
    if task_type == 'Go_Mod2':
        trial = GoMod1(num_trials, noise=noise)
    if task_type == 'Anti_Go_Mod1':
        trial = AntiGoMod1(num_trials, noise=noise)
    if task_type == 'Anti_Go_Mod2':
        trial = AntiGoMod2(num_trials, noise=noise)
    if task_type == 'Order1': 
        trial = Order1(num_trials, noise=noise)
    if task_type == 'Order2': 
        trial = Order2(num_trials, noise=noise)
    if task_type == 'DM':
        trial = DM(num_trials, noise=noise)
    if task_type == 'Anti_DM': 
        trial = AntiDM(num_trials, noise=noise)
    if task_type == 'RT_DM':
        trial = RTDM(num_trials, noise=noise)
    if task_type == 'Anti_RT_DM': 
        trial = AntiRTDM(num_trials, noise=noise)
    if task_type == 'MultiDM':
        trial = MultiDM(num_trials, noise=noise)
    if task_type == 'Anti_MultiDM': 
        trial = AntiMultiDM(num_trials, noise=noise)
    if task_type == 'DelayDM': 
        trial = DelayDM(num_trials, noise=noise)
    if task_type == 'Anti_DelayDM': 
        trial = DelayAntiDM(num_trials, noise=noise)
    if task_type == 'DelayMultiDM': 
        trial = DelayMultiDM(num_trials, noise=noise)
    if task_type == 'Anti_DelayMultiDM': 
        trial = DelayAntiMultiDM(num_trials, noise=noise)
    if task_type == 'DM_Mod1': 
        trial = DMMod1(num_trials, noise=noise)
    if task_type == 'DM_Mod2': 
        trial = DMMod2(num_trials, noise=noise)
    if task_type == 'Anti_DM_Mod1': 
        trial = AntiDMMod1(num_trials, noise=noise)
    if task_type == 'Anti_DM_Mod2': 
        trial = AntiDMMod2(num_trials, noise=noise)
    if task_type == 'COMP1': 
        trial = COMP1(num_trials, noise=noise)
    if task_type == 'COMP2': 
        trial = COMP2(num_trials, noise=noise)
    if task_type == 'MultiCOMP1': 
        trial = MultiCOMP1(num_trials, noise=noise)
    if task_type == 'MultiCOMP2': 
        trial = MultiCOMP2(num_trials, noise=noise)
    if task_type == 'COMP1_Mod1': 
        trial = COMP1Mod1(num_trials, noise=noise)
    if task_type == 'COMP1_Mod2': 
        trial = COMP1Mod2(num_trials, noise=noise)
    if task_type == 'COMP2_Mod1': 
        trial = COMP2Mod1(num_trials, noise=noise)
    if task_type == 'COMP2_Mod2': 
        trial = COMP2Mod2(num_trials, noise=noise)
    if task_type == 'DMS': 
        trial = DMS(num_trials, noise=noise)
    if task_type == 'DNMS': 
        trial = DNMS(num_trials, noise=noise)
    if task_type == 'DMC': 
        trial = DMC(num_trials, noise=noise)
    if task_type == 'DNMC': 
        trial = DNMC(num_trials, noise=noise)

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
