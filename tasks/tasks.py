import numpy as np
import torch
import tasks.task_factory as task_factory

def invert_task_dict(task_dict):
    inv_swap_dict = {}
    for k, v in task_dict.items():
        for task in v:
            inv_swap_dict[task] = k
    return inv_swap_dict

TASK_LIST = ['Go', 'Anti_Go', 'RT_Go', 'Anti_RT_Go', 
            
            'Go_Mod1', 'Anti_Go_Mod1', 'Go_Mod2', 'Anti_Go_Mod2',

            'DelayGo', 'Anti_DelayGo',

            'DM', 'Anti_DM', 'MultiDM', 'Anti_MultiDM', 

            'RT_DM', 'Anti_RT_DM', 

            'ConDM', 'Anti_ConDM', 'ConMultiDM', 'Anti_ConMultiDM',            

            'DelayDM', 'Anti_DelayDM', 'DelayMultiDM', 'Anti_DelayMultiDM',

            'DM_Mod1', 'Anti_DM_Mod1', 'DM_Mod2', 'Anti_DM_Mod2',
            
            #'RT_DM_Mod1', 'Anti_RT_DM_Mod1', 'RT_DM_Mod2', 'Anti_RT_DM_Mod2', 

            'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 

            'COMP1_Mod1', 'COMP2_Mod1', 'COMP1_Mod2', 'COMP2_Mod2',

            'DMS', 'DNMS', 'DMC', 'DNMC']

# SWAP_LIST = [            
#             ('Anti_DM_Mod2', 'RT_Go', 'Anti_ConDM', 'COMP1'), 
#             ('DelayDM', 'Anti_Go_Mod2', 'ConMultiDM', 'DMS'),
#             ('DM_Mod2', 'Anti_RT_Go',  'Go', 'MultiCOMP1'), 
#             ('Go_Mod2', 'Anti_ConMultiDM', 'Anti_DelayGo', 'COMP2'), 
#             ('Anti_DelayMultiDM', 'MultiDM', 'Anti_RT_DM', 'DMC'),             
#             ('DM', 'Anti_Go', 'Go_Mod1', 'MultiCOMP2'), 
#             ('Anti_DM', 'DelayMultiDM', 'Anti_Go_Mod1',  'DNMS'), 
#             ('RT_DM', 'Anti_MultiDM', 'DM_Mod1', 'DNMC'),
#             ('ConDM', 'Anti_DelayDM', 'DelayGo', 'Anti_DM_Mod1')
#             ]

SWAP_LIST = [
            ('Anti_DM_Mod2', 'Go_Mod1', 'Anti_Go', 'COMP1', 'ConMultiDM'), 
            ('Anti_Go_Mod1', 'DM_Mod2', 'Go', 'MultiCOMP1', 'ConDM'), 
            ('DM_Mod1', 'Anti_Go_Mod2', 'DelayGo', 'DMS', 'Anti_ConDM'),
            ('Go_Mod2', 'Anti_DM_Mod1', 'Anti_DelayGo', 'COMP2', 'Anti_ConMultiDM'), 
            ('Anti_DelayDM', 'MultiDM', 'Anti_RT_DM', 'DMC', 'COMP1_Mod1'),             
            ('DM', 'Anti_DelayMultiDM', 'RT_Go', 'MultiCOMP2', 'COMP1_Mod2'), 
            ('Anti_DM', 'DelayMultiDM', 'Anti_RT_Go',  'DNMS', 'COMP2_Mod2'), 
            ('RT_DM', 'Anti_MultiDM', 'DelayDM', 'DNMC', 'COMP2_Mod1')
            ]
            
ALIGNED_LIST = [
            ('DM', 'Anti_DM', 'MultiCOMP1', 'MultiCOMP2'), 
            ('Go', 'Anti_Go', 'COMP1_Mod1', 'COMP1_Mod2'), 
            ('DM_Mod1', 'DM_Mod2', 'COMP2_Mod1', 'COMP2_Mod2'), 
            ('Go_Mod1', 'Go_Mod2', 'ConDM', 'Anti_ConDM'), 
            ('Anti_Go_Mod1', 'Anti_Go_Mod2', 'DelayMultiDM', 'Anti_DelayMultiDM'), 
            ('DelayGo', 'Anti_DelayGo', 'ConMultiDM', 'Anti_ConMultiDM'), 
            ('MultiDM', 'Anti_MultiDM', 'DMS', 'DNMS'), 
            ('RT_DM', 'Anti_RT_DM', 'COMP1', 'COMP2'), 
            ('DelayDM', 'Anti_DelayDM', 'DMC', 'DNMC'), 
            ('RT_Go', 'Anti_RT_Go', 'Anti_DM_Mod1', 'Anti_DM_Mod2')
            ]


SWAPS_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], SWAP_LIST.copy()))
ALIGNED_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], ALIGNED_LIST.copy()))
INV_SWAPS_DICT = invert_task_dict(SWAPS_DICT)


class Task(): 
    def __init__(self, num_trials, noise, factory, **factory_kwargs):
        if noise is None: 
            noise = np.random.uniform(0.05, 0.1)
        self.num_trials = num_trials
        self.noise = noise
        self.factory = factory(num_trials, noise, **factory_kwargs)
        self.conditions_arr = self.factory.cond_arr
        self.target_dirs = self.factory.target_dirs
        self.inputs = self.factory.make_trial_inputs()
        self.targets = self.factory.make_trial_targets()
        self.masks = self.factory.make_loss_mask()

class Go(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'Go'


class AntiGo(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_Go'

class DelayGo(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing='delay', 
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'DelayGo'

class AntiDelayGo(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing='delay', 
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_DelayGo'

class DelayGoMod1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing='delay', 
                        mod=0,
                        multi=True,
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'DelayGo_Mod1'

class DelayGoMod2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing='delay', 
                        mod=1,
                        multi=True,
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'DelayGo_Mod2'


class AntiDelayGoMod1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing='delay', 
                        mod=0,
                        multi=True,
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_DelayGo_Mod1'

class AntiDelayGoMod2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing='delay', 
                        mod=1,
                        multi=True,
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_DelayGo_Mod2'


class RTGo(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        dir_chooser = task_factory.choose_pro
                        )
        self.task_type = 'RT_Go'

class AntiRTGo(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        dir_chooser = task_factory.choose_anti
                        )
        self.task_type = 'Anti_RT_Go'

class GoMod1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 0, 
                        dir_chooser = task_factory.choose_pro, 
                        multi=True
                        )
        self.task_type = 'Go_Mod1'

class GoMod2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 1, 
                        dir_chooser = task_factory.choose_pro,
                        multi=True
                        )
        self.task_type = 'Go_Mod2'

class AntiGoMod1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 0, 
                        dir_chooser = task_factory.choose_anti, 
                        multi=True
                        )
        self.task_type = 'Anti_Go_Mod1'

class AntiGoMod2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 1, 
                        dir_chooser = task_factory.choose_anti, 
                        multi=True
                        )
        self.task_type = 'Anti_Go_Mod2'


class DM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        )
        self.task_type = 'DM'

class AntiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        )
        self.task_type = 'Anti_DM'


class ConDM(Task):
    def __init__(self, num_trials, noise=None):
        noise = np.random.uniform(0.1, 0.4)
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmax,
                        threshold_folder = 'dm_noise_thresholds'
                        )
        self.task_type = 'ConDM'

class ConAntiDM(Task):
    def __init__(self, num_trials, noise=None): 
        noise = np.random.uniform(0.1, 0.4)

        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmin,
                        threshold_folder = 'anti_dm_noise_thresholds'
                        )
        self.task_type = 'Anti_ConDM'

class ConMultiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        threshold_folder = 'multi_dm_noise_thresholds'
                        )
        self.task_type = 'ConMultiDM'

class ConAntiMultiDM(Task):
    def __init__(self, num_trials, noise=None): 
        noise = np.random.uniform(0.01, 0.4) 
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmin,
                        multi=True,
                        threshold_folder = 'anti_multi_dm_noise_thresholds'
                        )
        self.task_type = 'Anti_ConMultiDM'

class RTDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        timing = 'RT'
                        )
        self.task_type = 'RT_DM'

class AntiRTDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        timing = 'RT'
                        )
        self.task_type = 'Anti_RT_DM'

class MultiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        multi=True
                        )
        self.task_type = 'MultiDM'

class AntiMultiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        multi=True
                        )
        self.task_type = 'Anti_MultiDM'

class DelayDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        timing='delay'
                        )
        self.task_type = 'DelayDM'

class DelayAntiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        timing='delay'
                        )
        self.task_type = 'Anti_DelayDM'

class DelayMultiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        timing='delay',
                        multi=True
                        )
        self.task_type = 'DelayMultiDM'

class DelayAntiMultiDM(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        timing='delay',
                        multi=True
                        )
        self.task_type = 'Anti_DelayMultiDM'

class DMMod1(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        mod=0, 
                        multi=True
                        )
        self.task_type = 'DM_Mod1'

class DMMod2(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        mod=1, 
                        multi=True
                        )
        self.task_type = 'DM_Mod2'
        
class AntiDMMod1(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=0, 
                        multi=True
                        )
        self.task_type = 'Anti_DM_Mod1'

class AntiDMMod2(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=1,
                        multi=True
                        )
        self.task_type = 'Anti_DM_Mod2'

class RTDMMod1(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        timing='RT',
                        mod=0,
                        multi=True
                        )
        self.task_type = 'RT_DM_Mod1'

class RTDMMod2(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        timing='RT',
                        mod=1,
                        multi=True
                        )
        self.task_type = 'RT_DM_Mod2'

class AntiRTDMMod1(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        timing='RT',
                        mod=0,
                        multi=True
                        )
        self.task_type = 'Anti_RT_DM_Mod1'

class AntiRTDMMod2(Task):
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        timing='RT',
                        mod=1,
                        multi=True
                        )
        self.task_type = 'Anti_RT_DM_Mod2'

class COMP1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 1
                        )
        self.task_type = 'COMP1'

class COMP2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 2
                        )
        self.task_type = 'COMP2'

class MultiCOMP1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 1, 
                        multi=True
                        )
        self.task_type = 'MultiCOMP1'

class MultiCOMP2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 2,
                        multi=True
                        )
        self.task_type = 'MultiCOMP2'


class COMP1Mod1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 1, 
                        mod=0, 
                        multi=True
                        )
        self.task_type = 'COMP1_Mod1'

class COMP1Mod2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 1, 
                        mod=1,
                        multi=True
                        )
        self.task_type = 'COMP1_Mod2'

class COMP2Mod1(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 2, 
                        mod=0,
                        multi=True
                        )
        self.task_type = 'COMP2_Mod1'

class COMP2Mod2(Task): 
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 2, 
                        mod=1,
                        multi=True
                        )
        self.task_type = 'COMP2_Mod2'

class DMS(Task):
    def __init__(self, num_trials, noise=None):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = True, match_type = 'stim')
        self.task_type = 'DMS'

class DNMS(Task):
    def __init__(self, num_trials, noise=None):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = False, match_type = 'stim')
        self.task_type = 'DNMS'

class DMC(Task):
    def __init__(self, num_trials, noise=None):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = True, match_type = 'cat')
        self.task_type = 'DMC'

class DNMC(Task):
    def __init__(self, num_trials, noise=None):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = False, match_type = 'cat')
        self.task_type = 'DNMC'
 
def construct_trials(task_type, num_trials, noise = None, return_tensor=False):
    assert task_type in TASK_LIST, "entered invalid task type"
    if task_type == 'Go':
        trial = Go(num_trials, noise=noise)
    if task_type == 'RT_Go':
        trial = RTGo(num_trials, noise=noise)
    if task_type == 'Anti_Go':
        trial = AntiGo(num_trials, noise=noise)
    if task_type == 'Anti_RT_Go':
        trial = AntiRTGo(num_trials, noise=noise)

    if task_type == 'DelayGo': 
        trial = DelayGo(num_trials, noise=noise)
    if task_type == 'Anti_DelayGo': 
        trial = AntiDelayGo(num_trials, noise=noise)

    if task_type == 'DelayGo_Mod1':
        trial = DelayGoMod1(num_trials, noise=noise)
    if task_type == 'DelayGo_Mod2':
        trial = DelayGoMod2(num_trials, noise=noise)
    if task_type == 'Anti_DelayGo_Mod1':
        trial = AntiDelayGoMod1(num_trials, noise=noise)
    if task_type == 'Anti_DelayGo_Mod2':
        trial = AntiDelayGoMod2(num_trials, noise=noise)

    if task_type == 'Go_Mod1':
        trial = GoMod1(num_trials, noise=noise)
    if task_type == 'Go_Mod2':
        trial = GoMod2(num_trials, noise=noise)
    if task_type == 'Anti_Go_Mod1':
        trial = AntiGoMod1(num_trials, noise=noise)
    if task_type == 'Anti_Go_Mod2':
        trial = AntiGoMod2(num_trials, noise=noise)

    if task_type == 'DM':
        trial = DM(num_trials, noise=noise)
    if task_type == 'Anti_DM': 
        trial = AntiDM(num_trials, noise=noise)

    if task_type == 'ConDM':
        trial = ConDM(num_trials, noise=noise)
    if task_type == 'Anti_ConDM': 
        trial = ConAntiDM(num_trials, noise=noise)
    if task_type == 'ConMultiDM':
        trial = ConMultiDM(num_trials, noise=noise)
    if task_type == 'Anti_ConMultiDM': 
        trial = ConAntiMultiDM(num_trials, noise=noise)

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

    if task_type == 'RT_DM_Mod1': 
        trial = RTDMMod1(num_trials, noise=noise)
    if task_type == 'RT_DM_Mod2': 
        trial = RTDMMod2(num_trials, noise=noise)
    if task_type == 'Anti_RT_DM_Mod1': 
        trial = AntiRTDMMod1(num_trials, noise=noise)
    if task_type == 'Anti_RT_DM_Mod2': 
        trial = AntiRTDMMod2(num_trials, noise=noise)

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
                TASK_LIST.index(task_type))


