import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import instructRNN.tasks.task_factory as task_factory

def invert_holdout_dict(task_dict):
    inv_swap_dict = {}
    for k, v in task_dict.items():
        for task in v:
            inv_swap_dict[task] = k
    return inv_swap_dict

def _check_test_list(exp_list):
    task_list = TASK_LIST.copy()
    for tasks in exp_list: 
        for task in tasks: 
            task_list.pop(task_list.index(task))
    return task_list

TASK_LIST = ['Go', 'AntiGo', 'RTGo', 'AntiRTGo', 
            'GoMod1', 'AntiGoMod1', 'GoMod2',  'AntiGoMod2',
            'RTGoMod1', 'AntiRTGoMod1',  'RTGoMod2','AntiRTGoMod2',
            'DM', 'AntiDM', 'MultiDM', 'AntiMultiDM', 
            'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2',
            'ConDM', 'ConAntiDM',
            'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 
            'AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2', 
            'COMP1Mod1', 'COMP2Mod1', 'COMP1Mod2', 'COMP2Mod2', 
            'Dur1', 'Dur2', 'MultiDur1', 'MultiDur2',
            'AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2',
            'Dur1Mod1', 'Dur2Mod1', 'Dur1Mod2', 'Dur2Mod2', 
            'DMS', 'DNMS', 'DMC', 'DNMC', 
            ]

SUBTASKS_DICT = {'small': ['Go', 'AntiGo', 'RTGo', 'AntiRTGo', 
                    'GoMod1', 'AntiGoMod1', 'GoMod2',  'AntiGoMod2',
                    'RTGoMod1', 'AntiRTGoMod1',  'RTGoMod2','AntiRTGoMod2',
                    'DM', 'AntiDM', 'MultiDM', 'AntiMultiDM', 
                    'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2',
                    'DMS', 'DNMS', 'DMC', 'DNMC'],
    
                'Go': ['Go', 'AntiGo', 'RTGo', 'AntiRTGo', 
                        'GoMod1', 'AntiGoMod1', 'GoMod2',  'AntiGoMod2',
                        'RTGoMod1', 'AntiRTGoMod1',  'RTGoMod2','AntiRTGoMod2'],

                'DM': ['DM', 'AntiDM', 'MultiDM', 'AntiMultiDM', 
                        'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2',
                        'ConDM', 'ConAntiDM'],

                'COMP': ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 
                        'AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2', 
                        'COMP1Mod1', 'COMP2Mod1', 'COMP1Mod2', 'COMP2Mod2'],

                'DUR': [ 'Dur1', 'Dur2', 'MultiDur1', 'MultiDur2',
                        'AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2',
                        'Dur1Mod1', 'Dur2Mod1', 'Dur1Mod2', 'Dur2Mod2'],

                'MATCH':['DMS', 'DNMS', 'DMC', 'DNMC', ]
                }


SUBTASKS_SWAP_DICT = {'Go': {'go_swap0': ('Go', 'AntiGoMod1', 'RTGoMod2'),
                        'go_swap1': ('AntiGo', 'GoMod1', 'AntiRTGoMod2' ),
                        'go_swap2': ('RTGo', 'AntiRTGoMod1', 'AntiGoMod2'), 
                        'go_swap3': ('AntiRTGo', 'RTGoMod1', 'GoMod2')},

                    'small': {'small_swap0': ('Go', 'AntiRTGoMod2', 'DMMod1', 'DNMS'),
                        'small_swap1': ('AntiGo', 'RTGoMod1', 'AntiDMMod2', 'DMS'),
                        'small_swap2': ('RTGo', 'AntiDMMod1', 'DMMod2', 'DNMC'), 
                        'small_swap3': ('AntiRTGo', 'RTGoMod2', 'MultiDM', 'AntiGoMod1'),
                        'small_swap4': ('GoMod1', 'AntiGoMod2', 'DMC', 'AntiDM' ), 
                        'small_swap5': ('AntiRTGoMod1', 'DM', 'GoMod2', 'AntiMultiDM')},
                }

NONCOND_CLAUSE_LIST = ['Go', 'AntiGo', 'RTGo', 'AntiRTGo', 
            'GoMod1', 'AntiGoMod1', 'GoMod2',  'AntiGoMod2',
            'RTGoMod1', 'AntiRTGoMod1',  'RTGoMod2','AntiRTGoMod2',
            'DM', 'AntiDM', 'MultiDM', 'AntiMultiDM', 
            'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2',
            ]

COND_CLAUSE_LIST = ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 
            'AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2', 
            'COMP1Mod1', 'COMP2Mod1', 'COMP1Mod2', 'COMP2Mod2', 
            'Dur1', 'Dur2', 'MultiDur1', 'MultiDur2',
            'AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2',
            'Dur1Mod1', 'Dur2Mod1', 'Dur1Mod2', 'Dur2Mod2', 
            'DMS', 'DNMS', 'DMC', 'DNMC', 'ConDM', 'ConAntiDM']

SWAP_LIST = [            
            ('AntiDMMod2', 'RTGo', 'DM', 'MultiCOMP2',  'AntiMultiDur1'), 
            ('COMP1Mod1', 'AntiGoMod2',  'DMS', 'AntiDur1', 'RTGoMod2'),
            ('RTGoMod1', 'AntiCOMP2', 'AntiRTGo', 'Dur2', 'MultiCOMP1'), 
            ('GoMod2', 'AntiMultiCOMP2', 'DMMod2', 'AntiRTGoMod1', 'AntiDur2'), 
            ('MultiDM', 'COMP2Mod2', 'AntiMultiCOMP1', 'AntiGoMod1', 'Dur1Mod1'),     
            ('AntiDM',  'AntiRTGoMod2', 'Dur2Mod2', 'AntiCOMP1', 'DNMS'), 
            ('MultiDur1',  'GoMod1', 'COMP2', 'DMC', 'Dur2Mod1'),
            ('COMP2Mod1', 'AntiMultiDM', 'DNMC', 'DMMod1', 'Dur1Mod2'),
            ('ConAntiDM', 'COMP1', 'MultiDur2', 'COMP1Mod2', 'Go'),
            ('AntiGo', 'Dur1', 'ConDM', 'AntiDMMod1', 'AntiMultiDur2'),            
            ]

FAMILY_LIST = [
            ('Go', 'AntiGo', 'RTGo', 'AntiRTGo'), 
            ('GoMod1',  'GoMod2', 'AntiGoMod1', 'AntiGoMod2'), 
            ('RTGoMod1', 'RTGoMod2', 'AntiRTGoMod1', 'AntiRTGoMod2'), 
            ('DM', 'AntiDM', 'MultiDM', 'AntiMultiDM', 'ConDM', 'ConAntiDM'),
            ('DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'),
            ('COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2'),
            ('AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2'),
            ('COMP1Mod1', 'COMP2Mod1', 'COMP1Mod2', 'COMP2Mod2'), 
            ('Dur1', 'Dur2', 'MultiDur1', 'MultiDur2'),
            ('AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2'),
            ('Dur1Mod1', 'Dur2Mod1', 'Dur1Mod2', 'Dur2Mod2'), 
            ('DMS', 'DNMS', 'DMC', 'DNMC')
            ]

TASK_GROUPS = {
            'Go': ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2',  'AntiGoMod2', 'RTGo', 'AntiRTGo', 'RTGoMod1', 'AntiRTGoMod1',  'RTGoMod2','AntiRTGoMod2'],
            'DM' : ['DM', 'AntiDM', 'MultiDM', 'AntiMultiDM', 'ConDM', 'ConAntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'],
            'COMP' : ['COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 'AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2', 'COMP1Mod1', 'COMP2Mod1', 'COMP1Mod2', 'COMP2Mod2'], 
            'Dur' : ['Dur1', 'Dur2', 'MultiDur1', 'MultiDur2', 'AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2', 'Dur1Mod1', 'Dur2Mod1', 'Dur1Mod2', 'Dur2Mod2'], 
            'Match' : ['DMS', 'DNMS', 'DMC', 'DNMC']
            }

SWAPS_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], SWAP_LIST.copy()))
FAMILY_DICT = dict(zip(['family'+str(num) for num in range(len(FAMILY_LIST))], FAMILY_LIST.copy()))
INV_SWAPS_DICT = invert_holdout_dict(SWAPS_DICT)
INV_GROUP_DICT = invert_holdout_dict(TASK_GROUPS)
MULTITASK_DICT = {'Multitask':[]}

DICH_DICT = {
    'dich0' : [('Go', 'AntiGo'), ('GoMod1', 'AntiGoMod1'), ('GoMod2', 'AntiGoMod2'), 
                            ('RTGo', 'AntiRTGo'),('RTGoMod1', 'AntiRTGoMod1'), ('RTGoMod2', 'AntiRTGoMod2')],

    'dich1' : [('Go', 'RTGo'), ('GoMod1', 'RTGoMod1'), ('GoMod2', 'RTGoMod2'), 
                            ('AntiGo', 'AntiRTGo'),('AntiGoMod1', 'AntiRTGoMod1'), ('AntiGoMod2', 'AntiRTGoMod2')],

    'dich2' : [('DM', 'AntiDM'), ('DMMod1', 'AntiDMMod1'), ('DMMod2', 'AntiDMMod2'), ('MultiDM', 'AntiMultiDM'), ('ConDM', 'ConAntiDM'),
                ('COMP1', 'AntiCOMP1'), ('COMP2', 'AntiCOMP2'), ('MultiCOMP1', 'AntiMultiCOMP1'), ('MultiCOMP2', 'AntiMultiCOMP2')],

    'dich3' : [('Dur1', 'AntiDur1'), ('Dur2', 'AntiDur2'), ('MultiDur1', 'AntiMultiDur1'), ('MultiDur2', 'AntiMultiDur2')],

    'dich4' : [('Dur1', 'Dur2'),  ('AntiDur1', 'AntiDur2'), ('MultiDur1', 'MultiDur2'), 
                    ('AntiMultiDur1', 'AntiMultiDur2'), ('Dur1Mod1', 'Dur2Mod1'), ('Dur1Mod2', 'Dur2Mod2'), 
                    ('COMP1', 'COMP2'), ('MultiCOMP1', 'MultiCOMP2'), ('AntiCOMP1', 'AntiCOMP2'), ('AntiMultiCOMP1', 'AntiMultiCOMP2'),
                    ('COMP1Mod1', 'COMP2Mod1'), ('COMP1Mod2', 'COMP2Mod2')],

    'dich5' : [('DMS', 'DNMS'), ('DMC', 'DNMC')],
    'dich6' : [('DNMS', 'DNMC'), ('DMS', 'DMC')],
    'dich7' : [('GoMod1', 'GoMod2'), ('AntiGoMod1', 'AntiGoMod2'), ('RTGoMod1', 'RTGoMod2'), ('AntiRTGoMod1', 'AntiRTGoMod2'),
                ('DMMod1', 'DMMod2'), ('AntiDMMod1', 'AntiDMMod2'), ('COMP1Mod1', 'COMP1Mod2'), 
                ('COMP2Mod1', 'COMP2Mod2'), ('Dur1Mod1', 'Dur1Mod2'), ('Dur2Mod1', 'Dur2Mod2')]
}

#rich vec dims
# pro/anti
# standard/rt
# mod1/mod2
# strongest/weakest
# multi
# con
# first/second
# longest/shortest
# matching/non-matching
# stimulus/category


class Task(): 
    def __init__(self, num_trials, noise, factory, **factory_kwargs):
        if noise is None: 
            noise = np.random.uniform(0.1, 0.15)
        self.num_trials = num_trials
        self.noise = noise
        self.factory = factory(num_trials, noise, **factory_kwargs)
        self.conditions_arr = self.factory.cond_arr
        self.target_dirs = self.factory.target_dirs
        self.inputs = self.factory.make_trial_inputs()
        self.targets = self.factory.make_trial_targets()
        self.masks = self.factory.make_loss_mask()
   
class Go(Task): 
    comp_ref_tasks = ('RTGo', 'AntiRTGo', 'AntiGo')
    rich_vector = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'Go'

class AntiGo(Task): 
    comp_ref_tasks = ('AntiRTGo', 'RTGo', 'Go')
    rich_vector = [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'AntiGo'

class RTGo(Task):
    comp_ref_tasks = ('Go', 'AntiGo', 'AntiRTGo')
    rich_vector = [1, -1, 0, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'RTGo'

class AntiRTGo(Task):
    comp_ref_tasks = ('AntiGo', 'Go', 'RTGo')
    rich_vector = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'AntiRTGo'

class GoMod1(Task): 
    comp_ref_tasks = ('RTGoMod1', 'AntiRTGoMod1', 'AntiGoMod1')
    rich_vector = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 0, multi=True,
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'GoMod1'

class GoMod2(Task): 
    comp_ref_tasks = ('RTGoMod2', 'AntiRTGoMod2', 'AntiGoMod2')
    rich_vector = [1, 1, -1, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 1, multi=True,
                        dir_chooser = task_factory.choose_pro,
                        **factory_kwargs
                        )
        self.task_type = 'GoMod2'

class AntiGoMod1(Task): 
    comp_ref_tasks = ('AntiRTGoMod1', 'RTGoMod1', 'GoMod1')
    rich_vector = [-1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 0, multi=True,
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs 
                        )
        self.task_type = 'AntiGoMod1'

class AntiGoMod2(Task): 
    comp_ref_tasks = ('AntiRTGoMod2', 'RTGoMod2', 'GoMod2')
    rich_vector = [-1, 1, -1, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 1, multi=True,
                        dir_chooser = task_factory.choose_anti, 
                        **factory_kwargs
                        )
        self.task_type = 'AntiGoMod2'

class RTGoMod1(Task):
    comp_ref_tasks = ('GoMod1', 'AntiGoMod1', 'AntiRTGoMod1')
    rich_vector = [1, -1, 1, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        mod=0,
                        multi=True,
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'RTGoMod1'

class AntiRTGoMod1(Task):
    comp_ref_tasks = ('AntiGoMod1', 'GoMod1', 'RTGoMod1')
    rich_vector = [-1, -1, 1, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        mod=0,
                        multi=True,
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'AntiRTGoMod1'

class RTGoMod2(Task):
    comp_ref_tasks = ('GoMod2', 'AntiGoMod2', 'AntiRTGoMod2')
    rich_vector = [1, -1, -1, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        mod=1,
                        multi=True,
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'RTGoMod2'

class AntiRTGoMod2(Task):
    comp_ref_tasks = ('AntiGoMod2', 'GoMod2', 'RTGoMod2')
    rich_vector = [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        mod=1,
                        multi=True,
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'AntiRTGoMod2'

class DM(Task):
    comp_ref_tasks = ('MultiDM', 'AntiMultiDM', 'AntiDM')
    rich_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        **factory_kwargs
                        )
        self.task_type = 'DM'

class AntiDM(Task):
    comp_ref_tasks = ('AntiMultiDM', 'MultiDM', 'DM')
    rich_vector = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None,**factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        **factory_kwargs
                        )
        self.task_type = 'AntiDM'

class MultiDM(Task):
    comp_ref_tasks = ('DM', 'AntiDM', 'AntiMultiDM')
    rich_vector = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'MultiDM'

class AntiMultiDM(Task):
    comp_ref_tasks = ('AntiDM', 'DM', 'MultiDM')
    rich_vector = [0, 0, 0, -1, 1, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None,**factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'AntiMultiDM'

class ConDM(Task):
    comp_ref_tasks = ('DM', 'AntiDM', 'ConAntiDM')
    rich_vector = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmax,
                        threshold_folder = 'dm_noise_thresholds',
                        **factory_kwargs
                        )
        self.task_type = 'ConDM'

class ConAntiDM(Task):
    comp_ref_tasks = ('AntiDM', 'DM', 'ConDM')
    rich_vector = [0, 0, 0, -1, 0, 1, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmin,
                        threshold_folder = 'anti_dm_noise_thresholds',
                        **factory_kwargs
                        )
        self.task_type = 'ConAntiDM'

class DMMod1(Task):
    comp_ref_tasks = ('DMMod2', 'AntiDMMod2', 'AntiDMMod1')
    rich_vector = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        mod=0, 
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'DMMod1'

class DMMod2(Task):
    comp_ref_tasks = ('DMMod1', 'AntiDMMod1', 'AntiDMMod2')
    rich_vector = [0, 0, -1, 1, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        mod=1, 
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'DMMod2'
        
class AntiDMMod1(Task):
    comp_ref_tasks = ('AntiDMMod2', 'DMMod2', 'DMMod1')
    rich_vector = [0, 0, 1, -1, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None,**factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=0, 
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'AntiDMMod1'

class AntiDMMod2(Task):
    comp_ref_tasks = ('AntiDMMod1', 'DMMod1', 'DMMod2')
    rich_vector = [0, 0, -1, -1, 0, 0, 0, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=1,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'AntiDMMod2'

class Dur1(Task): 
    comp_ref_tasks = ('MultiDur1', 'MultiDur2', 'Dur2')
    rich_vector = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'Dur1'


class Dur2(Task): 
    comp_ref_tasks = ('MultiDur2', 'MultiDur1', 'Dur1')
    rich_vector = [0, 0, 0, 0, 0, 0, -1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'Dur2'


class MultiDur1(Task): 
    comp_ref_tasks = ('Dur1', 'Dur2', 'MultiDur2')
    rich_vector = [0, 0, 0, 0, 1, 0, 1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True, 
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'MultiDur1'


class MultiDur2(Task): 
    comp_ref_tasks = ('Dur2', 'Dur1', 'MultiDur1')
    rich_vector = [0, 0, 0, 0, 1, 0, -1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True,
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'MultiDur2'

class AntiDur1(Task): 
    comp_ref_tasks = ('AntiMultiDur1', 'AntiMultiDur2', 'AntiDur2')
    rich_vector = [0, 0, 0, 0, 0, 0, 1, -1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 0,
                        tar = 'short',

                        **factory_kwargs
                        )
        self.task_type = 'AntiDur1'

class AntiDur2(Task): 
    comp_ref_tasks = ('AntiMultiDur2', 'AntiMultiDur1', 'AntiDur1')
    rich_vector = [0, 0, 0, 0, 0, 0, -1, -1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 1,
                        tar = 'short',
                        **factory_kwargs
                        )
        self.task_type = 'AntiDur2'

class AntiMultiDur1(Task): 
    comp_ref_tasks = ('AntiDur1', 'AntiDur2', 'AntiMultiDur2')
    rich_vector = [0, 0, 0, 0, 1, 0, 1, -1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True, 
                        resp_stim = 0,
                        tar = 'short',
                        **factory_kwargs
                        )
        self.task_type = 'AntiMultiDur1'


class AntiMultiDur2(Task): 
    comp_ref_tasks = ('AntiDur2', 'AntiDur1', 'AntiMultiDur1')
    rich_vector = [0, 0, 0, 0, 1, 0, -1, -1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True,
                        resp_stim = 1,
                        tar = 'short',
                        **factory_kwargs
                        )
        self.task_type = 'AntiMultiDur2'

class Dur1Mod1(Task): 
    comp_ref_tasks = ('Dur1Mod2', 'Dur2Mod2', 'Dur2Mod1')
    rich_vector = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True,
                        resp_stim = 0,
                        mod = 0, 
                        **factory_kwargs
                        )
        self.task_type = 'Dur1Mod1'


class Dur1Mod2(Task): 
    comp_ref_tasks = ('Dur1Mod1', 'Dur2Mod1', 'Dur2Mod2')
    rich_vector = [0, 0, -1, 0, 0, 0, 1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True,
                        resp_stim = 0,
                        mod=1,
                        **factory_kwargs
                        )
        self.task_type = 'Dur1Mod2'

class Dur2Mod1(Task): 
    comp_ref_tasks = ('Dur2Mod2', 'Dur1Mod2', 'Dur1Mod1')
    rich_vector = [0, 0, 1, 0, 0, 0, -1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True,
                        resp_stim = 1,
                        mod = 0, 
                        **factory_kwargs
                        )
        self.task_type = 'Dur2Mod1'


class Dur2Mod2(Task): 
    comp_ref_tasks = ('Dur2Mod1', 'Dur1Mod1', 'Dur1Mod2')
    rich_vector = [0, 0, -1, 0, 0, 0, -1, 1, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        multi=True,
                        resp_stim = 1,
                        mod=1,
                        **factory_kwargs
                        )
        self.task_type = 'Dur2Mod2'

class COMP1(Task): 
    comp_ref_tasks = ('MultiCOMP1', 'MultiCOMP2', 'COMP2')
    rich_vector = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'COMP1'

class COMP2(Task): 
    comp_ref_tasks = ('MultiCOMP2', 'MultiCOMP1', 'COMP1')
    rich_vector = [0, 0, 0, 1, 0, 0, -1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'COMP2'

class MultiCOMP1(Task): 
    comp_ref_tasks = ('COMP1', 'COMP2', 'MultiCOMP2')
    rich_vector = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        resp_stim = 0, 
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'MultiCOMP1'

class MultiCOMP2(Task): 
    comp_ref_tasks = ('COMP2', 'COMP1', 'MultiCOMP1')
    rich_vector = [0, 0, 0, 1, 1, 0, -1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        resp_stim = 1,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'MultiCOMP2'


class AntiCOMP1(Task): 
    comp_ref_tasks = ('AntiMultiCOMP1', 'AntiMultiCOMP2', 'AntiCOMP2')
    rich_vector = [0, 0, 0, -1, 0, 0, 1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmin,
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'AntiCOMP1'

class AntiCOMP2(Task): 
    comp_ref_tasks = ('AntiMultiCOMP2', 'AntiMultiCOMP1', 'COMP1')
    rich_vector = [0, 0, 0, -1, 0, 0, -1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmin,
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'AntiCOMP1'


class AntiMultiCOMP1(Task): 
    comp_ref_tasks = ('AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP2')
    rich_vector = [0, 0, 0, -1, 1, 0, 1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        multi=True, 
                        str_chooser = np.argmin,
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'AntiMultiCOMP1'

class AntiMultiCOMP2(Task): 
    comp_ref_tasks = ('AntiCOMP2', 'AntiCOMP1', 'AntiMultiCOMP1')
    rich_vector = [0, 0, 0, -1, 1, 0, -1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        multi=True, 
                        str_chooser = np.argmin,
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'AntiMultiCOMP2'


class COMP1Mod1(Task): 
    comp_ref_tasks = ('COMP1Mod2', 'COMP2Mod2', 'COMP2Mod1')
    rich_vector = [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        resp_stim = 0,
                        mod = 0,
                        **factory_kwargs
                        )
        self.task_type = 'COMP1Mod1'

class COMP1Mod2(Task): 
    comp_ref_tasks = ('COMP1Mod1', 'COMP2Mod1', 'COMP2Mod2')
    rich_vector = [0, 0, -1, 1, 0, 0, 1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        resp_stim = 0,
                        mod = 1,
                        **factory_kwargs
                        )
        self.task_type = 'COMP1Mod2'

class COMP2Mod1(Task): 
    comp_ref_tasks = ('COMP2Mod2', 'COMP1Mod2', 'COMP1Mod1')
    rich_vector = [0, 0, 1, 1, 0, 0, -1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        resp_stim = 1,
                        mod = 0,
                        **factory_kwargs
                        )
        self.task_type = 'COMP2Mod1'


class COMP2Mod2(Task): 
    comp_ref_tasks = ('COMP2Mod1', 'COMP1Mod1', 'COMP1Mod2')
    rich_vector = [0, 0, -1, 1, 0, 0, -1, 0, 0, 0]

    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        multi=True,
                        str_chooser = np.argmax,
                        resp_stim = 1,
                        mod = 1,
                        **factory_kwargs
                        )
        self.task_type = 'COMP2Mod2'

class DMS(Task):
    comp_ref_tasks = ('DMC', 'DNMC', 'DNMS')
    rich_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = True, match_type = 'stim',
                        **factory_kwargs)
        self.task_type = 'DMS'

class DNMS(Task):
    comp_ref_tasks = ('DNMC', 'DMC', 'DMS')
    rich_vector = [0, 0, 0, 0, 0, 0, 0, 0, -1, 1]

    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = False, match_type = 'stim',
                        **factory_kwargs)
        self.task_type = 'DNMS'

class DMC(Task):
    comp_ref_tasks = ('DMS', 'DNMS', 'DNMC')
    rich_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, -1]

    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = True, match_type = 'cat',
                        **factory_kwargs)
        self.task_type = 'DMC'

class DNMC(Task):
    comp_ref_tasks = ('DNMS', 'DMS', 'DMC')
    rich_vector = [0, 0, 0, 0, 0, 0, 0, 0, -1, -1]

    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = False, match_type = 'cat',
                        **factory_kwargs)
        self.task_type = 'DNMC'


def construct_trials(task_type, num_trials=None, return_tensor=False, **factory_kwargs):
    assert task_type in TASK_LIST, "entered invalid task type"
    trial = getattr(sys.modules[__name__], task_type)

    if num_trials is None: 
        return trial 
    else: 
        trial = trial(num_trials, **factory_kwargs)

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

