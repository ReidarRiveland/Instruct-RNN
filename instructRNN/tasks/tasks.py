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

ALIGNED_LIST = [
            ('DM', 'AntiDM', 'MultiCOMP1', 'MultiCOMP2'), 
            ('Go', 'AntiGo', 'DMMod2', 'AntiDMMod2'), 
            ('DMMod1', 'AntiDMMod1', 'RTGo', 'AntiRTGo'), 
            ('GoMod1', 'GoMod2', 'AntiMultiDur1', 'AntiMultiDur2'), 
            ('AntiGoMod1', 'AntiGoMod2', 'MultiDur1', 'MultiDur2'), 
            ('Dur1', 'Dur2', 'RTGoMod1', 'AntiRTGoMod1'), 
            ('MultiDM', 'AntiMultiDM', 'COMP1Mod2', 'COMP2Mod2'), 
            ('RTGoMod2', 'AntiRTGoMod2', 'COMP1Mod1', 'COMP2Mod1'), 
            ('Dur1Mod1', 'Dur2Mod1', 'DMC', 'DNMC'),
            ('AntiDur1', 'AntiDur2', 'COMP1', 'COMP2'), 
            ('Dur1Mod2', 'Dur2Mod2', 'ConDM', 'ConAntiDM'),
            ('AntiCOMP1', 'AntiCOMP2', 'DMS', 'DNMS'),
            ('AntiMultiCOMP1', 'AntiMultiCOMP2')
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



SWAPS_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], SWAP_LIST.copy()))
ALIGNED_DICT = dict(zip(['aligned'+str(num) for num in range(len(ALIGNED_LIST))], ALIGNED_LIST.copy()))
FAMILY_DICT = dict(zip(['family'+str(num) for num in range(len(FAMILY_LIST))], FAMILY_LIST.copy()))
INV_SWAPS_DICT = invert_holdout_dict(SWAPS_DICT)
MULTITASK_DICT = {'Multitask':[]}

# DICH_DICT = {
#     'dich0' : (['Go', 'GoMod1', 'GoMod2'], ['AntiGo', 'AntiGoMod1', 'AntiGoMod2']),
#     'dich1' : (['RTGo', 'RTGoMod1', 'RTGoMod2'], ['AntiRTGo', 'AntiRTGoMod1', 'AntiRTGoMod2']),
#     'dich2' : (['Go', 'GoMod1', 'GoMod2'], ['RTGo', 'RTGoMod1', 'RTGoMod2']),
#     'dich3' : (['AntiGo', 'AntiGoMod1', 'AntiGoMod2'], ['AntiRTGo', 'AntiRTGoMod1', 'AntiRTGoMod2']),

#     'dich4' : (['DM', 'ConDM', 'MultiDM', 'DMMod1', 'DMMod2'],
#                 ['AntiDM', 'ConAntiDM', 'AntiMultiDM', 'AntiDMMod1', 'AntiDMMod2']),

#     'dich5' : (['COMP1', 'MultiCOMP1', 'AntiCOMP1', 'AntiMultiCOMP1', 'COMP1Mod1', 'COMP1Mod2'],
#                 ['COMP2', 'MultiCOMP2', 'AntiCOMP2', 'AntiMultiCOMP2', 'COMP2Mod1', 'COMP2Mod2']),
                
#     'dich6' : (['COMP1', 'MultiCOMP1', 'COMP2', 'MultiCOMP2'],
#                 ['AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2']), 

#     'dich7' : (['Dur1', 'AntiDur1', 'MultiDur1', 'AntiMultiDur1', 'Dur1Mod1', 'Dur1Mod2'],
#                 ['Dur2', 'AntiDur2', 'MultiDur2', 'AntiMultiDur2', 'Dur2Mod1', 'Dur2Mod2']), 

#     'dich8' : (['Dur1', 'Dur2', 'MultiDur1', 'MultiDur2'],
#                 ['AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2']), 

#     'dich9' : (['DMS', 'DNMS'], ['DMC', 'DNMC']),
#     'dich10' : (['DNMS', 'DNMC'], ['DMS', 'DMC']), 
#     'dich11' : (['GoMod1', 'AntiGoMod1', 'RTGoMod1', 'AntiRTGoMod1', 'DMMod1', 'AntiDMMod1', 'COMP1Mod1', 'COMP2Mod1', 'Dur1Mod1', 'Dur2Mod1'], 
#                 ['GoMod2', 'AntiGoMod2', 'RTGoMod2', 'AntiRTGoMod2', 'DMMod2', 'AntiDMMod2', 'COMP1Mod2', 'COMP2Mod2', 'Dur1Mod2', 'Dur2Mod2'])
# }


DICH_DICT = {
    'dich0' : [('Go', 'AntiGo'), ('GoMod1', 'AntiGoMod1'), ('GoMod2', 'AntiGoMod2'), 
                            ('RTGo', 'AntiRTGo'),('RTGoMod1', 'AntiRTGoMod1'), ('RTGoMod2', 'AntiRTGoMod2')],
    'dich1' : (['RTGo', 'RTGoMod1', 'RTGoMod2'], ['AntiRTGo', 'AntiRTGoMod1', 'AntiRTGoMod2']),
    'dich2' : (['Go', 'GoMod1', 'GoMod2'], ['RTGo', 'RTGoMod1', 'RTGoMod2']),
    'dich3' : (['AntiGo', 'AntiGoMod1', 'AntiGoMod2'], ['AntiRTGo', 'AntiRTGoMod1', 'AntiRTGoMod2']),

    'dich4' : (['DM', 'ConDM', 'MultiDM', 'DMMod1', 'DMMod2'],
                ['AntiDM', 'ConAntiDM', 'AntiMultiDM', 'AntiDMMod1', 'AntiDMMod2']),

    'dich5' : (['COMP1', 'MultiCOMP1', 'AntiCOMP1', 'AntiMultiCOMP1', 'COMP1Mod1', 'COMP1Mod2'],
                ['COMP2', 'MultiCOMP2', 'AntiCOMP2', 'AntiMultiCOMP2', 'COMP2Mod1', 'COMP2Mod2']),
                
    'dich6' : (['COMP1', 'MultiCOMP1', 'COMP2', 'MultiCOMP2'],
                ['AntiCOMP1', 'AntiCOMP2', 'AntiMultiCOMP1', 'AntiMultiCOMP2']), 

    'dich7' : (['Dur1', 'AntiDur1', 'MultiDur1', 'AntiMultiDur1', 'Dur1Mod1', 'Dur1Mod2'],
                ['Dur2', 'AntiDur2', 'MultiDur2', 'AntiMultiDur2', 'Dur2Mod1', 'Dur2Mod2']), 

    'dich8' : (['Dur1', 'Dur2', 'MultiDur1', 'MultiDur2'],
                ['AntiDur1', 'AntiDur2', 'AntiMultiDur1', 'AntiMultiDur2']), 

    'dich9' : (['DMS', 'DNMS'], ['DMC', 'DNMC']),
    'dich10' : (['DNMS', 'DNMC'], ['DMS', 'DMC']), 
    'dich11' : (['GoMod1', 'AntiGoMod1', 'RTGoMod1', 'AntiRTGoMod1', 'DMMod1', 'AntiDMMod1', 'COMP1Mod1', 'COMP2Mod1', 'Dur1Mod1', 'Dur2Mod1'], 
                ['GoMod2', 'AntiGoMod2', 'RTGoMod2', 'AntiRTGoMod2', 'DMMod2', 'AntiDMMod2', 'COMP1Mod2', 'COMP2Mod2', 'Dur1Mod2', 'Dur2Mod2'])
}




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
    
    def plot_trial(self, index):
        ins = self.inputs
        tars = self.targets
        fix = ins[index, :, 0:1]
        mod1 = ins[index, :, 1:task_factory.STIM_DIM+1]
        mod2 = ins[index, :, 1+task_factory.STIM_DIM:1+(2*task_factory.STIM_DIM)]
        tars = tars[index, :, :]

        to_plot = (fix.T, mod1.T, mod2.T, tars.T)

        gs_kw = dict(width_ratios=[1], height_ratios=[1, 5, 5, 5])

        fig, axn = plt.subplots(4,1, sharex = True, gridspec_kw=gs_kw)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        ylabels = ('fix.', 'mod. 1', 'mod. 2', 'Target')
        for i, ax in enumerate(axn.flat):
            sns.heatmap(to_plot[i], yticklabels = False, cmap = 'Reds', ax=ax, cbar=i == 0, vmin=0, vmax=1.5, cbar_ax=None if i else cbar_ax)

            ax.set_ylabel(ylabels[i])
            if i == 0: 
                ax.set_title('%r Trial Info' %self.task_type)
            if i == 3: 
                ax.set_xlabel('time')
        plt.show()

   
class Go(Task): 
    comp_ref_tasks = ('RTGo', 'AntiRTGo', 'AntiGo')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'Go'

class AntiGo(Task): 
    comp_ref_tasks = ('AntiRTGo', 'RTGo', 'Go')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'AntiGo'

class RTGo(Task):
    comp_ref_tasks = ('Go', 'AntiGo', 'AntiRTGo')
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
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        **factory_kwargs
                        )
        self.task_type = 'DM'

class AntiDM(Task):
    comp_ref_tasks = ('AntiMultiDM', 'MultiDM', 'DM')
    def __init__(self, num_trials, noise=None,**factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        **factory_kwargs
                        )
        self.task_type = 'AntiDM'

class MultiDM(Task):
    comp_ref_tasks = ('DM', 'AntiDM', 'AntiMultiDM')
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
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=1,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'AntiDMMod2'

class Dur(Task): 
    comp_ref_tasks = ('MultiDur', 'AntiMultiDur', 'AntiDur')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMDurFactory, 
                        dur_chooser = np.greater,
                        **factory_kwargs
                        )
        self.task_type = 'Dur'

class AntiDur(Task): 
    comp_ref_tasks = ('AntiMultiDur', 'MultiDur', 'Dur')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMDurFactory, 
                        dur_chooser = np.less,
                        **factory_kwargs
                        )
        self.task_type = 'AntiDur'

class MultiDur(Task): 
    comp_ref_tasks = ('Dur', 'AntiDur', 'AntiMultiDur')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMDurFactory, 
                        dur_chooser = np.greater,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'MultiDur'

class AntiMultiDur(Task): 
    comp_ref_tasks = ('AntiDur', 'Dur', 'MultiDur')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMDurFactory, 
                        dur_chooser = np.less,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'AntiMultiDur'

class Dur1(Task): 
    comp_ref_tasks = ('MultiDur1', 'MultiDur2', 'Dur2')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'Dur1'


class Dur2(Task): 
    comp_ref_tasks = ('MultiDur2', 'MultiDur1', 'Dur1')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'Dur2'


class MultiDur1(Task): 
    comp_ref_tasks = ('Dur1', 'Dur2', 'MultiDur2')
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
    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = True, match_type = 'stim',
                        **factory_kwargs)
        self.task_type = 'DMS'

class DNMS(Task):
    comp_ref_tasks = ('DNMC', 'DMC', 'DMS')
    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = False, match_type = 'stim',
                        **factory_kwargs)
        self.task_type = 'DNMS'

class DMC(Task):
    comp_ref_tasks = ('DMS', 'DNMS', 'DNMC')
    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.MatchingFactory,                        
                        matching_task = True, match_type = 'cat',
                        **factory_kwargs)
        self.task_type = 'DMC'

class DNMC(Task):
    comp_ref_tasks = ('DNMS', 'DMS', 'DMC')
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

# trials = AntiDMMod2(100)
# trials.plot_trial(0)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('task')
#     parser.add_argument('-num', default=1)
#     args = parser.parse_args()

#     _task = construct_trials(args.task)
#     trials = _task(args.num)

#     for index in range(args.num):
#         trials.plot_trial(index)
