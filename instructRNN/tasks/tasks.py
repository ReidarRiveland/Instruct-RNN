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

TASK_LIST = ['Go', 'Anti_Go', 'RT_Go', 'Anti_RT_Go', 
            
            'Go_Mod1', 'Anti_Go_Mod1', 'Go_Mod2', 'Anti_Go_Mod2',

            'DM', 'Anti_DM', 'MultiDM', 'Anti_MultiDM', 

            'ConDM', 'Anti_ConDM', 'ConMultiDM', 'Anti_ConMultiDM',            

            'DM_Mod1', 'Anti_DM_Mod1', 'DM_Mod2', 'Anti_DM_Mod2',
            
            'COMP1', 'COMP2', 'MultiCOMP1', 'MultiCOMP2', 

            'COMP1_Mod1', 'COMP2_Mod1', 'COMP1_Mod2', 'COMP2_Mod2', 

            'Dur1', 'Dur2', 'DurLong', 'DurShort',
            
            'DMS', 'DNMS', 'DMC', 'DNMC']

SWAP_LIST = [            
            ('Anti_DM_Mod2', 'RT_Go', 'Anti_ConDM', 'COMP1'), 
            ('COMP1_Mod1', 'Anti_Go_Mod2', 'ConMultiDM', 'DMS'),
            ('DM_Mod2', 'Anti_RT_Go',  'Go', 'MultiCOMP1'), 
            ('Go_Mod2', 'Anti_ConMultiDM', 'DurLong', 'COMP2'), 
            ('Dur1', 'MultiDM', 'COMP2_Mod2', 'DMC'),             
            ('DM', 'Anti_Go', 'Go_Mod1', 'MultiCOMP2'), 
            ('Anti_DM', 'Dur2', 'Anti_Go_Mod1',  'DNMS'), 
            ('COMP2_Mod1', 'Anti_MultiDM', 'DM_Mod1', 'DNMC'),
            ('ConDM', 'DurShort', 'COMP1_Mod2', 'Anti_DM_Mod1')
            ]

            
ALIGNED_LIST = [
            ('DM', 'Anti_DM', 'MultiCOMP1', 'MultiCOMP2'), 
            ('Go', 'Anti_Go', 'Anti_DM_Mod1', 'Anti_DM_Mod2'), 
            ('DM_Mod1', 'DM_Mod2', 'RT_Go', 'Anti_RT_Go'), 
            ('Go_Mod1', 'Go_Mod2', 'ConDM', 'Anti_ConDM'), 
            ('Anti_Go_Mod1', 'Anti_Go_Mod2', 'DelayMultiDM', 'Anti_DelayMultiDM'), 
            ('DelayGo', 'Anti_DelayGo', 'ConMultiDM', 'Anti_ConMultiDM'), 
            ('MultiDM', 'Anti_MultiDM', 'DMS', 'DNMS'), 
            ('RT_DM', 'Anti_RT_DM', 'COMP1', 'COMP2'), 
            ('DelayDM', 'Anti_DelayDM', 'DMC', 'DNMC')
            ]


SWAPS_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], SWAP_LIST.copy()))
ALIGNED_DICT = dict(zip(['aligned'+str(num) for num in range(len(ALIGNED_LIST))], ALIGNED_LIST.copy()))
MULTITASK_DICT = {'Multitask':[]}

INV_SWAPS_DICT = invert_holdout_dict(SWAPS_DICT)


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
    comp_ref_tasks = ('RT_Go', 'Anti_RT_Go', 'Anti_Go')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'Go'


class AntiGo(Task): 
    comp_ref_tasks = ('Anti_RT_Go', 'RT_Go', 'Go')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'Anti_Go'


class RTGo(Task):
    comp_ref_tasks = ('Go', 'Anti_Go', 'Anti_RT_Go')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'RT_Go'

class AntiRTGo(Task):
    comp_ref_tasks = ('Anti_Go', 'Go', 'RT_Go')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        timing = 'RT',
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs
                        )
        self.task_type = 'Anti_RT_Go'


class GoMod1(Task): 
    comp_ref_tasks = ('Go_Mod2', 'Anti_Go_Mod2', 'Anti_Go_Mod1')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 0, multi=True,
                        dir_chooser = task_factory.choose_pro, 
                        **factory_kwargs
                        )
        self.task_type = 'Go_Mod1'

class GoMod2(Task): 
    comp_ref_tasks = ('Go_Mod1', 'Anti_Go_Mod1', 'Anti_Go_Mod2')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 1, multi=True,
                        dir_chooser = task_factory.choose_pro,
                        **factory_kwargs
                        )
        self.task_type = 'Go_Mod2'

class AntiGoMod1(Task): 
    comp_ref_tasks = ('Anti_Go_Mod2', 'Go_Mod2', 'Go_Mod1')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 0, multi=True,
                        dir_chooser = task_factory.choose_anti,
                        **factory_kwargs 
                        )
        self.task_type = 'Anti_Go_Mod1'

class AntiGoMod2(Task): 
    comp_ref_tasks = ('Anti_Go_Mod1', 'Go_Mod1', 'Go_Mod2')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.GoFactory, 
                        mod = 1, multi=True,
                        dir_chooser = task_factory.choose_anti, 
                        **factory_kwargs
                        )
        self.task_type = 'Anti_Go_Mod2'

class DM(Task):
    comp_ref_tasks = ('MultiDM', 'Anti_MultiDM', 'Anti_DM')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        **factory_kwargs
                        )
        self.task_type = 'DM'

class AntiDM(Task):
    comp_ref_tasks = ('Anti_MultiDM', 'MultiDM', 'DM')
    def __init__(self, num_trials, noise=None,**factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        **factory_kwargs
                        )
        self.task_type = 'Anti_DM'

class MultiDM(Task):
    comp_ref_tasks = ('DM', 'Anti_DM', 'Anti_MultiDM')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'MultiDM'

class AntiMultiDM(Task):
    comp_ref_tasks = ('Anti_DM', 'DM', 'MultiDM')
    def __init__(self, num_trials, noise=None,**factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'Anti_MultiDM'

class ConDM(Task):
    comp_ref_tasks = ('ConMultiDM', 'Anti_ConMultiDM', 'Anti_ConDM')
    def __init__(self, num_trials, noise=None, **factory_kwargs):
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmax,
                        threshold_folder = 'dm_noise_thresholds',
                        **factory_kwargs
                        )
        self.task_type = 'ConDM'

class ConAntiDM(Task):
    comp_ref_tasks = ('Anti_ConMultiDM', 'ConMultiDM', 'ConDM')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmin,
                        threshold_folder = 'anti_dm_noise_thresholds',
                        **factory_kwargs
                        )
        self.task_type = 'Anti_ConDM'

class ConMultiDM(Task):
    comp_ref_tasks = ('ConDM', 'Anti_ConDM', 'Anti_ConMultiDM')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmax,
                        multi=True,
                        threshold_folder = 'multi_dm_noise_thresholds',
                        **factory_kwargs
                        )
        self.task_type = 'ConMultiDM'

class ConAntiMultiDM(Task):
    comp_ref_tasks = ( 'Anti_ConDM', 'ConDM', 'ConMultiDM')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.ConDMFactory, 
                        str_chooser = np.argmin,
                        multi=True,
                        threshold_folder = 'anti_multi_dm_noise_thresholds',
                        **factory_kwargs
                        )
        self.task_type = 'Anti_ConMultiDM'

class DMMod1(Task):
    comp_ref_tasks = ('DM_Mod2', 'Anti_DM_Mod2', 'Anti_DM_Mod1')
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        mod=0, 
                        multi=True
                        )
        self.task_type = 'DM_Mod1'

class DMMod2(Task):
    comp_ref_tasks = ('DM_Mod1', 'Anti_DM_Mod1', 'Anti_DM_Mod2')
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmax,
                        mod=1, 
                        multi=True
                        )
        self.task_type = 'DM_Mod2'
        
class AntiDMMod1(Task):
    comp_ref_tasks = ('Anti_DM_Mod2', 'DM_Mod2', 'DM_Mod1')
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=0, 
                        multi=True
                        )
        self.task_type = 'Anti_DM_Mod1'

class AntiDMMod2(Task):
    comp_ref_tasks = ('Anti_DM_Mod1', 'DM_Mod1', 'DM_Mod2')
    def __init__(self, num_trials, noise=None): 
        super().__init__(num_trials, noise,
                        task_factory.DMFactory, 
                        str_chooser = np.argmin,
                        mod=1,
                        multi=True
                        )
        self.task_type = 'Anti_DM_Mod2'

class Dur1(Task): 
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 0,
                        dur_type = 'comp',
                        **factory_kwargs
                        )
        self.task_type = 'Dur1'


class Dur2(Task): 
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        resp_stim = 1,
                        dur_type = 'comp',
                        **factory_kwargs
                        )
        self.task_type = 'Dur2'


class DurLong(Task): 
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        dur_type = 'long',
                        **factory_kwargs
                        )
        self.task_type = 'DurLong'


class DurShort(Task): 
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.DurFactory, 
                        dur_type = 'short',
                        **factory_kwargs
                        )
        self.task_type = 'DurShort'

class COMP1(Task): 
    comp_ref_tasks = ('MultiCOMP1', 'MultiCOMP2', 'COMP2')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 0,
                        **factory_kwargs
                        )
        self.task_type = 'COMP1'

class COMP2(Task): 
    comp_ref_tasks = ('MultiCOMP2', 'MultiCOMP1', 'COMP1')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
                        resp_stim = 1,
                        **factory_kwargs
                        )
        self.task_type = 'COMP2'

class MultiCOMP1(Task): 
    comp_ref_tasks = ('COMP1', 'COMP2', 'MultiCOMP2')
    def __init__(self, num_trials, noise=None, **factory_kwargs): 
        super().__init__(num_trials, noise,
                        task_factory.COMPFactory, 
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
                        resp_stim = 1,
                        multi=True,
                        **factory_kwargs
                        )
        self.task_type = 'MultiCOMP2'

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


def construct_trials(task_type, num_trials=None, noise = None, return_tensor=False):
    assert task_type in TASK_LIST, "entered invalid task type"
    if task_type == 'Go':
        trial = Go
    if task_type == 'RT_Go':
        trial = RTGo
    if task_type == 'Anti_Go':
        trial = AntiGo
    if task_type == 'Anti_RT_Go':
        trial = AntiRTGo
    if task_type == 'Go_Mod1':
        trial = GoMod1
    if task_type == 'Go_Mod2':
        trial = GoMod2
    if task_type == 'Anti_Go_Mod1':
        trial = AntiGoMod1
    if task_type == 'Anti_Go_Mod2':
        trial = AntiGoMod2
    if task_type == 'DM':
        trial = DM
    if task_type == 'Anti_DM': 
        trial = AntiDM
    if task_type == 'ConDM':
        trial = ConDM
    if task_type == 'Anti_ConDM': 
        trial = ConAntiDM
    if task_type == 'ConMultiDM':
        trial = ConMultiDM
    if task_type == 'Anti_ConMultiDM': 
        trial = ConAntiMultiDM
    if task_type == 'MultiDM':
        trial = MultiDM
    if task_type == 'Anti_MultiDM': 
        trial = AntiMultiDM
    if task_type == 'DM_Mod1': 
        trial = DMMod1
    if task_type == 'DM_Mod2': 
        trial = DMMod2
    if task_type == 'Anti_DM_Mod1': 
        trial = AntiDMMod1
    if task_type == 'Anti_DM_Mod2': 
        trial = AntiDMMod2

    if task_type == 'Dur1': 
        trial = Dur1
    if task_factory == 'Dur2': 
        trial = Dur2

    if task_type == 'COMP1': 
        trial = COMP1
    if task_type == 'COMP2': 
        trial = COMP2
    if task_type == 'MultiCOMP1': 
        trial = MultiCOMP1
    if task_type == 'MultiCOMP2': 
        trial = MultiCOMP2
    if task_type == 'DMS': 
        trial = DMS
    if task_type == 'DNMS': 
        trial = DNMS
    if task_type == 'DMC': 
        trial = DMC
    if task_type == 'DNMC': 
        trial = DNMC
    if num_trials is None: 
        return trial
        
    if num_trials is None: 
        return trial 
    else: 
        trial = trial(num_trials, noise=noise)

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
