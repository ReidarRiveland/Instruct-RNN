from tasks import TASK_LIST
from task_factory import TUNING_DIRS
task_list = TASK_LIST
tuning_dirs = TUNING_DIRS

training_lists_dict={
'dual_holdouts' : [['RT Go', 'Anti Go'], ['Go',  'Anti RT Go'], 
                    ['AntiDM', 'MultiDM'], ['Anti MultiDM', 'DM'], 
                    ['COMP1', 'MultiCOMP2'], ['COMP2', 'MultiCOMP1'], 
                    ['DMS', 'DNMC'], ['DMC', 'DNMS']],

'aligned_holdouts' : [['RT Go', 'Anti RT Go'],['Go', 'Anti Go'], 
                        ['Anti DM', 'DM'], ['Anti MultiDM', 'MultiDM'], 
                        ['COMP1', 'COMP2'], ['MultiCOMP1', 'MultiCOMP2'],
                        ['DMS', 'DNMS'], ['DMC', 'DNMC']],

'alt_aligned_holdouts' : [['Go', 'RT Go'],['Anti Go', 'Anti RT Go'], 
                        ['Anti DM', 'Anti MultiDM'], ['DM', 'MultiDM'], 
                        ['COMP1', 'MultiCOMP1'], ['COMP2', 'MultiCOMP2'],
                        ['DMS', 'DMC'], ['DNMS', 'DNMC']],

'swap_holdouts' : [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], 
                    ['RT Go', 'DNMC'], ['DM', 'MultiCOMP2'], 
                    ['MultiDM', 'DNMS'], ['Anti MultiDM', 'COMP1'], 
                    ['COMP2', 'DMS'], ['Anti Go', 'MultiCOMP1']],

}


task_swaps_map = {'Go': 'Go_Anti_DM', 
                'Anti Go': 'Anti_Go_MultiCOMP1', 
                'RT Go': 'RT_Go_DNMC', 
                'Anti RT Go': 'Anti_RT_Go_DMC',
                'DM': 'DM_MultiCOMP2',
                'Anti DM': 'Go_Anti_DM',
                'MultiDM': 'MultiDM_DNMS',
                'Anti MultiDM': 'Anti_MultiDM_COMP1',
                'COMP1': 'Anti_MultiDM_COMP1', 
                'COMP2': 'COMP2_DMS',
                'MultiCOMP1': 'Anti_Go_MultiCOMP1',
                'MultiCOMP2': 'DM_MultiCOMP2',
                'DMS': 'COMP2_DMS', 
                'DNMS': 'MultiDM_DNMS', 
                'DMC': 'Anti_RT_Go_DMC', 
                'DNMC': 'RT_Go_DNMC',
                'Multitask':'Multitask'}

all_swaps = list(set(task_swaps_map.values()))

task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'goldenrod', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'simpleNetPlus': ('blue', '+'), 'bowNet': ('orange', None), 'gptNet': ('red', None), 'gptNet_tuned': ('red', 'v'), 'bertNet_tuned': ('green', 'v'),
                    'bertNet': ('green', None), 'sbertNet': ('purple', None), 'sbertNet_tuned': ('purple', 'v')}

def get_holdout_file_name(holdouts): 
    if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
    else: holdout_file = holdouts[0]
    holdout_file = holdout_file.replace(' ', '_')
    return holdout_file

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
