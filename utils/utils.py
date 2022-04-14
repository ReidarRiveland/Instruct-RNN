from task import Task
task_list = Task.TASK_LIST
swapped_task_list = Task.SWAPPED_TASK_LIST
tuning_dirs = Task.TUNING_DIRS

training_lists_dict={
'single_holdouts' :  [[item] for item in Task.TASK_LIST.copy()+['Multitask']],
'dual_holdouts' : [['RT Go', 'Anti Go'], ['Anti MultiDM', 'DM'], ['COMP1', 'MultiCOMP2'], ['DMC', 'DNMS']],
'aligned_holdouts' : [['Anti DM', 'Anti MultiDM'], ['COMP1', 'MultiCOMP1'], ['DMS', 'DNMS'],['Go', 'RT Go']],
'swap_holdouts' : [['Go', 'Anti DM'], ['Anti RT Go', 'DMC'], ['RT Go', 'DNMC'], ['DM', 'MultiCOMP2'], ['MultiDM', 'DNMS'], ['Anti MultiDM', 'COMP1'], ['COMP2', 'DMS'], ['Anti Go', 'MultiCOMP1']]
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

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'bow20Net': ('yellow', None), 'bowNet': ('orange', None), 'gptNet': ('red', None), 'gptNet_tuned': ('red', 'v'), 'bertNet_tuned': ('green', 'v'),
                    'bertNet': ('green', None), 'bertNet_layer_11': ('green', '.'), 'sbertNet': ('purple', None), 'sbertNet_tuned': ('purple', 'v'), 'sbertNet_layer_11': ('purple', '.')}

def get_holdout_file_name(holdouts): 
    if len(holdouts) > 1: holdout_file = '_'.join(holdouts)
    else: holdout_file = holdouts[0]
    holdout_file = holdout_file.replace(' ', '_')
    return holdout_file

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
