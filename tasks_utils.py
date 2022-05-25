from tasks import TASK_LIST
from task_factory import TUNING_DIRS
task_list = TASK_LIST
tuning_dirs = TUNING_DIRS

def invert_task_dict(task_dict):
    inv_swap_dict = {}
    for k, v in task_dict.items():
        for task in v:
            inv_swap_dict[task] = k
    return inv_swap_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

# SWAP_LIST = [('Anti_ConDM', 'DM_Mod2', 'Anti_RT_Go', 'DelayGo', 'MultiCOMP2'), 
#             ('DelayDM', 'Anti_ConMultiDM', 'RT_DM_Mod2', 'Anti_RT_DM', 'COMP2'), 
#             ('Anti_DM', 'MultiDM', 'Anti_RT_DM_Mod2', 'Go', 'DNMC'), 
#             ('DM', 'Anti_MultiDM', 'RT_DM_Mod1', 'Anti_Go', 'COMP1'), 
#             ('Anti_DelayGo', 'ConDM', 'Anti_DM_Mod1', 'DelayMultiDM', 'DMC'), 
#             ('RT_Go', 'Anti_RT_DM_Mod1', 'ConMultiDM', 'Anti_Go_Mod2', 'MultiCOMP1'),
#             ('Anti_DelayMultiDM', 'RT_DM', 'Anti_Go_Mod1', 'DM_Mod1', 'DNMS'), 
#             ('Go_Mod2', 'Anti_DelayDM', 'Go_Mod1', 'Anti_DM_Mod2', 'DMS')
#             ]

# SWAP_LIST = [('Anti_ConDM', 'DM_Mod2', 'Anti_RT_Go', 'DelayGo', 'MultiCOMP2'), 
#             ('DelayDM', 'Anti_ConMultiDM', 'RT_DM_Mod2', 'Anti_RT_DM', 'COMP2'), 
#             ('Anti_DM', 'MultiDM', 'Anti_RT_DM_Mod2', 'Go', 'DNMC'), 
#             ('DM', 'Anti_MultiDM', 'RT_DM_Mod1', 'Anti_Go', 'COMP1'), 

#             ('Anti_DelayGo', 'Go_Mod2', 'Anti_DM_Mod1', 'DelayMultiDM', 'DMC'), 

#             ('RT_Go', 'Anti_RT_DM_Mod1', 'ConMultiDM', 'Anti_Go_Mod2', 'MultiCOMP1'),
#             ('Anti_DelayMultiDM', 'RT_DM', 'Anti_Go_Mod1', 'DM_Mod1', 'DNMS'), 
#             ('ConDM', 'Anti_DelayDM', 'Go_Mod1', 'Anti_DM_Mod2', 'DMS')
#             ]

SWAP_LIST = [('Anti_ConDM', 'DM_Mod2', 'Anti_RT_Go', 'DelayGo', 'MultiCOMP2'), 
            ('DelayDM', 'Anti_ConMultiDM', 'COMP1_Mod2', 'Anti_RT_DM', 'DMC'), 
            ('Anti_DM', 'MultiDM', 'COMP1_Mod1', 'Go', 'DNMS'), 
            ('DM', 'Anti_MultiDM', 'COMP2_Mod1', 'Anti_Go', 'DNMC'), 

            ('Anti_DelayGo', 'Go_Mod2', 'Anti_DM_Mod1', 'DelayMultiDM', 'COMP2'), 

            ('RT_Go', 'COMP2_Mod2', 'ConMultiDM', 'Anti_Go_Mod2', 'MultiCOMP1'),
            ('Anti_DelayMultiDM', 'RT_DM', 'Anti_Go_Mod1', 'DM_Mod1', 'COMP1'), 
            ('ConDM', 'Anti_DelayDM', 'Go_Mod1', 'Anti_DM_Mod2', 'DMS')
            ]


SWAPS_DICT = dict(zip(['swap'+str(num) for num in range(len(SWAP_LIST))], SWAP_LIST.copy()))

INV_SWAPS_DICT = invert_task_dict(SWAPS_DICT)

ALIGNED_LIST=['Anti_ConDM', 'ConDM']

def get_swap_task(task):
    swap_label = INV_SWAPS_DICT[task]
    pos = SWAPS_DICT[swap_label].index(task)
    swap_index = (pos+1)%len(SWAPS_DICT[swap_label])
    return SWAPS_DICT[swap_label][swap_index]

task_colors = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange',
                        'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'goldenrod', 
                        'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold',
                        'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

MODEL_STYLE_DICT = {'simpleNet': ('blue', None), 'simpleNetPlus': ('blue', '+'), 'bowNet': ('orange', None), 'gptNet': ('red', None), 'gptNet_tuned': ('red', 'v'), 'bertNet_tuned': ('green', 'v'),
                    'bertNet': ('green', None), 'sbertNet': ('purple', None), 'sbertNet_tuned': ('purple', 'v')}
