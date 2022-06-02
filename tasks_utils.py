from tasks import TASK_LIST

def invert_task_dict(task_dict):
    inv_swap_dict = {}
    for k, v in task_dict.items():
        for task in v:
            inv_swap_dict[task] = k
    return inv_swap_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

SWAP_LIST = [            
            ('DM_Mod1', 'Anti_Go_Mod2', 'ConMultiDM', 'DMS'),
            ('Anti_DM_Mod2', 'Go_Mod1', 'Anti_Go', 'COMP1'), 
            ('Anti_Go_Mod1', 'DM_Mod2', 'Go', 'MultiCOMP1'), 
            ('Go_Mod2', 'Anti_DM_Mod1', 'Anti_DelayGo', 'COMP2'), 
            ('Anti_DelayMultiDM', 'MultiDM', 'Anti_RT_DM', 'DMC'),             
            ('DM', 'Anti_ConDM', 'RT_Go', 'MultiCOMP2'), 
            ('Anti_DM', 'DelayMultiDM', 'Anti_RT_Go',  'DNMS'), 
            ('RT_DM', 'Anti_MultiDM', 'DelayDM', 'DNMC'),
            ('ConDM', 'Anti_DelayDM', 'DelayGo', 'Anti_ConMultiDM')
            ]

task_list=TASK_LIST.copy()

for swap in SWAP_LIST: 
    for task in swap: 
        task_list.pop(task_list.index(task))
task_list
# SWAP_LIST = [
#             ('Anti_DM_Mod2', 'Go_Mod1', 'Anti_Go', 'COMP1', 'ConMultiDM'), 
#             ('Anti_Go_Mod1', 'DM_Mod2', 'Go', 'MultiCOMP1', 'ConDM'), 
#             ('DM_Mod1', 'Anti_Go_Mod2', 'DelayGo', 'DMS', 'Anti_ConDM'),
#             ('Go_Mod2', 'Anti_DM_Mod1', 'Anti_DelayGo', 'COMP2', 'Anti_ConMultiDM'), 
#             ('Anti_DelayDM', 'MultiDM', 'Anti_RT_DM', 'DMC', 'COMP1_Mod1'),             
#             ('DM', 'Anti_DelayMultiDM', 'RT_Go', 'MultiCOMP2', 'COMP1_Mod2'), 
#             ('Anti_DM', 'DelayMultiDM', 'Anti_RT_Go',  'DNMS', 'COMP2_Mod2'), 
#             ('RT_DM', 'Anti_MultiDM', 'DelayDM', 'DNMC', 'COMP2_Mod1')
#             ]

task_list = TASK_LIST.copy()


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


