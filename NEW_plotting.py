task_group_colors = defaultdict(dict)
task_group_colors['Go'] = { 'Go':'tomato', 'RT Go':'limegreen', 'Anti Go':'cyan', 'Anti RT Go':'orange'}
task_group_colors['Decision Making'] = { 'DM':'Red', 'Anti DM':'Green', 'MultiDM':'Blue', 'Anti MultiDM':'Yellow'}
task_group_colors['Comparison'] = { 'COMP1':'sienna', 'COMP2':'seagreen', 'MultiCOMP1':'skyblue', 'MultiCOMP2':'gold'}
task_group_colors['Delay'] = { 'DMS':'firebrick', 'DNMS':'lightgreen', 'DMC':'dodgerblue', 'DNMC':'darkorange'}

ALL_STYLE_DICT = {'Model1': ('blue', None), 'Model1shuffled': ('blue', '+'), 'SIF':('brown', None), 'BoW': ('orange', None), 'GPT_cat': ('red', '^'), 'GPT train': ('red', '.'), 
                        'BERT_cat': ('green', '^'), 'BERT train': ('green', '.'), 'S-Bert_cat': ('purple', '^'), 'S-Bert train': ('purple', '.'), 'S-Bert' : ('purple', None), 
                        'InferSent train': ('yellow', '.'), 'InferSent_cat': ('yellow', '^'), 'Transformer': ('pink', '.')}
COLOR_DICT = {'Model1': 'blue', 'SIF':'brown', 'BoW': 'orange', 'GPT': 'red', 'BERT': 'green', 'S-Bert': 'purple', 'InferSent':'yellow', 'Transformer': 'pink'}
MODEL_MARKER_DICT = {'SIF':None, 'BoW':None, 'shuffled':'+', 'cat': '^', 'train': '.', 'Transformer':'.'}
MARKER_DICT = {'^': 'task categorization', '.': 'end-to-end', '+':'shuffled'}
NAME_TO_PLOT_DICT = {'Model1': 'One-Hot Task Vec.','Model1shuffled': 'Shuffled One-Hot', 'SIF':'SIF', 'BoW': 'BoW', 'GPT_cat': 'GPT (task cat.)', 'GPT train': 'GPT (end-to-end)', 
                        'BERT_cat': 'BERT (task cat.)', 'BERT train': 'BERT (end-to-end)', 'S-Bert_cat': 'S-BERT (task cat.)', 'S-Bert train': 'S-BERT (end-to-end)',  
                        'S-Bert': 'S-BERT (raw)', 'InferSent train': 'InferSent (end-to-end)', 'InferSent_cat': 'InferSent (task cat.)', 'Transformer': 'Transformer (end-to-end)'}


def task_cmap(array): 
    all_task_dict = {}
    for task_colors in task_group_colors.values(): 
        all_task_dict.update(task_colors)
    color_list = []

    for index in array: 
        color_list.append(all_task_dict[task_list[index]])

    return color_list