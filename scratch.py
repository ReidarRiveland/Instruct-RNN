import torch
import numpy as np
from instructRNN.models.full_models import SimpleNetPlus
from instructRNN.tasks.tasks import SWAP_LIST, TASK_LIST, SWAPS_DICT, construct_trials
from instructRNN.plotting.plotting import plot_scatter
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.instructions.instruct_utils import one_hot_input_rule
from instructRNN.analysis.model_analysis import get_rule_embedder_reps
from instructRNN.trainers.mem_net_trainer import MemNet
import instructRNN.models.full_models as full_models

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


TASK_LIST
SWAP_LIST[3]

# EXP_FILE = '3.16models/swap_holdouts'
# simpleNet = SimpleNetPlus(rule_encoder_hidden=128)
# holdouts_file = 'swap3'
# simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')
# memNet = MemNet(64, 256)
# memNet.load_state_dict(torch.load('3.16models/swap_holdouts/swap0/simpleNetPlus/mem_net/simpleNetPlus_seed0_memNet.pt'))


def eval_mem_net_holdouts(model_name, folder_name, batch_size=100): 
    if 'swap' in folder_name: 
        exp_dict = SWAPS_DICT
    perf_array = np.full((5, len(TASK_LIST)), np.NaN)
    model = full_models.make_default_model(model_name)
    with torch.no_grad():
        for seed in range(5):
            for holdout_label, tasks in exp_dict.items(): 
                model.load_model(folder_name+'/'+holdout_label+'/'+model.model_name, suffix='_seed'+str(seed))
                memNet = MemNet(64, 256)
                memNet.load_state_dict(torch.load(folder_name+'/'+holdout_label+'/'+model.model_name+'/mem_net/'+model.model_name+'_seed'+str(seed)+'_memNet.pt'))
                for task in tasks: 
                    print(task)
                    ins, targets, _, target_dirs, _ = construct_trials(task, batch_size)
                    _, hid = model(torch.Tensor(ins), torch.Tensor(one_hot_input_rule(100, task)))
                    inferred_embed, _ = memNet(torch.tensor(ins), torch.tensor(targets), hid)
                    out, hid = model(torch.Tensor(ins), info_embedded=inferred_embed[:, -1, :])
                    perf = np.mean(isCorrect(out, torch.Tensor(targets), target_dirs))
                    perf_array[seed, TASK_LIST.index(task)] = perf

    return perf_array


perf_arr = eval_mem_net_holdouts('simpleNetPlus', '3.16models/swap_holdouts')

np.save(open('simpleNetPlus_comp_holdout_perf.npy', 'wb'), perf_arr)