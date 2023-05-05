from instructRNN.models.full_models import * 
from instructRNN.analysis.model_analysis import * 
import numpy.linalg as LA
from instructRNN.plotting.plotting import *
import matplotlib.pyplot as plt

# ####PC PLOTS
EXP_FILE = '7.20models/multitask_holdouts/Multitask/'

def get_rnn_svd(model): 
    r_hid, z_hid, c_hid = torch.split(model.recurrent_units.state_dict()['layers.0.cell.weight_hh'], 256)
    return LA.svd(c_hid)

def plot_svd_dist(model_list, seed=0, **plt_kwargs): 
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))

    axn.set_xticks(range(0, 20))
    axn.set_xticklabels([str(label) for label in range(1, 21)])

    for model_name in model_list:
        model = make_default_model(model_name)
        model.load_model(EXP_FILE+model.model_name, suffix='_seed'+str(seed))
        u, s, vh = get_rnn_svd(model)
        axn.plot(s[:20], color=MODEL_STYLE_DICT[model_name][0], linewidth=0.8, **plt_kwargs)

    axn.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list])
    plt.show()


def plot_svd_overlap(model_list, seed=0, **plt_kwargs): 
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))

    axn.set_xticks(range(0, 20))
    axn.set_xticklabels([str(label) for label in range(1, 21)])

    for model_name in model_list:
        model = make_default_model(model_name)
        model.load_model(EXP_FILE+model.model_name, suffix='_seed'+str(seed))
        u, s, vh = get_rnn_svd(model)
        overlaps = np.diag(np.matmul(u.T, vh))

        axn.scatter(s[:20], overlaps[:20], color=MODEL_STYLE_DICT[model_name][0], linewidth=0.8, **plt_kwargs)

    axn.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list])
    plt.show()

def plot_pc_var_exp_dist(model_list, holdout_file, seed=0, **plt_kwargs): 
    fig, axn = plt.subplots(1, 1, sharey = True, sharex=True, figsize =(6, 4))

    axn.set_xticks(range(0, 20))
    axn.set_xticklabels([str(label) for label in range(1, 21)])

    for model_name in model_list:
        model = make_default_model(model_name)
        model.load_model('7.20models/swap_holdouts/'+holdout_file+'/'+model.model_name, suffix='_seed'+str(seed))
        if 'XL' in model_name: depth = 24
        else: depth = 12
        reps  = get_instruct_reps(model.langModel, depth='full')
        _, var_explained = reduce_rep(reps, pcs = range(20))
        axn.plot(var_explained[:20], color=MODEL_STYLE_DICT[model_name][0], linewidth=0.8, **plt_kwargs)

    axn.legend(labels=[MODEL_STYLE_DICT[model_name][2] for model_name in model_list])
    plt.show()


plot_pc_var_exp_dist(['clipNet_lin', 'gptNetXL_lin', 'sbertNet_lin', 'bertNet_lin', 'gptNet_lin'], 'swap0')

plot_svd_overlap(['simpleNet', 'clipNet_lin', 'simpleNetPlus'])

plot_svd_dist(['simpleNet', 'clipNet_lin', 'simpleNetPlus'])