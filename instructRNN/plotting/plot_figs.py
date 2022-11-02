from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *
from instructRNN.analysis.decoder_analysis import get_novel_instruct_ratio




data = HoldoutDataFrame('7.20models', 'swap', 'gptNet_lin', mode = 'combined')


to_plot_models = ['simpleNet',  'bowNet_lin',  'bertNet_lin', 'gptNetXL_lin', 'gptNet_lin', 'sbertNet_lin', 'clipNet_lin']
tuned_to_plot = ['gptNetXL_lin_tuned', 'sbertNet_lin_tuned', 'clipNet_lin_tuned', 'bertNet_lin_tuned', 'bowNet_lin', 'simpleNet', 'gptNet_lin_tuned']


plot_all_training_curves('7.20models', 'multitask', 'Multitask', to_plot_models)

##VALIDATION
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models[1:][::-1], seeds =range(1), mode='validation')
plot_0_shot_task_hist('7.20models', 'swap', to_plot_models[1:], seeds =range(0,5), mode='validation')

###HOLDOUTS
plot_avg_holdout_curve('7.20models', 'swap', to_plot_models, seeds =range(0, 5), mode='combinedcomp')
plot_0_shot_task_hist('7.20models', 'swap', to_plot_models, seeds =range(0,5), mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models[::-1], seeds =range(0, 5), mode='combined')

##TUNED HOLDOUTS
plot_avg_holdout_curve('7.20models', 'swap', tuned_to_plot, seeds =range(0, 5), mode='combined')
plot_0_shot_task_hist('7.20models', 'swap', tuned_to_plot, seeds =range(0, 5), mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', tuned_to_plot[::-1], seeds =range(0, 5), mode='combined')

###SWAP HOLDOUTS
plot_avg_holdout_curve('7.20models', 'swap', to_plot_models, seeds =range(0, 5), mode='swap_combined')
plot_0_shot_task_hist('7.20models', 'swap', to_plot_models, seeds =range(0,5), mode='swap_combined')
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models[::-1], seeds =range(0, 5), mode='swap_combined')

###FAMILY
plot_avg_holdout_curve('7.20models', 'family', ['clipNet_lin'], seeds =range(5, 10), mode='combined')

plot_avg_holdout_curve('7.20models', 'family', to_plot_models, seeds =range(0, 5), mode='combined')
plot_0_shot_task_hist('7.20models', 'family', to_plot_models, seeds =range(0,5), mode='combined')
plot_all_task_lolli_v('7.20models', 'family', to_plot_models[::-1], seeds =range(0, 5), mode='combined')

###nonlinguitic variants
plot_simpleNet_comps('7.20models', 'swap', ['simpleNet', 'simpleNetPlus'])
plot_0_shot_task_hist('7.20models', 'swap', ['simpleNet', 'simpleNetPlus', 'simpleNet_comp', 'simpleNetPlus_comp'][::-1], seeds =range(0,5))
plot_all_task_lolli_v('7.20models', 'swap', ['simpleNet', 'simpleNetPlus', 'simpleNet_comp', 'simpleNetPlus_comp'][::-1], seeds =range(0,5))

'simpleNet_comp'.split('_')[0]

####PC PLOTS
EXP_FILE = '7.20models/swap_holdouts'

clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

plot_scatter(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

clipNet.load_model('7.20models/multitask_holdouts/Multitask/'+clipNet.model_name, suffix='_seed3')
plot_scatter(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, instruct_mode=None)
plot_scatter(clipNet, TASK_LIST, dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')


bertNet = BERTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
bertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+bertNet.model_name, suffix='_seed0')

plot_scatter(bertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(bertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

bertNet.load_model('7.20models/multitask_holdouts/Multitask/'+bertNet.model_name, suffix='_seed1')
plot_scatter(bertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, instruct_mode=None)
plot_scatter(bertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')


gptNet = GPTNetXL_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed2')

plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

gptNet.load_model('7.20models/multitask_holdouts/Multitask/'+gptNet.model_name, suffix='_seed3')
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, instruct_mode=None)
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')


simpleNet = SimpleNet(rnn_hidden_dim=256)
holdouts_file = 'swap9'
simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')

plot_scatter(simpleNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)

simpleNet.load_model('7.20models/multitask_holdouts/Multitask/'+simpleNet.model_name, suffix='_seed1')
plot_scatter(simpleNet, TASK_LIST, dims=3, pcs=[0, 1, 2], num_trials=50)

plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'sbertNet_lin', 'bertNet_lin',  'gptNetXL_lin', 'gptNet_lin',  'bowNet_lin', 'simpleNet', 'simpleNetPlus'][::-1], seeds=range(5), plot_multis=True)
plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'sbertNet_lin', 'bertNet_lin',  'gptNetXL_lin', 'gptNet_lin',  'bowNet_lin'][::-1], seeds=range(5), mode='swap_combined')

plot_ccgp_corr('7.20models', 'swap', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'sbertNet_lin', 'gptNet_lin', 'gptNetXL_lin', 'simpleNet'])

#############



plot_task_var_heatmap('7.20models/swap_holdouts/swap9', 'clipNet_lin', 2, cluster_info=cluster_info)

####
EXP_FILE = '7.20models/swap_holdouts'

clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

SWAP_LIST[-1]
task_var, cluters_dict, cluster_labels, sorted_indices = plot_task_var_heatmap('7.20models/swap_holdouts/swap9', 'clipNet_lin', 2)


for key, value in cluters_dict.items():
    if np.mean(task_var[value, TASK_LIST.index('AntiDMMod1')]) > 0.4:
        print(key)

cluters_dict

cluters_dict[4][np.argmax(task_var[cluters_dict[4], TASK_LIST.index('AntiDMMod1')])]


unit=96
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1, min_contrast=0)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1, min_contrast=0)




unit=41
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1, min_contrast=0)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1, min_contrast=0)
plot_tuning_curve(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], unit, [140]*4)




unit=51
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1, min_contrast=0)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1, min_contrast=0)
plot_tuning_curve(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], unit, [129]*4)




























clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

###Go Only Neuron 
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 2, [140]*6)

###Direction Tuning FLip
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 73, [140]*6)

###sort of
plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 11, [140]*6)
plot_neural_resp(clipNet, 'AntiGo','direction', 73, num_trials=25)


clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')
unit=9
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1)


unit=18
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1)



EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap5'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')



unit=4
plot_neural_resp(clipNet, 'DM','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDM','diff_strength', unit, num_trials=25, smoothing=1)

plot_tuning_curve(clipNet, ['DM', 'AntiDM'], 4, [120]*6, smoothing=1)



###decoder figs
plot_partner_perf()
confuse_mat = np.load('7.20models/multitask_holdouts/decoder_perf/clipNet_lin/sm_multidecoder_multi_confuse_mat.npy')
plot_decoding_confuse_mat(np.round(np.mean(confuse_mat, axis=0)/50, 2), fmt='.0%', annot_kws={'size':3}, linewidths=0.2)

get_novel_instruct_ratio(sm_holdout=True, decoder_holdout=False)
get_novel_instruct_ratio(sm_holdout=True, decoder_holdout=True)






