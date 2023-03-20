from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *
from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *
from instructRNN.analysis.decoder_analysis import get_novel_instruct_ratio, print_decoded_instruct
from instructRNN.data_loaders.perfDataFrame import *

to_plot_models = ['clipNet_lin', 'sbertNet_lin', 'gptNetXL_lin', 'gptNet_lin', 'bertNet_lin', 'bowNet_lin', 'simpleNetPlus', 'simpleNet']
tuned_to_plot = ['clipNet_lin_tuned', 'sbertNet_lin_tuned', 'gptNetXL_lin_tuned', 'gptNet_lin_tuned', 'bertNet_lin_tuned', 'bowNet_lin', 'simpleNet']

##ALL MODEL LEARNING CURVES
fig, axn = plot_curves('7.20models', 'multitask', to_plot_models, training_file='Multitask', linewidth=0.5)
fig.tight_layout()
plt.show()

##VALIDATION
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models[:-2], mode='val')
plt.show()

###HOLDOUTS
plot_curves('7.20models', 'swap', to_plot_models, mode='combined', avg=True, linewidth=0.8)
plot_k_shot_task_hist('7.20models', 'swap', to_plot_models[::-1], mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models, mode='combined')
plt.show()

##TUNED HOLDOUTS
plot_curves('7.20models', 'swap', tuned_to_plot, mode='combined', avg=True, linewidth=0.8)
plot_k_shot_task_hist('7.20models', 'swap', tuned_to_plot[::-1], mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', tuned_to_plot, mode='combined')
plt.show()

###SWAP HOLDOUTS
plot_curves('7.20models', 'swap', to_plot_models, mode='swap_combined', avg=True, linewidth=0.8)
plot_k_shot_task_hist('7.20models', 'swap', to_plot_models[::-1], mode='swap_combined')
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models, mode='swap_combined')
plt.show()

###FAMILY
plot_curves('7.20models', 'family', to_plot_models, mode='combined', avg=True, linewidth=0.8)
plot_k_shot_task_hist('7.20models', 'family', to_plot_models[::-1], mode='combined')
plot_all_task_lolli_v('7.20models', 'family', to_plot_models, mode='combined')
plt.show()

####PC PLOTS
EXP_FILE = '7.20models/swap_holdouts'


#CLIPNET
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'

clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')
plot_scatter(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='full')

clipNet.load_model('7.20models/multitask_holdouts/Multitask/'+clipNet.model_name, suffix='_seed3')
plot_scatter(clipNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, instruct_mode=None)

#GPTNET
gptNet = GPTNetXL_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'

gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed2')
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='full')

gptNet.load_model('7.20models/multitask_holdouts/Multitask/'+gptNet.model_name, suffix='_seed3')
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, instruct_mode=None)

#SIMPLENET
simpleNet = SimpleNet(rnn_hidden_dim=256)
holdouts_file = 'swap9'

simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')
plot_scatter(simpleNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)

simpleNet.load_model('7.20models/multitask_holdouts/Multitask/'+simpleNet.model_name, suffix='_seed1')
plot_scatter(simpleNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)

#CCGP PLOTS
plot_layer_ccgp('7.20models/swap_holdouts', to_plot_models)
plot_ccgp_corr('7.20models', 'swap', to_plot_models)

#############SINGLE UNIT TUING

EXP_FILE = '7.20models/swap_holdouts'
##TUNING
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

##ANTI GO
unit=39
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], unit, [145]*4, num_trials=80, smoothing=1, min_coh=0.01, max_coh=0.5)

##DM
unit=175
plot_neural_resp(clipNet, 'DMMod1', 'diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'AntiDMMod2','diff_strength', unit, num_trials=25)

###COMP
EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap6'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

unit=82
plot_neural_resp(clipNet, 'COMP2', 'diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'AntiCOMP2','diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'COMP1','diff_strength', unit, num_trials=25)
plot_neural_resp(clipNet, 'AntiCOMP1','diff_strength', unit, num_trials=25)


##matching
EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap1'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed2')

unit=147
plot_tuning_curve(clipNet, ['DMS', 'DNMS', 'DMC', 'DNMC'], unit, [149]*4, num_trials=50, smoothing=1.0)


###decoder figs
plot_partner_perf('clipNet_lin')

confuse_mat = np.load('7.20models/multitask_holdouts/decoder_perf/clipNet_lin/test_sm_multi_decoder_multi_confuse_mat.npy')
plot_decoding_confuse_mat(np.round(np.mean(confuse_mat, axis=0)/50, 2), linewidths=0.1, linecolor='#E5E4E2')

get_novel_instruct_ratio(sm_holdout=False, decoder_holdout=False)
get_novel_instruct_ratio(sm_holdout=True, decoder_holdout=False)
get_novel_instruct_ratio(sm_holdout=True, decoder_holdout=True)

###ADDITIONAL SUPP FIGS
###Unit Var
plot_task_var_heatmap('7.20models/swap_holdouts/swap9', 'clipNet_lin', 2)

###STRUCTURE FIG
plot_comp_bar('7.20models', 'swap', to_plot_models, ['ccgp', 'multi_ccgp', 'swap_ccgp'], y_lim=(0.5, 1.0))
plt.show()

plot_comp_bar('7.20models', 'swap', to_plot_models, ['combinedcomp', 'multi_comp'], y_lim=(0.0, 1.0))
plt.show()

###HOLDOUTS
plot_curves('7.20models', 'swap', to_plot_models, mode='combinedinputs_only', avg=True, linewidth=0.8)
plt.show()

multi_multi_instruct = pickle.load(open('7.20models/multitask_holdouts/decoder_perf/clipNet_lin/test_sm_multi_decoder_multi_instructs_dict', 'rb'))
holdout_multi_instruct = pickle.load(open('7.20models/swap_holdouts/decoder_perf/clipNet_lin/test_sm_holdout_decoder_multi_instructs_dict', 'rb'))
holdout_holdout_instruct = pickle.load(open('7.20models/swap_holdouts/decoder_perf/clipNet_lin/test_sm_holdout_decoder_holdout_instructs_dict', 'rb'))

print_decoded_instruct(multi_multi_instruct)
print_decoded_instruct(holdout_multi_instruct)
print_decoded_instruct(holdout_holdout_instruct)

###GPT COMPARISON
fig_axn = plot_curves('7.20models', 'swap', ['gptNetXL_lin'], mode='combined', avg=True, linewidth=0.8)
plot_curves('7.20models', 'swap', ['gptNetXL_L_lin'], mode='combined', fig_axn=fig_axn, avg=True, linewidth=0.8, linestyle='--')
plt.show()

plot_layer_ccgp('7.20models/swap_holdouts', ['gptNetXL_L_lin', 'gptNetXL_lin', 'clipNet_lin'], seeds=[0])
