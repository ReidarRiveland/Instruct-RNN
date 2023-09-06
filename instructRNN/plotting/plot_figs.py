from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *
from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *
from instructRNN.analysis.decoder_analysis import get_novel_instruct_ratio, print_decoded_instruct
from instructRNN.data_loaders.perfDataFrame import *
from instructRNN.analysis.model_analysis import calc_t_test

to_plot_models = ['combNet',  'sbertNetL_lin','sbertNet_lin',  'clipNet_lin', 'bertNet_lin', 'gptNetXL_lin', 'gptNet_lin',  'bowNet_lin', 'simpleNet']
non_lin_models = ['sbertNetL', 'sbertNet', 'clipNet', 'gptNetXL', 'gptNet', 'bertNet', 'bowNet']
tuned_to_plot = ['combNet', 'clipNet_lin_tuned', 'sbertNetL_lin_tuned', 'sbertNet_lin_tuned', 'gptNetXL_lin_tuned', 'gptNet_lin_tuned', 'bertNet_lin_tuned', 'bowNet_lin', 'simpleNet']
aux_models = ['combNet', 'combNetPlus', 'clipNetS_lin', 'bowNet_lin_plus', 'rawBertNet_lin', 'simpleNetPlus', 'simpleNet']

##ALL MODEL LEARNING CURVES
fig_axn = plot_curves('7.20models', 'multitask', to_plot_models, training_file='Multitask', linewidth=0.5)
fig_axn[0].tight_layout()
plt.show()

##VALIDATION
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models[:-2], mode='val')
plt.show()

###HOLDOUTS
plot_curves('7.20models', 'swap', to_plot_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('7.20models', 'swap', to_plot_models, mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='combined')
plot_significance(t_mat, p_mat, to_plot_models)
p_mat
plt.show()

##non-linear holdouts
plot_curves('7.20models', 'swap', non_lin_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('7.20models', 'swap', non_lin_models, mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', non_lin_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('7.20models', 'swap', non_lin_models, mode='combined')
plot_significance(is_sig, non_lin_models)
plt.show()

##langPlus 
plot_curves('7.20models', 'swap', aux_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('7.20models', 'swap', aux_models, mode='combined', hatch = '///', edgecolor='white')
plot_all_task_lolli_v('7.20models', 'swap', aux_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('7.20models', 'swap', aux_models, mode='combined')
plot_significance(t_mat, aux_models)
plt.show()

##TUNED HOLDOUTS
plot_curves('7.20models', 'swap', tuned_to_plot, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('7.20models', 'swap', tuned_to_plot, mode='combined')
plot_all_task_lolli_v('7.20models', 'swap', tuned_to_plot, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('7.20models', 'swap', tuned_to_plot, mode='combined')
plot_significance(t_mat, is_sig, tuned_to_plot)
plt.show()

###SWAP HOLDOUTS
plot_curves('7.20models', 'swap', to_plot_models, mode='swap_combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('7.20models', 'swap', to_plot_models, mode='swap_combined')
plot_all_task_lolli_v('7.20models', 'swap', to_plot_models, mode='swap_combined')
plt.show()

_, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='swap_combined')
plot_significance(is_sig, to_plot_models)
plt.show()

###FAMILY
plot_curves('7.20models', 'family', to_plot_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('7.20models', 'family', to_plot_models, mode='combined')
plot_all_task_lolli_v('7.20models', 'family', to_plot_models, mode='combined')
plt.show()

_, is_sig = calc_t_test('7.20models', 'family', to_plot_models, mode='combined')
plot_significance(is_sig, to_plot_models)
plt.show()

####PC PLOTS
EXP_FILE = '7.20models/swap_holdouts'

#SBERTNET
sbertNet = make_default_model('sbertNetL_lin')
holdouts_file = 'swap9'

sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed4')
plot_scatter(sbertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(sbertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='full')

sbertNet.load_model('7.20models/multitask_holdouts/Multitask/'+sbertNet.model_name, suffix='_seed0')
plot_scatter(sbertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, instruct_mode=None)

#GPTNET
gptNet = make_default_model('gptNetXL_lin')
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

#COMBNET
combNet = make_default_model('combNet')
holdouts_file = 'swap9'

combNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+combNet.model_name, suffix='_seed2')
plot_scatter(combNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)

#CCGP PLOTS
plot_layer_ccgp('7.20models', 'swap', to_plot_models)
plt.show()
plot_ccgp_corr('7.20models', 'swap', to_plot_models)

#############SINGLE UNIT TUING
EXP_FILE = '7.20models/swap_holdouts'
sbertNet = make_default_model('sbertNetL_lin')

##ANTI GO
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed2')
unit=42
plot_tuning_curve(sbertNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], unit, [149]*4, num_trials=80, smoothing=1, min_coh=0.01, max_coh=0.5)

##DM
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

#unit = 216, seed 3
#unit=3, seed 0
unit = 3
plot_neural_resp(sbertNet, 'DMMod1', 'diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'AntiDMMod1','diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'DMMod2','diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'AntiDMMod2','diff_strength', unit, num_trials=25)

###COMP REDO!
holdouts_file = 'swap6'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed1')

#unit = 201, seed 1
unit = 220
plot_neural_resp(sbertNet, 'COMP1','diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'COMP2', 'diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'AntiCOMP1','diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'AntiCOMP2','diff_strength', unit, num_trials=25)

##matching
holdouts_file = 'swap1'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed1')
unit=14
plot_tuning_curve(sbertNet, ['DMS', 'DNMS', 'DMC', 'DNMC'], unit, [149]*4, num_trials=50, smoothing=1.0)


###decoder figs
plot_partner_perf('clipNet_lin')

confuse_mat = np.load('7.20models/multitask_holdouts/decoder_perf/clipNet_lin/test_sm_multi_decoder_multi_confuse_mat.npy')
plot_decoding_confuse_mat(np.round(np.mean(confuse_mat, axis=0)/50, 2), linewidths=0.1, linecolor='#E5E4E2')

get_novel_instruct_ratio(sm_holdout=False, decoder_holdout=False)
get_novel_instruct_ratio(sm_holdout=True, decoder_holdout=False)
get_novel_instruct_ratio(sm_holdout=True, decoder_holdout=True)

###ADDITIONAL SUPP FIGS
###Unit Var
var = plot_task_var_heatmap('7.20models/swap_holdouts/swap6', 'sbertNetL_lin', 1)


###STRUCTURE FIG

_, _, stats_arr = plot_clauses_dots('7.20models', 'swap', to_plot_models, y_lim=(-0.3, 0.6))
plt.show()

plot_significance(stats_arr[0], stats_arr[1], to_plot_models)
plt.show()

plot_comp_dots('7.20models', 'swap', to_plot_models, ['combined'], y_lim=(0.0, 1.0))
plt.show()

plot_comp_dots('7.20models', 'swap', to_plot_models, ['ccgp', 'multi_ccgp', 'swap_ccgp'], y_lim=(0.5, 1.0))
plt.show()

t_mat, p_mat, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='ccgp')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()

p_mat, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='multi_ccgp')
plot_significance(is_sig, to_plot_models)
plt.show()

p_mat, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='swap_ccgp')
plot_significance(is_sig, to_plot_models)
plt.show()

plot_comp_dots('7.20models', 'swap', to_plot_models, ['combinedcomp', 'multi_comp'], y_lim=(0.0, 1.0))
plt.show()

p_mat, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='combinedcomp')
plot_significance(is_sig, to_plot_models)
plt.show()

p_mat, is_sig = calc_t_test('7.20models', 'swap', to_plot_models, mode='multi_comp')
plot_significance(is_sig, to_plot_models)
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


plot_all_comp_holdout_lolli_v('7.20models', 'swap', to_plot_models)