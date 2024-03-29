from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *
from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *
from instructRNN.analysis.decoder_analysis import get_novel_instruct_ratio, print_decoded_instruct, get_decoded_vec_cos_sim, _get_partner_perf_labels
from instructRNN.data_loaders.perfDataFrame import *
from instructRNN.analysis.model_analysis import calc_t_test

to_plot_models = ['combNet',  'sbertNetL_lin','sbertNet_lin', 'clipNetS_lin', 'bertNet_lin', 'gptNetXL_lin', 'gptNet_lin',  'bowNet_lin', 'simpleNet']
non_lin_models = ['combNet', 'sbertNetL', 'sbertNet', 'clipNetS', 'gptNetXL', 'gptNet', 'bertNet', 'bowNet', 'simpleNet']
tuned_to_plot = ['combNet', 'sbertNetL_lin_tuned', 'sbertNet_lin_tuned', 'clipNetS_lin_tuned',  
                    'gptNetXL_lin_tuned', 'gptNet_lin_tuned', 'bertNet_lin_tuned', 'bowNet_lin', 'simpleNet']
aux_models = ['combNet', 'combNetPlus', 'bowNet_lin', 'bowNetPlus', 'bertNet_lin', 'rawBertNet_lin', 'simpleNet', 'simpleNetPlus']



###FIG2
plot_curves('NN_simData', 'swap', to_plot_models, mode='combined', avg=True, linewidth=1.2)
plot_all_models_task_dist('NN_simData', 'swap', to_plot_models, mode='combined')
plt.show()

plot_comp_dots('NN_simData', 'swap', to_plot_models, mode='swap_combined')
plot_comp_dots('NN_simData', 'family', to_plot_models, mode='combined')
plot_comp_dots('NN_simData', 'swap', tuned_to_plot, mode='combined')
_, _, within_sig, stats_arr = plot_clauses_dots('NN_simData', 'swap', to_plot_models[:], y_lim=(-0.55, 0.28))
plt.show()

###FIG3
EXP_FILE = 'NN_simData/swap_holdouts'

#SBERTNET
sbertNet = make_default_model('sbertNetL_lin')
holdouts_file = 'swap9'

sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed4')
plot_scatter(sbertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(sbertNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='full')

#GPTNET
gptNet = make_default_model('gptNetXL_lin')
holdouts_file = 'swap9'

gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed2')
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(gptNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='full')

#SIMPLENET
simpleNet = make_default_model('simpleNet')
holdouts_file = 'swap9'

simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')
plot_scatter(simpleNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(simpleNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='rule', s=15)

#StructureNet
combNet = make_default_model('combNet')
holdouts_file = 'swap9'

combNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+combNet.model_name, suffix='_seed2')
plot_scatter(combNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3)
plot_scatter(combNet, ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, rep_depth='rule', s=10)

#CCGP PLOTS
plot_layer_ccgp('NN_simData', 'swap', to_plot_models, ylim=(0.48, 1.01), markersize=4.5)
plt.show()
plot_ccgp_corr('NN_simData', 'swap', to_plot_models)


###FIG4
#############SINGLE UNIT TUING
EXP_FILE = 'NN_simData/swap_holdouts'
sbertNet = make_default_model('sbertNetL_lin')

##ANTI GO
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed2')
unit=42
plot_tuning_curve(sbertNet, ['Go', 'AntiGo', 'RTGo', 'AntiRTGo'], unit, [149]*4, num_trials=80, smoothing=1, min_coh=0.01, max_coh=0.5)

##DM
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')


unit = 3
plot_neural_resp(sbertNet, 'DMMod1', 'diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'AntiDMMod1','diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'DMMod2','diff_strength', unit, num_trials=25)
plot_neural_resp(sbertNet, 'AntiDMMod2','diff_strength', unit, num_trials=25)

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


###Fig5
perf, _ = plot_partner_perf('NN_simData', 'sbertNetL_lin', figsize=(3, 3), s=12)
confuse_mat = np.load('NN_simData/multitask_holdouts/decoder_perf/sbertNetL_lin/test_sm_multi_decoder_multi_confuse_mat.npy')
plot_decoding_confuse_mat(np.round(np.mean(confuse_mat, axis=0)/50, 2), linewidths=0.1, linecolor='#E5E4E2')


get_novel_instruct_ratio('NN_simData', 'sbertNetL_lin')
get_novel_instruct_ratio('NN_simData', 'sbertNetL_lin', sm_holdout=True)
get_novel_instruct_ratio('NN_simData', 'sbertNetL_lin', sm_holdout=True, decoder_holdout=True)

####SUPPLEMENT####

##ALL MODEL LEARNING CURVES
fig_axn = plot_curves('NN_simData', 'multitask', to_plot_models, training_file='Multitask', linewidth=0.5)
fig_axn[0].tight_layout()
plt.show()

##VALIDATION
plot_all_task_lolli_v('NN_simData', 'swap', to_plot_models[1:-1], mode='val')
plt.show()


####ADDITIONAL HOLDOUT####

###HOLDOUTS
plot_curves('NN_simData', 'swap', to_plot_models, mode='combined', avg=True, linewidth=1.2)
plot_all_models_task_dist('NN_simData', 'swap', to_plot_models, mode='combined')
plot_all_task_lolli_v('NN_simData', 'swap', to_plot_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='combined')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()


###SWAP HOLDOUTS
plot_curves('NN_simData', 'swap', to_plot_models, mode='swap_combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('NN_simData', 'swap', to_plot_models, mode='swap_combined')
plot_all_task_lolli_v('NN_simData', 'swap', to_plot_models, mode='swap_combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='swap_combined')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()

###FAMILY
plot_curves('NN_simData', 'family', to_plot_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('NN_simData', 'family', to_plot_models, mode='combined')
plot_all_task_lolli_v('NN_simData', 'family', to_plot_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'family', to_plot_models, mode='combined')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()


##TUNED HOLDOUTS
plot_curves('NN_simData', 'swap', tuned_to_plot, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('NN_simData', 'swap', tuned_to_plot, mode='combined')
plot_all_task_lolli_v('NN_simData', 'swap', tuned_to_plot, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', tuned_to_plot, mode='combined')
plot_significance(t_mat, p_mat, tuned_to_plot)
plt.show()


##non-linear holdouts
plot_curves('NN_simData', 'swap', non_lin_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('NN_simData', 'swap', non_lin_models, mode='combined')
plot_all_task_lolli_v('NN_simData', 'swap', non_lin_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', non_lin_models, mode='combined')
plot_significance(t_mat, p_mat, non_lin_models)
plt.show()

##langPlus 
plot_curves('NN_simData', 'swap', aux_models, mode='combined', avg=True, linewidth=0.8)
plot_all_models_task_dist('NN_simData', 'swap', aux_models, mode='combined', hatch = '///', edgecolor='white')
plot_all_task_lolli_v('NN_simData', 'swap', aux_models, mode='combined')
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', aux_models, mode='combined')
plot_significance(t_mat, p_mat, aux_models)
plt.show()

###Unit Var
var = plot_task_var_heatmap('NN_simData/swap_holdouts/swap9', 'sbertNetL_lin', 3)


###CLAUSE SIG

_, _, within_sig, stats_arr = plot_clauses_dots('NN_simData', 'swap', to_plot_models[:], y_lim=(-0.55, 0.28))
plt.show()

plot_significance(stats_arr[0], stats_arr[1], to_plot_models, vmin=-80, vmax=80)
plt.show()

plot_significance(within_sig[0][None, :], within_sig[1][None,:], to_plot_models, vmin=-80, vmax=80)
plt.show()


###CCGP
plot_comp_dots('NN_simData', 'swap', to_plot_models, 'ccgp', y_lim=(0.5, 1.0))
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='ccgp')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()

plot_comp_dots('NN_simData', 'swap', to_plot_models, 'multi_ccgp', y_lim=(0.5, 1.0))
plt.show()

t_mat, p_mat, is_sig  = calc_t_test('NN_simData', 'multitask', to_plot_models, mode='multi_ccgp')
plot_significance(t_mat,p_mat, to_plot_models)
plt.show()

plot_comp_dots('NN_simData', 'swap', to_plot_models, 'swap_ccgp', y_lim=(0.5, 1.0))
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='swap_ccgp')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()

plot_comp_dots('NN_simData', 'swap', to_plot_models, 'embedding_ccgp', y_lim=(0.475, 1.01))
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='embedding_ccgp')
plot_significance(t_mat, p_mat, to_plot_models)
plt.show()


###compositional inputs
plot_comp_dots('NN_simData', 'swap', to_plot_models, 'combinedcomp', y_lim=(0.0, 1.0))
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='combinedcomp')
plot_significance(t_mat, p_mat,  to_plot_models)
plt.show()

plot_comp_dots('NN_simData', 'swap', to_plot_models, 'multi_comp', y_lim=(0.0, 1.0))
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', to_plot_models, mode='multi_comp')
plot_significance(t_mat, p_mat,  to_plot_models)
plt.show()


###INPUTS ONLY
plot_curves('NN_simData', 'swap', to_plot_models, mode='combinedinputs_only', avg=True, linewidth=0.8)
plt.show()

plot_curves('NN_simData', 'family', to_plot_models[::-1], mode='combinedinputs_only', avg=True)
plt.show()

#############SINGLE UNIT TUING################
##########FAILURE TO SWAP BY GPT XL########
EXP_FILE = 'NN_simData/swap_holdouts'
gptNet = make_default_model('gptNetXL_lin')

holdouts_file = 'swap9'
gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed2')

unit=97
plot_neural_resp(gptNet, 'DMMod1', 'diff_strength', unit, num_trials=25)
plot_neural_resp(gptNet, 'AntiDMMod1','diff_strength', unit, num_trials=25)
plot_neural_resp(gptNet, 'DMMod2','diff_strength', unit, num_trials=25)
plot_neural_resp(gptNet, 'AntiDMMod2','diff_strength', unit, num_trials=25)

###GPT COMPARISON
fig_axn = plot_curves('NN_simData', 'swap', ['gptNetXL_lin'], mode='combined', avg=True, linewidth=0.8)
plot_curves('NN_simData', 'swap', ['gptNetXL_L_lin'], mode='combined', fig_axn=fig_axn, avg=True, linewidth=0.8)
plt.show()

t_mat, p_mat, is_sig = calc_t_test('NN_simData', 'swap', ['gptNetXL_lin', 'gptNetXL_L_lin'])
plot_significance(t_mat, p_mat, ['gptNetXL_lin', 'gptNetXL_L_lin'])
plt.show()


###ADDITIONAL DECODER

###additional decoders 
mean_perfs, stats = plot_partner_perf('NN_simData', 'sbertNetL_lin', figsize=(3, 3), s=12)
plot_significance(stats[0], stats[1], fig_size = (14, 3), xticklabels=_get_partner_perf_labels(), yticklabels=_get_partner_perf_labels(), labelsize=8)
plt.show()

mean_perfs, stats_comb = plot_partner_perf('NN_simData', 'combNet', figsize=(3, 3), s=12)
plot_significance(stats_comb[0], stats_comb[1], fig_size = (14, 3), xticklabels=_get_partner_perf_labels(), yticklabels=_get_partner_perf_labels(), labelsize=8)
plt.show()

confuse_mat = get_decoded_vec_cos_sim('NN_simData')
plot_decoding_confuse_mat(confuse_mat, cos_sim=True)

mean_perfs, stats_emb = plot_partner_perf('NN_simData','sbertNetL_lin', figsize=(3, 3), s=12, decode_embeddings=True)
plot_significance(stats_emb[0][:6, :6], stats_emb[1][:6,:6], fig_size = (14, 3), xticklabels=_get_partner_perf_labels()[:6], yticklabels=_get_partner_perf_labels()[:6], labelsize=8)
plt.show()

confuse_mat = np.load('NN_simData/multitask_holdouts/decoder_perf/sbertNetL_lin/test_sm_multi_decoder_multi_confuse_matfrom_embeddings.npy')
plot_decoding_confuse_mat(np.round(np.mean(confuse_mat, axis=0)/50, 2), linewidths=0.1, linecolor='#E5E4E2')

multi_multi_instruct = pickle.load(open('NN_simData/multitask_holdouts/decoder_perf/sbertNetL_lin/test_sm_multi_decoder_multi_instructs_dict', 'rb'))
holdout_multi_instruct = pickle.load(open('NN_simData/swap_holdouts/decoder_perf/sbertNetL_lin/test_sm_holdout_decoder_multi_instructs_dict', 'rb'))
holdout_holdout_instruct = pickle.load(open('NN_simData/swap_holdouts/decoder_perf/sbertNetL_lin/test_sm_holdout_decoder_holdout_instructs_dict', 'rb'))

print_decoded_instruct(multi_multi_instruct)
print_decoded_instruct(holdout_multi_instruct)
print_decoded_instruct(holdout_holdout_instruct)






