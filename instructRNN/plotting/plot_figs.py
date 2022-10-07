from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *


plot_avg_holdout_curve('7.20models', 'swap', 
                                ['gptNetXL_lin', 'sbertNet_lin', 'clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'simpleNet', 'gptNet_lin'],
                                seeds =range(0, 5),
                                mode='combined'
                                )


plot_all_holdout_curves('7.20models', 'swap', 
                                [ 'clipNet_lin', 'bertNet_lin'],
                                seeds =range(0, 5),
                                mode='combined'
                                )


plot_0_shot_task_hist('7.20models', 'swap', 
                                [ 'clipNet_lin', 'sbertNet_lin',  'gptNetXL_lin', 'bertNet_lin',  'gptNet_lin', 'simpleNet'][::-1],
                                seeds =range(0,5),
                                mode='combined'
                                )

plot_avg_holdout_curve('7.20models', 'swap', 
                                ['gptNetXL_lin_tuned', 'sbertNet_lin_tuned', 'clipNet_lin_tuned', 'bertNet_lin_tuned', 'bowNet_lin', 'simpleNet', 'gptNet_lin_tuned'],
                                seeds =range(0, 5),
                                mode='combined'
                                )



EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')



plot_task_var_heatmap('7.20models/swap_holdouts/swap9', 'clipNet_lin', 2)

plot_scatter(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')



EXP_FILE = '7.20models/swap_holdouts'
bertNet = BERTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
bertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+bertNet.model_name, suffix='_seed2')

plot_scatter(bertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(bertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')




plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'gptNet_lin', 'sbertNet_lin', 'simpleNet', 'gptNetXL_lin'][::-1], seeds=range(5), plot_multis=True)

plot_ccgp_corr('7.20models', 'swap', ['clipNet_lin', ' bertNet_lin', 'bowNet_lin', 'sbertNet_lin', 'gptNet_lin', 'simpleNet'])




###Go Only Neuron 
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 2, [140]*6)

###Direction Tuning FLip
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 73, [140]*6)

###sort of
plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 11, [140]*6)


###sort of
plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 11, [140]*6)
plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 16, [140]*6)
plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 22, [140]*6)
plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 41, [140]*6, smoothing=1e-1)

plot_tuning_curve(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 97, [120]*6, smoothing=1)

plot_tuning_curve(clipNet, ['COMP1', 'COMP2'], 56, [140]*6, smoothing=1e-1)




plot_neural_resp(clipNet, 'AntiGo','direction', 73, num_trials=25)


unit=113
plot_neural_resp(clipNet, 'DMMod1','diff_strength', unit, num_trials=25, smoothing=1)
plot_neural_resp(clipNet, 'AntiDMMod1','diff_strength', unit, num_trials=25, smoothing=1)

