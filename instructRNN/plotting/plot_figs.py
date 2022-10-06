from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *


plot_avg_holdout_curve('7.20models', 'swap', 
                                ['gptNetXL_lin', 'sbertNet_lin', 'clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'simpleNet', 'gptNet_lin'],
                                seeds =range(0, 5),
                                mode='combined'
                                )

plot_0_shot_task_hist('7.20models', 'swap', 
                                [ 'clipNet_lin', 'sbertNet_lin',  'gptNetXL_lin', 'bertNet_lin',  'gptNet_lin', 'simpleNet'][::-1],
                                seeds =range(0,5),
                                mode='combined'
                                )


EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

plot_scatter(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

plot_task_var_heatmap(EXP_FILE+'/'+holdouts_file, 'clipNet_lin', 4)

EXP_FILE = '7.20models/swap_holdouts'
gptNet = GPTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed4')

plot_scatter(gptNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=100)
plot_scatter(gptNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

plot_task_var_heatmap(EXP_FILE+'/'+holdouts_file, 'gptNet_lin', 4)


plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'gptNet_lin', 'sbertNet_lin', 'simpleNet'][::-1], seeds=range(5), plot_multis=True)
plot_ccgp_corr('7.20models', 'swap', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'sbertNet_lin', 'gptNet_lin', 'simpleNet'])


###Go Only Neuron 
plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 2, [140]*6, np.linspace(0, np.pi*2, 100))



plot_tuning_curve(clipNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], 27, [149]*6, np.linspace(0, np.pi*2, 100))




plot_neural_resp(clipNet, 'MultiDM','diff_strength', 15, num_trials=25)
plot_neural_resp(clipNet, 'DMMod1','diff_strength', 15, num_trials=25)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', 15, num_trials=25)

