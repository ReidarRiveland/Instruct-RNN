from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *

plot_ccgp_corr('7.20models', 'swap', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'sbertNet_lin', 'simpleNet'])

plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'gptNet_lin', 'sbertNet_lin', 'simpleNet'][::-1], seeds=range(5,10))
plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'gptNet_lin', 'sbertNet_lin', 'simpleNet'][::-1], seeds=[5])

plot_avg_holdout_curve('7.20models', 'swap', 
                                ['gptNetXL_lin', 'sbertNet_lin', 'clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'simpleNet', 'gptNet_lin'],
                                seeds =range(0, 5),
                                mode='combined'
                                )

plot_avg_holdout_curve('7.20models', 'swap', 
                                ['sbertNet_lin',],
                                seeds =range(5, 9),
                                mode = 'combined'
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

data = HoldoutDataFrame('7.20models', 'swap', 'gptNet_lin', seeds=range(5), mode='combined')

data = HoldoutDataFrame('7.20models', 'swap', 'clipNet_lin', seeds=[0], mode='')

mean, _ = data.avg_seeds(k_shot=0)
np.mean(mean)


EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed4')

plot_neural_resp(sbertNet, 'DM','diff_strength', 15, num_trials=25)
plot_neural_resp(sbertNet, 'DMMod2','diff_strength', 15, num_trials=25)

plot_scatter(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')


EXP_FILE = '7.20models/swap_holdouts'
gptNet = GPTNetXL(rnn_hidden_dim=256)
holdouts_file = 'swap9'
gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed0')
plot_scatter(gptNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(gptNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed4')

plot_neural_resp(sbertNet, 'DM','diff_strength', 15, num_trials=25)
plot_neural_resp(sbertNet, 'DMMod2','diff_strength', 15, num_trials=25)

plot_scatter(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')


EXP_FILE = '7.20models/swap_holdouts'
clipNet = CLIPNet_lin(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
clipNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+clipNet.model_name, suffix='_seed4')

sim_scores = get_layer_sim_scores(clipNet, rep_depth='12')
plot_RDM(sim_scores)

plot_scatter(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50)
plot_scatter(clipNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

plot_scatter(clipNet, ['Go', 'RTGo', 'GoMod1', 'RTGoMod1', 'GoMod2', 'RTGoMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')


plot_tuning_curve()

plot_neural_resp(clipNet, 'DM','diff_strength', 15, num_trials=25)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', 15, num_trials=25)

