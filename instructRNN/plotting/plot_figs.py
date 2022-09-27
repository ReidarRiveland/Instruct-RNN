from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *



plot_0_shot_task_hist('7.20models', 'swap', 
                                [ 'clipNet_lin', 'sbertNet_lin', 'gptNetXL_lin'],
                                seeds =range(0,1)
                                )


plot_avg_holdout_curve('7.20models', 'swap', 
                                ['gptNetXL_lin', 'sbertNet_lin', 'clipNet_lin', 'bertNet_lin', 'simpleNet', 'simpleNetPlus', 'comNet', 'comNetPlus', 'bowNet_lin'],
                                seeds =range(0, 5)
                                )


plot_all_holdout_curves('7.20models', 'swap', 
                                [ 'clipNet_lin'],
                                seeds =range(8)
                                )

plot_avg_holdout_curve('7.20models', 'swap', 
                                [ 'gptNetXL_lin'],
                                seeds =range(0, 5)
                                )

plot_all_training_curves('7.20models', 'multitask', 'Multitask', ['gptNet_lin'])


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


