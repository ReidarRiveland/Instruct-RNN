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

plot_scatter(clipNet, ['Go', 'RTGo', 'GoMod1', 'RTGoMod1', 'GoMod2', 'RTGoMod2'], dims=3, pcs=[0, 1, 2], num_trials=50, rep_depth='full')

plot_ccgp_corr('7.20models', 'swap', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'sbertNet_lin', 'gptNet_lin', 'simpleNet'])

plot_layer_ccgp('7.20models/swap_holdouts', ['clipNet_lin', 'bertNet_lin', 'bowNet_lin', 'gptNet_lin', 'sbertNet_lin', 'simpleNet'][::-1], seeds=range(5), plot_multis=True)


plot_neural_resp(clipNet, 'DM','diff_strength', 15, num_trials=25)
plot_neural_resp(clipNet, 'DMMod2','diff_strength', 15, num_trials=25)
