from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *

#plot_avg_holdout_curve('7.20models', 'swap', ['sbertNet_lin_tuned', 'simpleNet', 'bowNet', 'gptNetXL', ], seeds=range(5))
coh_arr = max_var_coh(10, main_mod = 0)
coh_arr



EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

plot_tuning_curve(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'AntiGoMod2', 'GoMod2'], 13, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=5)


plot_scatter(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[1, 2, 3], num_trials=100)
plot_tuning_curve(sbertNet, ['DM', 'AntiDM'], 203, [140]*2, np.linspace(-0.5, 0.5, 100), num_repeats=1)


plot_tuning_curve(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 43, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=1)




holdouts_file = 'swap1'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')
plot_tuning_curve(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'AntiGoMod2', 'GoMod2'], 13, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=1)
plot_tuning_curve(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'AntiGoMod2', 'GoMod2'], 15, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=1)
