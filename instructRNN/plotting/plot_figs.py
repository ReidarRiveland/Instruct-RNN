from random import shuffle
from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *

max_var_dir(100, 1, False, 1)
trials = DM(100, max_var=True)

trials.plot_trial(0)

plot_avg_holdout_curve('7.20models', 'swap', ['sbertNet_lin_tuned', 'simpleNet', 'bowNet', 'gptNetXL', ], seeds=range(5))



EXP_FILE = '7.20models/swap_holdouts'
#sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
simpleNet = SimpleNet()


holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')
simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')

plot_scatter(sbertNet, ['GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], dims=2, pcs=[0, 1])
plot_tuning_curve(sbertNet, ['Go'], 100, [120]*4, np.linspace(0, 2*np.pi, 100), num_repeats=20)

