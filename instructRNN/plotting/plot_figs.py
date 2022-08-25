from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

plot_avg_holdout_curve('7.20models', 'swap', ['sbertNet_lin_tuned', 'simpleNet', 'bowNet', 'gptNetXL', ], seeds=range(5))



EXP_FILE = '7.20models/swap_holdouts'
#sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
simpleNet = SimpleNet()


holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')
simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')

plot_scatter(sbertNet, ['GoMod1', 'AntiGoMod1', 'GoMod2', 'AntiGoMod2'], dims=2, pcs=[0, 1])
