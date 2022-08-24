from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

plot_avg_holdout_curve('7.20models', 'swap', ['sbertNet_lin_tuned', 'simpleNet', 'bowNet', 'gptNetXL'], seeds=range(5))



EXP_FILE = '7.20models/swap_holdouts'
#sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)

holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

plot_scatter(sbertNet, ['MultiDM', 'AntiMultiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[0, 1, 2])
