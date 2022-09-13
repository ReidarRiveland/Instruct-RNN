from instructRNN.plotting.plotting import * 
from instructRNN.models.full_models import *

from instructRNN.tasks.tasks import *
from instructRNN.tasks.task_factory import *

plot_avg_holdout_curve('7.20models', 'swap', ['sbertNet_lin_tuned', 'simpleNet', 'bowNet', 'gptNetXL'])


plot_avg_holdout_curve('7.20models', 'swap', 
                                ['sbertNet_lin_tuned', 'sbertNet', 'sbertNet_tuned', 
                                'bertNet', 'bertNet_tuned', 'clipNet', 'clipNet_tuned', 'simpleNet', 'bowNet', 'gptNetXL'], 
                                emphasis_list=['sbertNet_lin_tuned', 'simpleNet', 'bowNet', 'gptNetXL']
                                )




EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)



holdouts_file = 'swap0'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

plot_tuning_curve(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1', 'AntiGoMod2', 'GoMod2'], 13, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=1, smoothing=0.8)


plot_scatter(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], dims=3, pcs=[1, 2, 3], num_trials=100)
plot_tuning_curve(sbertNet, ['DM', 'AntiDM'], 23, [140]*2, np.linspace(-0.5, 0.5, 100), num_repeats=1)


plot_tuning_curve(sbertNet, ['DM', 'AntiDM', 'DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2'], 43, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=1)






EXP_FILE = '7.20models/swap_holdouts'
sbertNet = SBERTNet_lin_tuned(LM_out_dim=64, rnn_hidden_dim=256)
holdouts_file = 'swap9'
sbertNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+sbertNet.model_name, suffix='_seed0')

plot_tuning_curve(sbertNet, ['Go', 'AntiGo'], 32, [140]*6, np.linspace(0, 2*np.pi, 100), num_repeats=1, smoothing=1)



plot_scatter(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1',  'GoMod2', 'AntiGoMod2'], dims=3, pcs=[1, 2, 3], num_trials=50)
plot_scatter(sbertNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1',  'GoMod2', 'AntiGoMod2'], dims=3, pcs=[1, 2, 3], num_trials=50, rep_depth='full')


EXP_FILE = '7.20models/swap_holdouts'
simpleNet = SimpleNet(rnn_hidden_dim=256)
simpleNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+simpleNet.model_name, suffix='_seed0')
plot_scatter(simpleNet, ['Go', 'AntiGo', 'GoMod1', 'AntiGoMod1',  'GoMod2', 'AntiGoMod2'], dims=3, pcs=[1, 2, 3], num_trials=50)


EXP_FILE = '7.20models/swap_holdouts'
gptNet = GPTNetXL(rnn_hidden_dim=256)
holdouts_file = 'swap9'
gptNet.load_model(EXP_FILE+'/'+holdouts_file+'/'+gptNet.model_name, suffix='_seed0')
sim_scores = get_layer_sim_scores(gptNet, rep_depth='full')
plot_RDM(sim_scores, cmap=sns.color_palette("Reds", as_cmap=True))