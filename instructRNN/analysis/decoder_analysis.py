import torch
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model


def get_holdout_instruct_set(foldername, model_name, seed): 


load_str = '7.20models/swap_holdouts/swap0/clipNet_lin/'



sm_model = make_default_model('clipNet_lin')
rnn_decoder = DecoderRNN(256, drop_p=0.0)

device = torch.device(0)

seed=1
sm_model.to(device)
rnn_decoder.to(device)
sm_model.eval()

rnn_decoder.load_model(load_str, suffix='_seed'+str(seed))
sm_model.load_model(load_str, suffix='_seed'+str(seed))

encoder = EncoderDecoder(sm_model, rnn_decoder)
encoder.load_foldername

encoder.init_context_set('7.20models/swap_holdouts/swap0', 1, 512)
encoder.contexts

decoded_set = encoder.plot_confuse_mat(100, 2)

decoded_set['RTGo']

partner = make_default_model('clipNet_lin')
partner.load_model('7.20models/swap_holdouts/swap1/clipNet_lin/', suffix='_seed'+str(seed))


partner.load_model('7.20models/multitask_holdouts/Multitask/clipNet_lin/', suffix='_seed'+str(0))

perf, _ = encoder.test_partner_model(partner, decoded_dict=decoded_set)







from instructRNN.tasks.tasks import TASK_LIST

perf['instructions'][TASK_LIST.index('AntiDMMod2')]

perf['others'].mean()

import numpy as np 
np.mean(perf['instructions'], axis=1)


