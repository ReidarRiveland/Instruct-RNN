import torch
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model


#def get_holdout_instruct_set(foldername, model_name, seed): 


load_str = '7.20models/swap_holdouts/swap0/clipNet_lin/'
sm_model = make_default_model('clipNet_lin')
rnn_decoder = DecoderRNN(256, drop_p=0.0)

encoder = EncoderDecoder(sm_model, rnn_decoder)
encoder.load_model_componenets(load_str, 1)
encoder.to(0)

decoded_set = encoder.plot_confuse_mat(100, 2)

decoded_set['RTGo']
