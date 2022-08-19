import pickle
import torch
from instructRNN.decoding_models.decoder_models import DecoderRNN
from instructRNN.decoding_models.encoder_decoder import EncoderDecoder
from instructRNN.models.full_models import make_default_model

from instructRNN.instructions.instruct_utils import inv_train_instruct_dict, train_instruct_dict
from instructRNN.tasks.tasks import SWAPS_DICT


load_str = '7.20models/swap_holdouts/swap0/sbertNet_lin_tuned/'

sm_model = make_default_model('sbertNet_lin_tuned')
rnn_decoder = DecoderRNN(256, drop_p=0.0)

device = torch.device(0)

seed=0
sm_model.to(device)
rnn_decoder.to(device)
sm_model.eval()

rnn_decoder.load_model(load_str, suffix='_seed'+str(seed))
sm_model.load_model(load_str, suffix='_seed'+str(seed))

encoder = EncoderDecoder(sm_model, rnn_decoder)

encoder.init_context_set('swap0', 0, 768)


decoded_set = encoder.plot_confuse_mat(128, 2, )

decoded_set['Go']['other']


partner = make_default_model('sbertNet_lin_tuned')
partner.load_model('7.20models/swap_holdouts/swap1/sbertNet_lin_tuned/', suffix='_seed'+str(seed))

perf, _ = encoder.test_partner_model(partner, decoded_dict=decoded_set, )
perf['instructions'].mean()

import numpy as np 
np.mean(perf['instructions'], axis=1)




def get_all_partner_model_perf(num_repeats=5): 
    task_file='Multitask'    
    sm_model = config_model('sbertNet_tuned')
    partner_model = config_model('sbertNet_tuned')

    rnn_decoder = DecoderRNN(64, drop_p=0.1)
    sm_model.to(device)
    rnn_decoder.to(device)
    sm_model.eval()
    all_seeds = [0, 1, 2, 3, 4]

    load_str = '_ReLU128_4.11/swap_holdouts/'+task_file
    
    for seed in [0]: 
        all_perf_dict = {}
        all_perf_dict['instructions'] = np.empty((4, 16, num_repeats))
        all_perf_dict['context'] = np.empty((4, 16, num_repeats))
        all_perf_dict['others'] = np.empty((4, 16, num_repeats))
        print('\n seed '+str(seed)+'\n')
        sm_model.set_seed(seed)
        sm_model.load_model(load_str)
        rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_lin_wHoldout')
        encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)
        encoder_decoder.init_context_set(task_file, seed)
        decoded, confuse_mat = encoder_decoder.decode_set(128, from_contexts=True)
        partner_seeds = all_seeds.copy()
        partner_seeds.remove(seed)
        for i, partner_seed in enumerate([1]): 
            print(partner_seeds)
            print('\n partner seed '+str(partner_seed)+'\n')
            partner_model.set_seed(partner_seed)
            partner_model.load_model(load_str)
            perf, _ = encoder_decoder.test_partner_model(partner_model, num_repeats=num_repeats)
            for mode in ['context', 'instructions', 'others']:
                all_perf_dict[mode][i,...] = perf[mode]
        # pickle.dump(all_perf_dict, open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_all_perf_dict', 'wb'))
        # pickle.dump(confuse_mat, open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_confuse_mat', 'wb'))
        # pickle.dump(decoded, open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_decoded_set', 'wb'))

    return all_perf_dict


def load_decoder_outcomes(num_repeats=5): 
    all_perf_dict = {}
    all_perf_dict['instructions'] = np.empty((5, 4, 16, num_repeats))
    all_perf_dict['context'] = np.empty((5, 4, 16, num_repeats))
    all_perf_dict['others'] = np.empty((5, 4, 16, num_repeats))
    all_decoded_sets = []
    all_confuse_mat = np.empty((5, 16, 17))
    task_file='Multitask'    

    load_str = '_ReLU128_4.11/swap_holdouts/'+task_file


    for seed in range(5): 
        perf_dict=pickle.load(open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_all_perf_dict', 'rb'))
        confuse_mat = pickle.load(open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_confuse_mat', 'rb'))
        decoded_set = pickle.load(open(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_decoded_set', 'rb'))

        for mode in ['context', 'instructions', 'others']:
            all_perf_dict[mode][seed,...] = perf_dict[mode]        
        all_confuse_mat[seed, ...] = confuse_mat
        all_decoded_sets.append(decoded_set)

    return all_perf_dict, all_confuse_mat, all_decoded_sets












def get_all_holdout_partners():
    num_repeats = 10
    all_perf_dict = {}
    all_perf_dict['instructions'] = np.empty((5, 16, 5, num_repeats))
    all_perf_dict['context'] = np.empty((5, 16, 5, num_repeats))
    all_perf_dict['others'] = np.empty((5, 16, 5, num_repeats))

    for seed in range(5):
        sm_model.set_seed(seed)
        sm_model.load_model(load_str)
        rnn_decoder.load_model(load_str+'/sbertNet_tuned/decoders/seed'+str(seed)+'_rnn_decoder_lin_wHoldout')
        encoder_decoder = EncoderDecoder(sm_model, rnn_decoder)
        encoder_decoder.init_context_set(task_file, seed)
        for i in range(5): 
            model1.set_seed(i) 
            for tasks in training_lists_dict['swap_holdouts']: 
                holdout_file = get_holdout_file(tasks)
                model1.load_model('_ReLU128_4.11/swap_holdouts/'+holdout_file)
                perf, _ = encoder_decoder.test_partner_model(model1, num_repeats=num_repeats, tasks=tasks)
                for mode in ['context', 'instructions', 'others']:
                    all_perf_dict[mode][seed, [Task.TASK_LIST.index(tasks[0]), Task.TASK_LIST.index(tasks[1])], i, :] = perf[mode]

    return all_perf_dict
