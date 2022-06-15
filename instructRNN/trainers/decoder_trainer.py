import numpy as np
from attrs import define
from tqdm import tqdm
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from instructRNN.models.full_models import make_default_model
from instructRNN.trainers.base_trainer import BaseTrainer
from instructRNN.instructions.instruct_utils import get_instructions
from instructRNN.data_loaders.dataset import TaskDataSet
from instructRNN.decoding_models.decoder_models import DecoderRNN

device = torch.device(0)

@define 
class DecoderTrainerConfig(): 
    file_path: str
    random_seed: int

    epochs: int = 80
    batch_len: int = 64
    num_batches: int = 1000
    stream_data: bool = False
    holdouts: list = []

    optim_alg: optim = optim.Adam
    lr: float = 1e-4
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.999}

    init_teacher_forcing_ratio: float = 0.5
    

class DecoderTrainer(BaseTrainer):
    def __init__(self, config:DecoderTrainerConfig=None, from_checkpoint_dict:dict=None): 
        super().__init__(config, from_checkpoint_dict)

    def _print_progress(self, decoder_loss, use_teacher_forcing, 
                                decoded_sentence, target_instruct): 
        print('Teacher forcing: ' + str(use_teacher_forcing))
        print('Decoder Loss: ' + str(decoder_loss/self.pad_len))

        print('target instruction: ' + target_instruct[-1])
        if use_teacher_forcing:
            try:
                eos_index = decoded_sentence.index('[EOS]')
            except ValueError: 
                eos_index = -1
            decoded_sentence = ' '.join(decoded_sentence[:eos_index])
        
        print('decoded instruction: ' + decoded_sentence + '\n')

    def _log_step(self, loss, use_teacher_forcing): 
        if use_teacher_forcing: 
            self.teacher_loss_data.append(loss)
        else: 
            self.loss_data.append(loss)

    def _record_session(self, decoder, mode):
        checkpoint_attrs = super()._record_session()
        record_file = self.file_path+'/'+decoder.decoder_name+'_'+self.seed_suffix

        with_holdouts = bool(self.holdouts)
        if with_holdouts: 
            holdouts_suffix = '_wHoldout'
        else: 
            holdouts_suffix = ''
        
        if os.path.exists(self.file_path):pass
        else: os.makedirs(self.file_path)

        if mode == 'CHECKPOINT':    
            pickle.dump(checkpoint_attrs, open(record_file+'_CHECKPOINT_attrs'+holdouts_suffix, 'wb'))
            decoder.save_model(record_file+'_CHECKPOINT'+holdouts_suffix)

        if mode=='FINAL': 
            pickle.dump(checkpoint_attrs, open(record_file+'_attrs'+holdouts_suffix, 'wb'))
            os.remove(record_file+'_CHECKPOINT_attrs'+holdouts_suffix)
            
            decoder.save_model(record_file+holdouts_suffix)
            os.remove(record_file+'_CHECKPOINT'+holdouts_suffix+'.pt')

    def _init_streamer(self):
        self.streamer = TaskDataSet(self.file_path.partition('/')[0]+'/training_data', 
                        self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts, 
                        None)

    def _init_optimizer(self, decoder):
        self.decoder_optimizer = self.optim_alg([
            {'params' : decoder.gru.parameters()},
            {'params' : decoder.out.parameters()},
            {'params' : decoder.embedding.parameters()},
            {'params' : decoder.sm_decoder.parameters(), 'lr': self.lr}
        ], lr=self.lr, weight_decay=0.0)
        self.scheduler = self.scheduler_class(self.decoder_optimizer, **self.scheduler_args)

    def train(self, sm_model, decoder): 

        criterion = nn.NLLLoss(reduction='mean')
        teacher_forcing_ratio = self.init_teacher_forcing_ratio
        self.pad_len  = decoder.tokenizer.pad_len 
         
        self._init_streamer()
        self._init_optimizer(decoder)

        for i in tqdm(range(self.epochs), desc='epochs'): 
            print('Epoch: ' + str(i)+'\n')
            for j, data in enumerate(self.streamer.stream_batch()): 
                ins, _, _, _, task_type = data
                decoder_loss=0

                use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
                target_instruct = get_instructions(self.batch_len, task_type, None)

                target_tensor = decoder.tokenizer(target_instruct).to(device)
                _, sm_hidden = sm_model.forward(ins.to(device), target_instruct)

                self.decoder_optimizer.zero_grad()

                if use_teacher_forcing:
                    decoder_input = torch.tensor([[decoder.tokenizer.sos_token_id]*self.batch_len]).to(device)
                    input_token_tensor  =  target_tensor
                    decoded_sentence = []
                    # Teacher forcing: Feed the target as the next input
                    for di in range(self.pad_len):
                        logits, _ = decoder._base_forward(decoder_input, sm_hidden)
                        decoder_output = decoder.softmax(logits)
                        next_index = decoder.draw_next(logits)
                        #get words for last sentence in the batch
                        last_word_index = next_index.detach()[-1].item()
                        last_word = decoder.tokenizer.index2word[last_word_index]
                        decoded_sentence.append(last_word)

                        decoder_loss += criterion(decoder_output, target_tensor[:, di])
                        decoder_input = torch.cat((decoder_input, input_token_tensor[:, di].unsqueeze(0)), dim=0)

                    loss=(decoder_loss)/self.pad_len
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    decoder_output, _, decoded_indices = decoder(sm_hidden)
                    for k in range(self.pad_len):
                        decoder_loss += criterion(decoder_output, target_tensor[:, k])
                    decoded_sentence = decoder.tokenizer.untokenize_sentence(decoded_indices)[-1]  # detach from history as input        
                    loss=(decoder_loss)/self.pad_len
                
                self._log_step(loss.item(), use_teacher_forcing)
                loss.backward()
                self.decoder_optimizer.step()

                if j%50==0: 
                    self._print_progress(loss.item(), use_teacher_forcing, 
                                decoded_sentence, target_instruct)
                    self._record_session(decoder, 'CHECKPOINT')
                    
            self.scheduler.step()
            teacher_forcing_ratio -= self.init_teacher_forcing_ratio/self.epochs
            print('Teacher Force Ratio: ' + str(teacher_forcing_ratio))
        self._record_session(decoder, 'FINAL')

def check_decoder_trained(file_name, seed, use_holdouts): 
    if use_holdouts: 
        holdouts_suffix = '_wHoldout'
    else: 
        holdouts_suffix = ''

    try: 
        pickle.load(open(file_name+'/rnn_decoder_seed'+str(seed)+holdouts_suffix+'_attrs', 'rb'))
        print('\n Model at ' + file_name + ' for seed '+str(seed)+' and holdouts ' + str(use_holdouts) +' aleady trained')
        return True
    except FileNotFoundError:
        return False

def train_decoder_set(exp_folder, model_names, seeds, holdout_dict, use_holdouts, overwrite=False, **train_config_kwargs): 
    for seed in seeds: 
        torch.manual_seed(seed)
        for label, holdouts in holdout_dict.items():
            for model_name in model_names: 
                file_name = exp_folder+'/'+label+'/'+model_name

                if not overwrite and check_decoder_trained(file_name+'/decoders', seed, use_holdouts):
                    continue 
                else:  
                    print('\n TRAINING DECODER at ' + file_name + ' with holdouts ' +str(use_holdouts)+  '\n')
                    model = make_default_model(model_name)   
                    model.load_model(file_name, suffix='_seed'+str(seed))
                    model.to(device)

                    decoder = DecoderRNN(128)
                    decoder.to(device)

                    if use_holdouts: trainer_config = DecoderTrainerConfig(file_name+'/decoders', seed, holdouts=holdouts, **train_config_kwargs)
                    else: trainer_config = DecoderTrainerConfig(file_name+'/decoders', seed, **train_config_kwargs)
                    
                    trainer = DecoderTrainer(trainer_config)
                    trainer.train(model, decoder)


