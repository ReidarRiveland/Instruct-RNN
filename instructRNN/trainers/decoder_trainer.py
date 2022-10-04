import numpy as np
from attrs import define, asdict
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
    batch_len: int = 32
    num_batches: int = 1200
    stream_data: bool = True
    holdouts: list = []

    optim_alg: str = 'adam'
    lr: float = 1e-4
    weight_decay: float = 0.0

    scheduler_type: str = 'exp'
    scheduler_gamma: float = 0.999

    init_teacher_forcing_ratio: float = 0.5
    

class DecoderTrainer(BaseTrainer):
    def __init__(self, config:DecoderTrainerConfig=None): 
        self.config = asdict(config, recurse=False)
        self.cur_epoch = 0 
        self.cur_step = 0
        self.teacher_loss_data = []
        self.loss_data = []

        for name, value in self.config.items(): 
            setattr(self, name, value)
        self.seed_suffix = 'seed'+str(self.random_seed)

    @classmethod
    def from_checkpoint(cls, checkpoint_path): 
        attr_dict = pickle.load(open(checkpoint_path+'_attrs', 'rb'))
        config = DecoderTrainerConfig(**attr_dict.pop('config'))
        cls.checkpoint_path = checkpoint_path
        cls = cls(config) 
        for name, value in attr_dict.items(): 
            setattr(cls, name, value)
        return cls

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
        record_attrs = ['config', 'cur_epoch', 'teacher_loss_data', 'loss_data']
        checkpoint_attrs = {}
        for attr in record_attrs: 
            checkpoint_attrs[attr]=getattr(self, attr)
        record_file = self.file_path+'/'+decoder.decoder_name+'_'+self.seed_suffix

        with_holdouts = bool(self.holdouts)
        if with_holdouts: 
            holdouts_suffix = '_wHoldout'
        else: 
            holdouts_suffix = ''
        
        if os.path.exists(self.file_path):pass
        else: os.makedirs(self.file_path)

        if mode == 'CHECKPOINT':    
            chk_attr_path = record_file+'_CHECKPOINT_attrs'+holdouts_suffix
            pickle.dump(checkpoint_attrs, open(chk_attr_path, 'wb'))
            decoder.save_model(chk_attr_path)
            torch.save(self.optimizer.state_dict(), chk_attr_path+'_opt')

        if mode=='FINAL': 
            os.remove(record_file+'_CHECKPOINT_attrs'+holdouts_suffix)
            os.remove(record_file+'_CHECKPOINT'+holdouts_suffix+'.pt')
            os.remove(record_file+'_CHECKPOINT'+holdouts_suffix+'_opt')

            pickle.dump(checkpoint_attrs, open(record_file+'_attrs'+holdouts_suffix, 'wb'))
            decoder.save_model(record_file+holdouts_suffix)

    def _init_streamer(self):
        self.streamer = TaskDataSet(self.file_path.partition('/')[0]+'/training_data', 
                        self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts, 
                        None)

    def _init_optimizer(self, decoder):
        if self.optim_alg == 'adam': 
            optim_alg = optim.Adam

        self.optimizer = optim_alg([
            {'params' : decoder.gru.parameters()},
            {'params' : decoder.out.parameters()},
            {'params' : decoder.embedding.parameters()},
            {'params' : decoder.sm_decoder.parameters(), 'lr': self.lr}
        ], lr=self.lr, weight_decay=0.0)

    def _init_scheduler(self): 
        if self.scheduler_type == 'exp': 
            scheduler_class = optim.lr_scheduler.ExponentialLR
        self.scheduler = scheduler_class(self.optimizer, gamma=self.scheduler_gamma)

    def init_optimizer(self, model):
        self._init_optimizer(model)      
        self._init_scheduler()
        if hasattr(self, 'checkpoint_path'):
            opt_path = self.checkpoint_path + '_opt'
            self.optimizer.load_state_dict(torch.load(opt_path))  

    def train(self, sm_model, decoder): 
        criterion = nn.NLLLoss(reduction='mean')
        teacher_forcing_ratio = self.init_teacher_forcing_ratio
        self.pad_len  = decoder.tokenizer.pad_len 
         
        self._init_streamer()
        self.init_optimizer(decoder)

        for self.cur_epoch in tqdm(range(self.cur_epoch, self.epochs), desc='epochs'): 
            print('Epoch: ' + str(self.cur_epoch)+'\n')
            for j, data in enumerate(self.streamer.stream_batch()): 
                ins, _, _, _, task_type = data
                decoder_loss=0

                use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
                target_instruct = get_instructions(self.batch_len, task_type, None)

                target_tensor = decoder.tokenizer(target_instruct).to(device)
                _, sm_hidden = sm_model.forward(ins.to(device), target_instruct)

                self.optimizer.zero_grad()

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
                self.optimizer.step()

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

def load_checkpoint(model, file_name, seed): 
    checkpoint_name = model.model_name+'_seed'+str(seed)+'_CHECKPOINT'
    checkpoint_model_path = file_name+'/'+checkpoint_name+'.pt'

    print('\n Attempting to load model CHECKPOINT')
    if not os.exists(checkpoint_model_path): 
        raise Exception('No model checkpoint found at ' + checkpoint_model_path)
    else:
        print(checkpoint_model_path)
        model.load_state_dict(torch.load(checkpoint_model_path), strict=False)
        print('loaded model at '+ checkpoint_model_path)
    
    checkpoint_path = file_name+'/attrs/'+checkpoint_name
    trainer = DecoderTrainer.from_checkpoint(checkpoint_path)
    return model, trainer

def train_decoder(exp_folder, model_name, seed, labeled_holdouts, use_holdouts, use_checkpoint = False, overwrite=False, **train_config_kwargs): 
    torch.manual_seed(seed)
    label, holdouts = labeled_holdouts
    file_name = exp_folder+'/'+label+'/'+model_name

    if not overwrite and check_decoder_trained(file_name+'/decoders', seed, use_holdouts):
        return True

    print('\n TRAINING DECODER at ' + file_name + ' with holdouts ' +str(use_holdouts)+  '\n')
    model = make_default_model(model_name)   
    model.load_model(file_name, suffix='_seed'+str(seed))
    model.to(device)


    decoder = DecoderRNN(256)
    decoder.to(device)

    if use_checkpoint:
        try: 
            model, trainer = load_checkpoint(model, file_name, seed)
        except: 
            'NO checkpoint found, training model from starting point'


    if use_holdouts: trainer_config = DecoderTrainerConfig(file_name+'/decoders', seed, holdouts=holdouts, **train_config_kwargs)
    else: trainer_config = DecoderTrainerConfig(file_name+'/decoders', seed, **train_config_kwargs)
    
    trainer = DecoderTrainer(trainer_config)
    trainer.train(model, decoder)

