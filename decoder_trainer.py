import itertools
from json import decoder
import numpy as np
import torch.optim as optim
from utils.utils import training_lists_dict, get_holdout_file_name
from utils.task_info_utils import get_instructions
from task import Task
from dataset import TaskDataSet
from decoding_models.decoder_models import DecoderRNN
from attrs import define, asdict
from models.full_models import make_default_model
from tqdm import tqdm
import pickle
import os


import torch
import torch.nn as nn

torch.cuda.is_available()

device = torch.device(0)


@define 
class DecoderTrainerConfig(): 
    file_path: str
    random_seed: int

    epochs: int = 100
    batch_len: int = 64
    num_batches: int = 1000
    holdouts = []

    optim_alg: optim = optim.Adam
    lr: float = 1e-4
    weight_decay: float = 0.0

    scheduler_class: optim.lr_scheduler = optim.lr_scheduler.ExponentialLR
    scheduler_args: dict = {'gamma': 0.999999999}

    init_teaching_forcing_ratio = 0.5

class DecoderTrainer():
    def __init__(self, config:DecoderTrainerConfig=None, from_checkpoint_dict:dict=None): 
        self.config = config
        self.cur_epoch = 0 
        self.cur_step = 0
        self.teacher_loss_data = []
        self.loss_data = []
        self.seed_suffix = '_seed'+str(self.seed)

        if from_checkpoint_dict is not None: 
            for name, value in from_checkpoint_dict.items(): 
                setattr(self, name, value)

        for name, value in asdict(self.config, recurse=False).items(): 
            setattr(self, name, value)

    def _print_progress(self, decoder_loss, use_teacher_forcing, 
                                decoded_sentence, target_instruct): 
        print('Teacher forcing: ' + str(use_teacher_forcing))
        print('Decoder Loss: ' + str(decoder_loss.item()/self.pad_len))

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

    def _record_session(self, mode):
        record_attrs = ['config', 'optimizer', 'scheduler', 'cur_epoch', 'cur_step', 'teacher_loss_data', 'loss_data']
        checkpoint_attrs = {}
        for attr in record_attrs: 
            checkpoint_attrs[attr]=getattr(self, attr)

        if mode == 'CHECKPOINT':
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+self.seed_suffix+'_CHECKPOINT_attrs', 'wb'))
            self.decoder.save_model(self.file_path, suffix='_'+self.seed_suffix+'_CHECKPOINT')

        if mode=='FINAL': 
            pickle.dump(checkpoint_attrs, open(self.file_path+'/'+self.model_file_path+self.seed_suffix+'_attrs', 'wb'))
            os.remove(self.file_path+'/'+self.model_file_path+self.seed_suffix+'_CHECKPOINT_attrs')
            pickle.dump(self.loss_data, open(self.file_path+'/'+self.seed_suffix+'_training_loss', 'wb'))
            pickle.dump(self.correct_data, open(self.file_path+'/'+self.seed_suffix+'_training_correct', 'wb'))
            decoder.save_model(self.file_path, suffix='_'+self.seed_suffix)
            os.remove(self.file_path+'/'+decoder.model_name+'_'+self.seed_suffix+'_CHECKPOINT.pt')


    def _init_streamer(self):
        self.streamer = TaskDataSet(self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts, 
                        self.set_single_task)

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

        for i in tqdm(range(self.epochs), desc='epochs'): 
            print('Epoch: ' + str(i)+'\n')
            for j, data in enumerate(self.streamer.stream_batch()): 
                ins, _, _, _, task_type = data
                decoder_loss=0

                use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
                target_instruct = get_instructions(self.batch_size, task_type, None)

                target_tensor = decoder.tokenizer(target_instruct).to(device)
                _, sm_hidden = sm_model.forward(ins.to(device), target_instruct)

                self.decoder_optimizer.zero_grad()

                if use_teacher_forcing:
                    decoder_input = torch.tensor([[decoder.tokenizer.sos_token_id]*self.batch_size]).to(device)
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
                    
            self.scheduler.step()
            teacher_forcing_ratio -= self.init_teacher_forcing_ratio/self.epochs
            print('Teacher Force Ratio: ' + str(teacher_forcing_ratio))

def check_decoder_trained():


def train_decoder_set(config, decoder_type='rnn'): 
# decoder_type = 'rnn'
# #config=(0, 'sbertNet_tuned', training_lists_dict['swap_holdouts'][0])
# config=(0, 'sbertNet_tuned', ['Multitask'])
# #config=(0, 'sbertNet_tuned', ['Go', 'Anti DM'])

    model_file = '_ReLU128_4.11/swap_holdouts/'
    seed, model_name, tasks = config 
    for holdout_train in [False, True]:
        # if tasks == ['Multitask'] and holdout_train == True:
        #     continue

        if holdout_train: 
            holdout_str = '_wHoldout'
            holdouts=tasks
        else: 
            holdout_str = ''
            holdouts = []


        print(seed, tasks, holdout_str, holdouts)

        task_file = get_holdout_file_name(tasks)
        filename = model_file + task_file+'/'+ model_name 

        sm_model = make_default_model(model_name)
        sm_model.load_model(filename, suffix='_seed'+str(seed))
        sm_model.to(device)

        decoder= DecoderRNN(64, drop_p = 0.1)
        decoder.to(device)
        decoder_optimizer = optim.Adam([
            #{'params' : decoder.embedding.parameters()},
            {'params' : decoder.gru.parameters()},
            {'params' : decoder.out.parameters()},
            {'params' : decoder.embedding.parameters()},
            {'params' : decoder.sm_decoder.parameters(), 'lr': 1e-4}
        ], lr=1e-4, weight_decay=0.0)



        sch = optim.lr_scheduler.ExponentialLR(decoder_optimizer, 0.99, verbose=False)
        for n, p in decoder.named_parameters(): 
            if p.requires_grad: print(n)


        train_rnn_decoder(sm_model, decoder, decoder_optimizer, sch, 100, 0.5, holdout_tasks=holdouts)
        decoder.save_model(filename+'/decoders/seed'+str(seed)+'_'+decoder_type+'_decoder_lin'+holdout_str)
        decoder.save_model_data(filename+'/decoders/seed'+str(seed)+'_'+decoder_type+'_decoder_lin'+holdout_str)


from utils.utils import training_lists_dict
seeds = [0]
model_file = '_ReLU128_4.11/swap_holdouts/'
to_train = list(itertools.product(seeds, ['sbertNet_tuned'], training_lists_dict['swap_holdouts']))
for config in to_train: 
    train_decoder_set(config, decoder_type='rnn')


