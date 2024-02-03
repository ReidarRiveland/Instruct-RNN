import torch
import torch.nn as nn
from torch import optim
import numpy as np
from yaml import warnings

import pickle
import copy
from tqdm import tqdm
from attrs import define, asdict
import os
import warnings
from os.path import exists
from instructRNN.analysis.model_eval import task_eval

from instructRNN.trainers.base_trainer import *
from instructRNN.data_loaders.dataset import *
from instructRNN.tasks.task_criteria import *
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.models.full_models import make_default_model
from instructRNN.analysis.model_analysis import get_rule_reps, get_instruct_reps
from instructRNN.models.sensorimotor_models import InferenceNet
from instructRNN.instructions.instruct_utils import make_one_hot
from instructRNN.tasks.tasks import SWAPS_DICT

from ruleLearning.rulelearner import get_held_in_indices

if torch.cuda.is_available():
    device = torch.device(0)
    print(torch.cuda.get_device_name(device), flush=True)
else: 
    device = torch.device('cpu')
    

@define
class InferenceTrainerConfig(): 
    file_path: str
    random_seed: int
    epochs: int = 30
    min_run_epochs: int = 1
    batch_len: int = 64
    num_batches: int = 1200
    set_single_task: str = None
    stream_data: bool = True

    optim_alg: str = 'adam'
    init_lr: float = 0.0001
    init_lang_lr: float = None
    weight_decay: float = 0.0

    scheduler_type: str = 'exp'
    scheduler_gamma: float = 0.93
    scheduler_args: dict = {}


class InferenceTrainer(BaseTrainer): 
    def __init__(self, training_config:InferenceTrainerConfig): 
        super().__init__(training_config)

    def _init_streamer(self):
        self.streamer = TaskDataSet(
                        self.stream_data, 
                        self.batch_len, 
                        self.num_batches, 
                        self.holdouts, 
                        self.set_single_task)

    def init_optimizer(self, model):
        if self.optim_alg == 'adam': 
            optim_alg = optim.Adam

        optimizer = optim_alg(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.optimizer = optimizer

    def _init_scheduler(self):
        if self.scheduler_type == 'exp': 
            scheduler_class = optim.lr_scheduler.ExponentialLR

        self.scheduler = scheduler_class(self.optimizer, gamma=self.scheduler_gamma, **self.scheduler_args)
        self.step_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=[self.epochs-5, self.epochs-2, self.epochs-1], gamma=0.25)

    def train(self, inference_model, sm_model, swap_label): 
        inference_model.train()
        inference_model.to(device)
        self.holdouts = SWAPS_DICT[swap_label]

        self._init_streamer()
        self.init_optimizer(inference_model)
        self.held_in_tasks = [TASK_LIST[i] for i in get_held_in_indices(swap_label)]
        softmax = nn.LogSoftmax(dim=1)

        #criteria = nn.MSELoss()
        criteria = nn.NLLLoss()

        # if hasattr(sm_model, 'langModel'): 
        #     rule_reps = torch.tensor(get_instruct_reps(sm_model.langModel, depth='last'))
        #     print(rule_reps.shape)
        # else: 
        #     rule_reps = torch.tensor(get_rule_reps(sm_model, rule_layer='last'))

        for self.cur_epoch in tqdm(range(self.cur_epoch, self.epochs), desc='epochs'):

            self.streamer.shuffle_stream_order()
            for self.cur_step, data in enumerate(self.streamer.stream_batch()): 
                ins, tar, mask, tar_dir, task_type = data

                self.optimizer.zero_grad()
                out, _ = inference_model(ins.to(device))
                #target = torch.tensor(make_one_hot(len(self.held_in_tasks), self.held_in_tasks.index(task_type))).repeat(self.batch_len, 1)
                #target = sm_model.expand_info(rule_reps[TASK_LIST.index(task_type), np.random.randint(rule_reps.shape[1])].unsqueeze(0).repeat(self.batch_len, 1), 150, 0).to(device)
                target = torch.tensor(self.held_in_tasks.index(task_type)).repeat(self.batch_len)

                loss = criteria(softmax(out[:, -1, :]), target.to(inference_model.__device__)) 

                loss.backward()
                self.optimizer.step()

                if self.cur_step%50 == 0:
                    print('\n')
                    print(task_type)
                    print(loss.item())
                    scores, indices = out[0, -1, :].topk(5)
                    print([self.held_in_tasks[i] for i in indices.squeeze()])
                    print(scores.detach())
                    #print(out[0, -1, ...])
                
                if self.cur_step%250 == 0: 
                    with torch.no_grad():
                        print('showing validation perf')
                        for task in self.holdouts: 
                            print('\n')
                            ins, tar, mask, tar_dir, task_type = construct_trials('AntiDM', 20, return_tensor=True)
                            print(task)
                            out, _ = inference_model(ins.to(device))
                            scores, indices = out[0, -1, :].topk(5)
                            print([self.held_in_tasks[i] for i in indices.squeeze()])
                            print(scores.detach())

        if not os.path.exists(self.file_path): 
            os.makedirs(self.file_path)
        torch.save(inference_model.state_dict(), f'{self.file_path}/infer_{sm_model.model_name}_seed{self.random_seed}.pt')



def train_inference_model(exp_folder, model_name, seed, labeled_holdouts, **train_config_kwargs): 
    assert model_name in ['sbertNetL_lin', 'simpleNetPlus'], 'not implemented for other models, need to initialize correct rule dim'

    torch.manual_seed(seed)
    label, holdouts = labeled_holdouts
    file_name = exp_folder+'/'+label+'/'+model_name   

    sm_model = make_default_model(model_name)
    sm_model.load_model(file_name, suffix=f'_seed{seed}')

    if model_name == 'sbertNetL_lin': 
        inference_model = InferenceNet(45, 256)
    else: 
        inference_model = InferenceNet(45, 256)

    training_config = InferenceTrainerConfig(f'inference_models/{file_name}/', seed, **train_config_kwargs)
    trainer = InferenceTrainer(training_config)

    for n, p in inference_model.named_parameters(): 
        if p.requires_grad: print(n)

    trainer.train(inference_model, sm_model, labeled_holdouts[0])

train_inference_model('NN_simData/swap_holdouts', 'simpleNetPlus', 0, list(SWAPS_DICT.items())[0])


# from instructRNN.tasks.tasks import SWAPS_DICT

# #list(SWAPS_DICT.items())[0]

# train_inference_model('NN_simData/swap_holdouts', 'simpleNetPlus', 0, list(SWAPS_DICT.items())[0])



# import torch.nn as nn

# from instructRNN.models.full_models import *
# from instructRNN.tasks.tasks import SWAPS_DICT
# ################################################RULELEARNING############################################################

# sm_model = make_default_model('sbertNetL_lin')
# sm_model.load_model('NN_simData/swap_holdouts/swap0/sbertNetL_lin', suffix='_seed0')

# SWAPS_DICT['swap0']

# inference_model = InferenceNet(1024, 256)

# trainer = InferenceTrainer(InferenceTrainerConfig('ruleLearning/', 0, holdouts=SWAPS_DICT['swap0']))
# trainer.train(inference_model, sm_model)



# sm_model.langModel

# #torch.save(inference_model.state_dict(), 'inferenceModel_sbert.pt')


# inference_model.load_state_dict(torch.load('inferenceModel_sbert.pt'))

# rules = get_rule_reps(sm_model, rule_layer='last')


# SWAPS_DICT['swap0']

# ins, _ , _, _, _ =construct_trials('AntiCOMP2', 64)
# out, _ = inference_model(torch.tensor(ins).to(0))

# #####ONLY ENCODE IN CLUSTER IF SIMILARITY?

# rules.shape

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# sims = cosine_similarity(out[:, -1, :].detach().cpu().numpy(), rules.squeeze())

# [TASK_LIST[i] for i in sims.argmax(1)]


# np.argmax(sims

# labelsize=4
# with sns.plotting_context(rc={ 'xtick.labelsize': labelsize,'ytick.labelsize': labelsize}):
#     #sns.heatmap(np.corrcoef(rules.squeeze()))

#     sns.heatmap(cosine_similarity(out[0:1, -1, :].detach().cpu().numpy(), rules.squeeze()).T, yticklabels=TASK_LIST)
# plt.savefig('heatmap')
# plt.close()
