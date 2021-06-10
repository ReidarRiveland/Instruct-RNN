import numpy as np

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import isCorrect


def masked_MSE_Loss(nn_out, nn_target, mask):
    """MSE loss (averaged over features then time) function with special weighting mask that prioritizes loss in response epoch 
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_num, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_num, seq_len, features)
        mask (Tensor): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        Tensor: weighted loss of neural network response; shape: (1x1)
    """

    mask_applied = torch.mul(torch.pow((nn_out - nn_target), 2), mask)
    avg_applied = torch.mean(torch.mean(mask_applied, 2), 1)
    return torch.mean(avg_applied)


def init_optimizer(model, lr, milestones, weight_decay=0.0, langLR=None):
    try:
        if langLR is None: langLR = lr 
        optimizer = optim.Adam([
                {'params' : model.rnn.parameters()},
                {'params' : model.langModel.model.parameters(), 'lr': langLR}
            ], lr=lr, weight_decay=weight_decay)
    except AttributeError: 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    
def train(self, model, streamer, epochs, optimizer, scheduler): 
    batch_len = streamer.batch_len 
    for i in range(epochs):
        print('epoch', i)
        streamer.permute_task_order()
        for j, data in enumerate(streamer.get_batch()): 
            
            ins, tar, mask, tar_dir, task_type = data

            optimizer.zero_grad()

            task_info = model.get_instructions(batch_len, task_type)
            out, _ = model(task_info, torch.Tensor(ins, device=device))

            loss = masked_MSE_Loss(out, torch.Tensor(tar, device=device), torch.Tensor(mask, device=device)) 
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
            optimizer.step()

            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
            model._loss_data_dict.append(loss.item())
            model._correct_data_dict.append(frac_correct)

            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                self.sort_perf_by_task()
                print('Frac Correct ' + str(frac_correct) + '\n')
            
        if scheduler is not None: 
            scheduler.step()    


from cog_rnns import InstructNet, SimpleNet
from nlp_models import BERT, SBERT

net = SimpleNet(128, 1, 'relu')
opt, sch = init_optimizer(net, 0.001, [5, 10, 15, 20])