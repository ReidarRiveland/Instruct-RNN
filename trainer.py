from torch.optim import optimizer
from data import TaskDataSet
import numpy as np

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import isCorrect

from task import Task
tuning_dirs = Task.TUNING_DIRS

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
                {'params' : model.recurrent_units.parameters()},
                {'params' : model.sensory_motor_outs.parameters()},
                {'params' : model.langModel.parameters(), 'lr': langLR}
            ], lr=lr, weight_decay=weight_decay)
    except AttributeError: 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    
def train(model, streamer, epochs, optimizer, scheduler): 
    model.to(device)
    batch_len = streamer.batch_len 
    for i in range(epochs):
        print('epoch', i)
        streamer.shuffle_stream_order()
        for j, data in enumerate(streamer.stream_batch()): 
            
            ins, tar, mask, tar_dir, task_type = data
            
            optimizer.zero_grad()

            task_info = model.get_task_info(batch_len, task_type)
            out, _ = model(task_info, ins)

            loss = masked_MSE_Loss(out, tar, mask) 
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)                    
            optimizer.step()

            frac_correct = round(np.mean(isCorrect(out, tar, tar_dir)), 3)
            model._loss_data_dict[task_type].append(loss.item())
            model._correct_data_dict[task_type].append(frac_correct)
            if j%50 == 0:
                print(task_type)
                print(j, ':', model.model_name, ":", "{:.2e}".format(loss.item()))
                print('Frac Correct ' + str(frac_correct) + '\n')


        if scheduler is not None: 
            scheduler.step()    



import torch
from cog_rnns import SimpleNet
model = SimpleNet(128, 1)
model.to(device)

opt, sch = init_optimizer(model, 0.001, [5, 10, 15, 20])
data = TaskDataSet()
data.data_to_device(device)
train(model, data, 30, opt, sch)


def popvec(act_vec):
    """Population vector decoder that reads the orientation of activity in vector of activities
    Args:      
        act_vec (np.array): output tensor of neural network model; shape: (features, 1)
    Returns:
        float: decoded orientation of activity (in radians)
    """

    act_sum = np.sum(act_vec, axis=1)

    temp_cos = np.sum(np.multiply(act_vec, np.cos(tuning_dirs)), axis=1)/act_sum

    temp_sin = np.sum(np.multiply(act_vec, np.sin(tuning_dirs)), axis=1)/act_sum

    loc = np.arctan2(temp_sin, temp_cos)

    return np.mod(loc, 2*np.pi)



def get_dist(angle1, angle2):
    """Returns the true distance between two angles mod 2pi
    Args:      
        angle1, angle2 (float): angles in radians
    Returns:
        float: distance between given angles mod 2pi
    """
    dist = angle1-angle2
    return np.minimum(abs(dist),2*np.pi-abs(dist))

def isCorrect(nn_out, nn_target, target_dirs): 
    """Determines whether a given neural network response is correct, computed by batch
    Args:      
        nn_out (Tensor): output tensor of neural network model; shape: (batch_size, seq_len, features)
        nn_target (Tensor): batch supervised target responses for neural network response; shape: (batch_size, seq_len, features)
        target_dirs (np.array): masks of weights for loss; shape: (batch_num, seq_len, features)
    
    Returns:
        np.array: weighted loss of neural network response; shape: (batch)
    """
    batch_size = nn_out.shape[0]
    if type(nn_out) == torch.Tensor: 
        nn_out = gpu_to_np(nn_out)
    if type(nn_target) == torch.Tensor: 
        nn_target = gpu_to_np(nn_target)

    isCorrect = np.empty(batch_size, dtype=bool)
    criterion = (2*np.pi)/10

    for i in range(batch_size):
        #checks response maintains fixataion
        isFixed = all(np.where(nn_target[i, :, 0] == 0.85, nn_out[i, :, 0] > 0.5, True))

        #checks trials that requiring repressing responses
        if np.isnan(target_dirs[i]): 
            isDir = all((nn_out[i, 114:119, :].flatten() < 0.15))
        
        #checks responses are coherent and in the correct direction
        else:
            is_response = np.max(nn_out[i, -1, 1:]) > 0.6
            loc = popvec(nn_out[i, -1, 1:])
            dist = get_dist(loc, target_dirs[i])        
            isDir = dist < criterion and is_response
        isCorrect[i] = isDir and isFixed
    return isCorrect


nn_out = np.random.randn(128, 120, 33)
nn_target = np.random.randn(128, 120, 33)
tar_dirs = np.random.randn(128)
i=0
#checks response maintains fixataion
isFixed = np.all(np.where(nn_target[:, :, 0] == 0.85, nn_out[:, :, 0] > 0.5, True), axis=1)

isFixed.shape


is_response = np.max(nn_out[:, -1, 1:], axis=1) > 0.6
loc = popvec(nn_out[:, -1, 1:])



loc.shape

criterion = (2*np.pi)/10


#checks trials that requiring repressing responses
if np.isnan(target_dirs[i]): 
    isDir = all((nn_out[:, 114:119, :].flatten() < 0.15))

test = nn_out[:, 114:119, :] < 0.15

#checks responses are coherent and in the correct direction
else:
    is_response = np.max(nn_out[:, -1, 1:]) > 0.6
    loc = popvec(nn_out[:, -1, 1:])
    dist = get_dist(loc, tar_dirs)        

    isDir = np.logical_and(dist < criterion, is_response)
is_correct = np.logical_and(isDir, isFixed)

is_correct