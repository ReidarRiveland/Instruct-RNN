
import numpy as np
import torch
from task_factory import TUNING_DIRS
def gpu_to_np(t):
    """removes tensor from gpu and converts to np.array""" 
    if t.get_device() >= 0: 
        t = t.detach().to('cpu').numpy()
    elif t.get_device() == -1: 
        t = t.detach().numpy()
    return t

def popvec(act_vec):
    """Population vector decoder that reads the orientation of activity in vector of activities
    Args:      
        act_vec (np.array): output tensor of neural network model; shape: (features, 1)
    Returns:
        float: decoded orientation of activity (in radians)
    """
    act_sum = np.sum(act_vec, axis=1)
    temp_cos = np.sum(np.multiply(act_vec, np.cos(TUNING_DIRS)), axis=1)/act_sum
    temp_sin = np.sum(np.multiply(act_vec, np.sin(TUNING_DIRS)), axis=1)/act_sum
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

    criterion = (2*np.pi)/10

    #checks response maintains fixataion
    isFixed = np.all(np.where(nn_target[:, :, 0] == 0.85, nn_out[:, :, 0] > 0.5, True), axis=1)

    #checks for repressed responses
    isRepressed = np.all(nn_out[:, 114:119, :].reshape(batch_size, -1) < 0.15, axis = 1)

    #checks is responses are in the correct direction 
    is_response = np.max(nn_out[:, -1, 1:]) > 0.6
    loc = popvec(nn_out[:, -1, 1:])
    dist = get_dist(loc, np.nan_to_num(target_dirs))        
    isDir = np.logical_and(dist < criterion, is_response)

    #checks if responses were correctly repressed or produced 
    correct_reponse = np.where(np.isnan(target_dirs), isRepressed, isDir)

    #checks if correct responses also maintained fixation 
    is_correct = np.logical_and(correct_reponse, isFixed)
    return is_correct