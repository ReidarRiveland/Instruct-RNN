from task import Task
tuning_dirs = Task.TUNING_DIRS

import numpy as np
from multiprocessing import Pool


def popvec(act_vec):
    """Population vector decoder that reads the orientation of activity in vector of activities
    Args:      
        act_vec (np.array): output tensor of neural network model; shape: (features, 1)
    Returns:
        float: decoded orientation of activity (in radians)
    """

    act_sum = np.sum(act_vec)
    temp_cos = np.sum(np.multiply(act_vec, np.cos(tuning_dirs)))/act_sum
    temp_sin = np.sum(np.multiply(act_vec, np.sin(tuning_dirs)))/act_sum
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

    isCorrect = np.empty(batch_size)
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

batch_size = 128
isCorrect = np.empty(batch_size)
criterion = (2*np.pi)/10

def check_criteria(nn_out, nn_target, target_dir): 
    #checks response maintains fixataion
    isFixed = all(np.where(nn_target[:, 0] == 0.85, nn_out[ :, 0] > 0.5, True))

    #checks trials that requiring repressing responses
    if np.isnan(target_dir): 
        isDir = all((nn_out[114:119, :].flatten() < 0.15))
    
    #checks responses are coherent and in the correct direction
    else:
        is_response = np.max(nn_out[ -1, 1:]) > 0.6
        loc = popvec(nn_out[ -1, 1:])
        dist = get_dist(loc, target_dir)        
        isDir = dist < criterion and is_response
    return isDir and isFixed
