from abc import ABC, abstractmethod
from collections import defaultdict
from attrs import asdict
import numpy as np 
import torch
from torch import Tensor

def masked_MSE_Loss(nn_out: Tensor, nn_target: Tensor, mask: Tensor):
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


class BaseTrainer(ABC): 
    def __init__(self, config, from_checkpoint_dict): 
        assert not (config is None and from_checkpoint_dict is None), \
            'trainer must be initialized from training_config or from a checkpoint'

        self.config = asdict(config, recurse=False)
        self.cur_epoch = 0 
        self.cur_step = 0
        self.correct_data = defaultdict(list)
        self.loss_data = defaultdict(list)

        if from_checkpoint_dict is not None: 
            for name, value in from_checkpoint_dict.items(): 
                setattr(self, name, value)

        for name, value in self.config.items(): 
            setattr(self, name, value)

        self.seed_suffix = 'seed'+str(self.random_seed)

    def _check_model_training(self, duration=5): 
        min_run_elapsed = (self.cur_epoch >= self.min_run_epochs) or \
                            (self.cur_epoch == self.min_run_epochs-1 and self.cur_step == self.num_batches-1)
        if min_run_elapsed: 
            latest_perf = np.array([task_perf[-duration:] for task_perf in self.correct_data.values()])
            threshold_reached = np.all(latest_perf>self.checker_threshold)
            return threshold_reached 
        else: 
            return False

    def _log_step(self, task_type, frac_correct, loss): 
        self.correct_data[task_type].append(frac_correct)
        self.loss_data[task_type].append(loss)
    
    def _print_training_status(self, task_type):
        print('\n Training Step: ' + str(self.cur_step)+
                ' ----- Task Type: '+task_type+
                ' ----- Performance: ' + str(self.correct_data[task_type][-1])+
                ' ----- Loss: ' + "{:.3e}".format(self.loss_data[task_type][-1])+'\n'
                )

    def _record_session(self): 
        record_attrs = ['config', 'optimizer', 'scheduler', 'cur_epoch', 'cur_step', 'correct_data', 'loss_data']
        checkpoint_attrs = {}
        for attr in record_attrs: 
            checkpoint_attrs[attr]=getattr(self, attr)
        return checkpoint_attrs

    @abstractmethod
    def train(self): 
        pass
