from cog_rnns import InstructNet, SimpleNet
from nlp_models import BERT, SBERT
from data import DataStreamer
import torch
import torch.nn as nn
from trainer import mp_train, init_optimizer
from task import Task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tuning_dirs = torch.Tensor(Task.TUNING_DIRS)


