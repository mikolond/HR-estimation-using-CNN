import torch.nn as nn
import torch
import numpy as np


class EstimatorLoss(nn.Module):
    def __init__(self):
        super(EstimatorLoss, self).__init__()

    def forward(self,x,y):
        # simple l2 loss
        return torch.mean(torch.abs(x-y))