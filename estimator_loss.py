import torch.nn as nn
import torch
import numpy as np


class EstimatorLoss(nn.Module):
    def __init__(self):
        super(EstimatorLoss, self).__init__()

    def forward(self,x,y):
        # simple l2 loss
        l = len(x)
        loss_sum = 0
        for i in range(l):
            loss_sum += torch.abs(x[i] - y[i])
        return loss_sum / l