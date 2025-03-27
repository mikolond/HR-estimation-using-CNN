import torch.nn as nn
import torch
import numpy as np


class EstimatorLoss(nn.Module):
    def __init__(self):
        super(EstimatorLoss, self).__init__()

    def forward(self,x,y):
        # simple l2 loss
        print(f"x: {x.shape}, y: {y.shape}")
        print(f"x: {x}, y: {y}")
        print(f"abs: {torch.abs(x-y)}")
        return torch.sum(torch.abs(x-y))/len(x)