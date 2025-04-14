import torch.nn as nn
import torch
import numpy as np


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self,x,y):
        # pearsons correlation coefficient
        xy = torch.stack((x,y))
        pearson = torch.corrcoef(torch.stack((x,y)))[0,1]
        # print("pearson", pearson)
        out = torch.abs(1 - pearson)
        # print("pearson loss", out)
        return out