import torch.nn as nn
import torch
import numpy as np

class ExtractorLoss(nn.Module):
    def __init__(self, n_s, n_f):
        super(ExtractorLoss, self).__init__()
        self.n = n_s
        
    
    def PSD(self, x, y):
        '''
        Returns power spectra density of given sequence inside list x
        params: x :sequence of numbers
                y :the ground true frequency
        return: 
        '''
        term1 = torch.zeros(self.n, 1)
        term2 = torch.zeros(self.n, 1)
        for i in range(n):
            term1[i] = torch.x[i]*np.cos(2 * np.pi* y)

    def forward(self, x, y):
        return self.loss(x, y)