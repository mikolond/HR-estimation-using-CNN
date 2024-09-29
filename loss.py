import torch.nn as nn
import torch
import numpy as np

class ExtractorLoss(nn.Module):
    def __init__(self):
        super(ExtractorLoss, self).__init__()
        
    
    def PSD(self, x, f_true, fs):
        '''
        Returns power spectra density of given sequence inside list x
        params: x :list of numbers
                f_true : ground true frequency
                fs : sampling frequency
        return: 
        '''
        term1 = 0
        term2 = 0
        for i in range(len(x)):
            term1 += x[i] * np.cos(2 * np.pi * f_true * i / fs)
            term2 += x[i] * np.sin(2 * np.pi * f_true * i / fs)
        return term1**2 + term2**2
    
    def SNR(self, x, f_true, fs, delta, f_range):
        '''
        Returns signal to noise ratio of given sequence inside list x
        params: x :list of numbers
                f_true : ground true frequency
                delta : offset from the true frequency
                f_range : list(f_min, f_max) all possible frequencies
        return: 
        '''
        f_min, f_max = f_range
        f_wanted =list(range(f_true - delta, f_true + delta + 1))
        f_unwanted = list(range(f_min, f_true - delta)) + list(range(f_true + delta + 1, f_max+1))
        term1 = 0
        term2 = 0
        for f in f_wanted:
            term1 += self.PSD(x, f, fs)
        for f in f_unwanted:
            term2 += self.PSD(x, f, fs)

        return 10 * torch.log10(term1 / term2)


    def forward(self, x, f_true, fs, delta, f_range):
        return - self.SNR(x, f_true, fs, delta, f_range)