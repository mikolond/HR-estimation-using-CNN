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
    
    def SNR(self, x, f_true, fs, delta, sampling_f, f_range):
        '''
        Returns signal to noise ratio of given sequence inside list x
        params: x :list of numbers
                f_true : ground true frequency
                delta : offset from the true frequency
                f_range : list(f_min, f_max) all possible frequencies
        return: 
        '''
        f_min, f_max = f_range
        # list of wanted and unwanted frequencies
        f_wanted =list(torch.arange(f_true - delta, f_true + delta + sampling_f, sampling_f))
        print("f_min",f_min, "f_true",f_true, "delta",delta, "f_max",f_max)
        f_unwanted = list(torch.arange(f_min, f_true - delta, sampling_f)) + list(torch.arange(f_true + delta + sampling_f, f_max + sampling_f, sampling_f))
        term1 = 0
        term2 = 0
        for f in f_wanted:
            term1 += self.PSD(x, f, fs)/len(f_wanted)
        for f in f_unwanted:
            term2 += self.PSD(x, f, fs)/len(f_unwanted)

        return 10 * torch.log10(term1 / term2)


    def forward(self, x, f_true, fs, delta, sampling_f, f_range):
        l = len(x)
        loss_sum = 0
        for i in range(l):
            loss_sum -= self.SNR(x[i], f_true[i], fs, delta, sampling_f, f_range)
        return loss_sum / l