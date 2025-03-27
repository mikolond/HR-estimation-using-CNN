import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtractorLoss(nn.Module):
    def __init__(self):
        super(ExtractorLoss, self).__init__()
    
    def PSD(self, x, f, fs):
        """
        Computes the power spectral density (PSD) of the signal x at frequency f.
        """
        indices = torch.arange(len(x), dtype=torch.float32, device=x.device)
        angles = 2 * torch.pi * f * indices / fs
        cos_terms = torch.sum(x * torch.cos(angles))
        sin_terms = torch.sum(x * torch.sin(angles))
        return cos_terms**2 + sin_terms**2
    
    def log_likelihoods(self, x, f_true, fs, sampling_f, f_range):
        """
        Computes log-likelihoods using softmax over the PSD spectrum.
        """
        f_min, f_max = f_range
        f_values = torch.arange(f_min, f_max + sampling_f, sampling_f, device=x.device)
        
        # Compute PSD values for all frequency candidates
        psd_values = torch.stack([self.PSD(x, f, fs) for f in f_values])
        
        # Apply softmax to get likelihoods
        likelihoods = F.softmax(psd_values, dim=0)
        log_likelihoods = torch.log(likelihoods + 1e-8)  # Avoid log(0)
        
        # Find the closest index to f_true and extract log-likelihood
        closest_idx = torch.argmin(torch.abs(f_values - f_true))
        return -log_likelihoods[closest_idx]  # Negative log-likelihood loss
    
    def forward(self, x, f_true, fs, deltas, sampling_f, f_range):
        """
        Computes the average negative log-likelihood loss over batch.
        """
        l = len(x)
        loss_sum = 0
        for i in range(l):
            loss_sum += self.log_likelihoods(x[i], f_true[i], fs[i], sampling_f, f_range)
        return loss_sum / l
