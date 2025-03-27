import torch.nn as nn
import torch

class ExtractorLoss(nn.Module):
    def __init__(self):
        super(ExtractorLoss, self).__init__()
        
    
    def PSD(self, x, f_true, fs):
        """
        Returns power spectral density of given sequence x.
        """
        indices = torch.arange(len(x), dtype=torch.float32, device=x.device)
        angles = 2 * torch.pi * f_true * indices / fs
        cos_terms = torch.sum(x * torch.cos(angles))
        sin_terms = torch.sum(x * torch.sin(angles))
        return cos_terms**2 + sin_terms**2
    
    def SNR(self, x, f_true, fs, delta, sampling_f, f_range):
        """
        Returns signal-to-noise ratio of given sequence x.
        """
        f_min, f_max = f_range
        # Generate wanted and unwanted frequency ranges as tensors
        f_wanted = torch.arange(f_true - delta, f_true + delta + sampling_f, sampling_f, device=x.device)
        f_unwanted_1 = torch.arange(f_min, max(f_true - delta - sampling_f, f_min), sampling_f, device=x.device)
        f_unwanted_2 = torch.arange(f_true + delta + sampling_f, f_max + sampling_f, sampling_f, device=x.device)
        f_unwanted = torch.cat((f_unwanted_1, f_unwanted_2))


        # Compute PSD values for wanted and unwanted frequencies
        psd_wanted = torch.stack([self.PSD(x, f, fs) for f in f_wanted])
        psd_unwanted = torch.stack([self.PSD(x, f, fs) for f in f_unwanted])

        # Sum PSD values
        term1 = torch.sum(psd_wanted)
        term2 = torch.sum(psd_unwanted)

        loss = 10 * torch.log10(term2 / term1)
        return loss


    def forward(self, x, f_true, fs, deltas, sampling_f, f_range):
        l = len(x)
        loss_sum = 0
        for i in range(l):
            loss_sum = self.SNR(x[i], f_true[i], fs[i], deltas[i], sampling_f, f_range)
        return loss_sum / l