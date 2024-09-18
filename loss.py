import torch.nn as nn


class ExtractorLoss(nn.Module):
    def __init__(self):
        super(ExtractorLoss, self).__init__()
        

    def forward(self, x, y):
        return self.loss(x, y)