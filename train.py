import torch
from model import Extractor
from loss import ExtractorLoss


if __name__ == "__main__":


    model = Extractor()
    loss_fc = ExtractorLoss()
    
