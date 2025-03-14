import torch

cudnn_enabled = torch.backends.cudnn.enabled
cudnn_version = torch.backends.cudnn.version() if cudnn_enabled else 'cuDNN not enabled'
print(f"cuDNN enabled: {cudnn_enabled}")
print(f"cuDNN version: {cudnn_version}")