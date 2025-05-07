import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layer(params):
    layer = []
    i = 0
    while i <= len(params) - 1:
        if params[i] == "BN": # Batch Normalization
            layer += [nn.BatchNorm1d(params[i + 1])]
            i += 1
            # print("BN added, i=",i)
        elif params[i] == "CONV": # Convolutional Layer
            layer += [nn.Conv1d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5], dilation=params[i + 6])]
            i += 6
            # print("CONV added, i=",i)
        elif params[i] == "MP": # Max Pooling
            layer += [nn.MaxPool1d(kernel_size=params[i + 1], stride=params[i + 2])]
            i += 2
            # print("MP added, i=",i)
        elif params[i] == "ELU": # Exponential Linear Unit
            layer += [nn.ELU(params[i + 1])]
            i += 1
            # print("ELU added, i=",i)
        elif params[i] == "DP":
            layer += [nn.Dropout(params[i + 1])]
            i += 1
            # print("DP added, i=",i)
        elif params[i] == "LIN":
            layer += [nn.Linear(params[i + 1], params[i + 2])]
            i += 2
            # print("LIN added, i=",i)
        i += 1
    
    return nn.Sequential(*layer)



class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        # convolution parameters
        self.bn_input = nn.BatchNorm1d(1)
        self.conv1 = create_layer(["CONV", 1, 64 , 3, 1, 0, 1,"BN",64,"ELU", 1.0])
        self.conv2 = create_layer(["CONV", 64, 64 , 6, 1, 0, 1,"BN",64,"ELU", 1.0])
        self.conv3 = create_layer(["CONV", 64, 64 , 12, 1, 0, 1,"MP", 12, 2, "BN", 64, "ELU", 1.0])
        self.conv4 = create_layer(["CONV", 64, 64 , 24, 1, 0, 1,"BN",64,"ELU", 1.0])
        self.conv5 = create_layer(["CONV", 64, 64 , 50, 1, 0, 1,"BN",64,"ELU", 1.0])
        self.conv6 = create_layer(["CONV", 64, 64 , 50, 1, 0, 1,"BN",64,"ELU", 1.0])
        self.conv7 = create_layer(["CONV", 64, 10 , 15, 1, 0, 1,"ELU", 1.0])
        self.ada_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.init_weights()


    def forward(self, x):
        # normalization to [-1, 1]
        x = x - torch.mean(x, dim=-1).unsqueeze(dim=-1)
        
        x = self.bn_input(x)
        x = self.conv1(F.dropout(x,p = 0, training=self.training))
        x = self.conv2(F.dropout(x,p = 0.1, training=self.training))
        x = self.conv3(F.dropout(x,p = 0.1, training=self.training))
        x = self.conv4(F.dropout(x,p = 0.1, training=self.training))
        x = self.conv5(F.dropout(x,p = 0.1, training=self.training))
        x = self.conv6(F.dropout(x,p = 0.3, training=self.training))
        x = self.conv7(F.dropout(x,p = 0.5, training=self.training))
        x = x.resize(x.shape[0], x.shape[1])
        freqs = torch.arange(0, 240/60,240/60/x.shape[1], device=x.device, requires_grad=True)

        freqs_N = freqs.unsqueeze(0).repeat(x.shape[0], 1)


        x = x * freqs_N
        x = x.sum(dim=1)


        return x

    def init_weights(self):
        print("Initializing weights")
        for layer in self.modules():
            if type(layer) == nn.Conv1d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.BatchNorm1d:
                nn.init.normal_(layer.weight, 0, 0.1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)