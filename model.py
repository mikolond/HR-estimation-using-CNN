import torch
import torch.nn as nn

def create_layer(params):
    layer = []
    for i in range(len(params) - 1):
        if params[i] == "BN": # Batch Normalization
            layer += [nn.BatchNorm2d(params[i + 1])]
            i += 1
        elif params[i] == "CONV": # Convolutional Layer
            layer += [nn.Conv2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5])]
            i += 5
        elif params[i] == "MP": # Max Pooling
            layer += [nn.MaxPool2d(kernel_size=params[i + 1], stride=params[i + 2])]
            i += 2
        elif params[i] == "ELU": # Exponential Linear Unit
            layer += [nn.ELU(params[i + 1])]
            i += 1
        elif params[i] == "DP": # Dropout
            layer += [nn.Dropout2d(params[i + 1])]
            i += 1
    
    return nn.Sequential(*layer)



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # convolution parameters
        c_k_size = (15,10) # convolution kernel size
        c_st = (1,1) # convolution stride
        pad = (1,1) # padding
        in_ch = 3 # input channels
        ch = 64 # channels inside the network
        out_ch = 1 # output channels

        m_k_size = (2,2) # max pooling kernel size
        m_st = (2,2) # max pooling stride

        alpha_elu = 1.0 # ELU alpha

        self.conv1 = create_layer(["BN", in_ch,"CONV",in_ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        self.conv2 = create_layer(["CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        self.conv3 = create_layer(["CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        self.conv4 = create_layer(["CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", out_ch,"ELU", alpha_elu])

        self.init_weights()

    def forward_single(self, x):
        # normalization to [-1, 1]
        x = x / x.max()*2 - 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x, n):
        output_vector = torch.zeros(n, 1)
        for i in range(n):
            output_vector[i] = self.forward_single(x[i])
        return output_vector

    def init_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
