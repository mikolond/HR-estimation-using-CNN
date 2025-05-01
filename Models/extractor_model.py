import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layer(params):
    layer = []
    i = 0
    while i <= len(params) - 1:
        if params[i] == "BN": # Batch Normalization
            layer += [nn.BatchNorm2d(params[i + 1])]
            i += 1
            # print("BN added, i=",i)
        elif params[i] == "CONV": # Convolutional Layer
            layer += [nn.Conv2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5])]
            i += 5
            # print("CONV added, i=",i)
        elif params[i] == "MP": # Max Pooling
            layer += [nn.MaxPool2d(kernel_size=params[i + 1], stride=params[i + 2])]
            i += 2
            # print("MP added, i=",i)
        elif params[i] == "ELU": # Exponential Linear Unit
            layer += [nn.ELU(params[i + 1])]
            i += 1
            # print("ELU added, i=",i)
        elif params[i] == "DP2": # Dropout
            layer += [nn.Dropout2d(params[i + 1])]
            i += 1
            # print("DP2 added, i=",i)
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



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # convolution parameters
        c_k_size = (15,10) # convolution kernel size
        c_k_last_size = (12,10)
        c_st = (1,1) # convolution stride
        pad = (0,0) # padding
        in_ch = 3 # input channels
        ch1 = 64 # channels inside the network
        ch2 = 128
        out_ch = 1 # output channels


        m_k_size = (15,10) # max pooling kernel size
        m_st = (1,1) # max pooling stride

        alpha_elu = 1.0 # ELU alpha
        self.ada_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(192, 128))

        # 192 x 128
        self.bn_input = nn.BatchNorm2d(in_ch)
        self.conv1 = create_layer(["CONV", in_ch, ch1, c_k_size, c_st, pad,"MP" ,m_k_size, (2,2), "BN", ch1,"ELU", alpha_elu])
        # 89 x 60
        self.conv2 = create_layer(["CONV",ch1, ch1, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch1,"ELU", alpha_elu])
        # 38 x 26
        self.conv3 = create_layer(["CONV",ch1, ch2, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch2,"ELU", alpha_elu])
        # 13 x 9
        self.conv4 = create_layer(["CONV",ch2, ch2, c_k_last_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch2,"ELU", alpha_elu])
        # 1 x 1
        self.conv5 = create_layer(["CONV",ch2, out_ch, 1, 1, 0])


        self.init_weights()


    def forward(self, x):
        # normalization to [-1, 1]
        # x = x / 255 * 2 - 1
        self.ada_avg_pool2d(x)
        x = self.bn_input(x)
        x = self.conv1(F.dropout2d(x,p = 0, training=self.training))
        x = self.conv2(F.dropout(x,p = 0, training=self.training))
        x = self.conv3(F.dropout(x,p = 0, training=self.training))
        x = self.conv4(F.dropout2d(x,p = 0.2, training=self.training))
        x = self.conv5(F.dropout(x,p = 0.5, training=self.training))
        # normalize x from [min(x), max(x)] to [0,1]
        # x = x - x.min()
        # x = x / (x.max() - x.min())
        return x

    def init_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.BatchNorm2d:
                nn.init.normal_(layer.weight, 0, 0.1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)