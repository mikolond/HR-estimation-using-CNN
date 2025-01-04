import torch
import torch.nn as nn

def create_layer(params):
    layer = []
    i = 0
    while i <= len(params) - 1:
        if params[i] == "BN": # Batch Normalization
            layer += [nn.BatchNorm2d(params[i + 1])]
            i += 1
            print("BN added, i=",i)
        elif params[i] == "CONV": # Convolutional Layer
            layer += [nn.Conv2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5])]
            i += 5
            print("CONV added, i=",i)
        elif params[i] == "MP": # Max Pooling
            layer += [nn.MaxPool2d(kernel_size=params[i + 1], stride=params[i + 2])]
            i += 2
            print("MP added, i=",i)
        elif params[i] == "ELU": # Exponential Linear Unit
            layer += [nn.ELU(params[i + 1])]
            i += 1
            print("ELU added, i=",i)
        elif params[i] == "DP2": # Dropout
            layer += [nn.Dropout2d(params[i + 1])]
            i += 1
            print("DP2 added, i=",i)
        elif params[i] == "DP":
            layer += [nn.Dropout(params[i + 1])]
            i += 1
            print("DP added, i=",i)
        i += 1
    
    return nn.Sequential(*layer)



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # convolution parameters
        c_k_size = (15,10) # convolution kernel size
        c_k_last_size = (14,10)
        c_st = (1,1) # convolution stride
        pad = (1,1) # padding
        in_ch = 3 # input channels
        ch = 64 # channels inside the network
        out_ch = 1 # output channels


        m_k_size = (2,2) # max pooling kernel size
        m_st = (2,2) # max pooling stride

        alpha_elu = 1.0 # ELU alpha

        self.conv1 = create_layer(["BN", in_ch,"DP2",0.05,"CONV",in_ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        self.conv2 = create_layer(["DP",0,"CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        self.conv3 = create_layer(["DP",0,"CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        self.conv4 = create_layer(["DP2",0,"CONV",ch, out_ch, c_k_last_size, c_st, pad,"MP",m_k_size, m_st, "BN", out_ch,"ELU", alpha_elu])

        self.init_weights()


    def forward(self, x):
        # normalization to [-1, 1]
        # x = x / 255 * 2 - 1
        x = self.conv1(x)
        print("forward x shape, conv1",x.shape)
        x = self.conv2(x)
        print("forward x shape, conv2",x.shape)
        x = self.conv3(x)
        print("forward x shape, conv3",x.shape)
        x = self.conv4(x)
        print("forward x shape, conv4",x.shape)
        return x

    def init_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
