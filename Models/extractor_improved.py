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
        elif params[i] == "CONV_dil": # Convolutional Layer with dilation
            layer += [nn.Conv2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5], dilation=params[i + 6], padding_mode="reflect")]    
            i += 6
        elif params[i] == "CONV1d_dil": # Convolutional Layer with dilation
            layer += [nn.Conv1d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5], dilation=params[i + 6], padding_mode="reflect")]    
            i += 6
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
        else:
            raise ValueError(f"Unknown layer type: {params[i]}")
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
        self.conv1 = create_layer(["DP2",0,"CONV", in_ch, ch1, c_k_size, c_st, pad,"MP" ,m_k_size, (2,2), "BN", ch1,"ELU", alpha_elu])
        # 82 x 55
        self.conv2 = create_layer(["DP",0.05,"CONV",ch1, ch1, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch1,"ELU", alpha_elu])
        # 54 x 37
        self.conv3 = create_layer(["DP",0.1,"CONV",ch1, ch2, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch2,"ELU", alpha_elu])
        # 26 x 19
        self.conv4 = create_layer(["DP2",0.2,"CONV",ch2, ch2, c_k_last_size, c_st, pad])
        # 15 x 10
        self.max_pool1 = create_layer(["MP",(10,7), (1,1),"BN", ch2,"ELU", alpha_elu])
        # 6 x 4
        # make 24 x bs
        self.conv5 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,24), (1,1), (1,0), (1,1)])
        self.conv6 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,24), (1,1), (2,0), (2,1)])
        self.conv7 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,24), (1,1), (3,0), (3,1)])
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3,1), stride=(2,1))

        self.conv8 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,3), (1,1), (1,0), (1,1)])
        self.conv9 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,3), (1,1), (2,0), (2,1)])
        self.conv10 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,3), (1,1), (3,0), (3,1)])
        self.conv11 = create_layer(["DP",0.2,"CONV_dil",ch2, ch2, (3,3), (1,1), (1,0), (1,1)])
        self.tran_conv1 = nn.ConvTranspose1d(ch2, ch2, kernel_size=(4), stride=(2), padding=(0), output_padding=(0))

        self.conv12 = create_layer(["DP",0.4,"CONV1d_dil",ch2, ch2, 3, 1, 1, 1])
        self.conv_last = create_layer(["DP",0.4,"CONV1d_dil",ch2, out_ch, 1, 1, 0,1])
        # 1 x 1 x bs



        self.init_weights()


    def forward(self, x):
        # normalization to [-1, 1]
        # x = x / 255 * 2 - 1
        x = self.bn_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(1,0,2)
        x = torch.unsqueeze(x, 0)
        x1 = self.conv5(x)
        x2 = self.conv6(x)
        x3 = self.conv7(x)
        x = torch.cat((x1,x2,x3), dim=3)
        x = self.max_pool2(x)
        x1 = self.conv8(x)
        x2 = self.conv9(x)
        x3 = self.conv10(x)
        x = torch.cat((x1,x2,x3), dim=3)
        x = self.conv11(x)
        x = x.squeeze(3)
        x = self.tran_conv1(x)
        x = self.conv12(x)
        x = self.conv_last(x)
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
            if type(layer) == nn.ConvTranspose1d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.Conv1d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)