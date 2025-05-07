import torch
import torch.nn as nn

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
        elif params[i] == "CONV_dil": # Convolutional Layer
            layer += [nn.Conv2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5], dilation=params[i + 6])]
            i += 6
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
        ch = 64 # channels inside the network
        out_ch = 1 # output channels


        m_k_size = (15,10) # max pooling kernel size
        m_st = (1,1) # max pooling stride

        alpha_elu = 1.0 # ELU alpha
        # 192 x 128
        self.conv1 = create_layer(["BN",in_ch,"DP2",0.05,"CONV", in_ch, ch, c_k_size, c_st, pad,"MP" ,m_k_size, (2,2), "BN", ch,"ELU", alpha_elu])
        # 82 x 55
        self.conv2 = create_layer(["DP",0.05,"CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        # 54 x 37
        self.conv3 = create_layer(["DP",0.2,"CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "BN", ch,"ELU", alpha_elu])
        # 26 x 19
        self.conv4 = create_layer(["DP2",0.2,"CONV",ch, ch, (12,8), c_st, pad,"MP",(12,8), m_st, "BN", ch,"ELU", alpha_elu])
        # 4 x 5
        # make 88 x 1 vector
        # all 12 x 1 vectors in batch concatenated to 12 x batch_size
        # 2d convolution 

        # batch_size x 12
        self.conv5 = create_layer(["DP",0.3,"CONV",ch, ch, (11,12), (1,1), (5,0), "ELU", alpha_elu])
        # batch_size x 9
        self.conv6 = create_layer(["DP",0.5,"CONV",ch, ch, (9,9), (1,1), (4,0), "ELU", alpha_elu])
        # batch_size x 1
        self.conv_last = create_layer(["DP",0.5,"CONV",ch, out_ch, 1, 1, 0])
        # batch_size x 1


        self.init_weights()


    def forward(self, x):
        # normalization to [-1, 1]
        # x = x / 255 * 2 - 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print("after conv3:",x.shape)
        x = self.conv4(x)
        # print("after conv4:",x.shape)
        x = x.reshape(1,x.shape[1],x.shape[0],20)
        # print("after conv4 cat:",x.shape)
        x = self.conv5(x)
        # print("after conv5:",x.shape)
        x = self.conv6(x)
        # print("after conv6:",x.shape)
        x = self.conv_last(x)
        # print("after conv_last:",x.shape)
        return x

    def init_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.BatchNorm2d:
                nn.init.uniform_(layer.weight, 0, 0.1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)