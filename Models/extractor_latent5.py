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
        elif params[i] == "CONV_dil": # Convolutional Layer
            layer += [nn.Conv2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5], dilation=params[i + 6])]
            i += 6
            # print("CONV added, i=",i)
        elif params[i] == "CONV_trans": # Transposed convolutional Layer
            layer += [nn.ConvTranspose2d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5], output_padding=params[i + 6])]
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
        i += 1
    
    return nn.Sequential(*layer)



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # convolution parameters
        c_k_size = (15,10) # convolution kernel size
        c_st = (1,1) # convolution stride
        pad = (0,0) # padding
        in_ch = 3 # input channels
        ch = 64 # channels inside the network
        out_ch = 1 # output channels


        m_k_size = (15,10) # max pooling kernel size
        m_st = (1,1) # max pooling stride

        alpha_elu = 1.0 # ELU alpha
        # 192 x 128
        self.bn_input = nn.BatchNorm2d(in_ch)
        self.conv1 = create_layer(["CONV", in_ch, ch, c_k_size, c_st, pad,"MP" ,m_k_size, (2,2),"ELU", alpha_elu])
        # 82 x 55
        self.conv2 = create_layer(["CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "ELU", alpha_elu])
        # 54 x 37
        self.conv3 = create_layer(["CONV",ch, ch, c_k_size, c_st, pad,"MP",m_k_size, m_st, "ELU", alpha_elu])
        # 26 x 19
        self.conv4 = create_layer(["CONV",ch, ch, (12,8), c_st, pad,"MP",(12,8), m_st, "ELU", alpha_elu])
        # 4 x 5
        # all 20 x 1 vectors in batch concatenated to 20 x batch_size
        # 2d convolution 

        # batch_size x 12
        self.conv5 = create_layer(["CONV_dil",ch, ch, (3,3), (1,1), (1,0), (1,1), "ELU", alpha_elu])
        self.conv6 = create_layer(["CONV_dil",ch, ch, (3,3), (1,1), (2,0), (2,1),"ELU", alpha_elu])
        self.conv7 = create_layer(["CONV_dil",ch, ch, (3,3), (1,1), (3,0), (3,1),"ELU", alpha_elu])

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0))

        self.conv8 = create_layer(["CONV_dil",ch, ch, (3,3), (1,1), (1,0), (1,1),"ELU", alpha_elu])
        self.conv9 = create_layer(["CONV_dil",ch, ch, (3,3), (1,1), (2,0), (2,1),"ELU", alpha_elu])
        self.conv10 = create_layer(["CONV_dil",ch, ch, (3,3), (1,1), (3,0), (3,1),"ELU", alpha_elu])

        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0))

        self.conv11 = create_layer(["DP",0.1,"CONV_dil",ch, ch, (3,3), (1,1), (1,0), (1,1),"ELU", alpha_elu])
        self.conv12 = create_layer(["DP",0.1,"CONV_dil",ch, ch, (3,3), (1,1), (2,0), (2,1),"ELU", alpha_elu])
        self.conv13 = create_layer(["DP",0.1,"CONV_dil",ch, ch, (3,3), (1,1), (3,0), (3,1),"ELU", alpha_elu])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0))

        self.conv_trans1 = create_layer(["CONV_trans",ch, ch, (4,1), (2,1), (0,0), (0,0), "ELU", alpha_elu])
        self.conv14 = create_layer(["CONV",ch, ch, (3,6), (1,2), (1,0), "ELU", alpha_elu])

        self.conv_trans2 = create_layer(["CONV_trans",ch, ch, (4,1), (2,1), (0,0), (0,0), "ELU", alpha_elu])
        self.conv15 = create_layer(["CONV",ch, ch, (3,6), (1,2), (1,0), "ELU", alpha_elu])

        self.conv_trans3 = create_layer(["CONV_trans",ch, ch, (4,1), (2,1), (0,0), (0,0), "ELU", alpha_elu])
        self.conv16 = create_layer(["CONV",ch, ch, (3,5), (1,2), (1,0),"ELU", alpha_elu])

        self.maxpool4 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,0))
        # self.conv17 = create_layer(["CONV",ch, ch, (3,3), (1,1), (1,0), "ELU", alpha_elu])

        self.conv_last = create_layer(["CONV",ch, out_ch, 1, 1, 0])




        self.init_weights()


    def forward(self, x):
        # normalization to [-1, 1]
        # x = x / 255 * 2 - 1
        x = self.bn_input(x)
        x = self.conv1(F.dropout2d(x,p = 0, training=self.training))
        x = self.conv2(F.dropout(x,p = 0, training=self.training))
        x = self.conv3(F.dropout(x,p = 0, training=self.training))
        # print("after conv3:",x.shape)
        x = self.conv4(F.dropout2d(x,p = 0.1, training=self.training))
        # print("after conv4:",x.shape)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(1,0,2)
        x = torch.unsqueeze(x, 0)
        # print("after flatten and dimension unsqueeze:",x.shape)
        x1 = self.conv5(F.dropout(x,p = 0, training=self.training))
        # print("after conv5:",x1.shape)
        x2 = self.conv6(F.dropout(x,p = 0, training=self.training))
        # print("after conv6:",x2.shape)
        x3 = self.conv7(F.dropout(x,p = 0, training=self.training))
        # print("after conv7:",x3.shape)
        x = torch.cat((x1,x2,x3),3)
        # print("after conv5,6,7 cat:",x.shape)
        x = self.maxpool1(x)
        # print("after maxpool1:",x.shape)

        x1 = self.conv8(F.dropout(x,p = 0, training=self.training))
        # print("after conv8:",x1.shape)
        x2 = self.conv9(F.dropout(x,p = 0, training=self.training))
        # print("after conv9:",x2.shape)
        x3 = self.conv10(F.dropout(x,p = 0, training=self.training))
        # print("after conv10:",x3.shape)
        x = torch.cat((x1,x2,x3),3)
        # print("after conv8,9,10 cat:",x.shape)
        x = self.maxpool2(x)
        # print("after maxpool2:",x.shape)

        x1 = self.conv11(F.dropout(x,p = 0, training=self.training))
        # print("after conv11:",x1.shape)
        x2 = self.conv12(F.dropout(x,p = 0, training=self.training))
        # print("after conv12:",x2.shape)
        x3 = self.conv13(F.dropout(x,p = 0, training=self.training))
        # print("after conv13:",x3.shape)
        x = torch.cat((x1,x2,x3),3)
        # print("after conv11,12,13 cat:",x.shape)
        x = self.maxpool3(x)
        # print("after maxpool3:",x.shape)

        x = self.conv_trans1(F.dropout(x,p = 0, training=self.training))
        # print("after conv_trans1:",x.shape)
        x = self.conv14(F.dropout(x,p = 0, training=self.training))
        # print("after conv14:",x.shape)

        x = self.conv_trans2(F.dropout(x,p = 0.1, training=self.training))
        # print("after conv_trans2:",x.shape)
        x = self.conv15(F.dropout(x,p = 0.1, training=self.training))
        # print("after conv15:",x.shape)

        x = self.conv_trans3(F.dropout(x,p = 0.2, training=self.training))
        # print("after conv_trans3:",x.shape)
        x = self.conv16(F.dropout(x,p = 0.2, training=self.training))
        # print("after conv16:",x.shape)
        x = self.maxpool4(x)
        # x = self.conv17(F.dropout(x,p = 0.3, training=self.training))
        # print("after maxpool4:",x.shape)


        x = self.conv_last(F.dropout(x,p = 0.5, training=self.training))
        # print("after conv_last:",x.shape)
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
            if type(layer) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
