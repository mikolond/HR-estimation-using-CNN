import torch
import torch.nn as nn
import torch.nn.functional as F

def make_3d(x):
    # convert 3d tensor to 2d tensor
    x = x.permute(1, 0, 2, 3)
    x = x.unsqueeze(0)
    return x

def make_2d(x):
    # convert 2d tensor to 3d tensor
    x = x.squeeze(0)
    x = torch.permute(x, (1, 0, 2, 3))
    return x


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # convolution parameters
        c_k_size = (3,15,10) # convolution kernel size
        c_k_last_size = (3,12,10)
        c_st = (1,1,1) # convolution stride
        pad = (1,0,0) # padding
        in_ch = 3 # input channels
        ch1 = 64 # channels inside the network
        ch2 = 64
        out_ch = 1 # output channels


        m_k_size = (15,10) # max pooling kernel size
        m_st = (1,1) # max pooling stride

        alpha_elu = 1.0 # ELU alpha
        self.ada_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(192, 128))

        # 192 x 128
        self.conv1 = nn.Conv3d(in_ch, ch1, kernel_size=c_k_size, stride=c_st, padding=pad)
        self.maxpool1 = nn.MaxPool2d(kernel_size=m_k_size, stride=(2,2))
        # 89 x 60
        self.conv2 = nn.Conv3d(ch1, ch1, kernel_size=c_k_size, stride=c_st, padding=pad)
        self.maxpool2 = nn.MaxPool2d(kernel_size=m_k_size, stride=m_st)
        # 38 x 26
        self.conv3 = nn.Conv3d(ch1, ch2, kernel_size=c_k_size, stride=c_st, padding=pad)
        self.maxpool3 = nn.MaxPool2d(kernel_size=m_k_size, stride=m_st)
        # 13 x 9
        self.conv4 = nn.Conv3d(ch2, ch2, kernel_size=c_k_last_size, stride=c_st, padding=pad)
        self.maxpool4 = nn.MaxPool2d(kernel_size=m_k_size, stride=m_st)
        # 1 x 1
        self.conv5 = nn.Conv2d(ch2, out_ch, kernel_size=(1,1), stride=(1,1), padding=0)


        self.init_weights()
    


    def forward(self, x):
        # normalization to [-1, 1]
        x = x / 255 * 2 - 1
        self.ada_avg_pool2d(x)
        # print("afetr ada_avg_pool2d", x.shape)
        x = make_3d(x)
        # print("after make3d", x.shape)
        x = self.conv1(x)
        # print("after conv1", x.shape)
        x = make_2d(x)
        # print("after make_2d", x.shape)
        x = F.elu(self.maxpool1(F.dropout2d(x,p = 0, training=self.training)))
        # print("after maxpool1", x.shape)

        x = make_3d(x)
        x = self.conv2(x)
        x = make_2d(x)
        x = F.elu(self.maxpool2(F.dropout(x,p = 0, training=self.training)))
        # print("after maxpool2", x.shape)

        x = make_3d(x)
        x = self.conv3(x)
        x = make_2d(x)
        x = F.elu(self.maxpool3(F.dropout(x,p = 0, training=self.training)))
        # print("after maxpool3", x.shape)

        x = make_3d(x)
        x = self.conv4(x)
        x = make_2d(x)
        x = F.elu(self.maxpool4(F.dropout2d(x,p = 0.2, training=self.training)))
        # print("after maxpool4", x.shape)

        x = self.conv5(F.dropout(x,p = 0.5, training=self.training))
        # print("after conv5", x.shape)

        return x

    def init_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Conv3d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.BatchNorm2d:
                nn.init.normal_(layer.weight, 0, 0.1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)