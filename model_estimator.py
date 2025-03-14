import torch
import torch.nn as nn

def create_layer(params):
    layer = []
    i = 0
    while i <= len(params) - 1:
        if params[i] == "BN": # Batch Normalization
            layer += [nn.BatchNorm1d(params[i + 1])]
            i += 1
            # print("BN added, i=",i)
        elif params[i] == "CONV": # Convolutional Layer
            layer += [nn.Conv1d(params[i + 1], params[i + 2], kernel_size=params[i + 3], stride=params[i + 4], padding=params[i + 5])]
            i += 5
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
        conv_kernel = 16
        max_pool_kernel = 5


        alpha_elu = 1.0 # ELU alpha
        channels_1 = 200
        channels_2 = 128
        channels_3 = 128
        channels_4 = 128
        channels_5 = 128



        # 150 (1-dimensional, 1 channel)
        self.conv1 = create_layer(["BN",1,"DP",0.1,"CONV", 1, channels_1 , conv_kernel, 1, 0,"BN",channels_1,"ELU", alpha_elu])
        # 141
        self.conv2 = create_layer(["DP",0.1,"CONV", channels_1, channels_1 , conv_kernel, 1, 0,"BN",channels_1,"ELU", alpha_elu])
        # 132
        self.conv3 = create_layer(["DP",0.1,"CONV", channels_1, channels_1 , conv_kernel, 1, 0,"MP",max_pool_kernel, 1 ,"BN", channels_1, "ELU", alpha_elu])
        # 114
        self.conv4 = create_layer(["DP",0.1,"CONV", channels_1, channels_2 , conv_kernel, 1, 0,"BN",channels_2,"ELU", alpha_elu])
        # 105
        self.conv5 = create_layer(["DP",0.1,"CONV", channels_2, channels_2 , conv_kernel, 1, 0, "BN", channels_2, "ELU", alpha_elu])
        # 96
        self.conv6 = create_layer(["DP",0.1,"CONV", channels_2, channels_2 , conv_kernel, 1, 0, "MP",max_pool_kernel, 1 ,"BN", channels_2, "ELU", alpha_elu])
        # 78
        self.conv7 = create_layer(["DP",0.2,"CONV", channels_2, channels_3 , conv_kernel, 1, 0, "BN", channels_3, "ELU", alpha_elu])
        # 69
        self.conv8 = create_layer(["DP",0.2,"CONV", channels_3, channels_3 , conv_kernel, 1, 0, "BN", channels_3, "ELU", alpha_elu])
        # 60
        self.conv9 = create_layer(["DP",0.2,"CONV", channels_3, channels_3 , conv_kernel, 1, 0, "MP",max_pool_kernel, 1 ,"BN", channels_3, "ELU", alpha_elu])
        # 42
        self.conv10 = create_layer(["DP",0.3,"CONV", channels_3, channels_4 , conv_kernel, 1, 0, "BN", channels_4, "ELU", alpha_elu])
        # 33
        self.conv11 = create_layer(["DP",0.3,"CONV", channels_4, channels_5 , conv_kernel, 1, 0, "BN", channels_5, "ELU", alpha_elu])
        # 24
        self.conv_last = create_layer(["DP",0.5,"CONV", channels_5, 1, conv_kernel, 1, 0,"MP",max_pool_kernel, 1, "BN", 1, "ELU", alpha_elu])
        # 6
        self.ada_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)


        # # 150 (1-dimensional, 1 channel)
        # self.conv1 = create_layer(["BN",1,"DP",0.1,"CONV", 1, channels_1 , (3), 1, 0,"BN",channels_1,"ELU", alpha_elu])
        # # 141
        # self.conv2 = create_layer(["DP",0.1,"CONV", channels_1, channels_1 , conv_kernel, 1, 0,"BN",channels_1,"ELU", alpha_elu])
        # # 132
        # self.conv3 = create_layer(["DP",0.1,"CONV", channels_1, channels_1 , conv_kernel, 1, 0,"MP",max_pool_kernel, 1 ,"BN", channels_1, "ELU", alpha_elu])
        # # 114
        # self.conv4 = create_layer(["DP",0.1,"CONV", channels_1, channels_2 , conv_kernel, 1, 0,"BN",channels_2,"ELU", alpha_elu])
        # # 105
        # self.conv5 = create_layer(["DP",0.1,"CONV", channels_2, channels_2 , conv_kernel, 1, 0, "BN", channels_2, "ELU", alpha_elu])
        # # 96
        # self.conv6 = create_layer(["DP",0.1,"CONV", channels_2, channels_2 , conv_kernel, 1, 0, "MP",max_pool_kernel, 1 ,"BN", channels_2, "ELU", alpha_elu])
        # # 78
        # self.conv7 = create_layer(["DP",0.2,"CONV", channels_2, channels_3 , conv_kernel, 1, 0, "BN", channels_3, "ELU", alpha_elu])
        # # 69
        # self.conv8 = create_layer(["DP",0.2,"CONV", channels_3, channels_3 , conv_kernel, 1, 0, "BN", channels_3, "ELU", alpha_elu])
        # # 60
        # self.conv9 = create_layer(["DP",0.2,"CONV", channels_3, channels_3 , conv_kernel, 1, 0, "MP",max_pool_kernel, 1 ,"BN", channels_3, "ELU", alpha_elu])
        # # 42
        # self.conv10 = create_layer(["DP",0.3,"CONV", channels_3, channels_4 , conv_kernel, 1, 0, "BN", channels_4, "ELU", alpha_elu])
        # # 33
        # self.conv11 = create_layer(["DP",0.3,"CONV", channels_4, channels_5 , conv_kernel, 1, 0, "BN", channels_5, "ELU", alpha_elu])
        # # 24
        # self.conv_last = create_layer(["DP",0.5,"CONV", channels_5, 1, conv_kernel, 1, 0,"MP",max_pool_kernel, 1, "BN", 1, "ELU", alpha_elu])
        # # 6
        # self.ada_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.init_weights()


    def forward(self, x):
        # normalization
        x = x - torch.mean(x, dim=-1).unsqueeze(dim=-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv_last(x)
        x = self.ada_avg_pool(x)

        return x

    def init_weights(self):
        print("Initializing weights")
        for layer in self.modules():
            if type(layer) == nn.Conv1d:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.BatchNorm1d:
                nn.init.uniform_(layer.weight, 0, 0.1)
                nn.init.zeros_(layer.bias)
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)