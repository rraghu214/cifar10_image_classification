import torch
import torch.nn as nn
import torch.nn.functional as F

def convolution_regular(in_channels, out_channels, kernels=3, stride=1, padding=1, dilation=1, dropout_val=0.05):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernels, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_val)
    )

def convolution_depthwise_separable(in_channels, out_channels, kernels=3, stride=1, padding=1, dilation=1, dropout_val=0.05):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernels, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_val) 
    )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # C1: Normal Conv
        self.conv1_1=convolution_regular(in_channels=3,out_channels=32,kernels=3,stride=1,padding=1, dilation=1) # 32 x 32 --> 32 x 32 | RF - 3
        self.conv1_2= convolution_regular(in_channels=32,out_channels=64,kernels=3,stride=2,padding=1, dilation=1) # 32 x 32 --> 16 x 16 | RF - 5
        self.conv1x1_1=convolution_regular(in_channels=64,out_channels=16,kernels=1,stride=1,padding=0, dilation=1) # 16 x 16 --> 16 x 16 | RF - 5

        #C2: Depthwise Separable Conv
        self.conv2_1=convolution_regular(in_channels=16,out_channels=32,kernels=3,stride=1,padding=1, dilation=1) # 16 x 16 --> 16 x 16 | RF - 9
        self.conv2_2=convolution_depthwise_separable(in_channels=32,out_channels=64,kernels=3,stride=2,padding=1, dilation=1) # 16 x 16 --> 8 x 8 | RF - 13
        self.conv1x1_2=convolution_regular(in_channels=64,out_channels=32,kernels=1,stride=1,padding=0, dilation=1) # 8 x 8 --> 8 x 8 | RF - 13

        # C3: Dilated Conv (receptive field booster)
        self.conv3_1=convolution_regular(in_channels=32,out_channels=64,kernels=3,stride=1,padding=1, dilation=1) # 8 x 8 --> 8 x 8 | RF - 21
        self.conv3_2=convolution_regular(in_channels=64,out_channels=128,kernels=3,stride=1,padding=2, dilation=2) # 8 x 8 --> 8 x 8 | RF - 37
        self.conv1x1_3=convolution_regular(in_channels=128,out_channels=32,kernels=1,stride=1,padding=0, dilation=1,dropout_val=0.1) # 8 x 8 --> 8 x 8 | RF - 37

        # C4: Normal Conv (downsample)
        self.conv4_1=convolution_regular(in_channels=32,out_channels=32,kernels=3,stride=1,padding=1, dilation=1) # 8 x 8 --> 8 x 8 | RF - 45
        self.conv4_2=convolution_regular(in_channels=32,out_channels=64,kernels=3,stride=2,padding=1, dilation=1) # 8 x 8 --> 4 x 4 | RF - 53
        self.conv1x1_4=convolution_regular(in_channels=64,out_channels=128,kernels=1,stride=1,padding=0, dilation=1, dropout_val=0.2) # 4 x 4 --> 4 x 4 | RF - 53

        # Output Block: GAP + FC
        self.gap     = nn.AdaptiveAvgPool2d(1) # 4 x 4 --> 1 x 1 | RF - 77
        self.fc      = nn.Linear(128, 10) # 1 x 1 --> 1 x 1 | RF - 77
        

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1x1_1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv1x1_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv1x1_3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv1x1_4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)        
        return F.log_softmax(x, dim=1)



def get_optimizer_and_scheduler(model, train_loader_len, EPOCHS,lr=0.05, momentum=0.9,):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=train_loader_len, epochs=EPOCHS, anneal_strategy='cos',pct_start=0.2,div_factor=10.0,final_div_factor=100.0 )
    return optimizer, scheduler