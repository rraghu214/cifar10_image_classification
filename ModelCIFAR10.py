import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # C1: Normal Conv
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # C2: Depthwise Separable Conv
        self.dw2   = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False)  # depthwise
        self.pw2   = nn.Conv2d(32, 64, 1, stride=1, bias=False)  # pointwise
        self.bn2   = nn.BatchNorm2d(64)

        # C3: Dilated Conv (receptive field booster)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=2, dilation=2, bias=False)
        self.bn3   = nn.BatchNorm2d(128)

        # C4: Normal Conv (downsample)
        self.conv4 = nn.Conv2d(128, 96, 3, stride=2, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(96)

        # Final 1x1 conv + GAP
        self.conv1x1 = nn.Conv2d(96, 10, 1, bias=False)
        self.gap     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))             # C1
        x = F.relu(self.bn2(self.pw2(self.dw2(x))))     # C2 (DW Sep)
        x = F.relu(self.bn3(self.conv3(x)))             # C3 (Dilated)
        x = F.relu(self.bn4(self.conv4(x)))             # C4
        x = self.conv1x1(x)                             # 1x1 conv
        x = self.gap(x)                                 # GAP
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)



def get_optimizer_and_scheduler(model, train_loader_len, EPOCHS,lr=0.01, momentum=0.9,):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=train_loader_len, epochs=EPOCHS, anneal_strategy='cos',pct_start=0.2,div_factor=10.0,final_div_factor=100.0 )
    return optimizer, scheduler