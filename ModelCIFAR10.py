import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)


        self.Trans_1_1 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=1)
        # self.bn4   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn5   = nn.BatchNorm2d(128)

        # self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.bn5   = nn.BatchNorm2d(128)

        self.conv1x1 = nn.Conv2d(128, 10, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # 3 -> 32
        x = F.relu(self.bn2(self.conv2(x)))   # 32 -> 64
        x = F.relu(self.bn3(self.conv3(x)))   # 64 -> 128
        x = self.Trans_1_1(x)               # 128 -> 32
        x = F.relu(self.bn4(self.conv4(x)))   # 32 -> 64
        x = F.relu(self.bn5(self.conv5(x)))   # 64 -> 128

        x = self.conv1x1(x)                   # 128 -> 10
        x = self.gap(x)                       # GAP
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def get_optimizer_and_scheduler(model, train_loader_len, EPOCHS,lr=0.01, momentum=0.9,):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=train_loader_len, epochs=EPOCHS, anneal_strategy='cos',pct_start=0.2,div_factor=10.0,final_div_factor=100.0 )
    return optimizer, scheduler