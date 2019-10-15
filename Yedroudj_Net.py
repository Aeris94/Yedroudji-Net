import torch
import torch.nn as nn
import torch.nn.functional as F
import SRM_Filters

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(SRM_conv2d, self).__init__()
     
        self.weight = nn.Parameter(torch.Tensor(30, 1, 5, 5))
        self.bias = nn.Parameter(torch.Tensor(30))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data = torch.FloatTensor(SRM_Filters.SRM_Filters)
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, bias = self.bias, stride=1, padding=2)

class Yedroudj_Net(nn.Module):
    def __init__(self):
        super(Yedroudj_Net, self).__init__()
        self.bias = False

        self.preprocessing = SRM_conv2d()

        self.conv1 = nn.Conv2d(30, 30, 5, stride=1, padding=2, bias=self.bias)
        self.batchNorm1 = nn.BatchNorm2d(30, affine=True)

        self.conv2 = nn.Conv2d(30, 30, 5, stride=1, padding=2, bias=self.bias)
        self.batchNorm2 = nn.BatchNorm2d(30, affine=True)
        self.avgPool2 = nn.AvgPool2d(5)

        self.conv3 = nn.Conv2d(30, 32, 3, stride=1, padding=2, bias=self.bias)
        self.batchNorm3 = nn.BatchNorm2d(32, affine=True)
        self.avgPool3 = nn.AvgPool2d(5)

        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=2, bias=self.bias)
        self.batchNorm4 = nn.BatchNorm2d(64, affine=True)
        self.avgPool4 = nn.AvgPool2d(5)

        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=2, bias=self.bias)
        self.batchNorm5 = nn.BatchNorm2d(128, affine=True)
        self.globalAvgPool = nn.AvgPool2d(4) # tu jest moze chyba nie teges???

        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        # PREPROCESSING MODULE
        x = self.preprocessing(x)

        # CONVOUTION MODULE
        # block 1
        x = self.conv1(x).abs()
        x = self.batchNorm1(x)
        x = torch.clamp(x, min=-3, max=3)
        
        # block 2
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = torch.clamp(x, min=-2, max=2)
        x = self.avgPool2(x)

        # block 3
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.avgPool3(x)

        # block 4
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)
        x = self.avgPool4(x)

        # block 5
        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = F.relu(x)
        x = self.globalAvgPool(x)

        # CLASYFFICATION MODULE
        #print(x.size())
        x = x.view(-1, 128 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x