import torch.nn as nn

from ..mpc_l import PCLayer


class Model(nn.Module):
    def __init__(
        self,
    ):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)  # 1, 6, 24, 24
        y = self.relu1(y)  # 1 6 24 24
        y = self.pool1(y)  # 1 6 12 12
        y = self.conv2(y)  # 1 16 8 8
        y = self.relu2(y)  # 1 16 8 8
        y = self.pool2(y)  # 1 16 4 4
        y = y.view(y.shape[0], -1)  # 1 256
        y = self.fc1(y)  # 1 120
        y = self.relu3(y)  # 1 120
        y = self.fc2(y)  # 1 84
        y = self.relu4(y)  # 1 84
        y = self.fc3(y)  # 1 10
        y = self.relu5(y)  # 1 10
        return y


class PCModel(nn.Module):
    def __init__(
        self,
    ):
        super(PCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pcl1 = PCLayer()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        #self.pcll1 = PCLayer()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pcl2 = PCLayer()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        #self.pcll2 = PCLayer()
        self.fc1 = nn.Linear(256, 120)
        self.pcl3 = PCLayer()
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.pcl4 = PCLayer()
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        #self.pcl5 = PCLayer()
        #self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)  # 1, 6, 24, 24
        y = self.pcl1(y)
        y = self.relu1(y)  # 1 6 24 24
        y = self.pool1(y)  # 1 6 12 12
        #y = self.pcll1(y)
        y = self.conv2(y)  # 1 16 8 8
        y = self.pcl2(y)
        y = self.relu2(y)  # 1 16 8 8
        y = self.pool2(y)  # 1 16 4 4
        #y = self.pcll2(y)
        y = y.view(y.shape[0], -1)  # 1 256
        y = self.fc1(y)  # 1 120
        y = self.pcl3(y)
        y = self.relu3(y)  # 1 120
        y = self.fc2(y)  # 1 84
        y = self.pcl4(y)
        y = self.relu4(y)  # 1 84
        y = self.fc3(y)  # 1 10
        #y = self.pcl5(y)
        #y = self.relu5(y)  # 1 10
        return y
