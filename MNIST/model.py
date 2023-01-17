import torch
import torch.nn as nn


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.cov1 = nn.Conv2d(1, 8, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d((2,2))
        self.cov2 = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d((2,2))
        self.cov3 = nn.Conv2d(16,32,kernel_size=(2,2))
        self.fc1 = nn.Linear(16*25,160)
        self.fc2 = nn.Linear(160,64)
        self.fc3 = nn.Linear(64,10)
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.relu(self.cov1(x))
        x = self.pool1(x)
        x = self.relu(self.cov2(x))
        x = self.pool2(x)
        x = x.view(-1,16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

