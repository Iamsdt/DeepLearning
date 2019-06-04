import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):

    def __init__(self, input_size):
        super(InceptionModule, self).__init__()

        # branch pool
        self.branch_pool_conv1 = nn.Conv2d(input_size, 16, kernel_size=1)

        # branch 1 1
        self.branch_11_conv1 = nn.Conv2d(input_size, 16, kernel_size=1)

        # branch 55
        self.branch_55_conv1 = nn.Conv2d(input_size, 16, kernel_size=1)
        self.branch_55_conv2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # branch 33
        self.branch_33_conv1 = nn.Conv2d(input_size, 16, kernel_size=1)
        self.branch_33_conv2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_33_conv3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_11 = self.branch_11_conv1(x)

        branch_55 = self.branch_55_conv1(x)
        branch_55 = self.branch_55_conv2(branch_55)

        branch_33 = self.branch_33_conv1(x)
        branch_33 = self.branch_33_conv2(branch_33)
        branch_33 = self.branch_33_conv3(branch_33)

        branch_avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_avg = self.branch_11_conv1(branch_avg)

        outputs = [branch_11, branch_55, branch_33, branch_avg]

        return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.inception1 = InceptionModule(input_size=10)
        self.inception2 = InceptionModule(input_size=10)

        self.pool = nn.MaxPool2d(2)
        # linear
        self.fc1 = nn.Linear(1480, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = self.inception1(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = self.inception2(x)
        # Flatten the tensor
        x = x.view(in_size, 0)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc4(x)
