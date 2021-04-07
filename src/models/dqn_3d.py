import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_3D(nn.Module):

    def __init__(self, w, h, frame_stacks, outputs):
        super(DQN_3D, self).__init__()

        self.frame_stacks = frame_stacks

        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 2, 2), stride=2)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=1, stride=2)
        self.bn3 = nn.BatchNorm3d(32)
        self.avg_pool = nn.AvgPool3d(1)
        self.head = nn.Linear(4032, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x.shape)
        x = torch.stack(torch.tensor_split(x, self.frame_stacks, 1), 4)
        x = x.permute(0, 1, 4, 2, 3)
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # #print(x.shape)
        #x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = F.relu(self.avg_pool(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return self.head(x)
