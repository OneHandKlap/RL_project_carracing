import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict


def MobileNet():
    model = models.mobilenet_v2(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False

    #summary(model, (3, 224, 224))

    #model.classifier[1] = nn.Linear(1280, outputs)
    # for p in model.classifier[1].parameters():
    #    p.requires_grad = True
    #model.num_classes = outputs

    # #print(model)

    return model


class DQN_Smort(nn.Module):

    def __init__(self, w, h, frame_stacks, outputs):
        super(DQN_Smort, self).__init__()

        self.frame_stacks = frame_stacks

        self.mid = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, kernel_size=5, stride=2)),
            ('bn1', nn.BatchNorm2d(16)),
            ('max1', nn.MaxPool2d(2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(16, 32, kernel_size=5, stride=2)),
            ('bn2', nn.BatchNorm2d(32)),
            ('max2', nn.MaxPool2d(2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            ('bn3', nn.BatchNorm2d(32)),
            ('avg1', nn.AvgPool2d(2)),
            ('relu2', nn.ReLU()),
        ]))

        self.rnn = nn.RNN(128, 128)

        self.head = nn.Linear(128 * 3, outputs)

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, 128).to("cuda"))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # divide into number of stacks
        x_ = torch.tensor_split(x, self.frame_stacks, 1)
        output = []
        for i in range(len(x_)):
            # print(x_[i].shape)
            x_temp = self.mid(x_[i])
            # print(x_temp.shape)
            x_temp = x_temp.view(x_temp.size(0), -1)
            # print(x_temp.shape)
            output.append(x_temp)

        x_rnn = torch.cat(output)
        # print(x_rnn.shape)
        x_rnn = x_rnn.unsqueeze(dim=1)
        # print(x_rnn.shape)
        self.hidden = self.init_hidden()
        out, self.hidden = self.rnn(x_rnn, self.hidden)
        print(out.shape)
        x = out.view(out.size(0), -1)
        print(x.shape)
        x = self.head(x)
        print(x.shape)
        return x
