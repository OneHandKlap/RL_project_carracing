import torch.nn as nn
import torch
import torch.nn.functional as F


class DQN_Deep(nn.Module):

    def __init__(self, w, h, frame_stacks, outputs):
        super(DQN_Deep, self).__init__()
        self.conv1 = nn.Conv2d(3 * frame_stacks, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(3)
        self.flatten = nn.Flatten()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.dense1 = nn.Linear(linear_input_size, 32)
        self.head = nn.Linear(32, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.head(x)


'''
x = torch.randn(128, 9, 40, 60)
model = DQN_Deep(60, 40, 3, 16)
model(x)
torch.onnx.export(model, x, "dqn_deep.onnx", input_names=['input'], output_names=['output'], do_constant_folding=True)
'''
