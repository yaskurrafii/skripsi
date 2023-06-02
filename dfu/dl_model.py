from torch import nn
import torch.nn.functional as F
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3
        )
        self.pool1 = nn.MaxPool2d(1, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=192, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)
        self.pool3 = nn.MaxPool2d(3, 2, padding=1)
        self.pool4 = nn.MaxPool2d(3, 2, padding=1)
        self.pool5 = nn.MaxPool2d(7, 1)
        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.LazyLinear(2)

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(self.pool1(output))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = self.pool2(output)
        output = self.parallel_conv(output, 192)
        output = self.pool3(output)
        output = self.parallel_conv(output, 224)
        output = self.parallel_conv(output, 224)
        output = self.pool4(output)
        output = self.parallel_conv(output, 224)
        output = self.pool5(output)
        output = output.flatten(1)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

    def parallel_conv(self, data, input_channel):
        parallel_conv1 = nn.Conv2d(
            in_channels=input_channel, out_channels=32, kernel_size=1, stride=1
        )
        parallel_conv2 = nn.Conv2d(
            in_channels=input_channel, out_channels=64, kernel_size=1, stride=1
        )
        parallel_conv3 = nn.Conv2d(
            in_channels=input_channel, out_channels=128, kernel_size=1, stride=1
        )

        out1 = parallel_conv1(data)
        out2 = parallel_conv2(data)
        out3 = parallel_conv3(data)

        # out2 = out2.view(out2.size(0), -1)
        # out3 = out3.view(out2.size(0), -1)

        full_out = torch.cat([out1, out2, out3], 1)

        return full_out

    def eval(self):
        return super().eval()
