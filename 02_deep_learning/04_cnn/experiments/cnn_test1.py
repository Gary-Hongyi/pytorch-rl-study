import torch
import torch.nn as nn


class CNNNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear( #最后这里的这一步，为了拿去分析，还是需要传入 神经网络层Linear
            16 * 7 * 7,
            10
        )

    def forward(self, x):

        print("input:", x.shape)

        x = self.conv1(x)
        print("after conv1:", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("after pool1:", x.shape)

        x = self.conv2(x)
        print("after conv2:", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("after pool2:", x.shape)

        x = self.flatten(x)
        print("after flatten:", x.shape)

        x = self.fc(x)
        print("output:", x.shape)

        return x


#shape test
model = CNNNet()

x = torch.randn(64, 1, 28, 28)

y = model(x)

print(y.shape)