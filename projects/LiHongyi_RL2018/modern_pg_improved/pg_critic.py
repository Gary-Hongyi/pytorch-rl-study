
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6400, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 80×80 → 6400
        x = torch.flatten(x, start_dim=-2)

        
        x = self.fc1(x)
        x = self.relu(x)

        #这里代表的是输出一个状态价值，也就是用于 Actor-Critic 的 V
        value = self.fc2(x)

        return value


