
import torch
import torch.nn as nn


class PongActor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6400, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        # 80×80 → 6400
        x = torch.flatten(x, start_dim=-2)

        # 隐藏层
        x = self.fc1(x)
        x = self.relu(x)

        # 输出两个动作的 logits
        logits = self.fc2(x)

        return logits


