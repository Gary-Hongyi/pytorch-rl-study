
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.action_dim = action_dim

        # Stage 1: CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Stage 2: Flatten + Fully Connected  展平然后加入全连接层 
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        features = self.features(x)
        q_values = self.head(features)

        return q_values
