
import random
import numpy as np
import torch

from environment import Environment
from pg_agent import Agent_PG


# 0. 固定随机种子，提高实验可复现性
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# 1. 创建 Pong 环境
env = Environment(
    "Pong-v0",
    atari_wrapper=False,
    test=True
)

# 为 Pong 环境设置随机种子
env.seed(seed)


# 2. 创建 Agent
agent = Agent_PG(env, args=None)


# 3. 正式训练：在已有 Checkpoint 基础上继续训练1000局
try:
    agent.train(num_episodes=500) 

finally:
    env.close()
