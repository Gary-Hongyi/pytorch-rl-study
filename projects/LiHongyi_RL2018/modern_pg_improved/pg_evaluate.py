
import random
import numpy as np
import torch

import sys
from pathlib import Path

MODERN_DIR = Path(__file__).resolve().parent.parent / "modern"
sys.path.append(str(MODERN_DIR))

from environment import Environment
from pg_agent import Agent_PG


# 1. 固定随机种子
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 2. 创建测试环境
env = Environment(
    "Pong-v0",
    atari_wrapper=False,
    test=True
)
env.seed(seed)

agent = Agent_PG(env, args=None)

# 3. 加载训练好的 Actor
checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pong_pg_10000.pt"
)

if not checkpoint_path.exists():
    env.close()
    raise FileNotFoundError(
        f"找不到模型文件：{checkpoint_path}"
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

agent.actor.load_state_dict(checkpoint["model"])
agent.actor.eval()

print("Loaded checkpoint:", checkpoint_path)
print(
    "Training episodes:",
    checkpoint["episodes_completed"]
)

# 4. 独立评估 100 局
episode_rewards = []

try:
    for episode in range(100):
        observation = env.reset()
        agent.init_game_setting()

        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                action = agent.make_action(
                    observation,
                    test=True
                )

            observation, reward, done, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

        print(
            f"Eval {episode + 1:02d}/100 | "
            f"Reward: {total_reward:.1f}"
        )

finally:
    env.close()

# 5. 汇总结果
rewards = np.asarray(episode_rewards, dtype=np.float32)
wins = int(np.sum(rewards > 0))

print("\n===== Evaluation Results =====")
print(f"Episodes: {len(rewards)}")
print(f"Mean Reward: {rewards.mean():.2f}")
print(f"Median Reward: {np.median(rewards):.2f}")
print(f"Std Reward: {rewards.std():.2f}")
print(f"Wins: {wins}/100")
print(f"Win Rate: {wins / len(rewards) * 100:.1f}%")
