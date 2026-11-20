
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# 1. 两个实验的 Checkpoint
ROOT = Path(__file__).resolve().parent

vanilla_path = ROOT / "modern_pg" / "checkpoints" / "pong_pg_latest.pt"
improved_path = ROOT / "modern_pg_improved" / "checkpoints" / "pong_pg_latest.pt"


# 2. 读取 Checkpoint，计算最近30局平均 Reward
def load_curve(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    rewards = checkpoint["episode_rewards"]
    timesteps = checkpoint["episode_timesteps"]

    window = 30
    avg_rewards = []
    avg_timesteps = []

    for i in range(window - 1, len(rewards)):
        avg_rewards.append(
            sum(rewards[i - window + 1:i + 1]) / window
        )
        avg_timesteps.append(timesteps[i])

    return np.array(avg_timesteps), np.array(avg_rewards)


# 3. 分别读取 Vanilla 和 Improved 的训练记录
vanilla_steps, vanilla_rewards = load_curve(vanilla_path)
improved_steps, improved_rewards = load_curve(improved_path)


# 4. 确定共同的 Time Steps 范围
# 公平比较：只展示两种算法都已经训练过的范围
max_steps = min(vanilla_steps[-1], improved_steps[-1])

vanilla_mask = vanilla_steps <= max_steps
improved_mask = improved_steps <= max_steps


# 5. 绘制对照曲线
plt.figure(figsize=(11, 5))

plt.plot(
    vanilla_steps[vanilla_mask],
    vanilla_rewards[vanilla_mask],
    label="Vanilla PG",
    linewidth=1.5
)

plt.plot(
    improved_steps[improved_mask],
    improved_rewards[improved_mask],
    label="Improved PG (MC Value Baseline)",
    linewidth=1.5
)

plt.axhline(
    y=3,
    color="red",
    linestyle="--",
    label="Official Target: Reward = 3"
)

plt.xlabel("Time Steps")
plt.ylabel("Average Reward (Last 30 Episodes)")
plt.title("Pong: Vanilla PG vs Improved PG")
plt.xlim(0, max_steps)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# 6. 保存对照图
output_path = (
    ROOT / "archive" / "legacy_figures" / "pong_comparisons"
    / "pong_pg_comparison_regenerated.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists():
    raise FileExistsError(f"Preserving existing figure: {output_path}")
plt.savefig(output_path, dpi=150)

# 在共同训练预算处比较两种算法
vanilla_at_same_steps = np.interp(
    max_steps, vanilla_steps, vanilla_rewards
)
improved_at_same_steps = np.interp(
    max_steps, improved_steps, improved_rewards
)

print(f"Compared Time Steps: {max_steps:,}")
print(f"Vanilla Avg30: {vanilla_at_same_steps:.2f}")
print(f"Improved Avg30: {improved_at_same_steps:.2f}")
print("Comparison saved:", output_path)

plt.show()
