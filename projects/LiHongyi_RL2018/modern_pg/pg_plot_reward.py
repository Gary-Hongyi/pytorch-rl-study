
from pathlib import Path

import torch
import matplotlib.pyplot as plt


# 1. 找到训练过程中保存的 Checkpoint
checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pong_pg_latest.pt"
)

# 2. 读取训练记录
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

rewards = checkpoint["episode_rewards"]
timesteps = checkpoint["episode_timesteps"]

# 确认两组数据一一对应
assert len(rewards) == len(timesteps), "Reward 和 Time Steps 数量不一致"

print("Completed episodes:", len(rewards))
print("Total time steps:", checkpoint["total_steps"])

# 3. 计算最近 30 局平均 Reward
window = 30

if len(rewards) < window:
    raise ValueError("至少需要训练 30 局才能绘制 Avg30 曲线")

avg30_rewards = []
avg30_timesteps = []

for i in range(window - 1, len(rewards)):

    # 取最近 30 局的 Reward
    recent_rewards = rewards[i - window + 1:i + 1]

    # 计算平均 Reward
    avg_reward = sum(recent_rewards) / window

    avg30_rewards.append(avg_reward)

    # 横轴使用这一局结束时的累计 Time Steps
    avg30_timesteps.append(timesteps[i])

# 4. 绘制官方要求的学习曲线
plt.figure(figsize=(10, 5))

plt.plot(
    avg30_timesteps,
    avg30_rewards,
    marker="o",
    markersize=3,
    label="Average Reward (Last 30 Episodes)"
)

# 5. 标出老师要求的基准线
plt.axhline(
    y=3,
    color="red",
    linestyle="--",
    label="Official Baseline: Reward = 3"
)

plt.xlabel("Time Steps")
plt.ylabel("Average Reward (Last 30 Episodes)")
plt.title("Pong Policy Gradient Learning Curve")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 6. 保存图表
output_path = (
    Path(__file__).resolve().parent
    / "figures" / "pong_vanilla_pg_avg30_regenerated.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists():
    raise FileExistsError(f"Preserving existing figure: {output_path}")

plt.savefig(output_path, dpi=150)

print("Latest Avg30 Reward:", round(avg30_rewards[-1], 2))
print("Learning curve saved:", output_path)

plt.show()
