
"""Plot DQN learning curve from checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parent

    checkpoint_path = (
        root / "checkpoints" / "double_dqn_cpu_01_latest.pt"
    )

    output_path = (
        root / "figures" / "double_dqn_cpu_01_curve.png"
    )

    # 只读取自己生成、来源可信的 Checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    steps = np.asarray(
        checkpoint["episode_timesteps"],
        dtype=np.int64
    )

    rewards = np.asarray(
        checkpoint["episode_rewards"],
        dtype=np.float32
    )

    if len(steps) != len(rewards):
        raise ValueError("Steps and rewards length mismatch")

    if len(rewards) == 0:
        print("No completed episodes to plot.")
        return

    print("Completed episodes:", len(rewards))
    print("Recorded steps:", steps[-1])

    fig, ax = plt.subplots(figsize=(10, 5))

    # 每个 Episode 的原始 Reward
    ax.plot(
        steps,
        rewards,
        alpha=0.35,
        label="Episode Reward"
    )

    # 官方要求：最近30个 Episode 的平均 Reward
    window = 30

    if len(rewards) >= window:
        avg30 = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )

        ax.plot(
            steps[window - 1:],
            avg30,
            linewidth=2,
            label="Average Reward (Last 30 Episodes)"
        )
    else:
        print(
            f"Only {len(rewards)} episodes. "
            "At least 30 are needed for Avg30."
        )

    ax.set_xlabel("Number of Time Steps")
    ax.set_ylabel("Unclipped Episode Reward")
    ax.set_title("Double DQN Breakout Learning Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print("Figure saved:", output_path)


if __name__ == "__main__":
    main()
