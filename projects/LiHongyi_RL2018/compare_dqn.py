
"""Compare DQN and Double DQN training learning curves."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ========== 绘图配置 ==========

WINDOW = 30

# True：只比较双方都训练到的步数范围
# False：显示双方各自的全部训练范围
COMMON_STEPS_ONLY = True

# ==============================


def load_training_log(csv_path):
    """Read episode steps and rewards from training CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到训练日志: {csv_path}"
        )

    steps = []
    rewards = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            steps.append(int(row["total_steps"]))
            rewards.append(float(row["reward"]))

    steps = np.asarray(steps, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)

    if len(rewards) < WINDOW:
        raise ValueError(
            f"{csv_path.name} 不足 {WINDOW} Episodes"
        )

    return steps, rewards


def moving_average(steps, rewards):
    """Calculate average reward over the last 30 episodes."""

    avg = np.convolve(
        rewards,
        np.ones(WINDOW) / WINDOW,
        mode="valid",
    )

    return steps[WINDOW - 1:], avg


def main():
    root = Path(__file__).resolve().parent

    baseline_log = (
        root / "modern_dqn"
        / "logs" / "dqn_baseline_cpu_01.csv"
    )

    double_log = (
        root / "modern_double_dqn"
        / "logs" / "double_dqn_cpu_01.csv"
    )

    output_path = (
        root / "archive" / "legacy_figures" / "breakout_comparisons"
        / "dqn_vs_double_dqn_curve_regenerated.png"
    )

    # 1. 分别读取两组实验数据
    base_steps, base_rewards = load_training_log(
        baseline_log
    )

    double_steps, double_rewards = load_training_log(
        double_log
    )

    # 2. 分别计算最近30个 Episode 的平均 Reward
    base_x, base_avg = moving_average(
        base_steps, base_rewards
    )

    double_x, double_avg = moving_average(
        double_steps, double_rewards
    )

    # 3. 默认只显示双方共同训练过的步数范围
    if COMMON_STEPS_ONLY:
        max_steps = min(
            base_steps[-1],
            double_steps[-1]
        )

        base_mask = base_x <= max_steps
        double_mask = double_x <= max_steps

        base_x = base_x[base_mask]
        base_avg = base_avg[base_mask]

        double_x = double_x[double_mask]
        double_avg = double_avg[double_mask]

    # 4. 绘制对照曲线
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        base_x,
        base_avg,
        linewidth=1.8,
        label="DQN Baseline (Avg30)",
    )

    ax.plot(
        double_x,
        double_avg,
        linewidth=1.8,
        label="Double DQN (Avg30)",
    )

    ax.set_xlabel("Number of Environment Steps")
    ax.set_ylabel("Average Unclipped Reward (Last 30 Episodes)")
    ax.set_title("DQN vs Double DQN - Breakout Training")

    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim(left=0)

    # 5. 保存图片
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if output_path.exists():
        raise FileExistsError(f"Preserving existing figure: {output_path}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print("DQN recorded steps:", base_steps[-1])
    print("Double DQN recorded steps:", double_steps[-1])
    print("Figure saved:", output_path)


if __name__ == "__main__":
    main()
