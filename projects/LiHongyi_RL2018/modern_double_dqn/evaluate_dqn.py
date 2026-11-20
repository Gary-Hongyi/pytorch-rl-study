
"""Compare DQN Baseline and Double DQN at the same training steps."""

import random
from pathlib import Path

import numpy as np
import torch

from environment import Environment
from agent_dqn import Agent_DQN


# ========== 评估配置：以后只改这里 ==========

EVAL_STEPS = 1750_000

EPISODES = 100
TEST_EPSILON = 0.025
TEST_SEED = 2026

# ========================================


def evaluate_one(checkpoint_path):
    """Evaluate one checkpoint using the HW4-2 test procedure."""

    env = Environment(
        "BreakoutNoFrameskip-v4",
        args=None,
        atari_wrapper=True,
        test=True,
    )

    try:
        agent = Agent_DQN(env, args=None)

        # 加载指定训练节点的模型
        agent.load_checkpoint(checkpoint_path)

        # 统一测试探索率
        agent.epsilon = TEST_EPSILON

        # 恢复固定随机种子
        random.seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        torch.manual_seed(TEST_SEED)
        env.seed(TEST_SEED)

        rewards = []

        for episode in range(1, EPISODES + 1):
            state = env.reset()
            agent.init_game_setting()

            done = False
            episode_reward = 0.0

            while not done:
                action = agent.make_action(state, test=True)
                state, reward, done, info = env.step(action)
                episode_reward += float(reward)

            rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}/{EPISODES}, "
                    f"Mean Reward = {np.mean(rewards):.2f}"
                )

        rewards = np.asarray(rewards, dtype=np.float64)

        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "median": float(np.median(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        }

    finally:
        env.close()


def main():
    project_root = Path(__file__).resolve().parent.parent

    # 根据 EVAL_STEPS 自动选择两个相同训练步数的模型
    baseline_checkpoint = (
        project_root
        / "modern_dqn"
        / "checkpoints"
        / f"dqn_baseline_cpu_01_latest_{EVAL_STEPS}.pt"
    )

    double_checkpoint = (
        project_root
        / "modern_double_dqn"
        / "checkpoints"
        / f"double_dqn_cpu_01_latest_{EVAL_STEPS}.pt"
    )

    checkpoints = [
        ("DQN Baseline", baseline_checkpoint),
        ("Double DQN", double_checkpoint),
    ]

    # 开始前检查两个 Checkpoint 是否都存在
    for name, checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{name} Checkpoint 不存在：\n"
                f"{checkpoint_path}\n"
                f"请确认已保存 {EVAL_STEPS:,} Steps 的快照。"
            )

    print("=" * 55)
    print("DQN vs DOUBLE DQN EVALUATION")
    print(f"Training Steps: {EVAL_STEPS:,}")
    print(f"Test Episodes:  {EPISODES}")
    print(f"Test Epsilon:   {TEST_EPSILON}")
    print(f"Test Seed:      {TEST_SEED}")
    print("=" * 55)

    results = {}

    for name, checkpoint_path in checkpoints:
        print("\n" + "=" * 55)
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {checkpoint_path.name}")
        print("=" * 55)

        result = evaluate_one(checkpoint_path)
        results[name] = result

        print(f"\n{name} Evaluation:")
        print(f"Mean Reward:   {result['mean']:.2f}")
        print(f"Std Reward:    {result['std']:.2f}")
        print(f"Median Reward: {result['median']:.2f}")
        print(f"Min Reward:    {result['min']:.2f}")
        print(f"Max Reward:    {result['max']:.2f}")
        print(f"Reward > 40:   {result['mean'] > 40}")

    # ========== 最终对比 ==========

    baseline = results["DQN Baseline"]
    double = results["Double DQN"]

    difference = double["mean"] - baseline["mean"]

    print("\n" + "=" * 55)
    print(f"DQN vs DOUBLE DQN - {EVAL_STEPS:,} STEPS")
    print("=" * 55)

    print(f"DQN Baseline: {baseline['mean']:.2f}")
    print(f"Double DQN:   {double['mean']:.2f}")
    print(f"Difference:   {difference:+.2f}")

    if difference > 0:
        print("Result: Double DQN scored higher.")
    elif difference < 0:
        print("Result: DQN Baseline scored higher.")
    else:
        print("Result: Both models have equal mean reward.")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
