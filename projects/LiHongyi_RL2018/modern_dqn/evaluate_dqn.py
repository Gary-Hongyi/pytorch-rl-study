
"""Compare two DQN checkpoints using the official HW4-2 evaluation loop."""

import random
from pathlib import Path

import numpy as np
import torch

from environment import Environment
from agent_dqn import Agent_DQN


# ===== 评估配置 =====

CHECKPOINTS = [
    "dqn_baseline_cpu_01_latest_1250000.pt",
    "dqn_baseline_cpu_01_latest_2000000.pt",
]

EPISODES = 100
TEST_EPSILON = 0.025
TEST_SEED = 11037

# ====================


def evaluate_one(checkpoint_path):
    """Evaluate one saved model over 100 episodes."""

    # 每个模型创建独立的测试环境
    env = Environment(
        "BreakoutNoFrameskip-v4",
        args=None,
        atari_wrapper=True,
        test=True
    )

    try:
        agent = Agent_DQN(env, args=None)

        # 1. 恢复模型参数
        agent.load_checkpoint(checkpoint_path)

        # 2. 固定评估探索率
        agent.epsilon = TEST_EPSILON

        # 3. 加载 Checkpoint 后，重置随机数状态
        random.seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        torch.manual_seed(TEST_SEED)

        # 4. 固定测试环境种子
        env.seed(TEST_SEED)

        rewards = []

        # 5. 按官方测试流程运行
        for episode in range(EPISODES):
            state = env.reset()
            agent.init_game_setting()

            done = False
            episode_reward = 0.0

            while not done:
                action = agent.make_action(
                    state, test=True
                )

                state, reward, done, info = env.step(action)

                # test=True: 使用未裁剪 Reward
                episode_reward += float(reward)

            rewards.append(episode_reward)

            if (episode + 1) % 20 == 0:
                print(
                    f"Episode {episode + 1}/{EPISODES}, "
                    f"Mean={np.mean(rewards):.2f}"
                )

        # 6. 统计评估结果
        rewards = np.asarray(rewards, dtype=np.float64)

        results = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "median": float(np.median(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        }

        return results

    finally:
        env.close()


def main():
    root = Path(__file__).resolve().parent
    checkpoint_dir = root / "checkpoints"

    all_results = {}

    for filename in CHECKPOINTS:
        checkpoint_path = checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到 Checkpoint: {checkpoint_path}"
            )

        print("\n" + "=" * 50)
        print("Evaluating:", filename)
        print("=" * 50)

        result = evaluate_one(checkpoint_path)

        all_results[filename] = result

        print("\nEvaluation result:")
        print(f"Mean Reward:   {result['mean']:.2f}")
        print(f"Std Reward:    {result['std']:.2f}")
        print(f"Median Reward: {result['median']:.2f}")
        print(f"Min Reward:    {result['min']:.2f}")
        print(f"Max Reward:    {result['max']:.2f}")
        print(f"Baseline > 40: {result['mean'] > 40}")

    # 最终对比
    first = all_results[CHECKPOINTS[0]]
    second = all_results[CHECKPOINTS[1]]

    difference = second["mean"] - first["mean"]

    print("\n" + "=" * 50)
    print("DQN CHECKPOINT COMPARISON")
    print("=" * 50)

    print(f"1250000 Steps: {first['mean']:.2f}")
    print(f"1750000 Steps: {second['mean']:.2f}")
    print(f"Difference:    {difference:+.2f}")

    if difference < 0:
        print("Result: 1500000-step model scored lower.")
    elif difference > 0:
        print("Result: 1500000-step model scored higher.")
    else:
        print("Result: Both models have equal mean reward.")

    print("\nComparison completed!")


if __name__ == "__main__":
    main()
