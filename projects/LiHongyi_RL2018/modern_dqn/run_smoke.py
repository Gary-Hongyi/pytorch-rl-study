
"""HW4-2 DQN training and checkpoint smoke test."""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from environment import make_breakout_env
from agent_dqn import Agent_DQN


def main():
    # 1. 固定随机种子
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. 创建真实 Breakout 环境
    env = make_breakout_env(
        test=False,
        seed=seed,
        render_mode=None
    )

    try:
        # 3. 短训练配置
        args = SimpleNamespace(
            train_steps=100,
            epsilon_decay_steps=1_000_000,
            gamma=0.99,

            run_name="dqn_smoke",
            checkpoint_interval=100
        )

        # 4. 创建 Agent 并训练
        agent = Agent_DQN(env, args)

        print("===== Start Training Smoke Test =====")

        agent.train()

        # 5. 检查训练结果
        print("\n===== Training Summary =====")
        print("Total steps:", agent.total_steps)
        print("Replay size:", len(agent.replay_buffer))
        print("Completed episodes:", len(agent.episode_rewards))
        print("Epsilon:", agent.epsilon)

        assert agent.total_steps == 100
        assert len(agent.replay_buffer) == 100

        print("Training loop test passed!")

        # ===== Checkpoint Test =====

        # 6. 设置存档路径
        checkpoint_path = (
            Path(__file__).resolve().parent
            / "checkpoints"
            / "dqn_smoke.pt"
        )

        # 7. 保存当前 Agent
        agent.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        print("Checkpoint file size:",
              round(checkpoint_path.stat().st_size / 1024**2, 2),
              "MiB")

        # 8. 创建新的 Agent
        restored_agent = Agent_DQN(env, args)

        # 9. 加载存档
        restored_agent.load_checkpoint(checkpoint_path)

        # 10. 检查训练进度
        assert restored_agent.total_steps == agent.total_steps
        assert restored_agent.epsilon == agent.epsilon

        assert (
            restored_agent.episode_rewards
            == agent.episode_rewards
        )

        # 11. 检查 Replay Buffer
        assert (
            len(restored_agent.replay_buffer)
            == len(agent.replay_buffer)
        )

        n = len(agent.replay_buffer)

        assert np.array_equal(
            restored_agent.replay_buffer.states[:n],
            agent.replay_buffer.states[:n]
        )

        assert np.array_equal(
            restored_agent.replay_buffer.actions[:n],
            agent.replay_buffer.actions[:n]
        )

        # 12. 检查 Online Network 参数
        for p1, p2 in zip(
            agent.q_network.parameters(),
            restored_agent.q_network.parameters()
        ):
            assert torch.equal(p1, p2)

        # 13. 检查 Target Network 参数
        for p1, p2 in zip(
            agent.target_network.parameters(),
            restored_agent.target_network.parameters()
        ):
            assert torch.equal(p1, p2)

        print("Checkpoint save/load test passed!")

        print("\nAll DQN smoke tests passed!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
