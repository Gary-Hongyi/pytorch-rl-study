
"""DQN 正式训练入口：直接在 VS Code 点击运行按钮。"""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from environment import make_breakout_env
from agent_dqn import Agent_DQN


# ==========================================
# 只需要修改这里的三个参数
# ==========================================

RUN_NAME = "double_dqn_cpu_01"  # 实验名称
RESUME = True            # False=新实验，True=断点续训
STEPS = 250000              # 本次额外训练多少步

# ==========================================


def main():
    root = Path(__file__).resolve().parent

    checkpoint_path = (
        root / "checkpoints" / f"{RUN_NAME}_latest.pt"
    )

    log_path = (
        root / "logs" / f"{RUN_NAME}.csv"
    )

    # 1. 防止错误覆盖实验
    if RESUME:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到续训存档：{checkpoint_path}"
            )
    else:
        if checkpoint_path.exists() or log_path.exists():
            raise FileExistsError(
                "实验名称已存在，请开启 RESUME 或更换 RUN_NAME"
            )

    if STEPS <= 0:
        raise ValueError("STEPS 必须大于 0")

    # 2. 随机种子
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 3. 训练参数
    args = SimpleNamespace(
        train_steps=STEPS,
        run_name=RUN_NAME,
        gamma=0.99,
        epsilon_decay_steps=1_000_000,
        checkpoint_interval=10_000,
        log_interval=1000
    )

    # 4. 创建环境
    env = make_breakout_env(
        test=False,
        seed=seed,
        render_mode=None
    )

    try:
        # 5. 创建 Agent
        agent = Agent_DQN(env, args)

        # 6. 可选：加载上次训练进度
        if RESUME:
            agent.load_checkpoint(checkpoint_path)

        print("\n===== DQN Training =====")
        print("Experiment:", RUN_NAME)
        print("Resume:", RESUME)
        print("Device:", agent.device)
        print("Starting steps:", agent.total_steps)
        print("Additional steps:", STEPS)

        # 7. 启动训练
        try:
            agent.train()

        except KeyboardInterrupt:
            print("\n训练中断，正在保存存档...")
            agent.save_checkpoint(checkpoint_path)
            return

        # 8. 输出结果
        print("\n===== Training Summary =====")
        print("Total steps:", agent.total_steps)
        print("Replay size:", len(agent.replay_buffer))
        print("Completed episodes:", len(agent.episode_rewards))
        print("Final epsilon:", agent.epsilon)

        print("\nTraining completed!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
