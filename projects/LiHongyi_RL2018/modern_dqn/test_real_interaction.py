
from agent_dqn import Agent_DQN
from environment import make_breakout_env, ACTION_MEANINGS


def main():
    # 1. 创建真实 Breakout 环境
    env = make_breakout_env(
        test=False,
        seed=42,
        render_mode="human"
    )

    try:
        # 2. 获取真实 Observation
        observation, info = env.reset()

        # 3. 创建 Agent
        agent = Agent_DQN(env, args=None)

        print("Initial observation:")
        print("Shape:", observation.shape)
        print("Dtype:", observation.dtype)
        print("Range:", observation.min(), observation.max())

        action_counts = [0] * env.action_space.n
        total_reward = 0.0
        total_raw_reward = 0.0
        ended_episodes = 0

        # 4. 执行 100 次真实环境交互
        for step in range(100):

            # 前10步使用 Greedy，确保真实画面经过 CNN
            # 后90步使用 Random，验证探索分支
            agent.epsilon = 0.0 if step < 10 else 1.0

            action = agent.make_action(
                observation,
                test=False
            )

            assert env.action_space.contains(action)

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            action_counts[action] += 1
            total_reward += float(reward)

            raw_reward = info.get("raw_reward", reward)
            total_raw_reward += float(raw_reward)

            if step in (0, 9, 99):
                print(
                    f"Step {step + 1}: "
                    f"Action={action}, "
                    f"Reward={reward}, "
                    f"Terminated={terminated}, "
                    f"Truncated={truncated}"
                )

            # 5. 如果回合结束，重新开始
            if terminated or truncated:
                ended_episodes += 1
                observation, info = env.reset()
            else:
                observation = next_observation

        print("\n===== Test Summary =====")
        print("Total steps:", 100)
        print("Action counts:", action_counts)
        print("Total reward:", total_reward)
        print("Total raw reward:", total_raw_reward)
        print("Ended episodes:", ended_episodes)
        print("Action meanings:", ACTION_MEANINGS)
        print("Real Breakout interaction passed!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
