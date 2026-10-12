import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1")

# 开始一个 episode
state, info = env.reset()

print("Initial state:", state)
print("Action space:", env.action_space)

total_reward = 0
for step in range(10):

    # 先随机选择动作
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {step + 1}")
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

    total_reward += reward

    state = next_state

    if terminated or truncated:
        print("Episode finished.")
        break

print("Total reward:", total_reward)

env.close()
