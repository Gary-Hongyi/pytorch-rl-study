
import numpy as np
import torch
from types import SimpleNamespace

from agent_dqn import Agent_DQN


# 1. 创建模拟环境
env = SimpleNamespace(
    action_space=SimpleNamespace(n=4)
)

agent = Agent_DQN(env, args=None)

# 2. 创建模拟 Observation
obs = np.random.rand(
    84, 84, 4
).astype(np.float32)


# Test A: Exploration
agent.epsilon = 1.0

actions = [
    agent.make_action(obs, test=False)
    for _ in range(20)
]

assert all(0 <= a < 4 for a in actions)
assert all(isinstance(a, int) for a in actions)

print("Exploration test passed:", actions)


# Test B: Exploitation
agent.epsilon = 0.0

state = agent._observation_to_tensor(obs)

with torch.no_grad():
    q_values = agent.q_network(state)

expected_action = q_values.argmax(dim=1).item()

action = agent.make_action(obs, test=True)

assert action == expected_action

print("Q-values:", q_values)
print("Greedy action:", action)
print("Exploitation test passed!")


# Test C: Testing mode still respects epsilon
agent.epsilon = 1.0

test_actions = [
    agent.make_action(obs, test=True)
    for _ in range(20)
]

assert all(0 <= a < 4 for a in test_actions)

print("Testing epsilon-greedy test passed!")
