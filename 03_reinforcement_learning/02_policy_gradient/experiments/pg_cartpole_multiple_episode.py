import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt

class CartpoleActor(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4,32)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(32,2)


    def forward(self,x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x



def compute_returns(rewards, gamma=0.99):

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.append(G)

    returns.reverse()

    return returns



actor = CartpoleActor()


optimizer = torch.optim.Adam(
    actor.parameters(),
    lr=0.001
)


env = gym.make("CartPole-v1")


# ===========================
# 【新增1】训练多少局
# ===========================

#可视化记录每一个episode的rewards
episode_rewards = []

for episode in range(1000):


    # ===========================
    # 【新增2】每一局重新初始化
    # ===========================

    done = False

    episode_reward = 0

    rewards = []

    log_probs = []

   


    state, info = env.reset()

    state = torch.tensor(
        state,
        dtype=torch.float32
    )


    # ===========================
    # 原来的while循环
    # 负责玩一局游戏
    # ===========================

    while not done:


        logits = actor(state)

        prob = torch.softmax(
            logits,
            dim=0
        )


        dist = torch.distributions.Categorical(prob)


        action = dist.sample()


        log_prob = dist.log_prob(action)

        log_probs.append(log_prob)



        next_state, reward, terminated, truncated, info = env.step(
            action.item()
        )


        rewards.append(reward)


        episode_reward += reward



        if terminated or truncated:

            done = True



        next_state = torch.tensor(
            next_state,
            dtype=torch.float32
        )


        state = next_state



    # ===========================
    # 一局结束，开始学习
    # ===========================


    returns = torch.tensor(
        compute_returns(rewards),
        dtype=torch.float32
    )


    log_probs = torch.stack(log_probs)


    loss = -(log_probs * returns).sum()



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    # ===========================
    # 【新增3】观察训练效果
    # ===========================

    if episode % 50 == 0:

        print(
            "Episode:",
            episode,
            "Reward:",
            episode_reward
        )

    episode_rewards.append(episode_reward)



env.close()

#可视化训练曲线

plt.plot(episode_rewards)

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.title("REINFORCE CartPole Training")

plt.show()