import torch
import torch.nn as nn
import gymnasium as gym


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


done = False
episode_reward = 0

states = []
actions = []
rewards = []
log_probs = []

state, info = env.reset()
state = torch.tensor(state,dtype=torch.float32)
while not done:
       
       states.append(state)
       logits = actor(state)    
       prob = torch.softmax(logits,dim=0)

       dist = torch.distributions.Categorical(prob)
       action = dist.sample()
       actions.append(action)
       log_prob = dist.log_prob(action)
       log_probs.append(log_prob)

       next_state, reward, terminated, truncated, info = env.step(action.item())
       rewards.append(reward)
       if terminated or truncated:
              done = True

       episode_reward += reward

       next_state = torch.tensor(next_state,dtype=torch.float32)
       state = next_state


returns = torch.tensor(
    compute_returns(rewards),
    dtype=torch.float32
)

log_probs = torch.stack(log_probs)

loss = -(log_probs * returns).sum()

optimizer.zero_grad()

loss.backward()

optimizer.step()

    
              

       


