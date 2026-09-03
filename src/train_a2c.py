import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

def train_a2c(episodes=500):
    env = gym.make("LunarLander-v3")
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    gamma = 0.99

    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []

        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_t)
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(m.log_prob(action))
            values.append(value)
            rewards.append(reward)
            masks.append(float(not done))
            state = next_state

        # Compute returns and advantages
        returns = []
        R = 0
        for r, mask in zip(reversed(rewards), reversed(masks)):
            R = r + gamma * R * mask
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        values = torch.cat(values)
        advantage = returns - values.detach()

        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_ep_reward = sum(rewards)
        episode_rewards.append(total_ep_reward)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes} | Average Reward: {np.mean(episode_rewards[-50:]):.2f}")

    env.close()
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame({"episode": range(1, episodes + 1), "reward": episode_rewards}).to_csv("logs/a2c_training_results.csv", index=False)
    torch.save(model.state_dict(), "src/a2c_lunar_lander.pth")
    print("Training complete. Results saved to logs/a2c_training_results.csv")

if __name__ == "__main__":
    train_a2c(episodes=300)
