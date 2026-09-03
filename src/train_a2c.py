import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. A2C Network ---
class A2CNetwork(nn.Module):
    def __init__(self):
        super(A2CNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

LEARNING_RATE = 1e-3
GAMMA = 0.99
NUM_EPISODES = 300
SEEDS = [42, 101, 2024]

os.makedirs("logs", exist_ok=True)
os.makedirs("docs", exist_ok=True)
all_results = []

# --- 2. Multi-Seed Training Loop ---
for seed in SEEDS:
    print(f"\n--- Training on Seed: {seed} ---")
    
    # Enforce reproducibility per seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make("LunarLander-v3")
    agent = A2CNetwork()
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        log_probs, values, rewards = [], [], []
        
        done = truncated = False
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = agent(state_tensor)
            
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            log_probs.append(action_distribution.log_prob(action))
            values.append(state_value)
            rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            
        # A2C Return Calculation & Optimization
        returns = []
        R = 0.0
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        actor_loss = 0
        critic_loss = 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R.item() - value.item()
            actor_loss -= log_prob * advantage
            critic_loss += F.mse_loss(value, torch.tensor([[R]], dtype=torch.float32))
            
        total_loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Record data
        all_results.append({"Seed": seed, "Episode": episode + 1, "Reward": episode_reward})
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES} | Reward: {episode_reward:.2f}")
            
    # Save the final seed model
    if seed == SEEDS[-1]:
        torch.save(agent.state_dict(), "src/a2c_lunar_lander.pth")
        
    env.close()

# --- 3. Export Combined CSV ---
df_results = pd.DataFrame(all_results)
df_results.to_csv("logs/a2c_multiseed_results.csv", index=False)

# --- 4. Plot Mean with Measure of Spread ---
summary_df = df_results.groupby("Episode")["Reward"].agg(['mean', 'std']).reset_index()

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Poppins', 'Arial']

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')

# Plot mean line and standard deviation spread
ax.plot(summary_df["Episode"], summary_df["mean"], label="Mean Reward", color="#0052cc", linewidth=2)
ax.fill_between(
    summary_df["Episode"], 
    summary_df["mean"] - summary_df["std"], 
    summary_df["mean"] + summary_df["std"], 
    color="#0052cc", alpha=0.2, label=r"$\pm 1$ Standard Deviation"
)

ax.axhline(y=200, color='gray', linestyle='--', label="Solved (+200)")
ax.set_title("A2C Training Curve (3 Seeds)", color="#333333", weight='bold')
ax.set_xlabel("Episode", color="#333333")
ax.set_ylabel("Total Reward", color="#333333")
ax.legend(frameon=False)
ax.grid(color='#e0e0e0', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig("docs/learning_curve.png", dpi=300)
print("\nMulti-seed training complete. Plot saved to docs/learning_curve.png")
