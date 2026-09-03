import os
import gymnasium as gym
import numpy as np
import pandas as pd

class HeuristicBaselineAgent:
    def __init__(self):
        pass
    def predict(self, state, deterministic=True):
        x, y, vx, vy, angle, omega, left_contact, right_contact = state
        target_angle = np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4)
        angle_todo = (target_angle - angle) * 2.0 - omega * 1.0
        hover_todo = 0.55 - vy * 0.5
        if left_contact or right_contact:
            hover_todo = -0.5
            angle_todo = 0.0
        if hover_todo > 0.5 and hover_todo > abs(angle_todo):
            return 2
        elif angle_todo < -0.05:
            return 3
        elif angle_todo > 0.05:
            return 1
        return 0

def evaluate_agent(agent, env_id="LunarLander-v3", seeds=[42, 101, 2024], num_episodes=30):
    all_results = []
    for seed in seeds:
        env = gym.make(env_id)
        for ep in range(num_episodes):
            state, _ = env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0
            fuel_used = 0.0
            steps = 0
            while not done:
                action = agent.predict(state, deterministic=True)
                if action == 2:
                    fuel_used += 0.3
                elif action in [1, 3]:
                    fuel_used += 0.03
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            landed = (reward >= 100.0) or (state[6] == 1.0 and state[7] == 1.0 and total_reward > 100)
            crashed = (reward <= -100.0) or (terminated and total_reward < 0)
            all_results.append({
                "seed": seed, "episode": ep + 1, "total_reward": total_reward,
                "fuel_consumption": fuel_used, "success": int(landed), "crash": int(crashed), "steps": steps
            })
        env.close()
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    baseline_agent = HeuristicBaselineAgent()
    df_results = evaluate_agent(baseline_agent, seeds=[42, 101, 2024], num_episodes=30)
    df_results.to_csv("logs/baseline_results.csv", index=False)
    print("Baseline evaluation complete and saved to logs/baseline_results.csv")
