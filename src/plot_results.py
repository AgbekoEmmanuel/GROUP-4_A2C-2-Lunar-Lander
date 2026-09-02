import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("docs", exist_ok=True)

# 1. Plot A2C Learning Curve
a2c_df = pd.read_csv("logs/a2c_training_results.csv")
plt.figure(figsize=(10, 5))
sns.lineplot(data=a2c_df, x="episode", y="reward", alpha=0.3, color="blue", label="Episode Reward")
sns.lineplot(x=a2c_df["episode"], y=a2c_df["reward"].rolling(50).mean(), color="red", label="50-Episode Moving Average")
plt.title("A2C Agent Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.axhline(y=200, color='g', linestyle='--', label="Solved Threshold")
plt.legend()
plt.savefig("docs/learning_curve.png")
plt.close()

# 2. Compare Baseline vs A2C (Last 50 Episodes)
baseline_df = pd.read_csv("logs/baseline_results.csv")
baseline_avg = baseline_df["total_reward"].mean()
a2c_avg = a2c_df["reward"].tail(50).mean()

plt.figure(figsize=(6, 5))
sns.barplot(x=["Heuristic Baseline", "Trained A2C"], y=[baseline_avg, a2c_avg], hue=["Heuristic Baseline", "Trained A2C"], legend=False, palette=["gray", "blue"])
plt.title("Average Reward Comparison")
plt.ylabel("Average Reward")
plt.savefig("docs/comparison_bar.png")
plt.close()

print("Plots successfully generated and saved to the docs/ folder.")
