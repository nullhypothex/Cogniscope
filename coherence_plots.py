import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load your dataset ===
df = pd.read_csv("simulated_AD_users_groq.csv")

# Ensure timestamp/date columns are sorted
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['user_id', 'date'])

# Create output directory
os.makedirs("coherence_plots", exist_ok=True)

# === Plot per user ===
users = df['user_id'].unique()

for user in users:
    user_df = df[df['user_id'] == user]

    # Average coherence per day
    daily_avg = user_df.groupby('day')['coherence_score'].mean().reset_index()
    daily_avg['day'] = daily_avg['day'].astype(int)

    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily_avg, x='day', y='coherence_score', marker='o', linewidth=2.5)
    plt.title(f"Semantic Coherence Drift: {user}", fontsize=14)
    plt.xlabel("Day")
    plt.ylabel("Average Coherence Score")
    plt.axhline(y=0.5, color='r', linestyle='--', label="EarlyAD Threshold")
    plt.axhline(y=0.85, color='g', linestyle='--', label="Healthy Threshold")
    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(f"coherence_plots/{user}_coherence_trend.png")
    plt.close()

print("✅ Coherence plots saved in ./coherence_plots/")
