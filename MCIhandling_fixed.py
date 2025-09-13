import pandas as pd
import numpy as np
from scipy.stats import entropy

# === Load your noisy daily dataset ===
df = pd.read_csv("noisy_200simulated_users_groq_metrics.csv")

# === Feature Engineering per user + label phase ===
def compute_features(sub_df):
    feats = {}
    feats["coherence_mean"] = sub_df["coherence"].mean()
    feats["coherence_std"] = sub_df["coherence"].std()

    # Behavioral entropy (how spread-out watch_time, pauses, replays are)
    behavior_cols = ["watch_time_secs", "pause_count", "replay_count", "liked", "shared"]
    behavior = sub_df[behavior_cols].sum(axis=0).values
    probs = behavior / (behavior.sum() + 1e-9)
    feats["behavior_entropy_mean"] = entropy(probs)

    # Coherence slope (drift over time within this phase)
    if len(sub_df) > 1:
        x = np.arange(len(sub_df))
        y = sub_df["coherence"].values
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0
    feats["slope"] = slope

    return pd.Series(feats)

# Group by user_id + label to preserve MCI
df_features = df.groupby(["user_id", "label"]).apply(compute_features).reset_index()

print("✅ Engineered features per user-phase:")
print(df_features.head())

# Save engineered features
df_features.to_csv("user_features1.csv", index=False)
print("✅ Saved to user_features1.csv")
