import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load dataset
df = pd.read_csv("simulated_AD_users_groq_with_metrics.csv")

# Initialize SBERT model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Drift simulation
drift_data = []

for user_id in df["user_id"].unique():
    user_df = df[df["user_id"] == user_id]
    baseline_map = user_df[user_df["day"] == 1].groupby("video_title")["summary"].first().to_dict()
    
    for _, row in user_df.iterrows():
        vid = row["video_title"]
        if vid not in baseline_map:
            continue
        
        curr_summary = row["summary"]
        label = row["label"]
        try:
            emb_curr = sbert.encode(curr_summary, convert_to_tensor=True)
            emb_base = sbert.encode(baseline_map[vid], convert_to_tensor=True)
            coh_clean = util.pytorch_cos_sim(emb_curr, emb_base).item()
            noise = np.random.normal(0, 0.015)
            coh_noisy = max(min(coh_clean + noise, 1.0), 0.0)
            drift_data.append({
                "label": label,
                "coherence_clean": coh_clean,
                "coherence_noisy": coh_noisy,
                "drift_clean": 1.0 - coh_clean,
                "drift_noisy": 1.0 - coh_noisy,
                "noise": noise
            })
        except:
            continue

drift_df = pd.DataFrame(drift_data)
avg_drift = drift_df.groupby("label")[["coherence_clean", "coherence_noisy", "drift_clean", "drift_noisy"]].mean().round(3)
print("\n📊 Average Coherence and Drift by Label:")
print(avg_drift)

drift_df.to_csv("coherence_drift_analysis.csv", index=False)
