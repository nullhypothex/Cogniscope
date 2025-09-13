import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_early_detection(df, target_label="MCI", o_list=[100,200], k_list=[50,100]):
    """
    Evaluate early detection metrics for a given target label (e.g., MCI).
    
    df: DataFrame with columns [user_id, day, true_label, pred_label]
    target_label: class to detect early (default="MCI")
    o_list: list of observation windows for ERDE (e.g., [100,200])
    k_list: list of cutoffs for early precision/recall (e.g., [50,100])
    """
    
    results = {}
    
    # --- ERDE + TTD ---
    def compute_ERDE_TTD(o=100, c_fp=1.0):
        erdes, ttds = [], []
        for user_id, group in df.groupby("user_id"):
            group = group.sort_values("day")
            true_days = group[group["true_label"] == target_label]["day"].values
            if len(true_days) == 0:
                continue
            detected_days = group[group["pred_label"] == target_label]["day"].values
            if len(detected_days) == 0:
                erde = c_fp
                ttd = np.nan
            else:
                d = detected_days[0]
                if d <= o:
                    erde = 1 - np.exp(-d / o)
                else:
                    erde = c_fp
                ttd = d
            erdes.append(erde)
            ttds.append(ttd)
        return np.mean(erdes), np.nanmean(ttds)
    
    for o in o_list:
        erde, ttd = compute_ERDE_TTD(o=o)
        results[f"ERDE@{o}"] = erde
        results[f"TTD@{o}"] = ttd
    
    # --- Early Precision & Recall ---
    def early_precision_recall(k=50):
        tp, fn = 0, 0
        for user_id, group in df.groupby("user_id"):
            group = group.sort_values("day")
            true_days = group[group["true_label"] == target_label]["day"].values
            if len(true_days) == 0:
                continue
            detected_days = group[group["pred_label"] == target_label]["day"].values
            if len(detected_days) == 0:
                fn += 1
            elif detected_days[0] <= k:
                tp += 1
            else:
                fn += 1
        
        # False positives: Healthy users predicted as target_label
        fp_users = df[df["true_label"] != target_label].groupby("user_id").filter(
            lambda g: (g["pred_label"] == target_label).any()
        )["user_id"].unique()
        
        precision = tp / (tp + len(fp_users) + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return precision, recall
    
    for k in k_list:
        ep, er = early_precision_recall(k=k)
        results[f"EP@{k}"] = ep
        results[f"ER@{k}"] = er
    
    # --- Detection Curve ---
    detections = []
    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("day")
        detected_days = group[group["pred_label"] == target_label]["day"].values
        if len(detected_days) > 0:
            detections.append(detected_days[0])
    detections = np.array(detections)
    days = np.arange(1, df["day"].max()+1)
    detection_rate = [(detections <= d).mean() for d in days]
    
    plt.figure(figsize=(10,5))
    plt.plot(days, detection_rate, label=f"{target_label} detection curve")
    plt.xlabel("Day")
    plt.ylabel("Proportion Detected")
    plt.title(f"Time-to-Detection Curve ({target_label})")
    plt.legend()
    plt.show()

    # --- Histogram of Detection Days ---
    if len(detections) > 0:
        plt.figure(figsize=(8,4))
        plt.hist(detections, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        plt.xlabel("First Detection Day")
        plt.ylabel("Number of Users")
        plt.title(f"Histogram of First Detection Days ({target_label})")
        plt.show()
    
    return results


# ========================
# Example usage:
# ========================

# Load your dataframe (replace with your CSV)
df = pd.read_csv("200simulated_users_groq_metrics.csv")

# Map to correct columns
df = df.rename(columns={"label":"true_label"})

# Add dummy pred_label column (replace with your model output!)
# Here we assume 'Healthy' stays Healthy, and randomly flag some as 'MCI'
np.random.seed(42)
df["pred_label"] = df["true_label"]
flip_mask = np.random.rand(len(df)) < 0.05   # simulate 5% misclassified
df.loc[flip_mask, "pred_label"] = "MCI"

# Run evaluation
report = evaluate_early_detection(df, target_label="MCI", o_list=[100,200], k_list=[50,100])

print(pd.Series(report))
