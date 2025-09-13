import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# === Load your dataframe ===
# Must contain: user_id, day, label, summary
df = pd.read_csv("noisy_200simulated_users_groq_metrics.csv")

# === Models ===
sbert = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1

def compute_similarity_metrics(group):
    """Compute BLEU, ROUGE-L, SBERT similarity for one user against baseline (days 1-5)."""
    # Define baseline: choose day 1–5 summary with highest coherence (or first summary)
    baseline_texts = group[group["day"] <= 5]["summary"].tolist()
    if not baseline_texts:
        return pd.DataFrame()
    
    baseline = baseline_texts[0]  # (alternative: pick best coherence score if available)
    
    baseline_emb = sbert.encode(baseline, convert_to_tensor=True)
    
    results = []
    for _, row in group.iterrows():
        text = row["summary"]
        
        # BLEU
        bleu = sentence_bleu([baseline.split()], text.split(), smoothing_function=smooth_fn)
        
        # ROUGE-L
        rouge_l = rouge.score(baseline, text)["rougeL"].fmeasure
        
        # Embedding similarity
        emb = sbert.encode(text, convert_to_tensor=True)
        sim = util.cos_sim(baseline_emb, emb).item()
        
        results.append({
            "user_id": row["user_id"],
            "label": row["label"],
            "day": row["day"],
            "BLEU": bleu,
            "ROUGE-L": rouge_l,
            "EmbeddingSim": sim
        })
    return pd.DataFrame(results)

# === Apply per user ===
metrics_df = df.groupby("user_id").apply(compute_similarity_metrics).reset_index(drop=True)

# === Aggregate by label ===
table = metrics_df.groupby("label")[["BLEU","ROUGE-L","EmbeddingSim"]].mean().reset_index()

print("📊 Linguistic Similarity Metrics Relative to Baseline (Days 1–5):")
print(table.round(3))

# Save to CSV if needed
table.to_csv("linguistic_similarity_results.csv", index=False)
