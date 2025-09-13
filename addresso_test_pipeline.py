import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch

# --------------------
# Config / paths
# --------------------
TRAIN_BASE = r"D:/conference_submit/realdata/ADReSSo21_train/diagnosis/train"
TEST_BASE  = r"D:/conference_submit/realdata/ADReSSo21_test/diagnosis/test-dist"

# Train subpaths
TRAIN_AUDIO = os.path.join(TRAIN_BASE, "audio")   # has subfolders ad/ cn/
TRAIN_MMSE  = os.path.join(TRAIN_BASE, "adresso-train-mmse-scores.csv")

# Test subpaths
TEST_AUDIO = os.path.join(TEST_BASE, "audio")     # test audio folder(s)
TEST_SEG   = os.path.join(TEST_BASE, "segmentation")  # if needed (not used for ASR here)

# Output dirs
OUT_DIR = r"D:/conference_submit/addresso_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
ASR_TASK1_DIR = os.path.join(OUT_DIR, "asr_task1")
ASR_TASK2_DIR = os.path.join(OUT_DIR, "asr_task2")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(ASR_TASK1_DIR, exist_ok=True)
os.makedirs(ASR_TASK2_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------
# Utils
# --------------------
np.random.seed(42)
torch.manual_seed(42)
smoothie = SmoothingFunction().method4

def load_mmse(mmse_file):
    df = pd.read_csv(mmse_file)
    # Normalize columns: ensure adressfname, dx, mmse exist
    if "adressfname" not in df.columns and "adressfname" not in df.columns:
        raise ValueError("MMSE CSV missing column 'adressfname'")
    df["dx"] = df["dx"].str.upper()
    df["label"] = df["dx"]   # AD / CN
    return df

# --------------------
# ASR (faster-whisper)
# --------------------
print("Loading faster-whisper ASR model (base)...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_and_save(wav_path, out_folder):
    file_id = os.path.splitext(os.path.basename(wav_path))[0]
    out_txt = os.path.join(out_folder, f"{file_id}.txt")
    if os.path.exists(out_txt):
        # if already transcribed, reuse
        with open(out_txt, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        return file_id, text

    try:
        segments, info = asr_model.transcribe(wav_path, beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip().lower()
    except Exception as e:
        print("ASR failed for", wav_path, e)
        text = ""

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(text)
    return file_id, text

# --------------------
# Build transcripts for train (if not already saved)
# --------------------
def transcribe_dir_for_group(audio_group_dir, label, out_asr_dir):
    recs = []
    paths = sorted(glob.glob(os.path.join(audio_group_dir, "*.wav")))
    for wav in tqdm(paths, desc=f"ASR {label}"):
        fid, txt = transcribe_and_save(wav, out_asr_dir)
        recs.append({"id": fid, "transcript": txt, "label": label})
    return recs

# Transcribe train audio (AD and CN)
print("Transcribing TRAIN audio (may take a while)...")
train_records = []
for grp in ["ad", "cn"]:
    grp_dir = os.path.join(TRAIN_AUDIO, grp)
    if os.path.isdir(grp_dir):
        train_records.extend(transcribe_dir_for_group(grp_dir, grp.upper(), ASR_TASK1_DIR))
    else:
        print("Warning: train group folder missing:", grp_dir)

train_df = pd.DataFrame(train_records)
print("Collected train transcripts:", train_df.shape)

# Transcribe test audio and save in asr_task1 and asr_task2 (same transcripts)
print("Transcribing TEST audio (test-dist)...")
test_records = []
# test structure may be flat or have subfolders; find all wavs under TEST_AUDIO
test_wavs = sorted(glob.glob(os.path.join(TEST_AUDIO, "**", "*.wav"), recursive=True))
for wav in tqdm(test_wavs, desc="ASR TEST"):
    fid, txt = transcribe_and_save(wav, ASR_TASK1_DIR)   # save into asr_task1
    # also duplicate into asr_task2 folder per submission requirement
    with open(os.path.join(ASR_TASK2_DIR, f"{fid}.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    test_records.append({"id": fid, "transcript": txt})

test_df = pd.DataFrame(test_records)
print("Collected test transcripts:", test_df.shape)

# --------------------
# NLP models for metrics
# --------------------
print("Loading SBERT model and ROUGE scorer...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# --------------------
# Prepare references: CN texts from train
# --------------------
cn_texts_train = train_df[train_df["label"]=="CN"]["transcript"].dropna().tolist()
cn_concat = " ".join(cn_texts_train) if cn_texts_train else ""

if len(cn_texts_train)==0:
    print("Warning: no CN references in train transcripts. Text metrics may be meaningless.")

# Precompute CN embeddings mean
if cn_texts_train:
    cn_embs = sbert.encode(cn_texts_train, convert_to_tensor=True)
    cn_mean_emb = cn_embs.mean(dim=0)
else:
    cn_mean_emb = None

# --------------------
# Feature computation function
# --------------------
def compute_text_metrics_for_df(df_in, cn_texts, cn_concat, cn_mean_emb):
    sbert_sims, bleus, rouges = [], [], []
    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Computing metrics"):
        text = (row["transcript"] or "").strip()
        if not text:
            sbert_sims.append(0.0)
            bleus.append(0.0)
            rouges.append(0.0)
            continue

        # SBERT similarity
        if cn_mean_emb is not None:
            try:
                emb = sbert.encode(text, convert_to_tensor=True)
                sim = float(util.cos_sim(emb, cn_mean_emb))
            except Exception:
                sim = 0.0
        else:
            sim = 0.0
        sbert_sims.append(sim)

        # BLEU: using every CN text as a reference (tokenized)
        try:
            bleu = sentence_bleu([t.split() for t in cn_texts], text.split(), smoothing_function=smoothie) if cn_texts else 0.0
        except Exception:
            bleu = 0.0
        bleus.append(bleu)

        # ROUGE-L: compare to concatenated CN text
        try:
            rouge = scorer.score(cn_concat, text)["rougeL"].fmeasure if cn_concat else 0.0
        except Exception:
            rouge = 0.0
        rouges.append(rouge)

    df_in = df_in.copy()
    df_in["sbert_sim"] = sbert_sims
    df_in["bleu"] = bleus
    df_in["rouge_l"] = rouges
    return df_in

# Compute metrics on train and test
print("Computing metrics on TRAIN...")
train_df = compute_text_metrics_for_df(train_df, cn_texts_train, cn_concat, cn_mean_emb)
print("Computing metrics on TEST...")
test_df = compute_text_metrics_for_df(test_df, cn_texts_train, cn_concat, cn_mean_emb)

# Save intermediate CSVs
train_df.to_csv(os.path.join(OUT_DIR, "train_transcripts_with_metrics.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test_transcripts_with_metrics.csv"), index=False)

# --------------------
# Merge with MMSE labels (train)
# --------------------
mmse_df = load_mmse(TRAIN_MMSE)
merged = train_df.merge(mmse_df, left_on="id", right_on="adressfname", how="inner")
# Clean columns
merged = merged.drop(columns=["adressfname", "dx"], errors="ignore")
# If both label columns exist, unify
if "label_x" in merged.columns and "label_y" in merged.columns:
    merged = merged.drop(columns=["label_x"], errors="ignore").rename(columns={"label_y":"label"})
elif "label" not in merged.columns:
    merged["label"] = merged["label_x"] if "label_x" in merged.columns else merged.get("label_y","")

# Map labels to numeric for classification
merged["label_bin"] = merged["label"].map({"CN":0, "AD":1})

# Save merged
merged.to_csv(os.path.join(OUT_DIR, "train_merged_with_mmse.csv"), index=False)

# --------------------
# Quick EDA plots
# --------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="sbert_sim", data=merged)
plt.title("SBERT similarity by label (train)")
plt.savefig(os.path.join(PLOTS_DIR, "sbert_sim_by_label.png"))
plt.close()

# Cross-validation: classifier on text features
X = merged[["sbert_sim", "bleu", "rouge_l"]].fillna(0.0)
y = merged["label_bin"].fillna(0).astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_log = LogisticRegression(max_iter=2000)
clf_rf  = RandomForestClassifier(n_estimators=200, random_state=42)

print("Cross-validating Logistic Regression...")
cv_scores_log = cross_val_score(clf_log, X, y, cv=skf, scoring="f1")
print("Logistic F1 scores:", cv_scores_log, "mean:", cv_scores_log.mean())

print("Cross-validating RandomForest...")
cv_scores_rf = cross_val_score(clf_rf, X, y, cv=skf, scoring="f1")
print("RandomForest F1 scores:", cv_scores_rf, "mean:", cv_scores_rf.mean())

# Save CV boxplot
plt.figure(figsize=(6,4))
sns.boxplot(data=[cv_scores_log, cv_scores_rf])
plt.xticks([0,1], ["LogReg", "RandomForest"])
plt.ylabel("F1 score (5-fold)")
plt.savefig(os.path.join(PLOTS_DIR, "cv_f1_boxplot.png"))
plt.close()

# Fit final classifier(s) on whole train
clf_log.fit(X, y)
clf_rf.fit(X, y)

# Save models
joblib.dump(clf_log, os.path.join(OUT_DIR, "clf_log.joblib"))
joblib.dump(clf_rf, os.path.join(OUT_DIR, "clf_rf.joblib"))

# --------------------
# Predict on TEST (classification task 1)
# --------------------
X_test_feats = test_df[["sbert_sim", "bleu", "rouge_l"]].fillna(0.0)
pred_log = clf_log.predict(X_test_feats)
pred_rf  = clf_rf.predict(X_test_feats)

# Format outputs to submission CSV format:
# ADReSSo expects two columns: filename / Prediction (0 for CN, 1 for AD)
# We'll create two submission files: one for logistic, one for RF (these are your up to-5 variants).
sub_ids = test_df["id"].tolist()

out_df_log = pd.DataFrame({"id": sub_ids, "Prediction": [int(x) for x in pred_log]})
out_df_rf  = pd.DataFrame({"id": sub_ids, "Prediction": [int(x) for x in pred_rf]})

out_df_log.to_csv(os.path.join(OUT_DIR, "test_results-task1-1.csv"), index=False)
out_df_rf.to_csv(os.path.join(OUT_DIR, "test_results-task1-2.csv"), index=False)
print("Saved task1 submission files:", 
      os.path.join(OUT_DIR, "test_results-task1-1.csv"),
      os.path.join(OUT_DIR, "test_results-task1-2.csv"))

# Also save ASR outputs for test (already saved per-file into ASR_TASK1_DIR and ASR_TASK2_DIR)
# Zip or leave folders as-is for submission.
print("ASR transcripts saved in:", ASR_TASK1_DIR, ASR_TASK2_DIR)

# --------------------
# Predict MMSE (regression) -> Task 2
# --------------------
# Use ridge regression on same features as a simple baseline
X_reg = merged[["sbert_sim", "bleu", "rouge_l"]].fillna(0.0)
y_reg = merged["mmse"].astype(float)

reg = Ridge(alpha=1.0, random_state=42)
# simple CV for MSE
from sklearn.model_selection import cross_val_score
mse_scores = -cross_val_score(reg, X_reg, y_reg, cv=5, scoring="neg_mean_squared_error")
print("MMSE CV MSE (5-fold):", mse_scores, "mean:", mse_scores.mean())

reg.fit(X_reg, y_reg)
pred_mmse = reg.predict(X_test_feats)

out_df_mmse = pd.DataFrame({"id": sub_ids, "Prediction": [float(x) for x in pred_mmse]})
out_df_mmse.to_csv(os.path.join(OUT_DIR, "test_results-task2-1.csv"), index=False)
print("Saved task2 submission file:", os.path.join(OUT_DIR, "test_results-task2-1.csv"))

# Save combined results and plots
test_df_out = test_df.copy()
test_df_out["pred_log"] = pred_log
test_df_out["pred_rf"] = pred_rf
test_df_out["pred_mmse"] = pred_mmse
test_df_out.to_csv(os.path.join(OUT_DIR, "test_predictions_with_metrics.csv"), index=False)

print("\nDone. Outputs written under:", OUT_DIR)
