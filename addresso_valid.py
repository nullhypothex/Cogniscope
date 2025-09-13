import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import torch

# === Reproducibility ===
np.random.seed(42)
torch.manual_seed(42)

# === Paths (adjust if needed) ===
base_dir = r"D:/conference_submit/realdata/ADReSSo21_train/diagnosis/train"
audio_dir = os.path.join(base_dir, "audio")
mmse_file = os.path.join(base_dir, "adresso-train-mmse-scores.csv")

# === Load MMSE + diagnosis labels ===
mmse_df = pd.read_csv(mmse_file)
mmse_df["dx"] = mmse_df["dx"].str.upper()   # AD / CN
mmse_df["label"] = mmse_df["dx"]           # unified column
print("📂 MMSE file loaded:")
print(mmse_df.head())

# === Load faster-whisper ASR model ===
print("🔊 Loading faster-whisper model...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(file_path):
    """Transcribe a WAV file using faster-whisper."""
    try:
        segments, info = asr_model.transcribe(file_path, beam_size=5)
        return " ".join([seg.text for seg in segments]).strip().lower()
    except Exception as e:
        print(f"⚠️ Transcription failed for {file_path}: {e}")
        return ""

# === Collect transcripts from audio files ===
records = []
for group in ["ad", "cn"]:
    files = glob.glob(os.path.join(audio_dir, group, "*.wav"))
    for f in tqdm(files, desc=f"Transcribing {group.upper()}"):
        file_id = os.path.splitext(os.path.basename(f))[0]
        text = transcribe_audio(f)
        records.append({"id": file_id, "transcript": text, "label": group.upper()})

df = pd.DataFrame(records)
print(f"✅ Collected {len(df)} transcripts")
print(df.head())

# === Load NLP models ===
sbert = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smoothie = SmoothingFunction().method4

# === Prepare CN reference texts ===
cn_texts = df[df["label"]=="CN"]["transcript"].tolist()
cn_text_concat = " ".join(cn_texts)
cn_embeddings = sbert.encode(cn_texts, convert_to_tensor=True)
baseline_embedding = cn_embeddings.mean(dim=0)

# === Compute metrics per transcript ===
sbert_sims, bleus, rouges = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing metrics"):
    text = row["transcript"]
    if not text:
        sbert_sims.append(0.0)
        bleus.append(0.0)
        rouges.append(0.0)
        continue

    # SBERT similarity vs mean CN embedding
    try:
        emb = sbert.encode(text, convert_to_tensor=True)
        sim = float(util.cos_sim(emb, baseline_embedding))
    except:
        sim = 0.0
    sbert_sims.append(sim)

    # BLEU vs all CN references
    bleu = sentence_bleu([t.split() for t in cn_texts], text.split(), smoothing_function=smoothie)
    bleus.append(bleu)

    # ROUGE-L vs concatenated CN
    rouge = scorer.score(cn_text_concat, text)["rougeL"].fmeasure
    rouges.append(rouge)

df["sbert_sim"] = sbert_sims
df["bleu"] = bleus
df["rouge_l"] = rouges

print(df.head())

# === Merge transcripts with MMSE labels ===
merged = df.merge(mmse_df, left_on="id", right_on="adressfname", how="inner")

# Drop duplicate columns and keep single label
merged = merged.drop(columns=["adressfname", "dx", "label_x"], errors='ignore')
merged.rename(columns={"label_y":"label"}, inplace=True)

# Encode labels for classifier
y = merged["label"].map({"CN":0, "AD":1})
X = merged[["sbert_sim", "bleu", "rouge_l"]]

print(f"✅ Merged dataset size: {merged.shape}")
print("Columns after merge:", merged.columns)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === Results ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["CN","AD"]))

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save dataset with metrics ===
merged.to_csv("adressso21_with_metrics.csv", index=False)
print("\n💾 Saved dataset with metrics -> adressso21_with_metrics.csv")
