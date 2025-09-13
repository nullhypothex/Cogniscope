import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ----------------------------
# Utility: Plot classification report
# ----------------------------
def save_classification_report(y_true, y_pred, use_coherence, use_behavior, target_names=None, suffix=""):
    report = classification_report(y_true, y_pred, digits=3, output_dict=True, target_names=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.3f', cbar=False, ax=ax)
    ax.set_title(f"Classification Report\nBehavior={use_behavior}, Coherence={use_coherence}")
    plt.tight_layout()
    plt.savefig(f"classification_report_Behavior={use_behavior}_Coherence={use_coherence}{suffix}.png")
    plt.close(fig)

# ----------------------------
# Utility: Confusion matrix
# ----------------------------
def save_confusion_matrix(y_true, y_pred, use_coherence, use_behavior, target_names=None, suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix\nBehavior={use_behavior}, Coherence={use_coherence}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_Behavior={use_behavior}_Coherence={use_coherence}{suffix}.png")
    plt.close(fig)

# ----------------------------
# Train + evaluate model
# ----------------------------
def run_ablation_model(df, use_coherence=True, use_behavior=True, label_col='label', suffix=""):
    features = []
    if use_behavior:
        features += ['watch_time_secs', 'pause_count', 'replay_count', 'liked', 'shared']
    if use_coherence:
        features += ['coherence', 'BLEU', 'ROUGE_L', 'embedding_similarity']

    X = df[features]
    y = df[label_col]

    # Encode labels if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        target_names = le.classes_
    else:
        y_enc = y.values
        target_names = None

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    print(f"\n✅ Setting: Behavior={use_behavior}, Coherence={use_coherence}")
    print(classification_report(y_val, y_pred, digits=3, target_names=target_names))

    save_classification_report(y_val, y_pred, use_coherence, use_behavior, target_names, suffix)
    save_confusion_matrix(y_val, y_pred, use_coherence, use_behavior, target_names, suffix)

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    # === Load CLEAN (no label-conditioned summaries) dataset ===
    # This dataset must be generated with summaries independent of label
    df = pd.read_csv("no_label_conditioned_users.csv")

    print("🔬 FULL MODEL")
    run_ablation_model(df, use_coherence=True, use_behavior=True, suffix="_noLabel")

    print("\n❌ COHERENCE ONLY")
    run_ablation_model(df, use_coherence=True, use_behavior=False, suffix="_noLabel")

    print("\n❌ BEHAVIOR ONLY")
    run_ablation_model(df, use_coherence=False, use_behavior=True, suffix="_noLabel")

    # === Optional: Compare against older label-conditioned dataset for delta ===
    df_lc = pd.read_csv("noisy_200simulated_users_groq_metrics.csv")
    print("\n🔬 FULL MODEL (Label-conditioned)")
    run_ablation_model(df_lc, use_coherence=True, use_behavior=True, suffix="_labelCond")
