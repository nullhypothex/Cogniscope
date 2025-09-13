import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind

# Load your engineered features
df = pd.read_csv("user_features1.csv")  # output from your earlier step

X = df.drop(columns=["label", "user_id"])
y = df["label"]

# --- Stratified split to preserve MCI, Healthy, EarlyAD ratios ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- Class-weighted Logistic Regression (doesn't change labels, just balances loss) ---
clf = LogisticRegression(
    max_iter=2000, 
    solver="lbfgs", 
    multi_class="multinomial", 
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# --- Evaluation ---
y_pred = clf.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Statistical separability (t-tests + Cohen's d) ---
def cohens_d(x, y):
    return (x.mean() - y.mean()) / np.sqrt((x.std()**2 + y.std()**2) / 2)

for feat in ["coherence_mean", "behavior_entropy_mean", "slope"]:
    healthy = df[df.label=="Healthy"][feat]
    mci = df[df.label=="MCI"][feat]
    earlyad = df[df.label=="EarlyAD"][feat]

    print(f"\n=== {feat} separability ===")
    print("Healthy vs MCI:", ttest_ind(healthy, mci, equal_var=False), 
          "Cohen's d:", cohens_d(healthy, mci))
    print("MCI vs EarlyAD:", ttest_ind(mci, earlyad, equal_var=False), 
          "Cohen's d:", cohens_d(mci, earlyad))
