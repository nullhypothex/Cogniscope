import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import statsmodels.api as sm

# === Load dataset ===
df = pd.read_csv("noisy_200simulated_users_groq_metrics.csv")

# --- Step 1: Drift slope (linear regression per user) ---
slopes = []
for uid, user_df in df.groupby("user_id"):
    if user_df['day'].nunique() < 2:  # skip if only one day
        continue
    X = sm.add_constant(user_df['day'])
    y = user_df['coherence']
    model = sm.OLS(y, X).fit()
    slopes.append({"user_id": uid, "slope": model.params['day']})
df_slopes = pd.DataFrame(slopes)

# --- Step 2: Behavioral entropy (per row) ---
df['behavior_entropy'] = df[['pause_count','replay_count','liked','shared']].apply(
    lambda row: entropy(np.array(row)+1e-6), axis=1
)

# --- Step 3: Aggregate user-level features ---
user_features = df.groupby("user_id").agg({
    "coherence":["mean","std"],
    "watch_time_secs":"mean",
    "pause_count":"mean",
    "replay_count":"mean",
    "behavior_entropy":"mean"
})
user_features.columns = ["_".join(col).strip() for col in user_features.columns.values]
user_features = user_features.merge(df_slopes, on="user_id", how="left")

# --- Step 4: Add majority label per user ---
user_labels = df.groupby("user_id")['label'].agg(lambda x: x.mode()[0])
user_features = user_features.merge(user_labels, on="user_id", how="left")

print("\n✅ Engineered features:\n", user_features.head())

# --- Step 5: Train/test split ---
X = user_features.drop(columns=["label"])
y = user_features["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# --- Step 6: Logistic Regression Classifier ---
clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial")
clf.fit(X_train, y_train)

# --- Step 7: Evaluation ---
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Step 8: AUC (macro average across classes) ---
roc_auc = roc_auc_score(pd.get_dummies(y_test), y_prob, average="macro")
print(f"\n🔥 Macro AUC: {roc_auc:.3f}")

# --- Save model ---
import joblib
joblib.dump(clf, "cogniscope_classifier.pkl")
print("\n✅ Model saved as cogniscope_classifier.pkl")

# Save engineered features for later statistical evaluation
user_features.to_csv("user_features.csv", index=False)
print("✅ Saved engineered features to user_features.csv")
