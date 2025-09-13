import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.metrics import classification_report
# --- Load CSV ---
df = pd.read_csv("simulated_AD_users_groq_with_metrics.csv")
df["timestamp"] = pd.to_datetime(df["date"])

# --- Load or train model ---
clf, scaler = joblib.load("models/alzheimers_model_noisy.pkl")  # Must be trained already

# --- Predict labels ---
feature_cols = ["watch_time_secs", "skipped_secs", "pause_count", "replay_count", "liked", "shared", "coherence_score"]
X = scaler.transform(df[feature_cols])
df["predicted_label"] = clf.predict(X)

# --- Visualize coherence per user ---
users = df["user_id"].unique()
for user in users:
    user_df = df[df["user_id"] == user]
    fig = px.line(user_df, x="timestamp", y="coherence_score", color="predicted_label", title=f"Coherence Drift - {user}")
    st.plotly_chart(fig)

# --- Optional: Display confusion matrix ---
st.text("Model Performance:\n")
st.text(classification_report(df["label"], df["predicted_label"]))
