import os
import random
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# === API Setup ===
GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

GROQ_USER_IDS = {1, 4, 7, 12, 15, 24, 26, 28, 34, 37, 40, 46, 50, 54, 57, 62, 64,
                 68, 72, 78, 82, 84, 89, 90, 92, 94, 96, 100, 108, 114, 116,
                 118, 120, 123, 129, 135, 144, 147, 150, 154, 163, 166,
                 171, 174, 178, 182, 188, 191, 196, 200}

TOTAL_USERS, TOTAL_DAYS, VIDEOS_PER_DAY = 200, 200, 5
CSV_PATH = "no_label_conditioned_users.csv"

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

video_descriptions = {
    "Funny Cat": "A funny cat jumps into a box and surprises its owner.",
    "Yoga Basics": "A yoga instructor guides a basic stretching session.",
    "Cooking Pasta": "Steps to cook creamy mushroom pasta in 10 minutes.",
    "World News": "A reporter discusses recent global economic events.",
    "Memory Tips": "An expert shares techniques to improve memory recall.",
    "Bird Migration": "Learn how birds navigate thousands of miles.",
    "Rainforest Sounds": "Relaxing rainforest ambient background sounds.",
    "DIY Bookshelf": "Build a simple bookshelf with minimal tools.",
    "Space Update": "NASA announces new space telescope discoveries.",
    "Classic Music": "A Mozart piano sonata performed by a child prodigy."
}
video_titles = list(video_descriptions.keys())

# === Neutral templates (no label leakage) ===
NEUTRAL_TEMPLATES = [
    "This video explained the topic briefly.",
    "I remember a few main points, but not all details.",
    "The video gave an overview of the subject.",
    "It discussed some ideas with examples.",
    "This was a short clip with some highlights."
]

# === Engagement priors (still label-dependent, clinically grounded) ===
def simulate_engagement_metrics(label):
    if label == "Healthy":
        return np.random.uniform(60, 75), np.random.randint(0, 2), np.random.randint(0, 1), np.random.binomial(1, 0.35), np.random.binomial(1, 0.20)
    if label == "MCI":
        return np.random.uniform(40, 60), np.random.randint(1, 3), np.random.randint(1, 2), np.random.binomial(1, 0.20), np.random.binomial(1, 0.10)
    if label == "EarlyAD":
        return np.random.uniform(20, 40), np.random.randint(2, 5), np.random.randint(2, 4), np.random.binomial(1, 0.08), np.random.binomial(1, 0.03)

# === Progression profiles ===
def get_label_by_day(progression_type, day):
    if progression_type == "StableHealthy":
        return "Healthy"
    elif progression_type == "GradualDecliner":
        return "Healthy" if day < TOTAL_DAYS//3 else "MCI" if day < 2*TOTAL_DAYS//3 else "EarlyAD"
    elif progression_type == "EarlyConverter":
        return "MCI" if day < TOTAL_DAYS//4 else "EarlyAD"

# === Summary generator ===
def generate_summary(video_title, use_groq=True, no_label=True, cognitive_state=None, max_retries=5):
    """If no_label=True → generate neutral text (no leakage)."""
    if no_label:
        # Neutral Groq prompt
        prompt = f"Generate a short, vague spoken-style summary for a video titled '{video_title}'. Keep it around 2–3 sentences."
    else:
        # Label-conditioned Groq prompt (original)
        prompt = f"Generate a short summary for a person with {cognitive_state} cognitive condition, after watching '{video_title}'."

    # Try Groq API with retries
    if use_groq:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    GROQ_API_URL,
                    headers=HEADERS,
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 128
                    },
                    timeout=20
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                else:
                    print(f"⚠️ Groq HTTP {response.status_code}, attempt {attempt+1}/{max_retries}")
            except Exception as e:
                print(f"⚠️ Groq error on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(1.5)  # backoff between retries

        print(f"⚠️ All {max_retries} Groq attempts failed → using fallback.")

    # Fallback: neutral or label-conditioned templates
    if no_label:
        return random.choice(NEUTRAL_TEMPLATES)
    else:
        label_templates = [
            f"I watched the video on '{video_title}', it was interesting.",
            f"The video titled '{video_title}' explained some details, I recall a bit.",
            f"'{video_title}' was shown, I think it covered a few points."
        ]
        return random.choice(label_templates)


# === Simulation loop ===
def simulate_users(no_label=True):
    # Make sure file exists with header
    if not os.path.exists(CSV_PATH):
        pd.DataFrame().to_csv(CSV_PATH, index=False)

    for user_id in tqdm(range(1, TOTAL_USERS + 1), desc="Simulating Users"):
        progression_type = (
            "StableHealthy" if user_id % 4 == 0 else
            "GradualDecliner" if user_id % 4 == 1 else
            "EarlyConverter"
        )
        use_groq = user_id in GROQ_USER_IDS

        baseline_coherence = np.random.uniform(0.7, 0.9)
        drift = np.random.uniform(-0.001, -0.0005)

        rows = []  # buffer for this user

        for day in range(1, TOTAL_DAYS + 1):
            label = get_label_by_day(progression_type, day)
            date = datetime.today() + timedelta(days=day)

            for v in range(VIDEOS_PER_DAY):
                video_title = random.choice(video_titles)
                summary = generate_summary(
                    video_title,
                    use_groq=use_groq,
                    no_label=no_label,
                    cognitive_state=label
                )

                # Coherence drift (no label conditioning)
                coherence = baseline_coherence + drift * day + np.random.normal(0, 0.02)
                coherence = float(np.clip(coherence, 0, 1))

                # Engagement (still label-based)
                wt, pause, replay, liked, shared = simulate_engagement_metrics(label)

                rows.append({
                    "user_id": user_id,
                    "day": day,
                    "date": date.strftime("%Y-%m-%d"),
                    "video_title": video_title,
                    "label": label,
                    "summary": summary,
                    "coherence": coherence,
                    "watch_time_secs": round(wt),
                    "pause_count": pause,
                    "replay_count": replay,
                    "liked": liked,
                    "shared": shared,
                    "used_groq": use_groq,
                    "no_label_mode": no_label
                })

            # Save every 20 days
            if day % 20 == 0:
                df = pd.DataFrame(rows)
                df.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
                rows.clear()
                print(f"💾 Saved User {user_id} at day {day}")

        # Save any remaining rows at end of user
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
            print(f"✅ Completed User {user_id} and saved data")


if __name__ == "__main__":
    simulate_users(no_label=True)
