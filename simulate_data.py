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
import os
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = "___"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}
headers = HEADERS

GROQ_USER_IDS = {1, 4, 7, 12, 15, 24, 26, 28, 34, 37, 40, 46, 50, 54, 57, 62, 64,
                 68, 72, 78, 82, 84, 89, 90, 92, 94, 96, 100, 108, 114, 116,
                 118, 120, 123, 129, 135, 144, 147, 150, 154, 163, 166,
                 171, 174, 178, 182, 188, 191, 196, 200}

CSV_PATH = "new_simulated_users_groq.csv"
TOTAL_USERS = 200
TOTAL_DAYS = 200
VIDEOS_PER_DAY = 5

summary_cache = {}
last_request_time = 0
RATE_LIMIT_SECONDS = 1.5

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


video_metadata = {
    "Funny Cat": ("Entertainment", "Animals"),
    "Yoga Basics": ("Health", "Fitness"),
    "Cooking Pasta": ("Food", "Cooking"),
    "World News": ("News", "Politics"),
    "Memory Tips": ("Health", "Cognitive"),
    "Bird Migration": ("Nature", "Animals"),
    "Rainforest Sounds": ("Nature", "Relaxation"),
    "DIY Bookshelf": ("DIY", "Home"),
    "Space Update": ("Science", "Space"),
    "Classic Music": ("Music", "Education")
}

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

def add_label_noise(summary, label, day):
    filler_phrases = [
        "Umm, I think?", "Then again, maybe not.", "Anyway...",
        "It was kind of odd.", "Sort of confusing.",
        "I forget a bit here.", "Not totally sure though."
    ]
    sentences = summary.split('.')
    cleaned = [s.strip() for s in sentences if s.strip()]

    if label == "Healthy":
        final = ". ".join(cleaned)
    elif label == "MCI":
        final = ". ".join([
            s + " " + random.choice(filler_phrases) if random.random() < 0.4 else s
            for s in cleaned
        ])
    else:
        final = ". ".join([
            s + " " + random.choice(filler_phrases) if random.random() < 0.8 else s
            for s in cleaned
        ])
    return f"Day {day}: {final.strip()}."

def simulate_engagement_metrics(label):
    if label == "Healthy":
        return np.random.normal(280, 20), np.random.normal(5, 2), np.random.poisson(1), np.random.binomial(1, 0.05), np.random.binomial(1, 0.8), np.random.binomial(1, 0.6)
    if label == "MCI":
        return np.random.normal(240, 30), np.random.normal(15, 5), np.random.poisson(3), np.random.binomial(1, 0.1), np.random.binomial(1, 0.5), np.random.binomial(1, 0.3)
    if label == "EarlyAD":
        return np.random.normal(200, 40), np.random.normal(30, 10), np.random.poisson(5), np.random.binomial(1, 0.15), np.random.binomial(1, 0.3), np.random.binomial(1, 0.1)

def get_label_by_day(progression_type, day):
    if progression_type == "StableHealthy":
        return "Healthy"
    elif progression_type == "GradualDecliner":
        return "Healthy" if day < TOTAL_DAYS//3 else "MCI" if day < 2*TOTAL_DAYS//3 else "EarlyAD"
    elif progression_type == "EarlyConverter":
        return "MCI" if day < TOTAL_DAYS//4 else "EarlyAD"

def wait_for_rate_limit():
    global last_request_time
    elapsed = time.time() - last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    last_request_time = time.time()

def generate_summary(video_title, cognitive_state, use_groq=True, max_retries=10):
    key = (video_title, cognitive_state)
    if key in summary_cache:
        return summary_cache[key]

    prompt = (
        f"Generate a short, somewhat vague, spoken-style summary for a person with {cognitive_state} cognitive condition, "
        f"after watching a video titled '{video_title}'. The summary should reflect working memory limitations typical of "
        f"{cognitive_state} individuals. Keep it around 2-3 sentences, as if recounted by someone trying to remember details."
    )

    models_to_try = ["llama3-8b-8192", "gemma-7b-it"]

    for model_name in models_to_try:
        if use_groq:
            for retry in range(max_retries):
                wait_for_rate_limit()
                try:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": "You are a memory-impaired video summarizer."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 256
                        },
                        timeout=30
                    )
                    if response.status_code == 200:
                        summary = response.json()["choices"][0]["message"]["content"].strip()
                        summary_cache[key] = summary
                        return summary
                    elif response.status_code == 429:
                        wait_time = min(2 ** retry, 60)
                        print(f"⚠️ Retry {retry+1}/{max_retries} (Rate limit): Waiting {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"⚠️ Retry {retry+1}/{max_retries}: Groq {model_name} failed with {response.status_code}")
                        print("Response:", response.text)
                        time.sleep(2)
                except Exception as e:
                    print(f"⚠️ Retry {retry+1}/{max_retries}: Exception - {e}")
                    time.sleep(2)

    print(f"⚠️ All models failed for {key} → using fallback.")
    fallback_templates = {
        "Healthy": f"I watched the video on '{video_title}', it was interesting and I remember quite a bit.",
        "MCI": f"There was a video about '{video_title}', I think... It had some useful things.",
        "EarlyAD": f"Umm... it was about '{video_title}', but I’m not sure I followed all of it."
    }
    summary = fallback_templates.get(cognitive_state, f"The video titled '{video_title}' was... I can't recall clearly.")
    summary_cache[key] = summary
    return summary

def simulate_users():
    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    existing_users_days = set(zip(existing["user_id"], existing["day"])) if not existing.empty else set()

    output_rows = []

    for user_id in tqdm(range(1, TOTAL_USERS + 1), desc="Simulating Users"):
        progression_type = (
            "StableHealthy" if user_id % 4 == 0 else
            "GradualDecliner" if user_id % 4 == 1 else
            "EarlyConverter"
        )
        use_groq = user_id in GROQ_USER_IDS

        for day in range(1, TOTAL_DAYS + 1):
            if (user_id, day) in existing_users_days:
                continue  # Skip already saved

            label = get_label_by_day(progression_type, day)
            date = datetime.today() + timedelta(days=day)

            for v in range(VIDEOS_PER_DAY):
                video_title = random.choice(video_titles)
                category_1, category_2 = video_metadata[video_title]
                summary = generate_summary(video_title, label, use_groq)

                coherence = round(util.cos_sim(
                    sbert_model.encode(summary, convert_to_tensor=True),
                    sbert_model.encode(video_descriptions[video_title], convert_to_tensor=True)
                ).item(), 3)

                watch_time, pause_count, replay_count, liked, shared, _ = simulate_engagement_metrics(label)

                row = {
                    "user_id": user_id,
                    "day": day,
                    "date": date.strftime("%Y-%m-%d"),
                    "video_title": video_title,
                    "category_1": category_1,
                    "category_2": category_2,
                    "label": label,
                    "summary": summary,
                    "coherence": coherence,
                    "watch_time_secs": round(watch_time),
                    "pause_count": int(pause_count),
                    "replay_count": int(replay_count),
                    "liked": liked,
                    "shared": shared,
                    "used_groq": use_groq
                }
                output_rows.append(row)

            # Save every day to CSV to allow resumability
            df_day = pd.DataFrame(output_rows)
            df_day.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
            output_rows.clear()

            # Checkpoint logging
            if day in [25, 50, 75, 100, 125, 150, 175, 200]:
                print(f"✅ User {user_id} completed day {day}")

        print(f"🎉 Completed User {user_id}")

if __name__ == "__main__":
    simulate_users()
