"""Layer 3 demo: discover issues from reviews (offline, varied synthetic data).

The themes are NOT told to the model — they emerge from TF-IDF + KMeans.
    python scripts/demo_issues.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ingestion import GooglePlayAdapter
from src.nlp import extract_issues
from src.preprocessing import build_cleaned_frame
from src.storage import ISSUES_TABLE, connect, create_tables, insert_reviews, read_table

pd.set_option("display.max_columns", None, "display.width", 240)

# Varied phrasings per latent theme so clusters are real, not duplicate points.
THEMES = {
    1: ["app crashes on launch", "keeps crashing when I open it", "freezes then closes",
        "constant crashes after update", "crashes every single time I tap search"],
    2: ["battery drains so fast", "phone overheats while using the app",
        "huge battery drain in the background", "eats battery and gets hot",
        "drains my battery overnight for no reason"],
    3: ["too many ads", "ads pop up constantly", "annoying full screen ads everywhere",
        "way too many video ads now", "ads interrupt everything I do"],
    4: ["cannot log in", "login keeps failing", "stuck on the login screen",
        "sign in loops forever", "password never works on login"],
}


def build_raw(n_per=11, seed=42):
    rnd = random.Random(seed)
    raw = []
    i = 0
    for theme, phrases in THEMES.items():
        for _ in range(n_per):
            text = f"{rnd.choice(phrases)}, {rnd.choice(phrases)}."
            raw.append({
                "reviewId": f"gp_{i:03d}", "content": text,
                "score": rnd.choice([1, 1, 2, 3]), "thumbsUpCount": rnd.randint(0, 20),
                "reviewCreatedVersion": "5.4.1",
                "at": f"2026-08-{10 + (i % 15):02d} 10:00:00",
            })
            i += 1
    rnd.shuffle(raw)
    return raw


def main():
    std = GooglePlayAdapter("com.example.app", "Example App").standardize(build_raw())
    cleaned, _ = build_cleaned_frame(std)
    res = extract_issues(cleaned)

    print(f"\nDiscovered K={res.k} issues (method={res.method}) from {len(cleaned)} reviews\n")
    print("=== Issue summary ===")
    print(res.summary[["issue_id", "issue_label", "cluster_size",
                       "cluster_avg_rating", "issue_keywords"]].to_string(index=False))

    con = connect()
    create_tables(con)
    insert_reviews(con, ISSUES_TABLE, res.assignments, replace=True)
    print(f"\nStored {len(res.assignments)} assignments in {ISSUES_TABLE}.")
    print("\n=== Sample stored rows ===")
    print(read_table(con, ISSUES_TABLE, limit=5)[
        ["review_id", "issue_id", "issue_label", "cluster_avg_rating"]].to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
