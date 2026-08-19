"""Sample data so ingestion / the full pipeline / tests run without network calls."""
from __future__ import annotations

import json
import random

import pandas as pd

from src.utils.config import load_config


def load_sample_raw(platform: str) -> list[dict]:
    """Return native-format sample records for 'google_play' or 'app_store'."""
    path = load_config().path("data_sample") / f"{platform}_raw.json"
    return json.loads(path.read_text(encoding="utf-8"))


# Latent themes for the synthetic end-to-end dataset. NOT told to the model —
# TF-IDF/KMeans must rediscover them. One theme (payment) deliberately emerges.
_THEMES = {
    "payment":  ["payment failed at checkout", "payment failed, checkout error again"],
    "crash":    ["app crashes on launch", "app crashes on every launch"],
    "battery":  ["battery drains fast and phone overheats", "battery drains, phone overheats badly"],
    "positive": ["love it, works great", "works great, love the design"],
}
# Weekly mention counts per theme over 12 weeks; payment is the rising warning.
_WEEKLY = {
    "payment":  [1, 1, 2, 2, 3, 4, 6, 8, 10, 13, 16, 20],
    "crash":    [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6],
    "battery":  [3, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3, 4],
    "positive": [8, 7, 9, 8, 7, 8, 9, 8, 7, 8, 9, 8],
}


def build_sample_standardized(seed: int = 42, week0: str = "2026-05-04") -> pd.DataFrame:
    """Build a standardized (Layer-1 schema) review frame spanning both platforms,
    several themes, and 12 weeks — enough for issues, trends, impact, and forecast.

    Deterministic for a given seed, so the whole pipeline is reproducible.
    """
    from src.ingestion.app_store import AppStoreAdapter
    from src.ingestion.google_play import GooglePlayAdapter

    rnd = random.Random(seed)
    base = pd.Timestamp(week0)
    gp_raw, as_raw, rid = [], [], 0
    for theme, weekly in _WEEKLY.items():
        phrases = _THEMES[theme]
        good = theme == "positive"
        for wk, n in enumerate(weekly):
            day = base + pd.Timedelta(days=7 * wk + rnd.randint(0, 6))
            for _ in range(n):
                text = f"{rnd.choice(phrases)}, {rnd.choice(phrases)}."
                rating = rnd.choice([4, 5, 5] if good else [1, 1, 2, 3])
                if rid % 2 == 0:  # half to Google Play
                    gp_raw.append({
                        "reviewId": f"gp_{rid:04d}", "content": text, "score": rating,
                        "thumbsUpCount": rnd.randint(0, 15), "reviewCreatedVersion": "5.4.1",
                        "at": day.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                else:            # half to App Store
                    as_raw.append({
                        "userName": f"user{rid}", "title": "", "review": text,
                        "rating": rating, "date": day.strftime("%Y-%m-%d %H:%M:%S"),
                        "isEdited": False,
                    })
                rid += 1

    g = GooglePlayAdapter("com.example.app", "Example App").standardize(gp_raw)
    a = AppStoreAdapter("123456789", "Example App").standardize(as_raw)
    return pd.concat([g, a], ignore_index=True)
