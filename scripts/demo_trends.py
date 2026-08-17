"""Layer 4 demo: which issues are becoming worse right now? (offline)

Builds several weeks of reviews where one issue quietly emerges, then shows the
time-series metrics and the explainable early-warning summary.
    python scripts/demo_trends.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.analytics import build_issue_trends, score_issues
from src.storage import (
    SUMMARY_TABLE,
    TRENDS_TABLE,
    connect,
    create_tables,
    insert_reviews,
)

pd.set_option("display.max_columns", None, "display.width", 240)

# Weekly mention counts per issue over 8 weeks; "crash" is the emerging warning.
PLAN = {
    "issue_00": ("crash, launch",     [1, 1, 2, 1, 3, 6, 11, 18], 1),
    "issue_01": ("checkout, payment", [6, 5, 6, 5, 6, 5, 6, 5], 3),
    "issue_02": ("ads, video",        [3, 4, 3, 4, 5, 4, 5, 6], 2),
}
WEEK0 = pd.Timestamp("2026-06-01")


def build():
    rnd = random.Random(1)
    reviews, assigns, rid = [], [], 0
    for issue_id, (label, counts, base_rating) in PLAN.items():
        for wk, n in enumerate(counts):
            day = WEEK0 + pd.Timedelta(days=7 * wk)
            for _ in range(n):
                reviews.append((f"r{rid}", day, min(5, max(1, base_rating + rnd.randint(-1, 1)))))
                assigns.append((f"r{rid}", issue_id, label))
                rid += 1
    cleaned = pd.DataFrame(reviews, columns=["review_id", "review_date", "rating"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    return cleaned, assignments


def main():
    cleaned, assignments = build()
    trends = build_issue_trends(cleaned, assignments)
    summary = score_issues(trends, labels=assignments)

    print("\n=== issue_00 trajectory (the emerging issue) ===")
    t = trends[trends["issue_id"] == "issue_00"].sort_values("date")
    cols = ["date", "mention_count", "issue_share", "rolling_baseline",
            "deviation_score", "growth_rate", "anomaly_flag"]
    print(t[cols].round({"issue_share": 3, "rolling_baseline": 3,
                         "deviation_score": 3, "growth_rate": 3}).to_string(index=False))

    print("\n=== Early-warning summary (ranked by risk) ===")
    print(summary[["issue_id", "issue_label", "latest_share", "previous_share",
                   "recent_growth", "deviation_score", "average_rating",
                   "negative_share", "risk_score", "risk_level",
                   "confidence_level"]].to_string(index=False))

    con = connect()
    create_tables(con)
    insert_reviews(con, TRENDS_TABLE, trends, replace=True)
    insert_reviews(con, SUMMARY_TABLE, summary, replace=True)
    con.close()
    print(f"\nStored {len(trends)} trend rows and {len(summary)} summary rows in DuckDB.")


if __name__ == "__main__":
    main()
