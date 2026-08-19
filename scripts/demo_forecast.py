"""Layer 6 demo: forward-looking rating-risk early warning (offline, end-to-end).

Runs Layers 4 -> 5 -> 6 on synthetic data where one issue is quietly emerging,
prints the early-warning statements, and backtests the share forecast.
    python scripts/demo_forecast.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.analytics import build_issue_trends
from src.prediction import backtest, compute_issue_impact, predict_rating_risk
from src.storage import PREDICTION_TABLE, connect, create_tables, insert_reviews

pd.set_option("display.max_columns", None, "display.width", 240)
WEEK0 = pd.Timestamp("2026-05-01")

# issue -> (label, weekly counts over 14 weeks, rating_low, rating_high)
PLAN = {
    "issue_00": ("payment, checkout", np.linspace(2, 17, 14).round().astype(int), 1, 3),
    "issue_01": ("dark mode, love", [7] * 14, 4, 6),
    "issue_02": ("ads, video", [4, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4], 2, 4),
}


def build(seed=5):
    rng = np.random.default_rng(seed)
    rows, assigns, rid = [], [], 0
    for issue_id, (label, counts, lo, hi) in PLAN.items():
        for wk, n in enumerate(counts):
            day = WEEK0 + pd.Timedelta(days=7 * wk)
            for _ in range(int(n)):
                rows.append((f"r{rid}", day, int(rng.integers(lo, hi)), int(rng.integers(20, 90))))
                assigns.append((f"r{rid}", issue_id, label)); rid += 1
    cleaned = pd.DataFrame(rows, columns=["review_id", "review_date", "rating", "review_length"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    return cleaned, assignments


def main():
    cleaned, assignments = build()
    trends = build_issue_trends(cleaned, assignments)
    impact = compute_issue_impact(cleaned, assignments)
    pred = predict_rating_risk(trends, impact)

    print("\n=== Forward-looking rating-risk prediction ===")
    print(pred[["issue_id", "issue_label", "current_share", "recent_growth",
                "historical_rating_impact", "predicted_rating_impact",
                "lower_bound", "upper_bound", "risk_level",
                "confidence_level", "horizon"]].to_string(index=False))

    print("\n=== Early-warning statements ===")
    for _, r in pred.iterrows():
        print(f"\n[{r['risk_level']} risk | {r['confidence_level']} confidence] "
              f"{r['issue_id']} ({r['issue_label']})")
        print(f"  {r['explanation']}")

    print("\n=== Backtest (rolling-origin share forecast) ===")
    print(backtest(trends, impact))

    con = connect()
    create_tables(con)
    insert_reviews(con, PREDICTION_TABLE, pred, replace=True)
    con.close()
    print(f"\nStored {len(pred)} rows in {PREDICTION_TABLE}.")


if __name__ == "__main__":
    main()
