"""Layer 5 demo: how strongly is each issue associated with lower ratings? (offline)

    python scripts/demo_impact.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.prediction import compute_issue_impact
from src.storage import IMPACT_TABLE, connect, create_tables, insert_reviews

pd.set_option("display.max_columns", None, "display.width", 240)

# issue -> (label, n, rating_low, rating_high)
PLAN = {
    "issue_00": ("crash, launch", 70, 1, 3),      # harmful
    "issue_01": ("checkout, payment", 55, 2, 4),  # moderately harmful
    "issue_02": ("dark mode, love", 65, 4, 6),    # benign / positive
    "issue_03": ("rare glitch", 4, 1, 3),         # too small -> unreliable
}


def build(seed=7):
    rng = np.random.default_rng(seed)
    rows, assigns, rid = [], [], 0
    base = pd.Timestamp("2026-06-01")
    for issue_id, (label, n, lo, hi) in PLAN.items():
        for _ in range(n):
            rows.append((f"r{rid}", base + pd.Timedelta(days=rid % 45),
                         int(rng.integers(lo, hi)), int(rng.integers(20, 120))))
            assigns.append((f"r{rid}", issue_id, label)); rid += 1
    cleaned = pd.DataFrame(rows, columns=["review_id", "review_date", "rating", "review_length"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    return cleaned, assignments


def main():
    cleaned, assignments = build()
    impact = compute_issue_impact(cleaned, assignments)

    print("\n=== Issue rating-impact table ===")
    print(impact[["issue_id", "issue_label", "sample_size", "average_issue_rating",
                  "average_non_issue_rating", "rating_difference", "test_used",
                  "p_value", "regression_effect", "regression_ci_low",
                  "regression_ci_high", "reliable", "confidence_level"]].to_string(index=False))

    print("\n=== Plain-language interpretation ===")
    for _, r in impact.iterrows():
        print(f"- {r['issue_id']} ({r['issue_label']}): {r['interpretation']}")

    con = connect()
    create_tables(con)
    insert_reviews(con, IMPACT_TABLE, impact, replace=True)
    con.close()
    print(f"\nStored {len(impact)} rows in {IMPACT_TABLE}.")


if __name__ == "__main__":
    main()
