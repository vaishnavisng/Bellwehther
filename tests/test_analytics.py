"""Layer 4 tests: issue trends, anomaly flags, and explainable risk scoring."""
import numpy as np
import pandas as pd

from src.analytics import build_issue_trends, score_issues
from src.storage import (
    SUMMARY_TABLE,
    TRENDS_TABLE,
    connect,
    create_tables,
    insert_reviews,
    read_table,
    row_count,
)

WEEK0 = pd.Timestamp("2026-06-01")  # a Monday


def _make_data():
    """Two issues over 8 weeks: one stable, one sharply rising (a real warning)."""
    stable = [5] * 8                      # issue_stable: flat volume, good ratings
    rising = [1, 1, 1, 1, 2, 4, 8, 14]    # issue_rising: emerging spike, bad ratings
    reviews, assigns, rid = [], [], 0
    for wk in range(8):
        day = WEEK0 + pd.Timedelta(days=7 * wk)
        for _ in range(stable[wk]):
            reviews.append((f"r{rid}", day, 4)); assigns.append((f"r{rid}", "issue_stable")); rid += 1
        for _ in range(rising[wk]):
            reviews.append((f"r{rid}", day, 1)); assigns.append((f"r{rid}", "issue_rising")); rid += 1
    cleaned = pd.DataFrame(reviews, columns=["review_id", "review_date", "rating"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id"])
    assignments["issue_label"] = assignments["issue_id"].map(
        {"issue_stable": "checkout, cart", "issue_rising": "crash, launch"})
    return cleaned, assignments


def test_empty_inputs():
    empty = pd.DataFrame(columns=["review_id", "review_date", "rating"])
    trends = build_issue_trends(empty, pd.DataFrame(columns=["review_id", "issue_id"]))
    assert len(trends) == 0
    assert len(score_issues(trends)) == 0


def test_trend_shape_and_columns():
    cleaned, assignments = _make_data()
    trends = build_issue_trends(cleaned, assignments)
    assert list(trends.columns) == [
        "issue_id", "date", "mention_count", "issue_share", "average_rating",
        "negative_share", "wow_change", "rolling_baseline", "rolling_std",
        "deviation_score", "growth_rate", "anomaly_flag"]
    # 2 issues x 8 weekly periods (grid is complete)
    assert len(trends) == 16
    assert trends["date"].nunique() == 8


def test_rising_issue_flags_anomaly_and_growth():
    cleaned, assignments = _make_data()
    trends = build_issue_trends(cleaned, assignments)
    rising = trends[trends["issue_id"] == "issue_rising"].sort_values("date")
    assert rising["anomaly_flag"].any()            # spike detected
    assert rising["growth_rate"].iloc[-1] > 0      # still growing at the end
    stable = trends[trends["issue_id"] == "issue_stable"]
    assert not stable["anomaly_flag"].any()        # flat issue never flags


def test_shares_sum_to_one_per_period():
    cleaned, assignments = _make_data()
    trends = build_issue_trends(cleaned, assignments)
    per_period = trends.groupby("date")["issue_share"].sum()
    assert np.allclose(per_period.values, 1.0)


def test_risk_ranks_rising_above_stable():
    cleaned, assignments = _make_data()
    trends = build_issue_trends(cleaned, assignments)
    summary = score_issues(trends, labels=assignments)
    top = summary.iloc[0]
    assert top["issue_id"] == "issue_rising"
    assert top["risk_level"] in ("HIGH", "MEDIUM")
    assert top["risk_score"] > summary.iloc[1]["risk_score"]
    # explainable components are all present
    for col in ("recent_growth", "deviation_score", "average_rating",
                "negative_share", "risk_score", "confidence"):
        assert col in summary.columns
    assert top["issue_label"] == "crash, launch"


def test_confidence_reflects_volume():
    cleaned, assignments = _make_data()
    summary = score_issues(build_issue_trends(cleaned, assignments))
    assert (summary["confidence"] >= 0).all() and (summary["confidence"] <= 1).all()


def test_reproducibility():
    cleaned, assignments = _make_data()
    a = build_issue_trends(cleaned, assignments)
    b = build_issue_trends(cleaned, assignments)
    pd.testing.assert_frame_equal(a, b)


def test_storage_roundtrip():
    cleaned, assignments = _make_data()
    trends = build_issue_trends(cleaned, assignments)
    summary = score_issues(trends, labels=assignments)

    con = connect(":memory:")
    create_tables(con)
    insert_reviews(con, TRENDS_TABLE, trends, replace=True)
    insert_reviews(con, SUMMARY_TABLE, summary, replace=True)
    assert row_count(con, TRENDS_TABLE) == 16
    assert row_count(con, SUMMARY_TABLE) == 2
    back = read_table(con, SUMMARY_TABLE)
    assert set(back["risk_level"]) <= {"HIGH", "MEDIUM", "LOW"}
    con.close()
