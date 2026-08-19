"""Layer 6 tests: forward-looking rating-risk prediction + backtesting."""
import numpy as np
import pandas as pd

from src.analytics import build_issue_trends
from src.prediction import (
    PREDICTION_COLUMNS,
    backtest,
    compute_issue_impact,
    forecast_share,
    predict_rating_risk,
)
from src.storage import (
    PREDICTION_TABLE,
    connect,
    create_tables,
    insert_reviews,
    row_count,
)

WEEK0 = pd.Timestamp("2026-05-01")


def _pipeline_data(weeks=12):
    """A harmful issue whose share rises steadily, plus a stable benign issue."""
    rng = np.random.default_rng(3)
    rising = np.linspace(1, 16, weeks).round().astype(int)   # emerging
    stable = [6] * weeks                                       # flat
    rows, assigns, rid = [], [], 0
    for wk in range(weeks):
        day = WEEK0 + pd.Timedelta(days=7 * wk)
        for _ in range(int(rising[wk])):
            rows.append((f"r{rid}", day, int(rng.integers(1, 3)), 60))
            assigns.append((f"r{rid}", "issue_bad", "crash, launch")); rid += 1
        for _ in range(stable[wk]):
            rows.append((f"r{rid}", day, int(rng.integers(4, 6)), 40))
            assigns.append((f"r{rid}", "issue_ok", "love, great")); rid += 1
    cleaned = pd.DataFrame(rows, columns=["review_id", "review_date", "rating", "review_length"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    trends = build_issue_trends(cleaned, assignments)
    impact = compute_issue_impact(cleaned, assignments)
    return trends, impact


# --- forecast_share primitive ---
def test_forecast_share_extrapolates_upward():
    fc = forecast_share([0.1, 0.15, 0.2, 0.25, 0.3], horizon=2, alpha=0.05, min_fit=4)
    assert fc["slope"] > 0
    assert fc["predicted"] > 0.3
    assert fc["low"] <= fc["predicted"] <= fc["high"]


def test_forecast_share_too_short_returns_none():
    assert forecast_share([0.1, 0.2], horizon=2, alpha=0.05, min_fit=4) is None


# --- prediction table ---
def test_empty_inputs():
    out = predict_rating_risk(pd.DataFrame(columns=["issue_id", "date", "issue_share"]),
                              pd.DataFrame())
    assert list(out.columns) == PREDICTION_COLUMNS
    assert len(out) == 0


def test_prediction_schema_and_numbers_from_data():
    trends, impact = _pipeline_data()
    out = predict_rating_risk(trends, impact)
    assert list(out.columns) == PREDICTION_COLUMNS
    assert set(out["issue_id"]) == {"issue_bad", "issue_ok"}
    bad = out.set_index("issue_id").loc["issue_bad"]
    # rising harmful issue -> negative projected impact, ordered CI, real explanation
    assert bad["historical_rating_impact"] < 0
    assert bad["predicted_rating_impact"] < 0
    assert bad["lower_bound"] <= bad["upper_bound"]
    assert bad["current_trend"] == "rising"
    assert bad["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")


def test_every_warning_has_human_explanation():
    trends, impact = _pipeline_data()
    out = predict_rating_risk(trends, impact)
    assert out["explanation"].str.len().gt(0).all()
    bad = out.set_index("issue_id").loc["issue_bad"]
    assert bad["explanation"].startswith("Risk is")
    assert "because" in bad["explanation"]


def test_benign_issue_low_risk():
    trends, impact = _pipeline_data()
    out = predict_rating_risk(trends, impact).set_index("issue_id")
    assert out.loc["issue_ok"]["risk_level"] == "LOW"


def test_insufficient_history_uses_current_burden():
    # Only 3 weeks (< min_history) -> no forecast, but risk still assessed from
    # the current burden (share x historical penalty), with a baseline group.
    rng = np.random.default_rng(0)
    rows, assigns, rid = [], [], 0
    for wk in range(3):
        day = WEEK0 + pd.Timedelta(days=7 * wk)
        for _ in range(10):  # harmful complaint, big share, 1-star
            rows.append((f"b{rid}", day, 1, 50))
            assigns.append((f"b{rid}", "issue_bad", "crash")); rid += 1
        for _ in range(10):  # satisfied baseline (unclustered-style, 5-star)
            rows.append((f"g{rid}", day, 5, 50))
            assigns.append((f"g{rid}", "issue_ok", "great")); rid += 1
    cleaned = pd.DataFrame(rows, columns=["review_id", "review_date", "rating", "review_length"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    trends = build_issue_trends(cleaned, assignments)
    impact = compute_issue_impact(cleaned, assignments)
    out = predict_rating_risk(trends, impact).set_index("issue_id")
    row = out.loc["issue_bad"]
    # no forecast produced (too few periods)
    assert row["predicted_rating_impact"] is None or pd.isna(row["predicted_rating_impact"])
    assert row["confidence_level"] == "low"
    # but risk is driven by current burden and the issue is clearly harmful
    assert row["reason_code"] == "current_burden"
    assert row["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")
    assert "current" in row["explanation"] and "drag" in row["explanation"]


# --- backtesting ---
def test_backtest_reports_metrics():
    trends, impact = _pipeline_data(weeks=14)
    report = backtest(trends, impact)
    assert report["n_backtests"] > 0
    for key in ("share_mae", "directional_accuracy", "interval_coverage",
                "target_coverage", "limitations"):
        assert key in report
    assert 0 <= report["directional_accuracy"] <= 1
    assert 0 <= report["interval_coverage"] <= 1


def test_backtest_honest_when_too_small():
    trends, impact = _pipeline_data(weeks=4)  # barely enough to fit, few windows
    report = backtest(trends, impact)
    if report["n_backtests"] == 0:
        assert "Insufficient" in report["note"]
    else:
        assert "indicative" in report["limitations"]


def test_reproducibility():
    trends, impact = _pipeline_data()
    a = predict_rating_risk(trends, impact)
    b = predict_rating_risk(trends, impact)
    pd.testing.assert_frame_equal(a, b)


def test_storage_roundtrip():
    trends, impact = _pipeline_data()
    out = predict_rating_risk(trends, impact)
    con = connect(":memory:")
    create_tables(con)
    insert_reviews(con, PREDICTION_TABLE, out, replace=True)
    assert row_count(con, PREDICTION_TABLE) == 2
    con.close()
