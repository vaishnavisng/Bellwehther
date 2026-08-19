"""Layer 5 tests: historical rating-impact analysis."""
import numpy as np
import pandas as pd

from src.prediction import IMPACT_COLUMNS, compute_issue_impact
from src.storage import IMPACT_TABLE, connect, create_tables, insert_reviews, row_count


def _dataset(seed=0):
    """One clearly harmful issue (low ratings) + one benign issue (high ratings),
    large enough to be 'reliable'."""
    rng = np.random.default_rng(seed)
    rows, assigns, rid = [], [], 0
    base = pd.Timestamp("2026-06-01")
    # harmful issue: ratings ~1-2
    for _ in range(60):
        r = int(rng.integers(1, 3))
        rows.append((f"r{rid}", base + pd.Timedelta(days=rid % 40), r, 40 + rid % 10))
        assigns.append((f"r{rid}", "issue_bad", "crash, launch")); rid += 1
    # benign issue: ratings ~4-5
    for _ in range(60):
        r = int(rng.integers(4, 6))
        rows.append((f"r{rid}", base + pd.Timedelta(days=rid % 40), r, 30 + rid % 10))
        assigns.append((f"r{rid}", "issue_good", "great, love")); rid += 1
    cleaned = pd.DataFrame(rows, columns=["review_id", "review_date", "rating", "review_length"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    return cleaned, assignments


def test_empty_inputs():
    out = compute_issue_impact(
        pd.DataFrame(columns=["review_id", "review_date", "rating"]),
        pd.DataFrame(columns=["review_id", "issue_id"]))
    assert list(out.columns) == IMPACT_COLUMNS
    assert len(out) == 0


def test_schema_and_row_per_issue():
    cleaned, assignments = _dataset()
    out = compute_issue_impact(cleaned, assignments)
    assert list(out.columns) == IMPACT_COLUMNS
    assert set(out["issue_id"]) == {"issue_bad", "issue_good"}


def test_harmful_issue_has_negative_difference_and_penalty():
    cleaned, assignments = _dataset()
    out = compute_issue_impact(cleaned, assignments).set_index("issue_id")
    bad = out.loc["issue_bad"]
    assert bad["rating_difference"] < 0                 # lower than non-issue
    assert bad["average_issue_rating"] < bad["average_non_issue_rating"]
    assert bad["regression_effect"] < 0                 # negative historical penalty
    assert bad["significant"]                           # clearly separated groups
    assert "lower ratings" in bad["interpretation"]
    assert "association, not causation" in bad["interpretation"]


def test_test_chosen_not_blind_ttest():
    cleaned, assignments = _dataset()
    out = compute_issue_impact(cleaned, assignments)
    # ordinal 1-5 ratings -> rank-based test, reasoning recorded
    assert (out["test_used"] == "mann_whitney_u").all()
    assert out["test_reasoning"].str.len().gt(0).all()


def test_small_sample_marked_unreliable():
    base = pd.Timestamp("2026-06-01")
    rows, assigns = [], []
    for i in range(6):  # tiny issue
        rows.append((f"a{i}", base + pd.Timedelta(days=i), 1 + i % 2, 20))
        assigns.append((f"a{i}", "issue_tiny", "x"))
    for i in range(40):  # background so a comparison group exists
        rows.append((f"b{i}", base + pd.Timedelta(days=i), 4, 20))
        assigns.append((f"b{i}", "issue_big", "y"))
    cleaned = pd.DataFrame(rows, columns=["review_id", "review_date", "rating", "review_length"])
    assignments = pd.DataFrame(assigns, columns=["review_id", "issue_id", "issue_label"])
    out = compute_issue_impact(cleaned, assignments).set_index("issue_id")
    tiny = out.loc["issue_tiny"]
    assert tiny["reliable"] == False
    assert tiny["confidence_level"] == "low"
    assert "reliable estimate" in tiny["interpretation"]


def test_confidence_interval_present_and_ordered():
    cleaned, assignments = _dataset()
    out = compute_issue_impact(cleaned, assignments)
    assert (out["diff_ci_low"] <= out["diff_ci_high"]).all()
    assert out["regression_ci_low"].notna().all()


def test_reproducibility():
    cleaned, assignments = _dataset()
    a = compute_issue_impact(cleaned, assignments)
    b = compute_issue_impact(cleaned, assignments)
    pd.testing.assert_frame_equal(a, b)


def test_storage_roundtrip():
    cleaned, assignments = _dataset()
    out = compute_issue_impact(cleaned, assignments)
    con = connect(":memory:")
    create_tables(con)
    insert_reviews(con, IMPACT_TABLE, out, replace=True)
    assert row_count(con, IMPACT_TABLE) == 2
    con.close()
