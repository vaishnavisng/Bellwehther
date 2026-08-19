"""Layer 8 tests: dashboard data layer reads precomputed tables (no recomputation)."""
import pandas as pd
import pytest

import run_pipeline
from dashboard.data import (
    compute_kpis,
    load_all,
    representative_reviews,
    warning_ranked,
)


@pytest.fixture(scope="module", autouse=True)
def _populated_db():
    run_pipeline.run(sample=True)  # ensure the DuckDB tables exist


def test_load_all_returns_expected_tables():
    data = load_all()
    for t in ("cleaned_reviews", "issue_prediction", "issue_trends", "issue_impact"):
        assert isinstance(data[t], pd.DataFrame)
        assert not data[t].empty


def test_compute_kpis_from_stored_rows():
    kpis = compute_kpis(load_all())
    assert kpis["total_reviews"] > 0
    assert 1 <= kpis["avg_rating"] <= 5
    assert 0 <= kpis["negative_pct"] <= 100
    assert kpis["n_issues"] >= 1
    assert kpis["n_emerging"] >= kpis["n_high_risk"]


def test_warning_ranked_orders_worst_first():
    ranked = warning_ranked(load_all()["issue_prediction"])
    risks = ranked["risk_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3})
    assert list(risks) == sorted(risks, reverse=True)  # non-increasing severity


def test_representative_reviews_has_no_pii_columns():
    ranked = warning_ranked(load_all()["issue_prediction"])
    ev = representative_reviews(ranked.iloc[0]["issue_id"])
    assert not ev.empty
    # only context columns, no reviewer identity
    assert set(ev.columns) == {"review_date", "rating", "source_platform", "review_text"}


def test_empty_inputs_handled():
    empty = {"cleaned_reviews": pd.DataFrame(), "issue_prediction": pd.DataFrame(),
             "issue_summary": pd.DataFrame()}
    kpis = compute_kpis(empty)
    assert kpis["total_reviews"] == 0
    assert kpis["avg_rating"] is None
    assert warning_ranked(pd.DataFrame()).empty
