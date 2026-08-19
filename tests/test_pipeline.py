"""Layer 7 tests: end-to-end pipeline over a small sample dataset."""
import run_pipeline
from src.ingestion.sample import build_sample_standardized
from src.storage import (
    CLEANED_TABLE,
    IMPACT_TABLE,
    ISSUES_TABLE,
    PREDICTION_TABLE,
    RAW_TABLE,
    SUMMARY_TABLE,
    TRENDS_TABLE,
    connect,
    read_table,
    row_count,
)

ALL_TABLES = [RAW_TABLE, CLEANED_TABLE, ISSUES_TABLE, TRENDS_TABLE,
              SUMMARY_TABLE, IMPACT_TABLE, PREDICTION_TABLE]


def test_end_to_end_populates_all_tables():
    result = run_pipeline.run(sample=True)
    con = connect()
    for t in ALL_TABLES:
        assert row_count(con, t) > 0, f"{t} is empty"
        assert result["row_counts"][t] == row_count(con, t)
    con.close()


def test_final_warnings_are_wellformed():
    run_pipeline.run(sample=True)
    con = connect()
    pred = read_table(con, PREDICTION_TABLE)
    con.close()
    assert set(pred["risk_level"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert pred["explanation"].str.len().gt(0).all()   # every warning explained
    assert pred["horizon"].notna().all()
    # the seeded "payment" theme is designed to emerge as a rating risk
    assert (pred["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])).any()


def test_backtest_reported():
    result = run_pipeline.run(sample=True)
    bt = result["backtest"]
    assert bt["n_backtests"] > 0
    assert "directional_accuracy" in bt


def test_reproducible_same_input_same_output():
    """Same sample dataset + config -> identical analytical outputs."""
    std = build_sample_standardized()
    r1 = run_pipeline.run(std_df=std)
    con = connect()
    pred1 = read_table(con, PREDICTION_TABLE).sort_values("issue_id").reset_index(drop=True)
    con.close()

    r2 = run_pipeline.run(std_df=std)
    con = connect()
    pred2 = read_table(con, PREDICTION_TABLE).sort_values("issue_id").reset_index(drop=True)
    con.close()

    assert r1["row_counts"] == r2["row_counts"]
    import pandas as pd
    pd.testing.assert_frame_equal(pred1, pred2)


def test_stages_callable_independently():
    std = build_sample_standardized()
    cleaned, report = run_pipeline.clean(std)
    assert len(cleaned) > 0
    issues = run_pipeline.extract(cleaned)
    assert len(issues.assignments) == len(cleaned)
    trend_df, summary_df = run_pipeline.trends(cleaned, issues.assignments)
    assert len(trend_df) > 0
    impact_df = run_pipeline.impact(cleaned, issues.assignments)
    pred_df, bt = run_pipeline.forward_risk(trend_df, impact_df)
    assert len(pred_df) == len(summary_df)
