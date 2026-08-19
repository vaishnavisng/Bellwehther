"""Bellwether end-to-end pipeline.

    ingest -> validate -> clean -> DuckDB(raw, cleaned)
           -> issue extraction (TF-IDF/KMeans) -> trends + anomaly
           -> historical rating impact -> forward-looking risk
           -> DuckDB(final analytical tables)

All expensive NLP/statistical work happens HERE. The Streamlit dashboard only
reads the precomputed tables — it never re-runs this pipeline.

Run:
    python run_pipeline.py            # ingest live sources from config
    python run_pipeline.py --sample   # offline synthetic dataset (no network)

Reproducibility: for a fixed input dataset and config, outputs are identical.
KMeans uses a fixed random_state (config: nlp.random_state); every other stage
is deterministic pandas/statsmodels arithmetic. The sample dataset is seeded, so
`--sample` reproduces byte-for-byte across runs.

Each stage below is a plain function and can be called independently.
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

from src.analytics import build_issue_trends, score_issues
from src.ingestion.factory import fetch_all_sources
from src.ingestion.sample import build_sample_standardized
from src.nlp import extract_issues
from src.prediction import backtest, compute_issue_impact, predict_rating_risk
from src.preprocessing import build_cleaned_frame
from src.storage import (
    CLEANED_TABLE,
    IMPACT_TABLE,
    ISSUES_TABLE,
    PREDICTION_TABLE,
    RAW_TABLE,
    SUMMARY_TABLE,
    TRENDS_TABLE,
    connect,
    create_tables,
    data_quality,
    insert_reviews,
    row_count,
)
from src.utils.logging_setup import get_logger

log = get_logger("bellwether.pipeline")


@contextmanager
def stage(name: str):
    log.info("=> %s ...", name)
    t = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("XX %s FAILED", name)
        raise
    log.info("== %s done (%.2fs)", name, time.perf_counter() - t)


# --- individually callable stages (no DB side effects) --------------------- #
def ingest(count: int = 500, sample: bool = False):
    return build_sample_standardized() if sample else fetch_all_sources(
        count=count, save_raw=True)


def clean(std_df):
    """Validate + clean. The returned report carries the raw-validation counts."""
    return build_cleaned_frame(std_df)


def extract(cleaned_df):
    return extract_issues(cleaned_df)


def trends(cleaned_df, assignments):
    t = build_issue_trends(cleaned_df, assignments)
    return t, score_issues(t, labels=assignments)


def impact(cleaned_df, assignments):
    return compute_issue_impact(cleaned_df, assignments)


def forward_risk(trend_df, impact_df):
    return predict_rating_risk(trend_df, impact_df), backtest(trend_df, impact_df)


# --- orchestration --------------------------------------------------------- #
def run(std_df=None, count: int = 500, sample: bool = False) -> dict:
    """Run the full pipeline and persist every analytical table. Returns a summary
    dict (row counts, backtest, warning count) for verification."""
    t0 = time.perf_counter()
    log.info("Bellwether pipeline starting (sample=%s)", sample)
    con = connect()
    create_tables(con)

    with stage("1-4 ingest + validate"):
        if std_df is None:
            std_df = ingest(count=count, sample=sample)
        log.info("Ingested %d standardized reviews", len(std_df))

    with stage("4 clean"):
        cleaned, report = clean(std_df)
        log.info("Validation/cleaning report: %s", report)

    with stage("5 store raw + cleaned"):
        insert_reviews(con, RAW_TABLE, std_df, replace=True)
        insert_reviews(con, CLEANED_TABLE, cleaned, replace=True)
        for tbl in (RAW_TABLE, CLEANED_TABLE):
            log.info("Quality[%s]: %s", tbl, data_quality(con, tbl))

    with stage("6 issue extraction"):
        issues = extract(cleaned)
        log.info("Discovered %d issues (k=%d via %s)",
                 len(issues.summary), issues.k, issues.method)
        insert_reviews(con, ISSUES_TABLE, issues.assignments, replace=True)

    with stage("7 trends + anomaly"):
        trend_df, summary_df = trends(cleaned, issues.assignments)
        insert_reviews(con, TRENDS_TABLE, trend_df, replace=True)
        insert_reviews(con, SUMMARY_TABLE, summary_df, replace=True)

    with stage("8 historical rating impact"):
        impact_df = impact(cleaned, issues.assignments)
        insert_reviews(con, IMPACT_TABLE, impact_df, replace=True)

    with stage("9 forward-looking risk"):
        pred_df, bt = forward_risk(trend_df, impact_df)
        insert_reviews(con, PREDICTION_TABLE, pred_df, replace=True)
        log.info("Backtest: %s", bt)

    tables = [RAW_TABLE, CLEANED_TABLE, ISSUES_TABLE, TRENDS_TABLE,
              SUMMARY_TABLE, IMPACT_TABLE, PREDICTION_TABLE]
    row_counts = {t: row_count(con, t) for t in tables}
    n_warnings = int(pred_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
    con.close()

    elapsed = time.perf_counter() - t0
    log.info("Pipeline finished in %.2fs | row counts: %s", elapsed, row_counts)
    log.info("%d HIGH/CRITICAL rating-risk warning(s)", n_warnings)
    return {"row_counts": row_counts, "backtest": bt,
            "n_warnings": n_warnings, "elapsed": elapsed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the Bellwether pipeline")
    ap.add_argument("--sample", action="store_true",
                    help="use the offline synthetic dataset (no network)")
    ap.add_argument("--count", type=int, default=500,
                    help="max reviews per source when ingesting live")
    args = ap.parse_args()
    run(count=args.count, sample=args.sample)
