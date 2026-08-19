"""DuckDB storage for Bellwether.

Two tables:
  raw_reviews     - standardized reviews as ingested (may contain duplicates)
  cleaned_reviews - validated + feature-enriched, review_id as primary key

Reusable helpers: connect, create_tables, insert_reviews, read_table,
row_count, data_quality.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

RAW_TABLE = "raw_reviews"
CLEANED_TABLE = "cleaned_reviews"
ISSUES_TABLE = "review_issues"
TRENDS_TABLE = "issue_trends"
SUMMARY_TABLE = "issue_summary"
IMPACT_TABLE = "issue_impact"
PREDICTION_TABLE = "issue_prediction"

_DDL = {
    RAW_TABLE: """
        CREATE TABLE IF NOT EXISTS raw_reviews (
            review_id       VARCHAR,
            review_text     VARCHAR,
            rating          INTEGER,
            review_date     TIMESTAMP,
            source_platform VARCHAR,
            app_id          VARCHAR,
            app_name        VARCHAR,
            app_version     VARCHAR,
            helpful_count   INTEGER
        )""",
    CLEANED_TABLE: """
        CREATE TABLE IF NOT EXISTS cleaned_reviews (
            review_id       VARCHAR PRIMARY KEY,
            review_text     VARCHAR,
            cleaned_text    VARCHAR,
            rating          INTEGER,
            rating_bucket   VARCHAR,
            review_date     TIMESTAMP,
            review_day      DATE,
            review_week     VARCHAR,
            review_month    VARCHAR,
            review_year     INTEGER,
            source_platform VARCHAR,
            app_id          VARCHAR,
            app_name        VARCHAR,
            app_version     VARCHAR,
            helpful_count   INTEGER,
            review_length   INTEGER,
            word_count      INTEGER
        )""",
    ISSUES_TABLE: """
        CREATE TABLE IF NOT EXISTS review_issues (
            review_id          VARCHAR PRIMARY KEY,
            cluster_id         INTEGER,
            issue_id           VARCHAR,
            issue_label        VARCHAR,
            issue_keywords     VARCHAR,
            cluster_size       INTEGER,
            cluster_avg_rating DOUBLE
        )""",
    TRENDS_TABLE: """
        CREATE TABLE IF NOT EXISTS issue_trends (
            issue_id         VARCHAR,
            date             DATE,
            mention_count    INTEGER,
            issue_share      DOUBLE,
            average_rating   DOUBLE,
            negative_share   DOUBLE,
            wow_change       DOUBLE,
            rolling_baseline DOUBLE,
            rolling_std      DOUBLE,
            deviation_score  DOUBLE,
            growth_rate      DOUBLE,
            anomaly_flag     BOOLEAN,
            PRIMARY KEY (issue_id, date)
        )""",
    SUMMARY_TABLE: """
        CREATE TABLE IF NOT EXISTS issue_summary (
            issue_id         VARCHAR PRIMARY KEY,
            issue_label      VARCHAR,
            latest_date      DATE,
            latest_share     DOUBLE,
            previous_share   DOUBLE,
            recent_growth    DOUBLE,
            deviation_score  DOUBLE,
            average_rating   DOUBLE,
            negative_share   DOUBLE,
            mention_count    INTEGER,
            total_mentions   INTEGER,
            anomaly_flag     BOOLEAN,
            risk_score       DOUBLE,
            risk_level       VARCHAR,
            confidence       DOUBLE,
            confidence_level VARCHAR
        )""",
    IMPACT_TABLE: """
        CREATE TABLE IF NOT EXISTS issue_impact (
            issue_id                 VARCHAR PRIMARY KEY,
            issue_label              VARCHAR,
            sample_size              INTEGER,
            non_issue_size           INTEGER,
            average_issue_rating     DOUBLE,
            average_non_issue_rating DOUBLE,
            overall_rating           DOUBLE,
            median_issue_rating      DOUBLE,
            median_non_issue_rating  DOUBLE,
            rating_difference        DOUBLE,
            low_rating_share_issue   DOUBLE,
            low_rating_share_non_issue DOUBLE,
            test_used                VARCHAR,
            test_reasoning           VARCHAR,
            p_value                  DOUBLE,
            significant              BOOLEAN,
            diff_ci_low              DOUBLE,
            diff_ci_high             DOUBLE,
            regression_effect        DOUBLE,
            regression_ci_low        DOUBLE,
            regression_ci_high       DOUBLE,
            regression_p             DOUBLE,
            reliable                 BOOLEAN,
            confidence_level         VARCHAR,
            interpretation           VARCHAR
        )""",
    PREDICTION_TABLE: """
        CREATE TABLE IF NOT EXISTS issue_prediction (
            issue_id                 VARCHAR PRIMARY KEY,
            issue_label              VARCHAR,
            n_periods                INTEGER,
            horizon                  VARCHAR,
            current_share            DOUBLE,
            recent_growth            DOUBLE,
            current_trend            VARCHAR,
            historical_rating_impact DOUBLE,
            predicted_share          DOUBLE,
            predicted_rating_impact  DOUBLE,
            lower_bound              DOUBLE,
            upper_bound              DOUBLE,
            risk_level               VARCHAR,
            confidence_level         VARCHAR,
            reason_code              VARCHAR,
            explanation              VARCHAR
        )""",
}


def connect(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Connect to the DuckDB file (from config unless overridden). ':memory:' ok."""
    if db_path is None:
        db_path = load_config().path("database")
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    for ddl in _DDL.values():
        con.execute(ddl)


def _table_columns(con, table: str) -> list[str]:
    return con.execute(f"PRAGMA table_info('{table}')").df()["name"].tolist()


def insert_reviews(con, table: str, df: pd.DataFrame, replace: bool = False) -> int:
    """Insert a frame into a table, aligning on the table's columns.

    replace=True clears the table first (idempotent pipeline re-runs).
    """
    if table not in _DDL:
        raise ValueError(f"Unknown table {table!r}")
    cols = _table_columns(con, table)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Frame missing columns for {table}: {missing}")
    if replace:
        con.execute(f"DELETE FROM {table}")
    con.register("_incoming", df[cols])
    col_list = ", ".join(cols)
    con.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _incoming")
    con.unregister("_incoming")
    n = len(df)
    log.info("Inserted %d rows into %s", n, table)
    return n


def read_table(con, table: str, limit: int | None = None) -> pd.DataFrame:
    q = f"SELECT * FROM {table}"
    if limit:
        q += f" LIMIT {int(limit)}"
    return con.execute(q).df()


def row_count(con, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def data_quality(con, table: str) -> dict:
    """Row count, duplicate ids, missing critical values, bad ratings, date range."""
    total = row_count(con, table)
    dup = con.execute(
        f"SELECT COUNT(*) FROM (SELECT review_id FROM {table} "
        f"GROUP BY review_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    missing_text = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE review_text IS NULL OR review_text = ''"
    ).fetchone()[0]
    missing_rating = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE rating IS NULL"
    ).fetchone()[0]
    invalid_rating = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE rating < 1 OR rating > 5"
    ).fetchone()[0]
    dmin, dmax = con.execute(
        f"SELECT MIN(review_date), MAX(review_date) FROM {table}"
    ).fetchone()
    return {
        "table": table,
        "row_count": total,
        "duplicate_review_ids": dup,
        "missing_text": missing_text,
        "missing_rating": missing_rating,
        "invalid_rating": invalid_rating,
        "date_min": str(dmin) if dmin else None,
        "date_max": str(dmax) if dmax else None,
    }
