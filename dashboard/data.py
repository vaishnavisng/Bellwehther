"""Read-only data access for the dashboard.

The dashboard NEVER runs NLP/statistics. It only reads the precomputed tables
that run_pipeline.py wrote to DuckDB. Everything here is a plain SELECT plus a
little KPI aggregation over already-computed rows.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.utils.config import load_config

TABLES = [
    "cleaned_reviews", "review_issues", "issue_trends",
    "issue_summary", "issue_impact", "issue_prediction",
]
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def db_path() -> Path:
    return load_config().path("database")


def _connect() -> duckdb.DuckDBPyConnection:
    path = db_path()
    if not Path(path).exists():
        raise FileNotFoundError(
            f"DuckDB not found at {path}. Run `python run_pipeline.py --sample` first.")
    return duckdb.connect(str(path), read_only=True)


def load_all() -> dict[str, pd.DataFrame]:
    """Load every analytical table (empty DataFrame if a table is missing)."""
    con = _connect()
    try:
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        return {t: (con.execute(f"SELECT * FROM {t}").df()
                    if t in existing else pd.DataFrame()) for t in TABLES}
    finally:
        con.close()


def representative_reviews(issue_id: str, n: int = 8) -> pd.DataFrame:
    """Worst-rated example reviews for an issue (no reviewer PII is stored)."""
    con = _connect()
    try:
        return con.execute(
            """
            SELECT c.review_date, c.rating, c.source_platform, c.review_text
            FROM review_issues r
            JOIN cleaned_reviews c USING (review_id)
            WHERE r.issue_id = ?
            ORDER BY c.rating ASC, c.review_length DESC
            LIMIT ?
            """, [issue_id, n]).df()
    finally:
        con.close()


def compute_kpis(data: dict[str, pd.DataFrame]) -> dict:
    """Executive KPIs aggregated from stored rows (no model recomputation)."""
    cleaned = data.get("cleaned_reviews", pd.DataFrame())
    pred = data.get("issue_prediction", pd.DataFrame())
    summary = data.get("issue_summary", pd.DataFrame())

    if cleaned.empty:
        return {"total_reviews": 0, "avg_rating": None, "rating_trend_delta": None,
                "rating_by_month": pd.DataFrame(), "negative_pct": None,
                "n_emerging": 0, "n_high_risk": 0, "n_issues": 0}

    ratings = pd.to_numeric(cleaned["rating"], errors="coerce")
    by_month = (cleaned.assign(rating=ratings)
                .groupby("review_month", as_index=False)["rating"].mean()
                .sort_values("review_month"))
    trend_delta = None
    if len(by_month) >= 2:
        trend_delta = float(by_month["rating"].iloc[-1] - by_month["rating"].iloc[-2])

    n_emerging = n_high_risk = 0
    if not pred.empty:
        n_high_risk = int(pred["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
        n_emerging = int(pred["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum())
    elif not summary.empty:
        n_high_risk = int((summary["risk_level"] == "HIGH").sum())
        n_emerging = int(summary["anomaly_flag"].sum())

    return {
        "total_reviews": int(len(cleaned)),
        "avg_rating": round(float(ratings.mean()), 2),
        "rating_trend_delta": round(trend_delta, 2) if trend_delta is not None else None,
        "rating_by_month": by_month,
        "negative_pct": round(float((ratings <= 2).mean() * 100), 1),
        "n_emerging": n_emerging,
        "n_high_risk": n_high_risk,
        "n_issues": int(data.get("issue_prediction", pd.DataFrame()).shape[0]),
    }


def warning_ranked(pred: pd.DataFrame) -> pd.DataFrame:
    """Predictions ordered worst-first: risk tier, then most-negative impact."""
    if pred.empty:
        return pred
    df = pred.copy()
    df["_risk"] = df["risk_level"].map(RISK_ORDER).fillna(0)
    df["_imp"] = pd.to_numeric(df["predicted_rating_impact"], errors="coerce").fillna(0)
    return df.sort_values(["_risk", "_imp"], ascending=[False, True]).drop(
        columns=["_risk", "_imp"]).reset_index(drop=True)
