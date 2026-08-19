"""Issue-level time series, anomaly detection, and transparent early-warning scoring.

An issue becomes a warning because its *presence is rising unusually*, not because
it is merely common. Every signal is a plain, inspectable number — no black-box
score. The risk score is an explicit weighted blend of stored components.

Inputs (from earlier layers):
  cleaned_df   - cleaned_reviews (review_id, review_date, rating)
  assignments  - review_issues  (review_id -> issue_id, issue_label)

Outputs:
  build_issue_trends(...) -> per issue, per period time series
  score_issues(...)       -> one-row-per-issue warning summary (latest period)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

TREND_COLUMNS = [
    "issue_id", "date", "mention_count", "issue_share", "average_rating",
    "negative_share", "wow_change", "rolling_baseline", "rolling_std",
    "deviation_score", "growth_rate", "anomaly_flag",
]


def _cfg() -> dict:
    a = load_config().get("analytics", {}) or {}
    return {
        "freq": a.get("freq", "W"),               # weekly buckets by default
        "rolling_periods": a.get("rolling_periods", 4),
        "growth_lookback": a.get("growth_lookback", 2),
        "anomaly_z": a.get("anomaly_z", 2.0),
        "min_mentions_for_anomaly": a.get("min_mentions_for_anomaly", 3),
        # risk weights (must be interpretable; sum ~1)
        "weights": a.get("weights", {
            "growth": 0.30, "deviation": 0.25, "rating": 0.20,
            "negative": 0.15, "volume": 0.10,
        }),
        "growth_cap": a.get("growth_cap", 1.0),   # 100% growth -> full signal
        "z_cap": a.get("z_cap", 3.0),
        "volume_cap": a.get("volume_cap", 20),
        "confidence_cap": a.get("confidence_cap", 30),
        "risk_high": a.get("risk_high", 66),
        "risk_medium": a.get("risk_medium", 40),
    }


def build_issue_trends(cleaned_df: pd.DataFrame,
                       assignments: pd.DataFrame) -> pd.DataFrame:
    """Per-issue, per-period metrics with rolling baseline + z-score anomaly flag."""
    c = _cfg()
    if len(cleaned_df) == 0 or len(assignments) == 0:
        return pd.DataFrame(columns=TREND_COLUMNS)

    df = assignments[["review_id", "issue_id"]].merge(
        cleaned_df[["review_id", "review_date", "rating"]], on="review_id", how="inner")
    df["period"] = df["review_date"].dt.to_period(c["freq"]).dt.start_time
    df["is_negative"] = pd.to_numeric(df["rating"], errors="coerce") <= 2

    # Period totals use ALL reviews (issues may be clustered from a subset, e.g.
    # only critical reviews), so issue_share = share of *all* reviews mentioning
    # it. The grid spans every period any review exists — an issue appearing from
    # zero stays visible against periods where it had no mentions.
    allr = cleaned_df[["review_id", "review_date"]].copy()
    allr["period"] = allr["review_date"].dt.to_period(c["freq"]).dt.start_time
    periods = pd.Index(sorted(allr["period"].unique()), name="date")
    totals = allr.groupby("period").size().reindex(periods, fill_value=0)

    per = df.groupby(["issue_id", "period"]).agg(
        mention_count=("review_id", "size"),
        average_rating=("rating", "mean"),
        negative_share=("is_negative", "mean"),
    )

    frames = []
    for issue_id, g in per.groupby(level=0):
        g = g.droplevel(0).reindex(periods)
        g["mention_count"] = g["mention_count"].fillna(0).astype(int)
        g["issue_share"] = (g["mention_count"] / totals.replace(0, np.nan)).fillna(0.0)
        g["issue_id"] = issue_id
        g = g.reset_index(names="date").sort_values("date")

        share = g["issue_share"]
        g["wow_change"] = share.diff()
        # baseline excludes the current period (shift) -> deviation vs the past.
        base = share.shift(1).rolling(c["rolling_periods"], min_periods=2)
        g["rolling_baseline"] = base.mean()
        g["rolling_std"] = base.std()
        dev = (share - g["rolling_baseline"]) / g["rolling_std"]
        g["deviation_score"] = dev.replace([np.inf, -np.inf], np.nan)

        prior = share.shift(c["growth_lookback"])
        growth = (share - prior) / prior.replace(0, np.nan)
        # emergence from zero -> treat as full growth signal, not undefined.
        growth = growth.where(~((prior == 0) & (share > 0)), c["growth_cap"])
        g["growth_rate"] = growth

        g["anomaly_flag"] = (
            (g["deviation_score"] >= c["anomaly_z"]) &
            (g["mention_count"] >= c["min_mentions_for_anomaly"])
        ).fillna(False)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    for col in ("rolling_baseline", "rolling_std", "deviation_score",
                "growth_rate", "wow_change"):
        out[col] = out[col].fillna(0.0)
    out["average_rating"] = out["average_rating"].astype(float)
    log.info("Built trends: %d issue-periods across %d periods",
             len(out), len(periods))
    return out[TREND_COLUMNS]


def _level(value, high, medium) -> str:
    return "HIGH" if value >= high else "MEDIUM" if value >= medium else "LOW"


def score_issues(trends: pd.DataFrame,
                 labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Explainable early-warning summary at the most recent period.

    risk_score = 100 * weighted blend of normalized, individually stored signals.
    """
    c = _cfg()
    w = c["weights"]
    if len(trends) == 0:
        return pd.DataFrame()

    latest_date = trends["date"].max()
    srt = trends.sort_values("date")
    # second-to-last period's share, indexed by issue_id (nth keeps row index)
    prev_by_issue = srt.groupby("issue_id")["issue_share"].apply(
        lambda s: s.iloc[-2] if len(s) >= 2 else np.nan)
    total_mentions = trends.groupby("issue_id")["mention_count"].sum()

    snap = trends[trends["date"] == latest_date].set_index("issue_id")
    rows = []
    for issue_id, r in snap.iterrows():
        avg_rating = r["average_rating"]
        # Normalize each signal to 0..1 (higher = more concerning).
        s_growth = np.clip(max(r["growth_rate"], 0) / c["growth_cap"], 0, 1)
        s_dev = np.clip(max(r["deviation_score"], 0) / c["z_cap"], 0, 1)
        s_rating = np.clip((5 - avg_rating) / 4, 0, 1) if pd.notna(avg_rating) else 0.0
        s_neg = float(r["negative_share"]) if pd.notna(r["negative_share"]) else 0.0
        s_vol = np.clip(r["mention_count"] / c["volume_cap"], 0, 1)

        risk = 100 * (w["growth"] * s_growth + w["deviation"] * s_dev +
                      w["rating"] * s_rating + w["negative"] * s_neg +
                      w["volume"] * s_vol)

        tm = int(total_mentions.get(issue_id, 0))
        confidence = float(np.clip(tm / c["confidence_cap"], 0, 1))

        rows.append({
            "issue_id": issue_id,
            "latest_date": latest_date,
            "latest_share": round(float(r["issue_share"]), 4),
            "previous_share": round(float(prev_by_issue.get(issue_id, np.nan)), 4)
                if pd.notna(prev_by_issue.get(issue_id, np.nan)) else None,
            "recent_growth": round(float(r["growth_rate"]), 4),
            "deviation_score": round(float(r["deviation_score"]), 3),
            "average_rating": round(float(avg_rating), 3) if pd.notna(avg_rating) else None,
            "negative_share": round(s_neg, 3),
            "mention_count": int(r["mention_count"]),
            "total_mentions": tm,
            "anomaly_flag": bool(r["anomaly_flag"]),
            "risk_score": round(float(risk), 1),
            "risk_level": _level(risk, c["risk_high"], c["risk_medium"]),
            "confidence": round(confidence, 3),
            "confidence_level": _level(confidence, 0.66, 0.33),
        })

    summary = pd.DataFrame(rows)
    if labels is not None and "issue_label" in labels.columns:
        lbl = labels.groupby("issue_id")["issue_label"].first()
        summary.insert(1, "issue_label", summary["issue_id"].map(lbl))
    summary = summary.sort_values("risk_score", ascending=False).reset_index(drop=True)
    log.info("Scored %d issues; %d HIGH risk",
             len(summary), int((summary["risk_level"] == "HIGH").sum()))
    return summary
