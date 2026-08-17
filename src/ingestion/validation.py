"""Validation for standardized reviews.

Splits a standardized frame into clean rows and a problem report. Rules cover
the failure modes both adapters can produce: missing text, out-of-range rating,
bad date, duplicate ids, missing platform/app_id, unknown platform.
"""
from __future__ import annotations

import pandas as pd

from src.ingestion.schema import VALID_PLATFORMS
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def validate_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). Clean = valid, unique rows keeping first dup."""
    bad = pd.Series(False, index=df.index)

    missing_text = df["review_text"].isna() | (df["review_text"].str.strip() == "")
    invalid_rating = df["rating"].isna() | ~df["rating"].between(1, 5)
    invalid_date = df["review_date"].isna()
    missing_platform = df["source_platform"].isna()
    unknown_platform = ~df["source_platform"].isin(VALID_PLATFORMS)
    missing_app_id = df["app_id"].isna() | (df["app_id"].str.strip() == "")
    missing_id = df["review_id"].isna() | (df["review_id"].str.strip() == "")
    duplicate_id = df["review_id"].duplicated(keep="first") & ~missing_id

    for flag in (missing_text, invalid_rating, invalid_date, missing_platform,
                 unknown_platform, missing_app_id, missing_id, duplicate_id):
        bad |= flag.fillna(True)

    report = {
        "total": int(len(df)),
        "missing_text": int(missing_text.sum()),
        "invalid_rating": int(invalid_rating.sum()),
        "invalid_date": int(invalid_date.sum()),
        "missing_platform": int(missing_platform.sum()),
        "unknown_platform": int(unknown_platform.sum()),
        "missing_app_id": int(missing_app_id.sum()),
        "missing_review_id": int(missing_id.sum()),
        "duplicate_review_id": int(duplicate_id.sum()),
        "dropped": int(bad.sum()),
    }
    report["clean"] = report["total"] - report["dropped"]

    if report["dropped"]:
        log.warning("Validation dropped %d/%d records: %s",
                    report["dropped"], report["total"],
                    {k: v for k, v in report.items()
                     if k not in ("total", "clean", "dropped") and v})

    return df[~bad].reset_index(drop=True), report
