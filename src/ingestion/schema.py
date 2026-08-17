"""Standardized, platform-independent review schema.

Both adapters emit records with exactly these fields, and `to_standard_frame`
coerces them to consistent dtypes. Everything after ingestion depends only on
this schema, never on the source platform's raw shape.
"""
from __future__ import annotations

import hashlib

import pandas as pd

# Field -> pandas dtype. Nullable dtypes where a source may not supply the value.
SCHEMA: dict[str, str] = {
    "review_id": "string",
    "review_text": "string",
    "rating": "Int64",          # 1-5, nullable so malformed rows survive to validation
    "review_date": "datetime64[ns]",
    "source_platform": "string",  # "google_play" | "app_store"
    "app_id": "string",
    "app_name": "string",
    "app_version": "string",    # NULL where the source omits it (e.g. App Store)
    "helpful_count": "Int64",   # NULL where the source omits it (e.g. App Store)
}
COLUMNS = list(SCHEMA)

VALID_PLATFORMS = {"google_play", "app_store"}


def synthesize_review_id(source_platform: str, app_id: str, *parts: object) -> str:
    """Deterministic id for sources that don't provide one (Apple App Store).

    Stable across runs so duplicate detection works, derived only from review
    content — no reviewer PII is stored.
    """
    raw = "|".join(str(p) for p in (source_platform, app_id, *parts))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def to_standard_frame(records: list[dict]) -> pd.DataFrame:
    """Build a schema-typed DataFrame from standard-key dicts (order/dtype fixed)."""
    df = pd.DataFrame(records, columns=COLUMNS)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["helpful_count"] = pd.to_numeric(df["helpful_count"], errors="coerce").astype("Int64")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    for col in ("review_id", "review_text", "source_platform", "app_id",
                "app_name", "app_version"):
        df[col] = df[col].astype("string")
    return df[COLUMNS]
