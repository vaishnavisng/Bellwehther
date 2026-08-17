"""Cleaning + baseline feature creation.

Input: the standardized frame from Layer 1 (any platform).
Output: a validated `cleaned_reviews` frame that preserves the original text and
adds a lightly-normalized `cleaned_text` plus analytical features.

Normalization is deliberately gentle — no stopword or word removal — so issue
detection in later layers still sees the real vocabulary.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.ingestion.validation import validate_reviews
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

_WS = re.compile(r"\s+")
_PUNCT_RUN = re.compile(r"([!?.,])\1{2,}")   # cap runs of the same punct at 2
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: object) -> str:
    """Whitespace/punctuation/unicode normalization + lowercase. Keeps all words."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _CTRL.sub(" ", s)
    s = _PUNCT_RUN.sub(r"\1\1", s)
    s = _WS.sub(" ", s).strip()
    return s.lower()


def _rating_bucket(r) -> str | None:
    if pd.isna(r):
        return None
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def build_cleaned_frame(std_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate + clean a standardized frame. Returns (cleaned_df, quality_report)."""
    clean, report = validate_reviews(std_df)
    df = clean.copy()

    df["cleaned_text"] = df["review_text"].map(clean_text)
    # Drop rows that are empty only after cleaning (e.g. pure punctuation/emoji).
    empty_after = df["cleaned_text"].str.len() == 0
    if empty_after.any():
        report["empty_after_cleaning"] = int(empty_after.sum())
        df = df[~empty_after].reset_index(drop=True)

    df["rating_bucket"] = df["rating"].map(_rating_bucket).astype("string")
    df["review_length"] = df["cleaned_text"].str.len().astype("Int64")
    df["word_count"] = df["cleaned_text"].str.split().map(len).astype("Int64")

    d = df["review_date"]
    df["review_day"] = d.dt.date
    df["review_week"] = d.dt.strftime("%G-W%V").astype("string")   # ISO year-week
    df["review_month"] = d.dt.strftime("%Y-%m").astype("string")
    df["review_year"] = d.dt.year.astype("Int64")

    report["cleaned_rows"] = int(len(df))
    log.info("Cleaned frame: %d rows (from %d standardized)",
             len(df), report["total"])
    return df, report
