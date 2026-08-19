"""Layer 9 cross-cutting edge cases (complements the per-layer test files)."""
import numpy as np
import pandas as pd
import pytest

from src.ingestion import to_standard_frame, validate_reviews
from src.preprocessing import build_cleaned_frame, clean_text


def _std(rows):
    """Build a standardized frame from partial dicts (missing keys -> NA)."""
    cols = ["review_id", "review_text", "rating", "review_date", "source_platform",
            "app_id", "app_name", "app_version", "helpful_count"]
    return to_standard_frame([{c: r.get(c) for c in cols} for r in rows])


# --- malformed / messy text ------------------------------------------------- #
def test_clean_text_handles_control_chars_and_unicode():
    assert clean_text("bad\x00\x07text\twith\x1fctrl") == "bad text with ctrl"
    assert clean_text("café naïve") == "café naïve"      # accents kept
    assert clean_text("WHY??????") == "why??"                     # same-char run capped
    assert clean_text("") == "" and clean_text(None) == ""


def test_emoji_or_punctuation_only_reviews_are_dropped():
    rows = [
        {"review_id": "e1", "review_text": "😀😀😀", "rating": 5,
         "review_date": "2026-08-01", "source_platform": "google_play", "app_id": "a"},
        {"review_id": "e2", "review_text": "!!!", "rating": 1,
         "review_date": "2026-08-01", "source_platform": "google_play", "app_id": "a"},
        {"review_id": "e4", "review_text": "​​ ​", "rating": 2,
         "review_date": "2026-08-01", "source_platform": "google_play", "app_id": "a"},
        {"review_id": "e3", "review_text": "genuinely useful review text", "rating": 4,
         "review_date": "2026-08-01", "source_platform": "google_play", "app_id": "a"},
    ]
    cleaned, report = build_cleaned_frame(_std(rows))
    assert list(cleaned["review_id"]) == ["e3"]
    assert report["empty_after_cleaning"] == 3


# --- invalid ratings -------------------------------------------------------- #
@pytest.mark.parametrize("bad_rating", [0, 6, -1, 99, np.nan])
def test_out_of_range_ratings_are_rejected(bad_rating):
    rows = [{"review_id": "x", "review_text": "some text here", "rating": bad_rating,
             "review_date": "2026-08-01", "source_platform": "app_store", "app_id": "1"}]
    _, report = validate_reviews(_std(rows))
    assert report["invalid_rating"] == 1
    assert report["dropped"] == 1


# --- missing values / malformed records ------------------------------------- #
def test_completely_empty_record_dropped():
    _, report = validate_reviews(_std([{}]))
    assert report["dropped"] == 1


def test_missing_platform_and_app_id_flagged():
    rows = [{"review_id": "r", "review_text": "text", "rating": 3,
             "review_date": "2026-08-01", "source_platform": None, "app_id": None}]
    _, report = validate_reviews(_std(rows))
    assert report["missing_platform"] == 1
    assert report["missing_app_id"] == 1


# --- duplicates across a mixed batch ---------------------------------------- #
def test_duplicate_ids_deduped_keeping_first():
    rows = [
        {"review_id": "d", "review_text": "first", "rating": 1,
         "review_date": "2026-08-01", "source_platform": "google_play", "app_id": "a"},
        {"review_id": "d", "review_text": "second (dup id)", "rating": 5,
         "review_date": "2026-08-02", "source_platform": "google_play", "app_id": "a"},
    ]
    clean, report = validate_reviews(_std(rows))
    assert report["duplicate_review_id"] == 1
    assert len(clean) == 1
    assert clean.iloc[0]["review_text"] == "first"


# --- unavailable database --------------------------------------------------- #
def test_dashboard_reports_missing_database(monkeypatch, tmp_path):
    import dashboard.data as dd
    monkeypatch.setattr(dd, "db_path", lambda: tmp_path / "nope.duckdb")
    with pytest.raises(FileNotFoundError):
        dd.load_all()
    with pytest.raises(FileNotFoundError):
        dd.representative_reviews("issue_00")
