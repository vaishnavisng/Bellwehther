"""Layer 2 tests: cleaning/features and DuckDB storage + quality checks."""
import pandas as pd

from src.ingestion import AppStoreAdapter, GooglePlayAdapter
from src.ingestion.sample import load_sample_raw
from src.preprocessing import build_cleaned_frame, clean_text
from src.storage import (
    CLEANED_TABLE,
    RAW_TABLE,
    connect,
    create_tables,
    data_quality,
    insert_reviews,
    read_table,
    row_count,
)


def sample_std():
    g = GooglePlayAdapter("com.example.app", "Example App").standardize(
        load_sample_raw("google_play"))
    a = AppStoreAdapter("123456789", "Example App").standardize(
        load_sample_raw("app_store"))
    return pd.concat([g, a], ignore_index=True)


# --- cleaning ---
def test_clean_text_normalizes_but_keeps_words():
    assert clean_text("  Login   FAILS!!!!!  ") == "login fails!!"
    assert clean_text(None) == ""
    assert clean_text("café") == "café"  # unicode preserved, not stripped


def test_build_cleaned_frame_features():
    df, report = build_cleaned_frame(sample_std())
    assert len(df) == 6
    for col in ("cleaned_text", "rating_bucket", "review_length", "word_count",
                "review_day", "review_week", "review_month", "review_year"):
        assert col in df.columns
    assert set(df["rating_bucket"]) <= {"negative", "neutral", "positive"}
    # original text preserved alongside cleaned text
    assert df["review_text"].notna().all()
    assert (df["word_count"] > 0).all()


def test_build_cleaned_frame_drops_bad_rows():
    bad = pd.DataFrame([{
        "review_id": "b1", "review_text": "   ", "rating": 3,
        "review_date": pd.Timestamp("2026-08-01"), "source_platform": "google_play",
        "app_id": "com.x", "app_name": "X", "app_version": None, "helpful_count": None,
    }])
    df, report = build_cleaned_frame(bad)
    assert len(df) == 0  # whitespace-only text rejected


# --- storage ---
def test_storage_roundtrip_and_quality():
    con = connect(":memory:")
    create_tables(con)

    std = sample_std()
    cleaned, _ = build_cleaned_frame(std)

    insert_reviews(con, RAW_TABLE, std, replace=True)
    insert_reviews(con, CLEANED_TABLE, cleaned, replace=True)

    assert row_count(con, RAW_TABLE) == 6
    assert row_count(con, CLEANED_TABLE) == 6

    back = read_table(con, CLEANED_TABLE)
    assert set(back["source_platform"]) == {"google_play", "app_store"}

    q = data_quality(con, CLEANED_TABLE)
    assert q["duplicate_review_ids"] == 0
    assert q["missing_text"] == 0
    assert q["invalid_rating"] == 0
    assert q["date_min"] is not None
    con.close()


def test_insert_replace_is_idempotent():
    con = connect(":memory:")
    create_tables(con)
    std = sample_std()
    insert_reviews(con, RAW_TABLE, std, replace=True)
    insert_reviews(con, RAW_TABLE, std, replace=True)  # re-run must not duplicate
    assert row_count(con, RAW_TABLE) == 6
    con.close()


def test_cleaned_primary_key_blocks_duplicates():
    con = connect(":memory:")
    create_tables(con)
    cleaned, _ = build_cleaned_frame(sample_std())
    insert_reviews(con, CLEANED_TABLE, cleaned)
    try:
        insert_reviews(con, CLEANED_TABLE, cleaned)  # same ids -> PK violation
        assert False, "expected primary key violation"
    except Exception:
        pass
    con.close()
