"""Layer 1 tests: adapters -> standardized schema, validation, cross-platform."""
import pandas as pd
import pytest

from src.ingestion import (
    COLUMNS,
    AppStoreAdapter,
    GooglePlayAdapter,
    validate_reviews,
)
from src.ingestion.sample import load_sample_raw

GP_APP = ("com.example.app", "Example App")
AS_APP = ("123456789", "Example App")


def gp():
    return GooglePlayAdapter(*GP_APP)


def apple():
    return AppStoreAdapter(*AS_APP)


# --- Google Play ---
def test_gp_successful_ingestion():
    df = gp().standardize(load_sample_raw("google_play"))
    assert len(df) == 3
    assert (df["source_platform"] == "google_play").all()
    assert df["review_text"].notna().all()


def test_gp_empty_response():
    df = gp().standardize([])
    assert len(df) == 0
    assert list(df.columns) == COLUMNS


def test_gp_malformed_response():
    # Missing content/score/date -> standardizes but validation must flag it.
    df = gp().standardize([{"reviewId": "x", "userName": "u"}])
    clean, report = validate_reviews(df)
    assert len(clean) == 0
    assert report["missing_text"] == 1


def test_gp_duplicate_reviews():
    raw = load_sample_raw("google_play")
    df = gp().standardize(raw + raw[:1])  # one duplicate reviewId
    clean, report = validate_reviews(df)
    assert report["duplicate_review_id"] == 1
    assert len(clean) == 3


def test_gp_schema_validation():
    df = gp().standardize(load_sample_raw("google_play"))
    assert list(df.columns) == COLUMNS
    assert df["rating"].between(1, 5).all()


# --- Apple App Store ---
def test_as_successful_ingestion():
    df = apple().standardize(load_sample_raw("app_store"))
    assert len(df) == 3
    assert (df["source_platform"] == "app_store").all()
    # Apple omits these fields -> NULL, but schema still present.
    assert df["app_version"].isna().all()
    assert df["helpful_count"].isna().all()


def test_as_empty_response():
    df = apple().standardize([])
    assert len(df) == 0
    assert list(df.columns) == COLUMNS


def test_as_malformed_response():
    df = apple().standardize([{"userName": "u", "rating": 9}])  # no text, bad rating
    clean, report = validate_reviews(df)
    assert len(clean) == 0
    assert report["missing_text"] == 1
    assert report["invalid_rating"] == 1


def test_as_synthesizes_stable_ids_and_dedupes():
    raw = load_sample_raw("app_store")
    df = apple().standardize(raw + raw)  # every record duplicated
    assert df["review_id"].notna().all()
    clean, report = validate_reviews(df)
    assert report["duplicate_review_id"] == 3
    assert len(clean) == 3


def test_as_schema_validation():
    df = apple().standardize(load_sample_raw("app_store"))
    assert list(df.columns) == COLUMNS
    assert df["rating"].between(1, 5).all()


# --- Cross-platform ---
def test_both_produce_same_schema():
    g = gp().standardize(load_sample_raw("google_play"))
    a = apple().standardize(load_sample_raw("app_store"))
    assert list(g.columns) == list(a.columns) == COLUMNS
    assert g.dtypes.to_dict() == a.dtypes.to_dict()


def test_combined_dataset_preserves_platform_and_app_ids():
    g = gp().standardize(load_sample_raw("google_play"))
    a = apple().standardize(load_sample_raw("app_store"))
    combined = pd.concat([g, a], ignore_index=True)
    assert set(combined["source_platform"]) == {"google_play", "app_store"}
    assert set(combined["app_id"]) == {"com.example.app", "123456789"}
    assert combined["rating"].between(1, 5).all()
    clean, report = validate_reviews(combined)
    assert report["dropped"] == 0
    assert len(clean) == 6


def test_missing_app_id_rejected_at_construction():
    with pytest.raises(ValueError):
        GooglePlayAdapter("", "x")
