"""Layer 3 tests: issue extraction (TF-IDF + KMeans)."""
import pandas as pd
import pytest

from src.nlp import extract_issues


def make_df(texts, ratings=None):
    ratings = ratings or [3] * len(texts)
    return pd.DataFrame({
        "review_id": [f"r{i}" for i in range(len(texts))],
        "cleaned_text": texts,
        "rating": ratings,
    })


# Three clearly separable themes, repeated to exceed min_reviews_for_clustering.
THEMES = {
    "crash": "the app keeps crashing and freezes on launch every time",
    "battery": "battery drains fast and phone overheats while using this app",
    "ads": "too many ads popping up constantly interrupting everything annoying",
}


def themed_df(reps=8):
    texts, rids, ratings = [], [], []
    for i in range(reps):
        for j, (_, base) in enumerate(THEMES.items()):
            texts.append(f"{base} number {i}")
            rids.append(f"{j}_{i}")
            ratings.append(1 + j)
    return pd.DataFrame({"review_id": rids, "cleaned_text": texts, "rating": ratings})


def test_empty_dataset():
    res = extract_issues(make_df([]))
    assert res.k == 0
    assert len(res.assignments) == 0
    assert len(res.summary) == 0


def test_very_small_dataset_single_cluster():
    res = extract_issues(make_df(["login broken", "cannot pay", "great app"]))
    assert res.k == 1
    assert res.method == "single"
    assert len(res.assignments) == 3
    assert res.assignments["issue_id"].nunique() == 1


def test_invalid_k_too_large():
    with pytest.raises(ValueError):
        extract_issues(make_df(["a b c", "d e f"]), k=5)


def test_invalid_k_zero():
    with pytest.raises(ValueError):
        extract_issues(make_df(["a b c", "d e f"]), k=0)


def test_reproducibility():
    df = themed_df()
    a = extract_issues(df)
    b = extract_issues(df)
    assert a.k == b.k
    pd.testing.assert_frame_equal(a.assignments, b.assignments)


def test_output_consistency():
    df = themed_df()
    res = extract_issues(df)
    # every review assigned exactly once
    assert len(res.assignments) == len(df)
    assert set(res.assignments["review_id"]) == set(df["review_id"])
    # cluster sizes in summary account for all reviews
    assert res.summary["cluster_size"].sum() == len(df)
    # labels + keywords populated for real clusters
    assert res.assignments["issue_label"].notna().all()
    assert (res.summary["issue_keywords"].str.len() > 0).all()
    # k discovered within configured range
    assert 2 <= res.k <= 8


def test_finds_multiple_distinct_issues():
    res = extract_issues(themed_df())
    assert res.k >= 2  # separable themes should not collapse to one cluster
    labels = " ".join(res.summary["issue_keywords"])
    # keywords should reflect the actual vocabulary, not hardcoded categories
    assert any(w in labels for w in ("crash", "battery", "ads", "overheats"))
