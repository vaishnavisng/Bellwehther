"""Issue extraction: cleaned reviews -> TF-IDF -> KMeans -> labeled issues.

Issues are *discovered*, never hardcoded: cluster labels come from each cluster's
top TF-IDF terms. K is chosen by silhouette score over a configurable range so
the choice is defensible, not arbitrary.

A cluster is not automatically a business issue — the summary keeps top terms,
size, rating distribution, and representative reviews so an analyst can judge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class IssueResult:
    assignments: pd.DataFrame   # one row per review, denormalized with cluster stats
    summary: pd.DataFrame       # one row per issue/cluster
    k: int                      # number of clusters used
    method: str                 # how k was chosen ("silhouette", "single", "fixed")


def _nlp_cfg() -> dict:
    cfg = load_config().get("nlp", {}) or {}
    return {
        "k_min": cfg.get("k_min", 2),
        "k_max": cfg.get("k_max", 8),
        "min_reviews_for_clustering": cfg.get("min_reviews_for_clustering", 10),
        "min_df": cfg.get("min_df", 1),
        "max_df": cfg.get("max_df", 0.9),
        "ngram_max": cfg.get("ngram_max", 2),
        "top_keywords": cfg.get("top_keywords", 8),
        "n_representative": cfg.get("n_representative", 3),
        "random_state": cfg.get("random_state", 42),
    }


def _vectorize(texts, c) -> tuple:
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, c["ngram_max"]),
        min_df=c["min_df"],
        max_df=c["max_df"],
    )
    X = vec.fit_transform(texts)
    return vec, X


def choose_k(X, k_min: int, k_max: int, random_state: int) -> tuple[int, str]:
    """Pick K by best silhouette over [k_min, k_max], capped at n_samples-1."""
    n = X.shape[0]
    hi = min(k_max, n - 1)
    lo = max(2, k_min)
    if hi < lo:
        return 1, "single"
    best_k, best_score = 1, -1.0
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, random_state=random_state,
                        n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        log.debug("k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_score, best_k = score, k
    method = "silhouette" if best_k > 1 else "single"
    return best_k, method


def _cluster_keywords(centroid: np.ndarray, terms: np.ndarray, top_n: int) -> list[str]:
    idx = np.argsort(centroid)[::-1][:top_n]
    return [terms[i] for i in idx if centroid[i] > 0]


def extract_issues(cleaned_df: pd.DataFrame, k: int | None = None) -> IssueResult:
    """Cluster cleaned reviews into labeled issues.

    k=None -> silhouette-chosen. Pass k to fix it (must be 1..n_reviews).
    """
    c = _nlp_cfg()
    df = cleaned_df.reset_index(drop=True)
    n = len(df)
    cols = ["review_id", "cluster_id", "issue_id", "issue_label",
            "issue_keywords", "cluster_size", "cluster_avg_rating"]

    if n == 0:
        empty = pd.DataFrame(columns=cols)
        return IssueResult(empty, pd.DataFrame(columns=cols[1:]), 0, "empty")

    if k is not None and (k < 1 or k > n):
        raise ValueError(f"k={k} invalid for {n} reviews (must be 1..{n})")

    texts = df["cleaned_text"].fillna("").tolist()
    ratings = pd.to_numeric(df["rating"], errors="coerce")

    # Too little data, or degenerate vocabulary -> one honest cluster.
    single = n < c["min_reviews_for_clustering"] and k is None
    try:
        vec, X = _vectorize(texts, c)
        if X.shape[1] == 0:
            single = True
    except ValueError:
        single = True

    if single or k == 1:
        labels = np.zeros(n, dtype=int)
        chosen_k, method = 1, "single"
        terms = vec.get_feature_names_out() if not single else np.array([])
    else:
        if k is None:
            chosen_k, method = choose_k(X, c["k_min"], c["k_max"], c["random_state"])
        else:
            chosen_k, method = k, "fixed"
        if chosen_k == 1:
            labels = np.zeros(n, dtype=int)
            method = "single"
        else:
            km = KMeans(n_clusters=chosen_k, random_state=c["random_state"],
                        n_init=10)
            labels = km.fit_predict(X)
        terms = vec.get_feature_names_out()

    df = df.assign(cluster_id=labels)
    Xd = X.toarray() if not single else None

    summary_rows, assign_map = [], {}
    for cid in sorted(set(labels)):
        members = np.where(labels == cid)[0]
        size = len(members)
        avg_rating = float(ratings.iloc[members].mean()) if size else float("nan")

        if Xd is not None and len(terms):
            centroid = Xd[members].mean(axis=0)
            keywords = _cluster_keywords(centroid, terms, c["top_keywords"])
            # representative = members most similar to their centroid
            sims = Xd[members] @ centroid
            order = members[np.argsort(sims)[::-1]]
            rep_ids = df["review_id"].iloc[order[:c["n_representative"]]].tolist()
        else:
            keywords, rep_ids = [], df["review_id"].iloc[members[:c["n_representative"]]].tolist()

        issue_id = f"issue_{cid:02d}"
        label = ", ".join(keywords[:3]) if keywords else "unclustered"
        kw_str = ", ".join(keywords)
        assign_map[cid] = (issue_id, label, kw_str, size, avg_rating)
        summary_rows.append({
            "cluster_id": int(cid),
            "issue_id": issue_id,
            "issue_label": label,
            "issue_keywords": kw_str,
            "cluster_size": size,
            "cluster_avg_rating": round(avg_rating, 3) if size else None,
            "representative_review_ids": ", ".join(map(str, rep_ids)),
        })

    assignments = df[["review_id", "cluster_id"]].copy()
    assignments["issue_id"] = assignments["cluster_id"].map(lambda c_: assign_map[c_][0])
    assignments["issue_label"] = assignments["cluster_id"].map(lambda c_: assign_map[c_][1])
    assignments["issue_keywords"] = assignments["cluster_id"].map(lambda c_: assign_map[c_][2])
    assignments["cluster_size"] = assignments["cluster_id"].map(lambda c_: assign_map[c_][3])
    assignments["cluster_avg_rating"] = assignments["cluster_id"].map(
        lambda c_: round(assign_map[c_][4], 3))
    assignments = assignments[cols]

    summary = pd.DataFrame(summary_rows)
    log.info("Extracted %d issues from %d reviews (k=%d via %s)",
             len(summary), n, chosen_k, method)
    return IssueResult(assignments, summary, chosen_k, method)
