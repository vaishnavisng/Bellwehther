"""Historical rating-impact analysis.

Business question, per issue: "When this issue appears, how strongly is it
associated with lower ratings?" — answered defensibly, and framed as
*association, not causation*.

Two complementary views:
  1. Group comparison  - ratings for issue vs non-issue reviews, with a test
     chosen from the data (not a reflexive t-test) and a CI on the difference.
  2. OLS regression    - issue dummy + controls (review length, time period) to
     estimate the adjusted rating penalty historically associated with the issue.

Small samples are marked unreliable rather than reported as precise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

IMPACT_COLUMNS = [
    "issue_id", "issue_label", "sample_size", "non_issue_size",
    "average_issue_rating", "average_non_issue_rating", "overall_rating",
    "median_issue_rating", "median_non_issue_rating", "rating_difference",
    "low_rating_share_issue", "low_rating_share_non_issue",
    "test_used", "test_reasoning", "p_value", "significant",
    "diff_ci_low", "diff_ci_high",
    "regression_effect", "regression_ci_low", "regression_ci_high", "regression_p",
    "reliable", "confidence_level", "interpretation",
]


def _cfg() -> dict:
    p = load_config().get("impact", {}) or {}
    return {
        "min_sample": p.get("min_sample", 30),     # below this -> unreliable
        "hard_floor": p.get("hard_floor", 5),      # below this -> no test at all
        "alpha": p.get("alpha", 0.05),
        "low_rating_threshold": p.get("low_rating_threshold", 2),
        "normal_min_n": p.get("normal_min_n", 20),  # min n to even consider a t-test
    }


def _choose_and_run_test(issue_r: np.ndarray, other_r: np.ndarray, c: dict):
    """Pick a test from the data. Ratings are ordinal 1-5, usually non-normal,
    so the default is Mann-Whitney U; Welch's t is used only when both groups
    are reasonably large AND normality is not rejected."""
    n1, n2 = len(issue_r), len(other_r)
    if n1 < c["hard_floor"] or n2 < c["hard_floor"]:
        return "none", "sample too small for a meaningful test", None, None

    use_t = False
    if n1 >= c["normal_min_n"] and n2 >= c["normal_min_n"]:
        # Shapiro needs some variance; guard constant groups.
        try:
            p1 = stats.shapiro(issue_r).pvalue if np.ptp(issue_r) > 0 else 0.0
            p2 = stats.shapiro(other_r).pvalue if np.ptp(other_r) > 0 else 0.0
            use_t = p1 > c["alpha"] and p2 > c["alpha"]
        except Exception:
            use_t = False

    if use_t:
        stat, p = stats.ttest_ind(issue_r, other_r, equal_var=False)
        reason = ("large samples, normality not rejected -> Welch's t-test "
                  "(unequal variance)")
        return "welch_t", reason, float(stat), float(p)

    stat, p = stats.mannwhitneyu(issue_r, other_r, alternative="two-sided")
    reason = ("ordinal 1-5 ratings / non-normal or moderate n -> "
              "Mann-Whitney U (rank-based)")
    return "mann_whitney_u", reason, float(stat), float(p)


def _diff_ci(issue_r: np.ndarray, other_r: np.ndarray, alpha: float):
    """95% CI for the difference in mean rating (Welch / Satterthwaite)."""
    n1, n2 = len(issue_r), len(other_r)
    if n1 < 2 or n2 < 2:
        return None, None
    v1, v2 = issue_r.var(ddof=1), other_r.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    diff = issue_r.mean() - other_r.mean()
    if se == 0:
        return float(diff), float(diff)
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(diff - tcrit * se), float(diff + tcrit * se)


def _regression(data: pd.DataFrame, alpha: float):
    """OLS rating ~ is_issue + controls. Returns (effect, ci_low, ci_high, p)."""
    if data["is_issue"].nunique() < 2 or len(data) < 10:
        return None, None, None, None
    terms = ["is_issue"]
    if data["review_length"].nunique() > 1:
        terms.append("review_length")
    if data["period_idx"].nunique() > 1:
        terms.append("period_idx")
    formula = "rating ~ " + " + ".join(terms)
    try:
        model = smf.ols(formula, data=data).fit(cov_type="HC3")
        ci = model.conf_int(alpha=alpha).loc["is_issue"]
        return (float(model.params["is_issue"]), float(ci[0]), float(ci[1]),
                float(model.pvalues["is_issue"]))
    except Exception:
        log.exception("Regression failed")
        return None, None, None, None


def _confidence_level(n_issue: int, sig: bool, reg_excludes_zero: bool, c: dict) -> str:
    if n_issue < c["min_sample"]:
        return "low"
    if sig and reg_excludes_zero:
        return "high"
    if sig or reg_excludes_zero:
        return "medium"
    return "low"


def compute_issue_impact(cleaned_df: pd.DataFrame,
                         assignments: pd.DataFrame) -> pd.DataFrame:
    """One row per issue: group comparison + regression + reliability flags."""
    c = _cfg()
    if len(cleaned_df) == 0 or len(assignments) == 0:
        return pd.DataFrame(columns=IMPACT_COLUMNS)

    df = cleaned_df.copy()
    if "review_length" not in df.columns:
        df["review_length"] = df.get("cleaned_text", "").astype(str).str.len()
    df["review_length"] = pd.to_numeric(df["review_length"], errors="coerce").fillna(0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["period_idx"] = ((df["review_date"] - df["review_date"].min())
                        .dt.days // 7).astype("Int64").fillna(0).astype(int)
    df = df.dropna(subset=["rating"])

    labels = assignments.groupby("issue_id")["issue_label"].first() \
        if "issue_label" in assignments.columns else {}
    issue_of = assignments.set_index("review_id")["issue_id"]
    df["issue_id"] = df["review_id"].map(issue_of)
    df = df.dropna(subset=["issue_id"])

    overall = float(df["rating"].mean())
    low_t = c["low_rating_threshold"]
    rows = []
    for issue_id, members in df.groupby("issue_id"):
        issue_r = members["rating"].to_numpy(float)
        other = df[df["issue_id"] != issue_id]
        other_r = other["rating"].to_numpy(float)
        n1, n2 = len(issue_r), len(other_r)

        test_used, reason, _, p = _choose_and_run_test(issue_r, other_r, c)
        ci_lo, ci_hi = _diff_ci(issue_r, other_r, c["alpha"])

        reg_data = df.assign(is_issue=(df["issue_id"] == issue_id).astype(int))
        eff, r_lo, r_hi, r_p = _regression(reg_data, c["alpha"])

        sig = bool(p is not None and p < c["alpha"])
        reg_excl_zero = bool(r_lo is not None and (r_lo > 0 or r_hi < 0))
        reliable = n1 >= c["min_sample"]
        conf = _confidence_level(n1, sig, reg_excl_zero, c)

        if reliable and eff is not None:
            direction = "lower" if eff < 0 else "higher"
            interp = (f"Reviews mentioning this issue are historically associated "
                      f"with approximately {abs(eff):.1f}-star {direction} ratings "
                      f"(association, not causation).")
        else:
            interp = ("Too few observations for a reliable estimate; "
                      "treat as indicative only.")

        rows.append({
            "issue_id": issue_id,
            "issue_label": labels.get(issue_id) if hasattr(labels, "get") else None,
            "sample_size": n1,
            "non_issue_size": n2,
            "average_issue_rating": round(float(issue_r.mean()), 3),
            "average_non_issue_rating": round(float(other_r.mean()), 3) if n2 else None,
            "overall_rating": round(overall, 3),
            "median_issue_rating": float(np.median(issue_r)),
            "median_non_issue_rating": float(np.median(other_r)) if n2 else None,
            "rating_difference": round(float(issue_r.mean() - other_r.mean()), 3) if n2 else None,
            "low_rating_share_issue": round(float((issue_r <= low_t).mean()), 3),
            "low_rating_share_non_issue": round(float((other_r <= low_t).mean()), 3) if n2 else None,
            "test_used": test_used,
            "test_reasoning": reason,
            "p_value": round(p, 5) if p is not None else None,
            "significant": sig,
            "diff_ci_low": round(ci_lo, 3) if ci_lo is not None else None,
            "diff_ci_high": round(ci_hi, 3) if ci_hi is not None else None,
            "regression_effect": round(eff, 3) if eff is not None else None,
            "regression_ci_low": round(r_lo, 3) if r_lo is not None else None,
            "regression_ci_high": round(r_hi, 3) if r_hi is not None else None,
            "regression_p": round(r_p, 5) if r_p is not None else None,
            "reliable": reliable,
            "confidence_level": conf,
            "interpretation": interp,
        })

    out = pd.DataFrame(rows, columns=IMPACT_COLUMNS)
    out = out.sort_values("rating_difference").reset_index(drop=True)
    log.info("Computed impact for %d issues (%d reliable)",
             len(out), int(out["reliable"].sum()))
    return out
