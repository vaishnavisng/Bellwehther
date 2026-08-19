"""Forward-looking rating-risk prediction — the core, deliberately transparent layer.

PREDICTION TARGET
    Expected change in the product's *overall average rating* over the next H
    periods that is attributable to a single issue's changing prevalence.

TRANSPARENT MODEL (no black box)
    overall_avg = share * mean_issue_rating + (1 - share) * mean_non_issue_rating
    so a change in prevalence maps to a rating change by simple arithmetic:

        predicted_rating_impact = (predicted_share_{t+H} - current_share)
                                   * historical_penalty

    where:
      * predicted_share_{t+H} comes from a simple OLS trend extrapolation of the
        issue's share series (with a prediction interval for uncertainty), and
      * historical_penalty = (mean_issue_rating - mean_non_issue_rating), taken
        from Layer 5 (adjusted regression effect when reliable, else the raw
        group difference).

    Every input is an observable number from earlier layers; the forecast is
    linear extrapolation an analyst can reproduce by hand. This is *association*
    carried forward, never a causal claim.

HONESTY
    If the dataset has too few historical periods, we do not invent a forecast —
    the issue is marked insufficient_history. Backtesting reports real error and
    is explicit when the sample is too small for strong claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

PREDICTION_COLUMNS = [
    "issue_id", "issue_label", "n_periods", "horizon",
    "current_share", "recent_growth", "current_trend",
    "historical_rating_impact", "predicted_share", "predicted_rating_impact",
    "lower_bound", "upper_bound", "risk_level", "confidence_level",
    "reason_code", "explanation",
]

_FREQ_UNIT = {"W": "weeks", "D": "days", "M": "months"}


def _cfg() -> dict:
    f = load_config().get("forecast", {}) or {}
    a = load_config().get("analytics", {}) or {}
    return {
        "horizon_periods": f.get("horizon_periods", 2),
        "min_history_periods": f.get("min_history_periods", 5),
        "min_fit_points": f.get("min_fit_points", 4),
        "alpha": f.get("alpha", 0.05),
        # predicted overall-rating-impact magnitude (stars) -> base risk tier
        "impact_critical": f.get("impact_critical", 0.30),
        "impact_high": f.get("impact_high", 0.15),
        "impact_medium": f.get("impact_medium", 0.05),
        "disproportion_gap": f.get("disproportion_gap", 0.20),
        "freq": a.get("freq", "W"),
    }


def forecast_share(shares, horizon: int, alpha: float, min_fit: int) -> dict | None:
    """OLS trend extrapolation of an issue's share series, H periods ahead.

    Returns point + prediction interval (clipped to [0,1]), or None if too short.
    """
    shares = np.asarray(shares, float)
    n = len(shares)
    if n < min_fit:
        return None
    x = np.arange(n)
    model = sm.OLS(shares, sm.add_constant(x)).fit()
    future = np.array([[1.0, n - 1 + horizon]])
    fr = model.get_prediction(future).summary_frame(alpha=alpha)
    return {
        "predicted": float(np.clip(fr["mean"].iloc[0], 0, 1)),
        "low": float(np.clip(fr["obs_ci_lower"].iloc[0], 0, 1)),
        "high": float(np.clip(fr["obs_ci_upper"].iloc[0], 0, 1)),
        "slope": float(model.params[1]),
        "n": n,
    }


def _penalty(imp_row, alpha: float):
    """Historical per-mention rating penalty + whether it's significant/reliable."""
    if imp_row is None:
        return None, False, False, None, None
    reliable = bool(imp_row.get("reliable", False))
    eff = imp_row.get("regression_effect")
    rd = imp_row.get("rating_difference")
    penalty = eff if (reliable and pd.notna(eff)) else (rd if pd.notna(rd) else None)
    reg_p = imp_row.get("regression_p")
    significant = bool(imp_row.get("significant", False)) or (
        pd.notna(reg_p) and reg_p < alpha)
    return penalty, significant, reliable, imp_row.get("low_rating_share_issue"), \
        imp_row.get("low_rating_share_non_issue")


def _bump(level: str, by: int) -> str:
    return LEVELS[int(np.clip(LEVELS.index(level) + by, 0, len(LEVELS) - 1))]


def _classify_risk(pred_impact, penalty, reliable, significant, growth, anomaly, c) -> str:
    """Documented tiers: base tier from predicted harmful impact magnitude, then
    downgrade if the historical association is weak, upgrade if it's actively
    spiking. Only issues historically associated with LOWER ratings (penalty < 0)
    are treated as rating-deterioration risks — a benign issue merely losing
    share is not a warning."""
    if pred_impact is None or pred_impact >= 0 or penalty is None or penalty >= 0:
        return "LOW"
    mag = abs(pred_impact)
    if mag >= c["impact_critical"]:
        base = "CRITICAL"
    elif mag >= c["impact_high"]:
        base = "HIGH"
    elif mag >= c["impact_medium"]:
        base = "MEDIUM"
    else:
        base = "LOW"
    # weak evidence can't sustain the top tiers
    if base in ("HIGH", "CRITICAL") and not (reliable and significant):
        base = _bump(base, -1)
    # actively accelerating + anomalous -> one tier worse
    if anomaly and growth > 0 and base != "LOW":
        base = _bump(base, 1)
    return base


def _confidence(reliable, significant, enough_history, has_forecast) -> str:
    if not has_forecast:
        return "low"
    score = int(reliable) + int(significant) + int(enough_history)
    return "high" if score >= 3 else "medium" if score == 2 else "low"


def predict_rating_risk(trends: pd.DataFrame,
                        impact: pd.DataFrame) -> pd.DataFrame:
    """One forward-looking risk row per issue. Numbers derive from the data."""
    c = _cfg()
    if len(trends) == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    unit = _FREQ_UNIT.get(c["freq"], "periods")
    horizon_text = f"next {c['horizon_periods']} {unit}"
    n_periods = trends["date"].nunique()
    enough_history = n_periods >= c["min_history_periods"]
    imp_idx = impact.set_index("issue_id") if len(impact) else pd.DataFrame()

    rows = []
    for issue_id, g in trends.groupby("issue_id"):
        g = g.sort_values("date")
        shares = g["issue_share"].to_numpy(float)
        current_share = float(shares[-1])
        recent_growth = float(g["growth_rate"].iloc[-1])
        anomaly = bool(g["anomaly_flag"].iloc[-1])
        label = imp_idx.loc[issue_id]["issue_label"] if issue_id in imp_idx.index \
            and "issue_label" in imp_idx.columns else None

        imp_row = imp_idx.loc[issue_id].to_dict() if issue_id in imp_idx.index else None
        penalty, significant, reliable, low_iss, low_non = _penalty(imp_row, c["alpha"])

        fc = forecast_share(shares, c["horizon_periods"], c["alpha"],
                            c["min_fit_points"]) if enough_history else None
        slope = fc["slope"] if fc else float(np.polyfit(np.arange(len(shares)), shares, 1)[0]
                                             if len(shares) >= 2 else 0.0)
        trend = "rising" if slope > 1e-4 else "falling" if slope < -1e-4 else "flat"

        if fc and penalty is not None:
            predicted_share = fc["predicted"]
            d_share = predicted_share - current_share
            pred_impact = d_share * penalty
            b1 = (fc["low"] - current_share) * penalty
            b2 = (fc["high"] - current_share) * penalty
            lower, upper = sorted([b1, b2])
        else:
            predicted_share = pred_impact = lower = upper = None

        # Current burden = how much this issue drags the overall rating right now
        # (share x penalty). Used for risk when there's no forecast yet, so a
        # single scrape is still actionable.
        current_burden = current_share * penalty if penalty is not None else None
        risk_basis = pred_impact if fc and penalty is not None else current_burden

        risk = _classify_risk(risk_basis, penalty, reliable, significant,
                              recent_growth, anomaly, c)
        confidence = _confidence(reliable, significant, enough_history, fc is not None)

        reason, explanation = _explain(
            risk, current_share, recent_growth, trend, penalty, significant,
            pred_impact, current_burden, horizon_text, enough_history,
            fc is not None, low_iss, low_non, c)

        rows.append({
            "issue_id": issue_id,
            "issue_label": label,
            "n_periods": int(n_periods),
            "horizon": horizon_text,
            "current_share": round(current_share, 4),
            "recent_growth": round(recent_growth, 4),
            "current_trend": trend,
            "historical_rating_impact": round(penalty, 3) if penalty is not None else None,
            "predicted_share": round(predicted_share, 4) if predicted_share is not None else None,
            "predicted_rating_impact": round(pred_impact, 3) if pred_impact is not None else None,
            "lower_bound": round(lower, 3) if lower is not None else None,
            "upper_bound": round(upper, 3) if upper is not None else None,
            "risk_level": risk,
            "confidence_level": confidence,
            "reason_code": reason,
            "explanation": explanation,
        })

    out = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    # Rank worst-first: risk tier, then most-negative predicted impact, then
    # most-negative historical impact (so single-scrape burden issues still sort).
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    out["_r"] = out["risk_level"].map(rank).fillna(9)
    out = out.sort_values(
        ["_r", "predicted_rating_impact", "historical_rating_impact"],
        ascending=True, na_position="last").drop(columns="_r").reset_index(drop=True)
    log.info("Predicted risk for %d issues (%d HIGH+)",
             len(out), int(out["risk_level"].isin(["HIGH", "CRITICAL"]).sum()))
    return out


def _explain(risk, share, growth, trend, penalty, significant, pred_impact,
             current_burden, horizon_text, enough_history, has_forecast,
             low_iss, low_non, c) -> tuple:
    if penalty is None:
        return ("no_historical_impact",
                f"Issue share is {share:.1%} and {trend}, but there is no reliable "
                f"historical rating association to assess.")
    if penalty >= 0:
        return ("no_negative_pressure",
                f"Issue share is {share:.1%} and {trend}; it is historically "
                f"associated with {penalty:+.2f}-star (non-negative) ratings, so it "
                f"is not a rating-deterioration risk.")
    if has_forecast and pred_impact is not None and pred_impact >= 0:
        return ("no_negative_pressure",
                f"Issue share is {share:.1%} and {trend} ({growth:+.0%} recent "
                f"growth); its projected impact over the {horizon_text} is not "
                f"negative, so rating risk is low.")

    # A real complaint (penalty < 0). Build the shared evidence clauses.
    parts = [f"issue share is {share:.1%} and {trend}"]
    if has_forecast:
        parts[0] += f" ({growth:+.0%} vs recent baseline)"
    sig = "statistically significant" if significant else "not statistically significant"
    parts.append(f"it is historically associated with a {penalty:+.2f}-star "
                 f"rating change ({sig})")
    if low_iss is not None and low_non is not None and pd.notna(low_iss) \
            and pd.notna(low_non) and low_iss >= low_non + c["disproportion_gap"]:
        parts.append(f"it is disproportionately present in low-rated reviews "
                     f"({low_iss:.0%} vs {low_non:.0%})")

    if has_forecast and pred_impact is not None:
        parts.append(f"the projected impact over the {horizon_text} is "
                     f"{pred_impact:+.2f} stars")
        reason = ("accelerating_rating_risk"
                  if growth > 0 and risk in ("HIGH", "CRITICAL")
                  else "elevated_rating_risk" if risk in ("HIGH", "CRITICAL", "MEDIUM")
                  else "low_rating_risk")
    else:
        # No forecast yet: assess by current burden on the overall rating.
        parts.append(f"it currently accounts for an estimated {current_burden:+.2f}-"
                     f"star drag on the overall rating")
        parts.append("(too little history yet for a forward forecast — based on "
                     "the current snapshot)")
        reason = "current_burden"
    return reason, f"Risk is {risk} because " + "; ".join(parts) + "."


# --------------------------------------------------------------------------- #
# Backtesting: rolling-origin evaluation of the share forecast.
# --------------------------------------------------------------------------- #
def backtest(trends: pd.DataFrame, impact: pd.DataFrame | None = None) -> dict:
    """Simulate: at each past cutoff, forecast share H ahead and compare to what
    actually happened. Reports share error, directional accuracy, and prediction-
    interval coverage. Honest about small samples."""
    c = _cfg()
    horizon, min_fit, alpha = c["horizon_periods"], c["min_fit_points"], c["alpha"]
    imp_idx = impact.set_index("issue_id") if impact is not None and len(impact) else pd.DataFrame()

    abs_err, hits, covered, ri_abs_err = [], [], [], []
    for issue_id, g in trends.groupby("issue_id"):
        g = g.sort_values("date")
        shares = g["issue_share"].to_numpy(float)
        n = len(shares)
        penalty = None
        if issue_id in imp_idx.index:
            penalty, *_ = _penalty(imp_idx.loc[issue_id].to_dict(), alpha)
        for t in range(min_fit - 1, n - horizon):
            fc = forecast_share(shares[:t + 1], horizon, alpha, min_fit)
            if fc is None:
                continue
            actual = shares[t + horizon]
            base = shares[t]
            abs_err.append(abs(fc["predicted"] - actual))
            hits.append(np.sign(fc["predicted"] - base) == np.sign(actual - base))
            covered.append(fc["low"] <= actual <= fc["high"])
            if penalty is not None:
                ri_abs_err.append(abs((fc["predicted"] - base) * penalty
                                      - (actual - base) * penalty))

    n_bt = len(abs_err)
    if n_bt == 0:
        return {"n_backtests": 0,
                "note": "Insufficient history for backtesting; need more periods."}
    report = {
        "n_backtests": n_bt,
        "share_mae": round(float(np.mean(abs_err)), 4),
        "directional_accuracy": round(float(np.mean(hits)), 3),
        "interval_coverage": round(float(np.mean(covered)), 3),
        "target_coverage": round(1 - alpha, 3),
        "rating_impact_mae": round(float(np.mean(ri_abs_err)), 4) if ri_abs_err else None,
    }
    report["limitations"] = (
        "Small sample — treat metrics as indicative, not conclusive."
        if n_bt < 10 else
        "Linear share extrapolation; assumes recent trend persists over the horizon.")
    log.info("Backtest: %d windows, share_MAE=%.4f, dir_acc=%.2f, coverage=%.2f",
             n_bt, report["share_mae"], report["directional_accuracy"],
             report["interval_coverage"])
    return report
