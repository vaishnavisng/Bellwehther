# Bellwether — A Product-Feedback Early-Warning System

**A reusable product-feedback early-warning system using NLP, statistical analysis, and predictive analytics.**

Bellwether reads product reviews (Google Play, Apple App Store, or any source via an
adapter) and answers one question:

> **"What customer issue is becoming dangerous right now, and is it likely to hurt the product's rating in the near future?"**

It discovers issues from the reviews themselves (nothing hardcoded), tracks how each
issue is trending, flags the ones rising unusually fast, measures how strongly each is
*historically associated* with lower ratings, and produces a transparent, explainable
forward-looking risk estimate — surfaced in a product-analytics dashboard.

---

## Business problem

Product teams drown in reviews. By the time an app-store rating visibly drops, the
damage is already done and the root issue has been live for weeks. Teams need to know
**which emerging issue to act on first** — before it moves the rating — and they need a
reason they can defend to stakeholders, not a black-box score.

## Solution

An end-to-end analytical pipeline that turns raw reviews into a ranked, explained
early-warning list:

1. **Ingest** reviews through platform adapters into one standard schema.
2. **Clean & validate** text and store it in DuckDB.
3. **Discover issues** from the text with TF-IDF + KMeans (issues *emerge*; none are hardcoded).
4. **Track trends** — issue prevalence over time, week-over-week change, rolling baseline.
5. **Detect anomalies** — issues rising unusually fast (rolling mean/std z-score).
6. **Measure historical rating impact** — group comparison + regression (association, not causation).
7. **Predict** the near-term rating impact if the trend persists, with confidence intervals.
8. **Score risk** transparently (LOW → CRITICAL) with a human-readable explanation.
9. **Present** everything in a read-only Streamlit dashboard.

## Architecture

```
 Google Play Adapter ┐                      ┌ raw JSON preserved (data/raw)
                     ├─ Standardized Review ─┤
 App Store Adapter  ─┘        Schema         └ DuckDB: raw_reviews
                                   │
                     clean + validate + features
                                   │
                             DuckDB: cleaned_reviews
                                   │
              TF-IDF + KMeans  →  review_issues        (issue discovery)
                                   │
              trends + anomaly  →  issue_trends, issue_summary
                                   │
              rating impact     →  issue_impact         (group test + regression)
                                   │
              forward-looking   →  issue_prediction     (forecast + risk + explanation)
                                   │
                    Streamlit dashboard (READ-ONLY)
```

Only the ingestion adapters are platform-specific; everything after ingestion is
platform-independent. **All heavy computation happens in `run_pipeline.py`** and is
written to DuckDB — the dashboard never re-runs NLP or modeling, it only reads results.

## Tech stack

| Concern | Tools |
|---|---|
| Core | Python, Pandas, NumPy |
| Storage | DuckDB (SQL) |
| NLP | scikit-learn — TF-IDF, KMeans; basic text preprocessing |
| Statistics / prediction | SciPy, Statsmodels — OLS regression, Mann-Whitney U / Welch's t, rolling averages, z-score anomalies, linear trend extrapolation, confidence intervals |
| Visualization / dashboard | Plotly, Streamlit |
| Ingestion | google-play-scraper, app-store-scraper |
| Tests | pytest |

No deep learning, no transformers/embeddings, no orchestration or cloud infrastructure —
deliberately simple and interview-defensible.

---

## Methodology

### 1. Text preprocessing
Gentle, non-destructive normalization: Unicode NFKC, control-char removal, whitespace
collapse, capping runs of repeated punctuation, lowercasing. **Stopwords and words are
kept** so issue detection sees the real vocabulary. Reviews with no analyzable text
(empty, pure punctuation/emoji) are dropped. Original text is preserved alongside
`cleaned_text`.

### 2. Issue discovery — TF-IDF + KMeans
- **TF-IDF** (english stopwords, unigrams + bigrams, configurable `min_df`/`max_df`)
  turns reviews into sparse term-weight vectors — no training, fully inspectable.
- **KMeans** groups reviews into issues. **K is chosen by the best silhouette score** over
  a configurable range (not an arbitrary K).
- Each cluster is labeled by its **top TF-IDF terms** — so "issues" are grounded in the
  data, never a hardcoded taxonomy. A cluster is *not* automatically a business issue: we
  keep top terms, size, rating distribution, and representative reviews for the analyst to judge.

### 3. Trend detection
Per issue, per week: mention count, **issue share** (% of that week's reviews), average
rating, negative-review share, week-over-week change, and a rolling baseline (mean/std).

### 4. Anomaly detection
Transparent **rolling mean + rolling standard deviation + z-score**. The baseline is
*shifted* to exclude the current period (no leakage). `anomaly_flag = z ≥ threshold AND
enough mentions`. No anomaly-detection libraries — an analyst can reproduce it by hand.

### 5. Historical rating impact
For each issue we compare ratings of reviews mentioning it vs not:
- Mean/median, rating difference, low-rating share, sample sizes, and a **95% CI on the
  mean difference** (Welch/Satterthwaite).
- **The test is chosen from the data**, not applied blindly: default **Mann-Whitney U**
  (ratings are ordinal 1–5, non-normal); **Welch's t** only when both groups are large and
  Shapiro doesn't reject normality. The chosen test *and its reasoning* are stored.
- A simple **OLS regression** `rating ~ issue + review_length + time_period` (HC3 robust
  SE) estimates the *adjusted* rating penalty. This is **association, not causation** — the
  output always says "historically associated with approximately X-star lower ratings."

### 6. Forward-looking prediction
**Prediction target:** expected change in the product's overall average rating over the
next H periods attributable to one issue's changing prevalence. Because
`overall_avg = share·mean_issue + (1−share)·mean_non_issue`, this reduces to simple arithmetic:

```
predicted_rating_impact = (predicted_share_{t+H} − current_share) × historical_penalty
```

`predicted_share` comes from an **OLS linear extrapolation** of the issue's share series
with a 95% **prediction interval**; that interval is carried through to the impact CI. If
there are too few historical periods, **no forecast is invented** — the issue is marked
`insufficient_history`.

### 7. Risk scoring (transparent tiers)
Base tier from predicted harmful-impact magnitude (CRITICAL ≥ 0.30, HIGH ≥ 0.15,
MEDIUM ≥ 0.05 stars), then **downgraded** if the historical association isn't
reliable+significant and **upgraded** if the issue is actively anomalous and growing. Only
issues historically associated with *lower* ratings can be flagged — a benign issue merely
losing share is not a warning. Every warning carries a plain-English explanation citing the
actual numbers.

---

## Example output

From `python run_pipeline.py --sample` (288 synthetic reviews, 12 weeks, 2 platforms):

```
Discovered 4 issues (k=4 via silhouette)
Backtest: 28 windows, share_MAE=0.047, dir_acc=0.75, coverage=0.96

Issue: payment, payment failed, failed checkout
  Current share:        52.6%   (rising, +17% vs recent baseline)
  Historical impact:    -1.24 stars   (statistically significant)
  Low-rating presence:  83% vs 42% of other reviews
  Predicted impact:     -0.09 stars over the next 2 weeks   (CI [-0.40, -0.05])
  Risk: MEDIUM   Confidence: HIGH

Explanation:
  "Risk is MEDIUM because issue share is 52.6% and rising (+17% vs recent baseline);
   it is historically associated with a -1.24-star rating change (statistically
   significant); it is disproportionately present in low-rated reviews (83% vs 42%);
   the projected impact over the next 2 weeks is -0.09 stars."
```

## Dashboard overview

`streamlit run dashboard/app.py` — a read-only product-analytics view:

1. **Executive KPIs** — total reviews, average rating + trend, negative %, issues tracked, emerging count, high-risk count.
2. **Bellwether early warning** — the top emerging issue, prominent, fully explained.
3. **Emerging issues** — interactive, filterable, sortable table.
4. **Issue trends** — Plotly prevalence over time with rolling baseline and anomaly markers.
5. **Rating impact** — issue-present vs absent ratings and the regression penalty with 95% CIs.
6. **Explanation** — "Why is Bellwether warning us?" for a selected issue, in plain English.
7. **Review evidence** — representative reviews (no reviewer names / PII stored or shown).

---

## Setup & run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py --sample      # 1. compute analytical tables (offline, no network)
streamlit run dashboard/app.py       # 2. open the read-only dashboard

pytest                               # run the full test suite
```

Ingest live sources instead of the sample by editing `config/config.yaml` (`sources:`) and
running `python run_pipeline.py`. All tunable parameters (K range, rolling window, anomaly
threshold, risk weights, sample sizes, forecast horizon) live in `config/config.yaml`.

### Reproducibility
For a fixed input dataset and config, the pipeline is deterministic: KMeans uses a fixed
`nlp.random_state`; every other stage is plain pandas/statsmodels arithmetic. The seeded
`--sample` dataset reproduces byte-for-byte (`tests/test_pipeline.py`).

## Project structure

```
config/          config.yaml — paths + all analytical parameters
data/            raw/  processed/ (DuckDB)  sample/
src/
  ingestion/     platform adapters -> standardized schema, validation
  preprocessing/ cleaning + feature engineering
  nlp/           TF-IDF + KMeans issue extraction
  analytics/     trends, anomaly detection, risk summary
  prediction/    historical rating impact + forward-looking forecast
  storage/       DuckDB schema + read/write/quality helpers
  utils/         config + logging
dashboard/       Streamlit app (read-only) + data access layer
scripts/         per-layer offline demos
tests/           pytest suite (78 tests)
run_pipeline.py  end-to-end orchestration
```

---

## Statistical validation, assumptions & limitations

Bellwether is intentionally honest about what it can and cannot claim.

- **Association, not causation.** The rating-impact regression measures *association*
  between an issue and lower ratings, adjusting for review length and time period. It does
  **not** prove the issue *causes* rating decline — reviewers who hit an issue may differ in
  unobserved ways. All wording reflects this.
- **Minimum sample sizes.** Issues below `impact.min_sample` (default 30) are marked
  `reliable = False` and shown as indicative only; below `hard_floor` (5) no significance
  test is run at all. Clustering below `nlp.min_reviews_for_clustering` collapses to one
  honest cluster rather than inventing segments.
- **Test choice is data-driven.** Ratings are ordinal and non-normal, so the default is
  Mann-Whitney U; Welch's t is used only when normality checks and sample sizes justify it.
- **Confidence intervals everywhere.** Rating differences (Welch), regression effects (HC3),
  and share forecasts (OLS prediction interval) all report 95% CIs; these propagate into the
  predicted-impact CI.
- **Prediction uncertainty.** The forecast is a *linear extrapolation of recent share
  trend* — it assumes the recent trend persists and does not model seasonality, saturation,
  or external shocks. The horizon is capped by available history; with too few periods no
  prediction is made.
- **Backtesting.** A rolling-origin backtest re-forecasts share at each past cutoff and
  reports share MAE, directional accuracy, and prediction-interval coverage vs the target.
  On small datasets the report explicitly says metrics are indicative, not conclusive.
- **Data limits.** Results are only as representative as the reviews ingested; app-store
  reviews skew toward extremes, and KMeans requires choosing a granularity (silhouette-based
  here) that may split or merge conceptually related issues.
- **Issues = complaints.** Only reviews at/below `nlp.max_issue_rating` (default 3★) are
  clustered — praise is not an "issue." Each issue is then compared against *all* other
  reviews (including satisfied ones) for its rating impact.
- **History must span time for forecasting.** Trends and forecasts need reviews spread over
  several weeks. For very popular apps, even a few thousand *newest* reviews may cover only
  a few days, so no forecast is produced — the tool then reports a **current-burden** risk
  (share × historical penalty = the issue's drag on the rating right now) and says so. To
  build real trend history, run the pipeline periodically and let DuckDB accumulate, or
  analyze apps with a lower review volume.
- **Label quality / language.** Cluster labels are top TF-IDF terms; generic filler is
  removed via `nlp.extra_stopwords`, but noisy or non-English reviews can still yield fuzzy
  labels. A curated multilingual stopword list would improve this.

## Future improvements

- Richer forecasting (e.g. controlling for saturation, or a simple additive seasonal term) once more history is available.
- Per-platform issue comparison (Android vs iOS) — the schema already preserves `source_platform`.
- Human-in-the-loop issue labeling to refine auto-generated cluster labels.
- Scheduled incremental ingestion and pipeline runs.
- Larger real-world validation with longer review histories to make forecasting claims stronger.

---

## How I would explain this project in an interview

**The business problem.** Product teams find out about damaging issues only after the
app-store rating drops, by which point it's expensive to fix. Bellwether is an early-warning
system: it tells you which emerging complaint is most likely to hurt your rating soon, and
gives a defensible reason.

**The analytical approach.** It's a pipeline: ingest reviews → clean/store in DuckDB →
discover issues with TF-IDF + KMeans → track each issue's prevalence over time → flag
unusual growth with a z-score → measure each issue's historical link to ratings with a
statistical test and a regression → extrapolate the trend forward to estimate near-term
rating impact → score risk with a transparent rule and a written explanation → show it in a
dashboard.

**Why TF-IDF / KMeans (not deep learning).** The requirement is that issues *emerge from the
data* and that the method is explainable to a PM. TF-IDF + KMeans is transparent (I can show
the exact terms that define each cluster), needs no training data or GPUs, and is trivially
reproducible. Embeddings/BERTopic would be heavier and harder to defend for a data-analyst
portfolio — and unnecessary for this scale.

**Why the statistical analysis.** "This issue appears in low ratings" is a correlation. To
say something credible I compare issue vs non-issue ratings with a test chosen for ordinal
data (Mann-Whitney U by default, Welch's t only when justified), report a confidence
interval, and run a regression that adjusts for review length and time. That lets me quantify
the *historical* rating penalty per issue while being explicit that it's association, not
causation.

**How the prediction works.** The overall average rating is just a weighted blend of "issue
present" and "issue absent" ratings. So if I forecast how an issue's *share* will change
(simple OLS extrapolation with a prediction interval) and multiply by its historical penalty,
I get an expected rating change over a short horizon — all arithmetic I can reproduce by hand.

**How uncertainty is handled.** Confidence intervals at every step; small samples flagged
unreliable rather than reported precisely; no forecast when history is too short; and a
rolling-origin backtest that honestly reports error and coverage — including when the dataset
is too small for strong claims.

**Limitations.** It measures association, not causation; the forecast assumes the recent
trend persists; and results depend on review coverage. I'd strengthen the forecasting only
once there's enough history to justify it.

**Positioning.** It's not an "AI system" — it's a reusable product-feedback early-warning
system built with NLP, statistical analysis, and predictive analytics, designed to be
transparent and defensible end to end.
