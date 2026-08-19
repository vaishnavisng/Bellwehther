# Bellwether — A Product-Feedback Early-Warning System

**Bellwether answers one question:**
> *"What customer issue is becoming dangerous right now, and is it likely to hurt the product's rating in the near future?"*

It ingests reviews/feedback from any digital product (app, website, SaaS, e-commerce)
and surfaces **emerging issues before they drag the rating down**.

## Business problem

Product teams drown in reviews. By the time a rating drops, the damage is done.
Bellwether continuously watches review text, finds issues as they *emerge*, flags the
ones growing unusually fast, and estimates their likely impact on future ratings —
so teams can act early.

**Design principle:** issues are *not* hardcoded (no "payments", "crashes", "login"
lists). They emerge from the dataset itself, so the system is generic and reusable
across products via pluggable ingestion adapters.

## Pipeline

```
ingest → preprocess → nlp (issue extraction) → analytics (trend/anomaly)
       → prediction (rating impact) → storage → dashboard
```

1. **Ingest** reviews via source-specific adapters (Google Play, Apple App Store)
   normalized to one standard schema.
2. **Preprocess** — clean and normalize text.
3. **NLP** — extract recurring issues (TF-IDF + KMeans).
4. **Analytics** — track issue frequency over time, detect abnormally fast growth.
5. **Prediction** — measure each issue's historical link to ratings and estimate
   future rating impact, with risk + confidence levels.
6. **Dashboard** — PM / Data-Analyst-friendly view of emerging risks.

## Tech stack

- **Core:** Python, Pandas, NumPy
- **Storage:** SQL, DuckDB
- **NLP:** scikit-learn (TF-IDF, KMeans), basic text preprocessing
- **Stats / prediction:** SciPy, Statsmodels, linear regression, rolling averages,
  trend analysis, confidence intervals, basic anomaly detection
- **Visualization:** Plotly, Matplotlib
- **Dashboard:** Streamlit

## Project layout

```
config/          config.yaml (paths, pipeline settings)
data/            raw/  processed/  sample/
src/
  ingestion/     source adapters -> standard schema
  preprocessing/ text cleaning
  nlp/           issue extraction
  analytics/     trend + anomaly detection
  prediction/    rating-impact modelling
  storage/       DuckDB read/write
  utils/         config + logging
dashboard/       Streamlit app
scripts/         one-off helpers
tests/           pytest suite
run_pipeline.py  orchestrates stages end to end
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --sample     # offline end-to-end run (no network)
pytest
```

`python run_pipeline.py` ingests the live sources in `config/config.yaml`; add
`--sample` to run the whole chain on a seeded synthetic dataset with no network.
All expensive NLP/statistical work runs here and writes the analytical tables to
DuckDB — the dashboard only reads them.

### Reproducibility

For a fixed input dataset and `config/config.yaml`, the pipeline is
deterministic: KMeans uses a fixed `nlp.random_state`, and every other stage is
plain pandas/statsmodels arithmetic. The `--sample` dataset is seeded, so it
reproduces byte-for-byte (`test_pipeline.py::test_reproducible_same_input_same_output`).

> **Status:** Layer 7 (end-to-end pipeline) complete — `run_pipeline.py` chains
> ingest → validate → clean → DuckDB → issues → trends/anomaly → rating impact →
> forward-looking risk, with staged logging, timing, and per-stage functions.
> Seven analytical tables land in DuckDB; the dashboard reads them next.
>
> Try it: `python run_pipeline.py --sample` · demo scripts in `scripts/`
