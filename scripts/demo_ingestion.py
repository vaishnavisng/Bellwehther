"""Layer 1 demo: sample ingestion from both platforms -> one standardized dataset.

Runs offline against data/sample (no external API calls).
    python scripts/demo_ingestion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow direct `python scripts/...`

import pandas as pd

from src.ingestion import AppStoreAdapter, GooglePlayAdapter, validate_reviews
from src.ingestion.sample import load_sample_raw

pd.set_option("display.max_columns", None, "display.width", 200)


def main():
    gp = GooglePlayAdapter("com.example.app", "Example App")
    apple = AppStoreAdapter("123456789", "Example App")

    g = gp.standardize(load_sample_raw("google_play"))
    a = apple.standardize(load_sample_raw("app_store"))

    print("\n=== Google Play (standardized) ===")
    print(g[["review_id", "rating", "review_date", "source_platform", "app_version"]])

    print("\n=== Apple App Store (standardized) ===")
    print(a[["review_id", "rating", "review_date", "source_platform", "app_version"]])

    combined = pd.concat([g, a], ignore_index=True)
    clean, report = validate_reviews(combined)

    print("\n=== Combined standardized dataset ===")
    print(f"rows={len(combined)} platforms={sorted(combined['source_platform'].unique())} "
          f"app_ids={sorted(combined['app_id'].unique())}")
    print("validation:", report)

    print("\n=== Standardized schema (dtypes) ===")
    print(clean.dtypes)


if __name__ == "__main__":
    main()
