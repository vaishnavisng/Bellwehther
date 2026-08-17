"""Layer 2 demo: sample reviews -> clean -> DuckDB -> verify tables (offline).

    python scripts/demo_preprocess.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ingestion import AppStoreAdapter, GooglePlayAdapter
from src.ingestion.sample import load_sample_raw
from src.storage import CLEANED_TABLE, RAW_TABLE, connect, data_quality, read_table
import run_pipeline

pd.set_option("display.max_columns", None, "display.width", 220)


def main():
    g = GooglePlayAdapter("com.example.app", "Example App").standardize(
        load_sample_raw("google_play"))
    a = AppStoreAdapter("123456789", "Example App").standardize(
        load_sample_raw("app_store"))
    std = pd.concat([g, a], ignore_index=True)

    run_pipeline.run(std_df=std)  # clean + populate DuckDB

    con = connect()
    print("\n=== cleaned_reviews (selected columns) ===")
    df = read_table(con, CLEANED_TABLE)
    print(df[["review_id", "rating", "rating_bucket", "word_count",
              "review_month", "source_platform", "cleaned_text"]].to_string(index=False))

    print("\n=== Data quality ===")
    for table in (RAW_TABLE, CLEANED_TABLE):
        print(table, "->", data_quality(con, table))
    con.close()


if __name__ == "__main__":
    main()
