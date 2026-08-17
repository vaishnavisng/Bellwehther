"""Bellwether pipeline entrypoint.

Stage order: ingest -> preprocess -> store. NLP, analytics, prediction, and
dashboard plug in here in later layers.

    python run_pipeline.py            # ingest live sources from config, then store
"""
from src.ingestion.factory import fetch_all_sources
from src.preprocessing import build_cleaned_frame
from src.storage import (
    CLEANED_TABLE,
    RAW_TABLE,
    connect,
    create_tables,
    data_quality,
    insert_reviews,
)
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def run(std_df=None, count: int = 200):
    """Run preprocess + store. Ingests live sources unless std_df is supplied
    (tests/demos pass a standardized frame to stay offline)."""
    log.info("Bellwether pipeline starting")

    if std_df is None:
        std_df = fetch_all_sources(count=count, save_raw=True)
    log.info("Ingested %d standardized reviews", len(std_df))

    cleaned, report = build_cleaned_frame(std_df)
    log.info("Cleaning report: %s", report)

    con = connect()
    create_tables(con)
    insert_reviews(con, RAW_TABLE, std_df, replace=True)
    insert_reviews(con, CLEANED_TABLE, cleaned, replace=True)
    for table in (RAW_TABLE, CLEANED_TABLE):
        log.info("Quality[%s]: %s", table, data_quality(con, table))
    con.close()
    log.info("Pipeline finished")
    return report


if __name__ == "__main__":
    run()
