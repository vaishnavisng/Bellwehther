"""Layer 0 smoke test: config loads, paths resolve, logging works, pipeline runs."""
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger


def test_config_loads_and_paths_resolve():
    cfg = load_config()
    db = cfg.path("database")
    assert isinstance(db, Path)
    assert db.is_absolute()
    assert cfg["pipeline"]["rolling_window_days"] > 0


def test_expected_directories_exist():
    for d in ("data/raw", "data/processed", "data/sample", "src", "config"):
        assert (PROJECT_ROOT / d).is_dir()


def test_logger_works():
    get_logger("test").info("smoke")


def test_pipeline_runs():
    """Full preprocess+store pipeline, offline via sample standardized data."""
    import pandas as pd

    import run_pipeline
    from src.ingestion import GooglePlayAdapter
    from src.ingestion.sample import load_sample_raw

    std = GooglePlayAdapter("com.example.app", "Example App").standardize(
        load_sample_raw("google_play"))
    report = run_pipeline.run(std_df=std)
    assert report["cleaned_rows"] == 3
