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
    """Full pipeline end to end, offline via the seeded sample dataset."""
    import run_pipeline

    result = run_pipeline.run(sample=True)
    assert result["row_counts"]["cleaned_reviews"] > 0
    assert result["row_counts"]["issue_prediction"] > 0
