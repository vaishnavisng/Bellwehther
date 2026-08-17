"""Base review source adapter.

Split so the network boundary (`_fetch_raw`) is isolated from mapping
(`_map_record`): standardization runs offline against sample/raw data, and the
scraper libraries are imported lazily by subclasses only when actually fetching.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ingestion.schema import to_standard_frame
from src.utils.config import load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


class BaseReviewSource:
    source_platform: str = ""  # set by subclass: "google_play" | "app_store"

    def __init__(self, app_id: str, app_name: str | None = None):
        if not app_id:
            raise ValueError("app_id is required")
        self.app_id = str(app_id)
        self.app_name = app_name

    # --- to implement in subclasses ---
    def _fetch_raw(self, count: int, **kwargs) -> list[dict]:
        """Hit the platform and return its native review dicts (network boundary)."""
        raise NotImplementedError

    def _map_record(self, raw: dict) -> dict:
        """Translate one native record into a standard-schema dict."""
        raise NotImplementedError

    # --- shared ---
    def standardize(self, raw_records: list[dict]) -> pd.DataFrame:
        rows = [self._map_record(r) for r in raw_records]
        return to_standard_frame(rows)

    def fetch_reviews(self, count: int = 200, save_raw: bool = False,
                      **kwargs) -> pd.DataFrame:
        log.info("Fetching up to %d reviews from %s (app_id=%s)",
                 count, self.source_platform, self.app_id)
        try:
            raw = self._fetch_raw(count=count, **kwargs)
        except Exception:
            log.exception("Fetch failed for %s app_id=%s",
                          self.source_platform, self.app_id)
            raise
        log.info("Fetched %d raw records from %s", len(raw), self.source_platform)
        if save_raw:
            self._save_raw(raw)
        return self.standardize(raw)

    def _save_raw(self, raw: list[dict]) -> Path:
        """Preserve original source data for debugging/reproducibility."""
        out_dir = load_config().path("data_raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{self.source_platform}_{self.app_id}_{stamp}.json"
        path.write_text(json.dumps(raw, default=str, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        log.info("Saved %d raw records -> %s", len(raw), path)
        return path
