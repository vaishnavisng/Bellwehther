"""Load bundled sample raw data so ingestion/tests run without network calls."""
from __future__ import annotations

import json

from src.utils.config import load_config


def load_sample_raw(platform: str) -> list[dict]:
    """Return native-format sample records for 'google_play' or 'app_store'."""
    path = load_config().path("data_sample") / f"{platform}_raw.json"
    return json.loads(path.read_text(encoding="utf-8"))
