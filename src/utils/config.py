"""Config + path handling. One loader, cached, resolves paths against project root."""
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def path(self, *keys) -> Path:
        """Resolve a path from config['paths'] against the project root."""
        value = self._data["paths"][keys[0] if keys else "data_raw"]
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
