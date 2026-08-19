"""Config-driven source selection. Keeps the rest of Bellwether platform-agnostic."""
from __future__ import annotations

import pandas as pd

from src.ingestion.app_store import AppStoreAdapter
from src.ingestion.base import BaseReviewSource
from src.ingestion.google_play import GooglePlayAdapter
from src.utils.config import load_config

ADAPTERS: dict[str, type[BaseReviewSource]] = {
    "google_play": GooglePlayAdapter,
    "app_store": AppStoreAdapter,
}


def get_adapter(source_platform: str, app_id: str,
                app_name: str | None = None, **kwargs) -> BaseReviewSource:
    try:
        cls = ADAPTERS[source_platform]
    except KeyError:
        raise ValueError(
            f"Unknown source_platform {source_platform!r}; "
            f"expected one of {sorted(ADAPTERS)}"
        )
    return cls(app_id=app_id, app_name=app_name, **kwargs)


def fetch_all_sources(count: int = 200, save_raw: bool = False,
                      sources: list[dict] | None = None) -> pd.DataFrame:
    """Fetch every configured source and stack into one standardized frame.

    `sources` overrides config['sources'] when given (e.g. from the dashboard's
    "Analyze an app" control). Supports one platform or the same product across
    several platforms; source_platform is preserved so layers can split/combine.
    """
    if sources is None:
        sources = load_config().get("sources", []) or []
    frames = []
    for s in sources:
        adapter = get_adapter(
            s["source_platform"], s["app_id"], s.get("app_name"),
            **{k: v for k, v in s.items()
               if k in ("country",)},  # only pass adapter-supported extras
        )
        frames.append(adapter.fetch_reviews(count=count, save_raw=save_raw))
    if not frames:
        from src.ingestion.schema import to_standard_frame
        return to_standard_frame([])
    return pd.concat(frames, ignore_index=True)
