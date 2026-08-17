"""Apple App Store ingestion adapter (uses app-store-scraper).

Apple exposes fewer fields than Google Play: no review id, app version, or
helpful count. The adapter synthesizes a stable review_id and leaves the
unavailable fields NULL — that's the adapter's whole job, translating Apple's
shape into the standard schema.
"""
from __future__ import annotations

from src.ingestion.base import BaseReviewSource
from src.ingestion.schema import synthesize_review_id
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


class AppStoreAdapter(BaseReviewSource):
    source_platform = "app_store"

    def __init__(self, app_id: str, app_name: str | None = None,
                 country: str = "us"):
        super().__init__(app_id, app_name)
        self.country = country
        # app-store-scraper needs a slug; app_name works, else a placeholder.
        self._app_slug = (app_name or "app").lower().replace(" ", "-")

    def _fetch_raw(self, count: int, country: str | None = None,
                   **kwargs) -> list[dict]:
        from app_store_scraper import AppStore

        scraper = AppStore(
            country=country or self.country,
            app_name=self._app_slug,
            app_id=self.app_id,
        )
        scraper.review(how_many=count)  # library handles pagination internally
        return list(scraper.reviews)

    def _map_record(self, raw: dict) -> dict:
        date = raw.get("date")
        return {
            "review_id": synthesize_review_id(
                self.source_platform, self.app_id,
                raw.get("userName"), date, raw.get("review"),
            ),
            "review_text": raw.get("review"),
            "rating": raw.get("rating"),
            "review_date": date,
            "source_platform": self.source_platform,
            "app_id": self.app_id,
            "app_name": self.app_name,
            "app_version": None,      # not provided by App Store reviews
            "helpful_count": None,    # not provided by App Store reviews
        }
