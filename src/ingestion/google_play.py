"""Google Play ingestion adapter (uses google-play-scraper)."""
from __future__ import annotations

from src.ingestion.base import BaseReviewSource
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


class GooglePlayAdapter(BaseReviewSource):
    source_platform = "google_play"

    def _fetch_raw(self, count: int, lang: str = "en", country: str = "us",
                   **kwargs) -> list[dict]:
        # Lazy import: keeps offline standardization/tests free of the dependency.
        from google_play_scraper import Sort, reviews

        collected: list[dict] = []
        token = None
        while len(collected) < count:
            batch, token = reviews(
                self.app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=min(200, count - len(collected)),  # library batches at 200
                continuation_token=token,
            )
            if not batch:
                break
            collected.extend(batch)
            if token is None:
                break
        return collected[:count]

    def _map_record(self, raw: dict) -> dict:
        return {
            "review_id": raw.get("reviewId"),
            "review_text": raw.get("content"),
            "rating": raw.get("score"),
            "review_date": raw.get("at"),
            "source_platform": self.source_platform,
            "app_id": self.app_id,
            "app_name": self.app_name,
            "app_version": raw.get("reviewCreatedVersion") or raw.get("appVersion"),
            "helpful_count": raw.get("thumbsUpCount"),
        }
