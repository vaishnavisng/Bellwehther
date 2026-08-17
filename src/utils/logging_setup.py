"""Minimal logging setup. Call get_logger(__name__) anywhere."""
import logging

from src.utils.config import load_config

_CONFIGURED = False


def _configure():
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = load_config().get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
