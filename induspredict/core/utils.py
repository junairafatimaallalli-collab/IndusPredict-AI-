"""
IndusPredict AI - Core Utilities.
Provides standard timezone-safe UTC timestamps and formatting helpers.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return timezone-naive current UTC datetime compatible with SQLite."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
