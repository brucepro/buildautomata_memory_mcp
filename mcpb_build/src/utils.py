"""
Utility functions for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

import sqlite3
import logging
from datetime import datetime
from typing import Union, Optional

logger = logging.getLogger("buildautomata-memory.utils")


def _adapt_datetime(dt: datetime) -> str:
    """Convert datetime to ISO format string for SQLite storage"""
    return dt.isoformat()


def _convert_timestamp(val: Union[str, bytes]) -> Optional[datetime]:
    """Convert timestamp string to datetime with error handling"""
    try:
        decoded = val.decode() if isinstance(val, bytes) else val
        return datetime.fromisoformat(decoded)
    except (ValueError, AttributeError) as e:
        # Log warning but don't crash - return None for malformed timestamps
        logger.warning(f"Failed to convert timestamp: {val}, error: {e}")
        return None


def register_sqlite_adapters():
    """Register SQLite adapters for datetime handling"""
    sqlite3.register_adapter(datetime, _adapt_datetime)
    sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
