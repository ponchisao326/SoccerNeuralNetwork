"""Internal DataFrame-to-dict flattening utilities.

Not part of the public adapter API. Import only from within the adapters package.
"""

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe_value(value: Any) -> Any:
    """Convert a pandas cell value to a BSON-safe Python scalar.

    Args:
        value: Raw value from a pandas DataFrame cell.

    Returns:
        A Python built-in (str, int, float, bool, None).
        ``pd.Timestamp`` becomes an ISO-8601 string.
        ``float('nan')`` and pandas NA become ``None``.
        Any remaining unknown type is coerced to ``str``.
    """
    if isinstance(value, (list, dict)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if not isinstance(value, (bool, int, float, str, type(None))):
        return str(value)
    return value


def flatten_dataframe(
    df: pd.DataFrame,
    source: str,
    league: str,
    season: str,
) -> list[dict[str, Any]]:
    """Flatten a soccerdata DataFrame to a list of BSON-safe Python dicts.

    Handles three structural patterns returned by soccerdata:

    1. MultiIndex on the row index only (e.g. ``read_schedule``).
    2. MultiIndex on columns only (e.g. some FBref stat tables).
    3. MultiIndex on both index and columns (e.g. ``read_team_season_stats``).

    Args:
        df: Raw DataFrame from a soccerdata reader method.
        source: Provenance tag injected into every record (e.g. ``"fbref"``).
        league: League identifier injected for traceability.
        season: Season identifier injected for traceability.

    Returns:
        List of flat dicts, one per row. Returns an empty list for empty input.
    """
    if df is None or df.empty:
        return []

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(filter(None, (str(s).strip() for s in col))).strip("_")
            for col in df.columns
        ]

    df = df.reset_index()

    records: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        record: dict[str, Any] = {k: _safe_value(v) for k, v in raw.items()}
        record["_source"] = source
        record["_league"] = league
        record["_season"] = season
        records.append(record)

    return records
