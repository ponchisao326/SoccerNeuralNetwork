"""Internal DataFrame-to-dict flattening utilities.

Not part of the public adapter API. Import only from within the adapters package.
"""

import json
import math
from pathlib import Path
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

    df = _flatten_columns(df.copy())

    records: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        record: dict[str, Any] = {k: _safe_value(v) for k, v in raw.items()}
        record["_source"] = source
        record["_league"] = league
        record["_season"] = season
        records.append(record)

    return records


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Join MultiIndex columns with ``_`` and move the row index into columns.

    Args:
        df: DataFrame to normalise. Mutated in place for the column rename;
            ``reset_index`` returns a new frame.

    Returns:
        DataFrame with flat string columns and the index promoted to columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(filter(None, (str(s).strip() for s in col))).strip("_")
            for col in df.columns
        ]
    return df.reset_index()


def spill_dataframe_to_jsonl(
    df: pd.DataFrame,
    path: str,
    source: str,
    league: str,
    season: str,
    chunk_size: int = 5_000,
) -> int:
    """Stream a soccerdata DataFrame to a line-delimited JSON file.

    Unlike :func:`flatten_dataframe`, this never materialises the full list of
    records in memory: rows are converted and written in ``chunk_size`` slices,
    so peak memory stays flat regardless of row count. Intended for very large
    batches (e.g. a WhoScored season) where the producing process must avoid
    both a large in-memory list and a large cross-process pickle.

    The caller owns ``df`` and is expected to discard it afterwards; the column
    rename mutates it in place to avoid the memory cost of a defensive copy.

    Args:
        df: Raw DataFrame from a soccerdata reader method.
        path: Destination file path. One JSON object is written per line.
        source: Provenance tag injected into every record.
        league: League identifier injected for traceability.
        season: Season identifier injected for traceability.
        chunk_size: Rows converted to dicts at once before being written.

    Returns:
        Number of records written. Writes an empty file (0 rows) for empty input.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        out.write_text("", encoding="utf-8")
        return 0

    df = _flatten_columns(df)

    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for start in range(0, len(df), chunk_size):
            sub = df.iloc[start:start + chunk_size]
            for raw in sub.to_dict(orient="records"):
                record: dict[str, Any] = {k: _safe_value(v) for k, v in raw.items()}
                record["_source"] = source
                record["_league"] = league
                record["_season"] = season
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
                total += 1

    return total
