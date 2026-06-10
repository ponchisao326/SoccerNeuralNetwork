"""Summarise multi-seed training runs per tag from ``metrics_history.jsonl``.

Run:
    uv run python scripts/summarize_metrics.py

Prints, for every ``--tag`` recorded by ``ml.train``, the across-seed mean and
standard deviation of the gate metrics (validation accuracy/log-loss first --
phase gates are decided on validation -- then the test figures for reference).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_HISTORY_PATH = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "metrics_history.jsonl"


def _mean_std(values: List[float]) -> str:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    runs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with _HISTORY_PATH.open(encoding="utf-8") as history:
        for line in history:
            record = json.loads(line)
            runs[str(record.get("tag", ""))].append(record)

    for tag, records in runs.items():
        seeds = [int(r["seed"]) for r in records]
        features = sorted({int(r["num_numeric_features"]) for r in records})
        print(f"tag={tag!r}  seeds={seeds}  numeric_features={features}")
        for label, getter in (
            ("val accuracy ", lambda r: float(r["val"]["accuracy"])),
            ("val log-loss ", lambda r: float(r["val"]["log_loss"])),
            ("val macro-F1 ", lambda r: float(r["val"]["macro_f1"])),
            ("test accuracy", lambda r: float(r["test_argmax"]["accuracy"])),
            ("test log-loss", lambda r: float(r["test_prob"]["log_loss"])),
            ("test rps     ", lambda r: float(r["test_prob"]["rps"])),
            ("test macro-F1", lambda r: float(r["test_argmax"]["macro_f1"])),
        ):
            print(f"  {label}: {_mean_std([getter(r) for r in records])}")
        print()


if __name__ == "__main__":
    main()
