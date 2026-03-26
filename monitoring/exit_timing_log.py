from pathlib import Path
import csv
import threading
import time
from typing import Optional

_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "exit_timing_log.csv"
_LOCK = threading.Lock()


def _ensure_file() -> None:
    if not _LOG_PATH.parent.exists():
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        with _LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp_ms",
                    "symbol",
                    "account_index",
                    "side",
                    "reason",
                    "trigger_price",
                    "hit_price",
                    "detected_ts_ms",
                    "submit_ts_ms",
                    "submit_delay_ms",
                    "timing_tag",
                ]
            )


def log_exit_timing(
    symbol: str,
    account_index: Optional[int],
    side: str,
    reason: str,
    trigger_price: Optional[float],
    hit_price: Optional[float],
    detected_ts_ms: Optional[float],
    submit_ts_ms: Optional[float],
    submit_delay_ms: Optional[float],
    timing_tag: str,
) -> None:
    _ensure_file()

    ts_ms = int(time.time() * 1000)

    row = [
        ts_ms,
        str(symbol),
        int(account_index) if account_index is not None else "",
        str(side),
        str(reason),
        f"{float(trigger_price):.12g}" if trigger_price is not None else "",
        f"{float(hit_price):.12g}" if hit_price is not None else "",
        f"{float(detected_ts_ms):.0f}" if detected_ts_ms is not None else "",
        f"{float(submit_ts_ms):.0f}" if submit_ts_ms is not None else "",
        f"{float(submit_delay_ms):.2f}" if submit_delay_ms is not None else "",
        str(timing_tag),
    ]

    with _LOCK:
        with _LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(row)
