from pathlib import Path
import csv
import threading
import time
from typing import Optional

_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "entry_timing_log.csv"
_LOCK = threading.Lock()


def _ensure_file() -> None:
    if not _LOG_PATH.parent.exists():
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        with _LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_ms",
                    "timestamp_iso",
                    "symbol",
                    "side",
                    "account_index",
                    "account_name",
                    "signal_ts_raw",
                    "signal_ts_ms",
                    "signal_ts_iso",
                    "start_ts_ms",
                    "start_ts_iso",
                    "delay_from_signal_ms",
                    "leverage_ms",
                    "order_ms",
                    "total_ms",
                    "order_attempts",
                    "result",
                    "error",
                ]
            )


def _iso_from_ms(ts_ms: Optional[float]) -> str:
    if ts_ms is None:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(ts_ms) / 1000.0))
    except Exception:
        return ""


def log_entry_timing(
    symbol: str,
    side: str,
    account_index: Optional[int],
    account_name: str,
    signal_ts_raw: Optional[float],
    start_ts_ms: float,
    leverage_ms: Optional[float],
    order_ms: Optional[float],
    total_ms: Optional[float],
    order_attempts: int,
    result: str,
    error: str = "",
) -> None:
    _ensure_file()

    ts_ms = int(time.time() * 1000)
    ts_iso = _iso_from_ms(ts_ms)

    signal_ts_ms = None
    if signal_ts_raw is not None:
        try:
            raw = float(signal_ts_raw)
            if raw > 1e12:
                signal_ts_ms = raw
            elif raw > 1e9:
                signal_ts_ms = raw * 1000.0
        except Exception:
            signal_ts_ms = None

    delay_ms = None
    if signal_ts_ms is not None:
        try:
            delay_ms = float(start_ts_ms) - float(signal_ts_ms)
        except Exception:
            delay_ms = None

    row = [
        ts_ms,
        ts_iso,
        symbol,
        side,
        int(account_index) if account_index is not None else "",
        account_name,
        signal_ts_raw if signal_ts_raw is not None else "",
        f"{float(signal_ts_ms):.0f}" if signal_ts_ms is not None else "",
        _iso_from_ms(signal_ts_ms) if signal_ts_ms is not None else "",
        f"{float(start_ts_ms):.0f}",
        _iso_from_ms(start_ts_ms),
        f"{float(delay_ms):.0f}" if delay_ms is not None else "",
        f"{float(leverage_ms):.2f}" if leverage_ms is not None else "",
        f"{float(order_ms):.2f}" if order_ms is not None else "",
        f"{float(total_ms):.2f}" if total_ms is not None else "",
        int(order_attempts),
        result,
        error,
    ]

    with _LOCK:
        with _LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
