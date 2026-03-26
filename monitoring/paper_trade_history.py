import csv
import threading
import time
from pathlib import Path
from typing import Optional

_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "paper_trade_history"
_LOCK = threading.Lock()


def _ensure_file(account_index: int) -> Path:
    if not _BASE_DIR.exists():
        _BASE_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASE_DIR / f"account_{account_index}.csv"
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_ms",
                    "timestamp_iso",
                    "event",
                    "symbol",
                    "side",
                    "qty",
                    "entry_price",
                    "exit_price",
                    "pnl_pct",
                    "reason",
                ]
            )
    return path


def log_paper_entry(
    account_index: int,
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float] = None,
    reason: str = "ENTRY",
) -> None:
    try:
        path = _ensure_file(account_index)
    except Exception:
        return

    ts_ms = int(time.time() * 1000)
    ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_ms / 1000.0))

    try:
        qty_val = float(qty) if qty is not None else 0.0
    except (TypeError, ValueError):
        qty_val = 0.0

    try:
        entry_val: Optional[float] = float(entry_price) if entry_price is not None else None
    except (TypeError, ValueError):
        entry_val = None

    row = [
        ts_ms,
        ts_iso,
        "ENTRY",
        symbol,
        side,
        qty_val,
        entry_val if entry_val is not None else "",
        "",
        "",
        reason,
    ]

    with _LOCK:
        try:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            return


def log_paper_exit(
    account_index: int,
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float],
    exit_price: Optional[float],
    pnl_pct: Optional[float],
    reason: str,
) -> None:
    try:
        path = _ensure_file(account_index)
    except Exception:
        return

    ts_ms = int(time.time() * 1000)
    ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_ms / 1000.0))

    try:
        qty_val = float(qty) if qty is not None else 0.0
    except (TypeError, ValueError):
        qty_val = 0.0

    try:
        entry_val: Optional[float] = float(entry_price) if entry_price is not None else None
    except (TypeError, ValueError):
        entry_val = None

    try:
        exit_val: Optional[float] = float(exit_price) if exit_price is not None else None
    except (TypeError, ValueError):
        exit_val = None

    try:
        pnl_val: Optional[float] = float(pnl_pct) if pnl_pct is not None else None
    except (TypeError, ValueError):
        pnl_val = None

    row = [
        ts_ms,
        ts_iso,
        "EXIT",
        symbol,
        side,
        qty_val,
        entry_val if entry_val is not None else "",
        exit_val if exit_val is not None else "",
        pnl_val if pnl_val is not None else "",
        reason,
    ]

    with _LOCK:
        try:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            return
