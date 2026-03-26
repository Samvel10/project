from pathlib import Path
import csv
import threading
from typing import Any, Dict, List, Optional

_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "signal_log.csv"
_LOG_LOCK = threading.Lock()


def _ensure_file() -> None:
    if not _LOG_PATH.parent.exists():
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        with _LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_ms",
                    "symbol",
                    "direction",
                    "interval",
                    "entry",
                    "sl",
                    "tp1",
                    "tp2",
                    "tp3",
                    "confidence",
                    "rsi",
                    "momentum",
                    "acceleration",
                    "volatility",
                    "atr",
                    "structure",
                    "range_type",
                    "fib_direction",
                    "label",
                    "outcome_type",
                    "pnl_pct",
                    "hold_minutes",
                    "sl_hit",
                    "tp1_hit",
                    "tp2_hit",
                    "tp3_hit",
                    "sl_alerted",
                    "tp1_alerted",
                    "tp2_alerted",
                    "tp3_alerted",
                    "tp1_sent",
                    "tp2_sent",
                    "tp3_sent",
                ]
            )


def get_log_path() -> Path:
    return _LOG_PATH


def log_signal(
    symbol: str,
    direction: str,
    entry: float,
    sl: Optional[float],
    tps: List[float],
    confidence: float,
    features: Dict[str, Any],
    last_candle: Dict[str, Any],
    interval: str,
    category: str = "small",
) -> None:
    _ensure_file()

    ts = last_candle.get("close_time") or last_candle.get("open_time")
    rsi = features.get("rsi")
    momentum = features.get("momentum")
    acceleration = features.get("acceleration")
    volatility = features.get("volatility")
    atr = features.get("atr")

    structure = features.get("structure")
    range_info = features.get("range") or {}
    range_type = None
    if isinstance(range_info, dict):
        range_type = range_info.get("type")

    fib = features.get("fibonacci") or {}
    fib_direction = None
    if isinstance(fib, dict):
        fib_direction = fib.get("direction")

    tp1 = tps[0] if len(tps) >= 1 else None
    tp2 = tps[1] if len(tps) >= 2 else None
    tp3 = tps[2] if len(tps) >= 3 else None

    row = [
        ts,
        symbol,
        direction,
        interval,
        entry,
        sl,
        tp1,
        tp2,
        tp3,
        confidence,
        rsi,
        momentum,
        acceleration,
        volatility,
        atr,
        structure,
        range_type,
        fib_direction,
        str(category).lower(),
    ]

    with _LOG_LOCK:
        with _LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
