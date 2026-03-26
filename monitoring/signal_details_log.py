from pathlib import Path
import csv
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "signal_details_log.csv"
_LOG_LOCK = threading.Lock()


def _format_am_time_from_ms(ts_ms: Any) -> Optional[str]:
    try:
        ts_int = int(ts_ms)
    except (TypeError, ValueError):
        return None

    try:
        dt_utc = datetime.utcfromtimestamp(ts_int / 1000.0)
        dt_am = dt_utc + timedelta(hours=4)
        return dt_am.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


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
                    "vol_1h_pct",
                    "last_5m_trades",
                    "avg_5m_trades",
                    "activity_status",
                    "followup_ts_ms",
                    "followup_vol_1h_pct",
                    "timestamp_am",
                    "followup_ts_am",
                    "last_activity_check_ts_ms",
                    "last_activity_check_ts_am",
                    "last_activity_trades",
                    "last_activity_avg_trades",
                    "last_activity_status",
                ]
            )


def log_signal_details(
    symbol: str,
    direction: str,
    interval: str,
    entry: float,
    sl: Optional[float],
    tps: List[float],
    confidence: float,
    last_candle: Dict[str, Any],
    vol_1h: Optional[float],
    act_5m: Optional[Dict[str, Any]],
) -> None:
    _ensure_file()

    ts = last_candle.get("close_time") or last_candle.get("open_time")

    ts_am = _format_am_time_from_ms(ts)

    tp1 = tps[0] if len(tps) >= 1 else None
    tp2 = tps[1] if len(tps) >= 2 else None
    tp3 = tps[2] if len(tps) >= 3 else None

    last_trades = None
    avg_trades = None
    status = None

    if isinstance(act_5m, dict):
        v1 = act_5m.get("last_trades")
        v2 = act_5m.get("avg_trades")
        v3 = act_5m.get("status")
        if isinstance(v1, (int, float)):
            last_trades = int(v1)
        if isinstance(v2, (int, float)):
            avg_trades = float(v2)
        if isinstance(v3, str):
            status = v3

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
        vol_1h,
        last_trades,
        avg_trades,
        status,
        None,
        None,
        ts_am,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    with _LOG_LOCK:
        with _LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def update_signal_followup(
    symbol: str,
    direction: str,
    ts_ms: int,
    followup_ts_ms: int,
    followup_vol_1h: Optional[float],
) -> None:
    _ensure_file()

    ts_key = str(ts_ms)

    with _LOG_LOCK:
        try:
            with _LOG_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except FileNotFoundError:
            return

        if not rows:
            return

        header = rows[0]
        target_len = max(len(header), 23)

        updated = False
        for i in range(1, len(rows)):
            row = rows[i]
            if not row:
                continue

            row_ts = row[0] if len(row) > 0 else None
            row_sym = row[1] if len(row) > 1 else None
            row_dir = row[2] if len(row) > 2 else None

            if str(row_ts) != ts_key:
                continue
            if row_sym is not None and row_sym != symbol:
                continue
            if row_dir is not None and row_dir != direction:
                continue

            while len(row) < target_len:
                row.append("")

            row[14] = str(followup_ts_ms) if followup_ts_ms is not None else ""
            if followup_vol_1h is not None:
                try:
                    row[15] = f"{float(followup_vol_1h):.2f}"
                except (TypeError, ValueError):
                    row[15] = ""
            else:
                row[15] = ""

            am_followup = _format_am_time_from_ms(followup_ts_ms) if followup_ts_ms is not None else None
            if am_followup:
                if len(row) <= 17:
                    while len(row) <= 17:
                        row.append("")
                row[17] = am_followup

            rows[i] = row
            updated = True
            break

        if not updated:
            return

        with _LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def get_signals_since_minutes(minutes: float) -> List[Dict[str, Any]]:
    _ensure_file()

    try:
        mins = float(minutes)
    except (TypeError, ValueError):
        mins = 0.0
    if mins <= 0:
        return []

    now_ms = int(time.time() * 1000.0)
    cutoff_ms = now_ms - int(mins * 60_000.0)

    with _LOG_LOCK:
        try:
            with _LOG_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except FileNotFoundError:
            return []

    if not rows:
        return []

    header = rows[0]
    records: List[Dict[str, Any]] = []

    for row in rows[1:]:
        if not row:
            continue

        ts_val = row[0] if len(row) > 0 else None
        try:
            ts_ms = int(float(ts_val)) if ts_val not in (None, "") else None
        except (TypeError, ValueError):
            ts_ms = None

        if ts_ms is None:
            continue
        if ts_ms < cutoff_ms:
            continue

        rec: Dict[str, Any] = {}
        for idx, name in enumerate(header):
            if idx < len(row):
                rec[name] = row[idx]
            else:
                rec[name] = None

        records.append(rec)

    records.sort(key=lambda r: int(float(r.get("timestamp_ms") or 0)))
    return records


def update_signal_activity(
    symbol: str,
    direction: str,
    ts_ms: int,
    check_ts_ms: int,
    last_trades: Optional[Any],
    avg_trades: Optional[Any],
    status: Optional[str],
) -> None:
    _ensure_file()

    ts_key = str(ts_ms)

    with _LOG_LOCK:
        try:
            with _LOG_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except FileNotFoundError:
            return

        if not rows:
            return

        header = rows[0]
        target_len = max(len(header), 23)

        updated = False
        for i in range(1, len(rows)):
            row = rows[i]
            if not row:
                continue

            row_ts = row[0] if len(row) > 0 else None
            row_sym = row[1] if len(row) > 1 else None
            row_dir = row[2] if len(row) > 2 else None

            if str(row_ts) != ts_key:
                continue
            if row_sym is not None and row_sym != symbol:
                continue
            if row_dir is not None and row_dir != direction:
                continue

            while len(row) < target_len:
                row.append("")

            row[18] = str(check_ts_ms) if check_ts_ms is not None else ""

            am_check = _format_am_time_from_ms(check_ts_ms) if check_ts_ms is not None else None
            row[19] = am_check or ""

            try:
                if isinstance(last_trades, (int, float)):
                    row[20] = str(int(last_trades))
                else:
                    row[20] = ""
            except Exception:
                row[20] = ""

            try:
                if isinstance(avg_trades, (int, float)):
                    row[21] = f"{float(avg_trades):.2f}"
                else:
                    row[21] = ""
            except Exception:
                row[21] = ""

            row[22] = status or ""

            rows[i] = row
            updated = True
            break

        if not updated:
            return

        with _LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
