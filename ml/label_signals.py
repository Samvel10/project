import sys
from pathlib import Path
import csv
import os

from ruamel.yaml import YAML

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.historical_loader import load_klines
from monitoring.signal_log import get_log_path
from monitoring.telegram import send_telegram, forward_telegram
from monitoring.subscribers import get_subscribers
from monitoring.signal_messages import get_signal_messages


def _format_price(price: float) -> str:
    """Pretty price formatting similar to main.format_price."""

    abs_p = abs(price)
    if abs_p >= 100:
        return f"{price:.2f}"
    if abs_p >= 1:
        return f"{price:.3f}"
    if abs_p >= 0.01:
        return f"{price:.4f}"
    return f"{price:.8f}"


def label_signals() -> None:
    log_path = get_log_path()
    if not log_path.exists():
        print("No signal log file found.")
        return False

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows or not fieldnames:
        print("Signal log is empty.")
        return False

    # Ensure new columns exist in header so that outcome details are persisted
    if "outcome_type" not in fieldnames:
        fieldnames.append("outcome_type")
    if "pnl_pct" not in fieldnames:
        fieldnames.append("pnl_pct")
    if "hold_minutes" not in fieldnames:
        fieldnames.append("hold_minutes")
    # Per-target hit flags so we can see, for each signal, which levels were
    # reached (TP1/TP2/TP3 and SL), independent of which one defined the
    # final outcome label.
    if "sl_hit" not in fieldnames:
        fieldnames.append("sl_hit")
    if "tp1_hit" not in fieldnames:
        fieldnames.append("tp1_hit")
    if "tp2_hit" not in fieldnames:
        fieldnames.append("tp2_hit")
    if "tp3_hit" not in fieldnames:
        fieldnames.append("tp3_hit")
    if "sl_minutes" not in fieldnames:
        fieldnames.append("sl_minutes")
    if "tp1_minutes" not in fieldnames:
        fieldnames.append("tp1_minutes")
    if "tp2_minutes" not in fieldnames:
        fieldnames.append("tp2_minutes")
    if "tp3_minutes" not in fieldnames:
        fieldnames.append("tp3_minutes")

    yaml = YAML()
    with open("config/trading.yaml") as f:
        trading_cfg = yaml.load(f)

    timeframe_cfg = trading_cfg.get("timeframe", {})
    interval = timeframe_cfg.get("base", "1m")

    notifications_cfg = trading_cfg.get("notifications", {})
    outcome_alerts_enabled = bool(notifications_cfg.get("outcome_alerts_enabled", False))

    telegram_token = os.environ.get("TELEGRAM_TOKEN") or trading_cfg.get("telegram_token")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID") or trading_cfg.get("telegram_chat_id")

    if outcome_alerts_enabled and telegram_token:
        try:
            subscribers = get_subscribers(telegram_chat_id, token=telegram_token)
        except Exception:
            subscribers = []
    else:
        subscribers = []

    future_limit = 50
    updated = False

    for row in rows:
        # Normalize rows that were previously marked as outcome_type=NONE
        # into "pending" state again, so they can be re-evaluated when more
        # future candles become available. Earlier versions would set
        # label=0, outcome_type=NONE even if the trade was still in progress.
        outcome_prev = (row.get("outcome_type") or "").strip()
        label_prev = (row.get("label") or "").strip()
        if outcome_prev == "NONE" and label_prev in ("0", ""):
            if label_prev != "":
                row["label"] = ""
                updated = True

        # Only work on rows that do not yet have a confirmed outcome label
        if row.get("label") not in (None, "", " "):
            continue

        symbol = row.get("symbol")
        direction = row.get("direction")
        ts_raw = row.get("timestamp_ms")
        entry_raw = row.get("entry")
        sl_raw = row.get("sl")
        tp1_raw = row.get("tp1")
        tp2_raw = row.get("tp2")
        tp3_raw = row.get("tp3")

        if not symbol or not direction or not ts_raw or not entry_raw or not sl_raw or not tp1_raw:
            continue

        try:
            ts = int(float(ts_raw))
            entry = float(entry_raw)
            sl = float(sl_raw)
            tp1 = float(tp1_raw)
            tp2 = float(tp2_raw) if tp2_raw not in (None, "", " ") else None
            tp3 = float(tp3_raw) if tp3_raw not in (None, "", " ") else None
        except ValueError:
            continue

        future = load_klines(symbol=symbol, interval=interval, limit=future_limit, start_time=ts + 1)
        if not future:
            continue

        idx_sl = None
        idx_tp1 = None
        idx_tp2 = None
        idx_tp3 = None

        for idx, c in enumerate(future):
            high = c["high"]
            low = c["low"]
            if direction == "BUY":
                if low <= sl and idx_sl is None:
                    idx_sl = idx
                if tp1 is not None and high >= tp1 and idx_tp1 is None:
                    idx_tp1 = idx
                if tp2 is not None and high >= tp2 and idx_tp2 is None:
                    idx_tp2 = idx
                if tp3 is not None and high >= tp3 and idx_tp3 is None:
                    idx_tp3 = idx
            else:
                if high >= sl and idx_sl is None:
                    idx_sl = idx
                if tp1 is not None and low <= tp1 and idx_tp1 is None:
                    idx_tp1 = idx
                if tp2 is not None and low <= tp2 and idx_tp2 is None:
                    idx_tp2 = idx
                if tp3 is not None and low <= tp3 and idx_tp3 is None:
                    idx_tp3 = idx

        indices = [
            (idx_sl, "SL"),
            (idx_tp1, "TP1"),
            (idx_tp2, "TP2"),
            (idx_tp3, "TP3"),
        ]
        valid = [(i, t) for (i, t) in indices if i is not None]

        # If no TP/SL has been hit yet in the available future candles,
        # leave this signal pending so it can be re-evaluated on later runs
        # when more data is available.
        if not valid:
            continue

        # Earliest event decides trade outcome. If SL and TP hit in the same candle,
        # the ordering in the list above gives SL priority (conservative).
        valid.sort(key=lambda it: it[0])
        winner_idx, outcome_type = valid[0]
        outcome = "0" if outcome_type == "SL" else "1"

        def _minutes_for_index(idx_local):
            if idx_local is None:
                return None
            try:
                c = future[idx_local]
            except (IndexError, TypeError):
                return None
            exit_ts_raw_local = c.get("close_time") or c.get("open_time")
            try:
                exit_ts_local = int(float(exit_ts_raw_local))
                return max(0.0, (exit_ts_local - ts) / 60000.0)
            except (TypeError, ValueError):
                return None

        sl_minutes = _minutes_for_index(idx_sl)
        tp1_minutes = _minutes_for_index(idx_tp1)
        tp2_minutes = _minutes_for_index(idx_tp2)
        tp3_minutes = _minutes_for_index(idx_tp3)

        hold_val = _minutes_for_index(winner_idx)
        hold_minutes = hold_val if hold_val is not None else 0.0

        # Compute percentage PnL relative to entry price
        pnl_pct = 0.0
        if outcome_type != "NONE":
            if outcome_type == "SL":
                exit_price = sl
            elif outcome_type == "TP1" and tp1 is not None:
                exit_price = tp1
            elif outcome_type == "TP2" and tp2 is not None:
                exit_price = tp2
            elif outcome_type == "TP3" and tp3 is not None:
                exit_price = tp3
            else:
                exit_price = tp1

            if entry != 0:
                if direction == "BUY":
                    pnl_pct = (exit_price - entry) / entry * 100.0
                else:
                    pnl_pct = (entry - exit_price) / entry * 100.0

        row["label"] = outcome
        row["outcome_type"] = outcome_type
        row["pnl_pct"] = f"{pnl_pct:.4f}"
        row["hold_minutes"] = f"{hold_minutes:.2f}"
        # Record, for analysis/monitoring, which levels were ever touched.
        row["sl_hit"] = "1" if idx_sl is not None else "0"
        row["tp1_hit"] = "1" if idx_tp1 is not None else "0"
        row["tp2_hit"] = "1" if idx_tp2 is not None else "0"
        row["tp3_hit"] = "1" if idx_tp3 is not None else "0"
        row["sl_minutes"] = f"{sl_minutes:.2f}" if sl_minutes is not None else ""
        row["tp1_minutes"] = f"{tp1_minutes:.2f}" if tp1_minutes is not None else ""
        row["tp2_minutes"] = f"{tp2_minutes:.2f}" if tp2_minutes is not None else ""
        row["tp3_minutes"] = f"{tp3_minutes:.2f}" if tp3_minutes is not None else ""
        updated = True

        if outcome_alerts_enabled and subscribers and outcome_type != "NONE":
            try:
                base_symbol = (
                    symbol.replace("USDT", "/USDT") if symbol.endswith("USDT") else symbol
                )
                direction_text = "Long🟢" if direction == "BUY" else "Short🔴"

                result_map = {
                    "TP1": "Target 1 hit ✅",
                    "TP2": "Target 2 hit ✅",
                    "TP3": "Target 3 hit ✅",
                    "SL": "Stop Loss hit ❌",
                }
                result_line = result_map.get(outcome_type, "Outcome updated")

                lines = [
                    f"#{base_symbol} - {direction_text}",
                    "",
                    f"Result: {result_line}",
                    "",
                    f"Entry: {_format_price(entry)}",
                ]

                if sl is not None:
                    lines.append(f"Stop Loss: {_format_price(sl)}")
                if tp1 is not None:
                    lines.append(f"Target 1: {_format_price(tp1)}")
                if tp2 is not None:
                    lines.append(f"Target 2: {_format_price(tp2)}")
                if tp3 is not None:
                    lines.append(f"Target 3: {_format_price(tp3)}")

                # Add holding duration and PnL information for easier monitoring
                sign = "+" if pnl_pct > 0 else ""
                lines.append("")
                if hold_minutes > 0:
                    lines.append(f"Duration: {hold_minutes:.2f} min")
                lines.append(f"PnL: {sign}{pnl_pct:.2f}%")

                msg = "\n".join(lines)

                # Try to forward original signal message first, then send summary
                try:
                    msg_ids = get_signal_messages(symbol, ts)
                except Exception:
                    msg_ids = {}

                for chat_id in subscribers:
                    try:
                        original_msg_id = msg_ids.get(chat_id)
                        if original_msg_id is not None:
                            try:
                                forward_telegram(telegram_token, chat_id, chat_id, original_msg_id)
                            except Exception:
                                pass

                        send_telegram(msg, telegram_token, chat_id)
                    except Exception:
                        continue
            except Exception:
                # Outcome alerts must never break labeling
                pass

    if not updated:
        print("No unlabeled signals to update.")
        return False

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Signal log updated with outcome labels.")
    return True


def main() -> None:
    label_signals()


if __name__ == "__main__":
    main()
