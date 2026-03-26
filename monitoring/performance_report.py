import sys
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datetime as dt


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.signal_log import get_log_path


@dataclass
class Stats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_sum: float = 0.0
    hold_sum_min: float = 0.0

    def add(self, label: Optional[str], pnl_pct: Optional[float], hold_min: Optional[float]) -> None:
        self.trades += 1
        if label == "1":
            self.wins += 1
        elif label == "0":
            self.losses += 1

        if pnl_pct is not None:
            self.pnl_sum += pnl_pct
        if hold_min is not None:
            self.hold_sum_min += hold_min

    def summarize(self) -> Dict[str, float]:
        if self.trades <= 0:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "winrate": 0.0,
                "avg_pnl": 0.0,
                "avg_hold_min": 0.0,
            }

        winrate = (self.wins / self.trades) * 100.0 if self.trades > 0 else 0.0
        avg_pnl = self.pnl_sum / self.trades if self.trades > 0 else 0.0
        avg_hold = self.hold_sum_min / self.trades if self.trades > 0 else 0.0
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "winrate": winrate,
            "avg_pnl": avg_pnl,
            "avg_hold_min": avg_hold,
        }


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_timestamp_ms(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _load_rows() -> List[Dict[str, str]]:
    log_path = get_log_path()
    if not log_path.exists():
        print("[REPORT] No signal_log.csv found.")
        return []

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_time_buckets(rows: List[Dict[str, str]]) -> Tuple[Stats, Dict[str, Stats], Dict[str, Stats], Dict[str, Stats]]:
    """Return overall, daily, weekly, monthly stats."""

    overall = Stats()
    daily: Dict[str, Stats] = defaultdict(Stats)
    weekly: Dict[str, Stats] = defaultdict(Stats)
    monthly: Dict[str, Stats] = defaultdict(Stats)

    for row in rows:
        label = (row.get("label") or "").strip()
        outcome_type = (row.get("outcome_type") or "").strip()

        # Only consider trades where we have a concrete outcome (TP/SL),
        # skip rows with no outcome yet.
        if outcome_type in ("", "NONE"):
            continue

        pnl_pct = _parse_float(row.get("pnl_pct"))
        hold_min = _parse_float(row.get("hold_minutes"))

        ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
        if ts_ms is None:
            # Still aggregate into overall, but not into time buckets
            overall.add(label, pnl_pct, hold_min)
            continue

        # Treat timestamps as UTC for reporting
        ts = dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
        day_key = ts.date().isoformat()  # YYYY-MM-DD
        iso_year, iso_week, _ = ts.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        month_key = f"{ts.year}-{ts.month:02d}"

        overall.add(label, pnl_pct, hold_min)
        daily[day_key].add(label, pnl_pct, hold_min)
        weekly[week_key].add(label, pnl_pct, hold_min)
        monthly[month_key].add(label, pnl_pct, hold_min)

    return overall, daily, weekly, monthly


def _print_stats_section(title: str, stats: Stats) -> None:
    s = stats.summarize()
    print(f"\n=== {title} ===")
    print(f"Trades:   {s['trades']}")
    print(f"Wins:     {s['wins']}")
    print(f"Losses:   {s['losses']}")
    print(f"Winrate:  {s['winrate']:.2f}%")
    print(f"Avg PnL:  {s['avg_pnl']:.4f}%")
    print(f"Avg Hold: {s['avg_hold_min']:.2f} min")


def _print_grouped_section(title: str, grouped: Dict[str, Stats], limit: Optional[int] = None) -> None:
    print(f"\n=== {title} ===")
    if not grouped:
        print("(no trades)")
        return

    # Sort keys chronologically (string keys are ISO-like so lexical sort works)
    keys = sorted(grouped.keys())
    if limit is not None and len(keys) > limit:
        keys = keys[-limit:]

    for key in keys:
        s = grouped[key].summarize()
        print(f"{key} -> trades={s['trades']}, winrate={s['winrate']:.2f}%, avg_pnl={s['avg_pnl']:.4f}%, avg_hold={s['avg_hold_min']:.2f} min")


def build_performance_summary(
    daily_limit: int = 30,
    weekly_limit: int = 26,
    monthly_limit: int = 12,
) -> str:
    """Build a compact text summary of performance for use in Telegram.

    This reuses the same logic as the CLI report but returns a single
    string instead of printing to stdout.
    """

    rows = _load_rows()
    if not rows:
        return "No performance data available yet."

    overall, daily, weekly, monthly = _build_time_buckets(rows)

    lines = []

    # Overall section
    o = overall.summarize()
    lines.append("Performance summary (from signal_log.csv)")
    lines.append("")
    lines.append("OVERALL:")
    lines.append(f"Trades: {o['trades']}, Wins: {o['wins']}, Losses: {o['losses']}")
    lines.append(f"Winrate: {o['winrate']:.2f}%, Avg PnL: {o['avg_pnl']:.4f}%, Avg hold: {o['avg_hold_min']:.2f} min")

    # Helper to format grouped stats
    def _append_group(title: str, grouped: Dict[str, Stats], limit: int) -> None:
        if not grouped:
            return
        keys = sorted(grouped.keys())
        if limit is not None and len(keys) > limit:
            keys_subset = keys[-limit:]
        else:
            keys_subset = keys

        lines.append("")
        lines.append(title)
        for key in keys_subset:
            s = grouped[key].summarize()
            lines.append(
                f"{key}: trades={s['trades']}, winrate={s['winrate']:.2f}%, avg_pnl={s['avg_pnl']:.4f}%, avg_hold={s['avg_hold_min']:.2f} min"
            )

    _append_group("BY DAY (recent)", daily, daily_limit)
    _append_group("BY WEEK (recent)", weekly, weekly_limit)
    _append_group("BY MONTH (recent)", monthly, monthly_limit)

    return "\n".join(lines)


def build_performance_summary_since(
    hours: float,
    daily_limit: int = 30,
    weekly_limit: int = 26,
    monthly_limit: int = 12,
    max_detailed_trades: Optional[int] = None,
) -> str:
    rows = _load_rows()
    if not rows:
        return "No performance data available yet."

    now = dt.datetime.utcnow()
    cutoff = now - dt.timedelta(hours=hours)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
        if ts_ms is None or ts_ms < cutoff_ms:
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        return f"No performance data in the last {hours:.1f} hours."

    # 1) Stats for the selected time window (OVERALL + detailed trades)
    overall_window, _daily_win, _weekly_win, _monthly_win = _build_time_buckets(filtered_rows)

    # 2) Stats over full history for BY DAY / BY WEEK / BY MONTH sections,
    # so that these reflect true daily/weekly/monthly performance instead of
    # being limited only to the last `hours`.
    _overall_all, daily_all, weekly_all, monthly_all = _build_time_buckets(rows)

    lines: List[str] = []

    o = overall_window.summarize()
    window_start = cutoff.strftime("%Y-%m-%d %H:%M")
    window_end = now.strftime("%Y-%m-%d %H:%M")
    lines.append(f"Performance (last {hours:.1f}h)")
    lines.append(f"Window: {window_start} -> {window_end} (UTC)")
    lines.append("")
    lines.append("OVERALL:")
    lines.append(f"Trades: {o['trades']}, Wins: {o['wins']}, Losses: {o['losses']}")
    lines.append(
        f"Winrate: {o['winrate']:.2f}%, Avg PnL: {o['avg_pnl']:.4f}%, Avg hold: {o['avg_hold_min']:.2f} min"
    )

    def _append_group(title: str, grouped: Dict[str, Stats], limit: int) -> None:
        if not grouped:
            return
        keys = sorted(grouped.keys())
        if limit is not None and len(keys) > limit:
            keys_subset = keys[-limit:]
        else:
            keys_subset = keys

        lines.append("")
        lines.append(title)
        for key in keys_subset:
            s = grouped[key].summarize()
            lines.append(
                f"{key}: trades={s['trades']}, winrate={s['winrate']:.2f}%, avg_pnl={s['avg_pnl']:.4f}%, avg_hold={s['avg_hold_min']:.2f} min"
            )

    _append_group("BY DAY (recent)", daily_all, daily_limit)
    _append_group("BY WEEK (recent)", weekly_all, weekly_limit)
    _append_group("BY MONTH (recent)", monthly_all, monthly_limit)

    # Detailed per-trade section: list each completed trade in the window
    # with its outcome, holding time and percentage PnL so that Telegram
    # /stats output shows not only aggregates but also individual deals.
    detailed_rows: List[Dict[str, str]] = []
    for row in filtered_rows:
        outcome_type = (row.get("outcome_type") or "").strip()
        if outcome_type in ("", "NONE"):
            continue
        detailed_rows.append(row)

    if detailed_rows:
        detailed_rows.sort(
            key=lambda r: _parse_timestamp_ms(r.get("timestamp_ms")) or 0
        )

        # Optionally limit how many detailed trades we list, while still
        # respecting the time window filtering above.
        if max_detailed_trades is not None and max_detailed_trades > 0:
            if len(detailed_rows) > max_detailed_trades:
                detailed_rows = detailed_rows[-max_detailed_trades:]

        lines.append("")
        lines.append(f"TRADES (detailed, last {hours:.1f}h):")
        lines.append(
            f"Showing {len(detailed_rows)} of {o['trades']} completed trades in this window."
        )

        for row in detailed_rows:
            symbol = row.get("symbol") or "?"
            direction = row.get("direction") or "?"
            outcome_type = (row.get("outcome_type") or "").strip()

            pnl = _parse_float(row.get("pnl_pct"))
            if pnl is None:
                pnl = 0.0
            hold_min = _parse_float(row.get("hold_minutes"))
            if hold_min is None:
                hold_min = 0.0

            ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
            if ts_ms is not None:
                ts = dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
                ts_str = ts.strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = "n/a"

            entry = _parse_float(row.get("entry"))
            sl_price = _parse_float(row.get("sl"))
            tp1_price = _parse_float(row.get("tp1"))
            tp2_price = _parse_float(row.get("tp2"))
            tp3_price = _parse_float(row.get("tp3"))

            sl_min = _parse_float(row.get("sl_minutes"))
            tp1_min = _parse_float(row.get("tp1_minutes"))
            tp2_min = _parse_float(row.get("tp2_minutes"))
            tp3_min = _parse_float(row.get("tp3_minutes"))

            timeline_events = []

            def _add_event(minutes_val, label, price_level):
                if minutes_val is None or entry is None or price_level is None:
                    return
                try:
                    if direction == "BUY":
                        pnl_event = (price_level - entry) / entry * 100.0
                    else:
                        pnl_event = (entry - price_level) / entry * 100.0
                except Exception:
                    pnl_event = 0.0
                timeline_events.append((minutes_val, label, pnl_event))

            _add_event(tp1_min, "TP1", tp1_price)
            _add_event(tp2_min, "TP2", tp2_price)
            _add_event(tp3_min, "TP3", tp3_price)
            _add_event(sl_min, "SL", sl_price)

            if timeline_events:
                timeline_events.sort(key=lambda e: e[0])
                parts = []
                for mins_ev, name_ev, pnl_ev in timeline_events:
                    sign_ev = "+" if pnl_ev > 0 else ""
                    parts.append(f"{name_ev} @ {mins_ev:.2f}m ({sign_ev}{pnl_ev:.2f}%)")
                timeline_str = " | ".join(parts)
            else:
                timeline_str = ""

            sign = "+" if pnl > 0 else ""
            base_line = (
                f"{ts_str} | {symbol} {direction} -> {outcome_type}, PnL: {sign}{pnl:.2f}%, Duration: {hold_min:.2f} min"
            )
            if timeline_str:
                base_line = base_line + f" | Timeline: {timeline_str}"
            lines.append(base_line)

    return "\n".join(lines)


def main() -> None:
    """Print overall, daily, weekly and monthly performance based on signal_log.csv.

    Usage:
        python -m monitoring.performance_report
    or
        python monitoring/performance_report.py
    """

    rows = _load_rows()
    if not rows:
        return

    overall, daily, weekly, monthly = _build_time_buckets(rows)

    print("[REPORT] Performance summary from signal_log.csv")

    _print_stats_section("OVERALL", overall)

    # By default, limit to last N periods to keep output readable.
    # You can change these constants if you want *all* history printed.
    DAILY_LIMIT = 60
    WEEKLY_LIMIT = 52
    MONTHLY_LIMIT = 24

    _print_grouped_section("BY DAY (last ~60 days)", daily, limit=DAILY_LIMIT)
    _print_grouped_section("BY WEEK (last ~52 weeks)", weekly, limit=WEEKLY_LIMIT)
    _print_grouped_section("BY MONTH (last ~24 months)", monthly, limit=MONTHLY_LIMIT)


if __name__ == "__main__":
    main()
