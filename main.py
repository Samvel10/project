import os
import sys
import time
import threading
import subprocess
import signal
import csv
import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLIENTS_DIR = os.path.join(_ROOT_DIR, "clients")
try:
    if os.path.isdir(_CLIENTS_DIR):
        for _client_id in os.listdir(_CLIENTS_DIR):
            _libs = os.path.join(_CLIENTS_DIR, _client_id, "libs")
            if os.path.isdir(_libs) and _libs not in sys.path:
                sys.path.insert(0, _libs)
except Exception:
    pass

from instance_security import register_startup, INSTANCE_ID
from data.live_stream import fetch_candles
from data.symbols import get_all_usdt_futures
from data.ws_price_stream import start_price_stream, get_last_price
from data.delisting_filter import is_delisting_symbol
from data.symbol_blocklist import is_symbol_blocked, get_mqa_info as _get_mqa_info
from data.historical_loader import load_klines
from signals.ensemble import generate_signal
from execution.binance_futures import (
    adjust_price,
    adjust_quantity,
    place_order,
    place_algo_order,
    set_leverage,
    set_leverage_for_account,
    get_open_positions_snapshot,
    close_position_market,
    get_public_mark_price,
    get_public_mark_prices,
    update_fibo_after_exit,
    startup_cleanup_accounts,
    room_sizing_pop_assignment,
    room_sizing_sync_all_accounts,
    compute_distance_based_sl_tp,
    get_symbol_lot_constraints,
    place_exchange_stop_order,
    cancel_order_by_id,
    cancel_symbol_open_orders,
    get_account_wallet_balance,
    record_symbol_loss,
    _get_clients,
)
from execution.paper_trading import open_paper_trades, check_paper_exits
from risk.live_risk import allow_trade
from monitoring.logger import log
from monitoring.telegram import send_telegram
from monitoring.subscribers import update_subscribers, get_subscribers, remove_subscriber
from monitoring.signal_log import log_signal
from monitoring.signal_details_log import (
    log_signal_details,
    update_signal_followup,
    update_signal_activity,
)
from monitoring.signal_messages import record_signal_message
from monitoring.stats_bot import run_stats_bot
from monitoring.accounts_bot import run_accounts_bot
from monitoring.account_report_bot import run_account_report_bot
from monitoring.trade_history import log_trade_exit
from monitoring.entry_timing_log import log_entry_timing
from monitoring.exit_timing_log import log_exit_timing
from ml.inference import InferenceEngine
from ml.train_sl_tp_nn import run_training as run_sl_tp_nn_training
from features.feature_store import build_features
from execution.sl_tp import compute_sl_tp
from execution.sl_tp_nn import suggest_sl_tp, on_trade_exit
from execution.ai_trade_manager import run_ai_trade_manager
from config import settings as app_settings
from config.proxies import get_random_proxy, get_working_proxies

try:
    from monitoring.news_guard import (
        is_news_window_active,
        get_active_news_events,
        get_us_high_impact_events_for_today_window,
    )
except Exception:
    def is_news_window_active() -> bool:
        return False

    def get_active_news_events():
        return []

    def get_us_high_impact_events_for_today_window():  # type: ignore[return-type]
        return []


# Single authority switch:
# when enabled, non-AITM modules stay in analysis-only mode for open-trade lifecycle.
AITM_MASTER_ENABLED = bool(getattr(app_settings, "AITM_MASTER_ENABLED", False))

# Preserve raw exchange mutators; wrap local calls in this module with AITM guard.
_RAW_PLACE_EXCHANGE_STOP_ORDER = place_exchange_stop_order
_RAW_CANCEL_ORDER_BY_ID = cancel_order_by_id
_RAW_CANCEL_SYMBOL_OPEN_ORDERS = cancel_symbol_open_orders
_RAW_CLOSE_POSITION_MARKET = close_position_market


def _skip_non_aitm_trade_action(action: str, details: str = "") -> None:
    if not AITM_MASTER_ENABLED:
        return
    suffix = f" | {details}" if details else ""
    log(f"[SKIP] {action} blocked (AITM master active){suffix}")


def place_exchange_stop_order(*args, **kwargs):  # type: ignore[override]
    if AITM_MASTER_ENABLED:
        _skip_non_aitm_trade_action("SL/TP modification")
        return None
    return _RAW_PLACE_EXCHANGE_STOP_ORDER(*args, **kwargs)


def cancel_order_by_id(*args, **kwargs):  # type: ignore[override]
    if AITM_MASTER_ENABLED:
        _skip_non_aitm_trade_action("Order cancellation")
        return None
    return _RAW_CANCEL_ORDER_BY_ID(*args, **kwargs)


def cancel_symbol_open_orders(*args, **kwargs):  # type: ignore[override]
    if AITM_MASTER_ENABLED:
        _skip_non_aitm_trade_action("Open-order cleanup")
        return None
    return _RAW_CANCEL_SYMBOL_OPEN_ORDERS(*args, **kwargs)


def close_position_market(*args, **kwargs):  # type: ignore[override]
    if AITM_MASTER_ENABLED:
        _skip_non_aitm_trade_action("Exit manager prevented from closing trade")
        return False
    return _RAW_CLOSE_POSITION_MARKET(*args, **kwargs)

# Configuration (կարդացվում է config/trading.yaml)
try:
    from ruamel.yaml import YAML  # type: ignore
except Exception:
    YAML = None
import yaml as _pyyaml

if YAML is not None:
    _yaml = YAML(typ="safe")
    with open("config/trading.yaml") as f:
        config = _yaml.load(f)
else:
    with open("config/trading.yaml") as f:
        config = _pyyaml.safe_load(f)

trading_cfg = config.get("trading", {})
TRADING_ENABLED = bool(trading_cfg.get("enabled", True))

capital_cfg = config.get("capital", {})
CAPITAL = capital_cfg.get("initial", 10000)

timeframe_cfg = config.get("timeframe", {})
INTERVAL = timeframe_cfg.get("base", "1m")

leverage_cfg = config.get("leverage", {})
DEFAULT_LEVERAGE = leverage_cfg.get("default", 10)

execution_cfg = config.get("execution", {})
try:
    TP_MODE = int(execution_cfg.get("tp_mode", 1))
except (TypeError, ValueError):
    TP_MODE = 1
if TP_MODE not in (1, 2, 3, 4):
    TP_MODE = 1
AUTO_SL_TP_ENABLED = bool(execution_cfg.get("auto_sl_tp_enabled", True))
# When true, PASSIVE market state from SLTP-AI can block entries.
# Default is false (advisory-only mode): PASSIVE is reported in signals
# but does not stop trading.
SLTP_AI_ENFORCE_TRADE_BLOCK = bool(execution_cfg.get("sltp_ai_trade_block_enabled", False))
DYNAMIC_ENTRY_ENABLED = bool(execution_cfg.get("dynamic_entry_enabled", False))

position_timeout_cfg = config.get("position_timeout") or {}
POSITION_TIMEOUT_ENABLED = bool(position_timeout_cfg.get("enabled", False))
try:
    POSITION_TIMEOUT_MAX_HOURS = float(position_timeout_cfg.get("max_hours", 0.0))
except (TypeError, ValueError):
    POSITION_TIMEOUT_MAX_HOURS = 0.0
if POSITION_TIMEOUT_MAX_HOURS < 0:
    POSITION_TIMEOUT_MAX_HOURS = 0.0

EXIT_MANAGER_INTERVAL_SECONDS = 1.0
ENTRY_MANAGER_INTERVAL_SECONDS = 1.0

# Internal state for manual multi-TP exits per account & symbol.
# Key: (account_index, symbol) -> {
#   "initial_amt": float,
#   "long": bool,
#   "tp_prices": [float, ...],
#   "sl_price": float,
#   "num_legs": int,
# }
_EXIT_POS_STATE = {}

_EXIT_LAST_NONEMPTY_POS_TS = 0.0

# Persistent memory for adaptive "added size" handling across restarts.
# Stores per (account, symbol, side): core/base qty + base entry reference so
# recovery TP for added qty remains deterministic after process restart.
_ADAPTIVE_ADD_MEM_PATH = os.path.join(_ROOT_DIR, "data", "adaptive_add_memory.json")
_ADAPTIVE_ADD_MEM_LOCK = threading.Lock()
_ADAPTIVE_ADD_MEM_CACHE: dict = {}
_ADAPTIVE_ADD_MEM_LOADED = False


def _adaptive_add_mem_load() -> dict:
    global _ADAPTIVE_ADD_MEM_LOADED, _ADAPTIVE_ADD_MEM_CACHE
    if _ADAPTIVE_ADD_MEM_LOADED:
        return _ADAPTIVE_ADD_MEM_CACHE
    with _ADAPTIVE_ADD_MEM_LOCK:
        if _ADAPTIVE_ADD_MEM_LOADED:
            return _ADAPTIVE_ADD_MEM_CACHE
        try:
            if os.path.exists(_ADAPTIVE_ADD_MEM_PATH):
                with open(_ADAPTIVE_ADD_MEM_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    _ADAPTIVE_ADD_MEM_CACHE = raw
                else:
                    _ADAPTIVE_ADD_MEM_CACHE = {}
            else:
                _ADAPTIVE_ADD_MEM_CACHE = {}
        except Exception:
            _ADAPTIVE_ADD_MEM_CACHE = {}
        _ADAPTIVE_ADD_MEM_LOADED = True
    return _ADAPTIVE_ADD_MEM_CACHE


def _adaptive_add_mem_flush() -> None:
    with _ADAPTIVE_ADD_MEM_LOCK:
        try:
            os.makedirs(os.path.dirname(_ADAPTIVE_ADD_MEM_PATH), exist_ok=True)
            tmp = _ADAPTIVE_ADD_MEM_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_ADAPTIVE_ADD_MEM_CACHE, f, ensure_ascii=False)
            os.replace(tmp, _ADAPTIVE_ADD_MEM_PATH)
        except Exception:
            return


def _adaptive_add_mem_key(acc_idx: int, symbol: str, long_side: bool) -> str:
    return f"{int(acc_idx)}|{str(symbol).upper()}|{'LONG' if long_side else 'SHORT'}"


def _adaptive_add_mem_get(acc_idx: int, symbol: str, long_side: bool) -> dict:
    mem = _adaptive_add_mem_load()
    row = mem.get(_adaptive_add_mem_key(acc_idx, symbol, long_side))
    return row if isinstance(row, dict) else {}


def _adaptive_add_mem_upsert(
    acc_idx: int,
    symbol: str,
    long_side: bool,
    base_qty: float,
    base_entry: float,
    extra_qty: float,
) -> None:
    mem = _adaptive_add_mem_load()
    k = _adaptive_add_mem_key(acc_idx, symbol, long_side)
    mem[k] = {
        "base_qty": float(max(0.0, base_qty)),
        "base_entry": float(max(0.0, base_entry)),
        "extra_qty": float(max(0.0, extra_qty)),
        "updated_ts": float(time.time()),
    }
    _adaptive_add_mem_flush()


def _adaptive_add_mem_remove(acc_idx: int, symbol: str, long_side: bool | None = None) -> None:
    mem = _adaptive_add_mem_load()
    keys = []
    if long_side is None:
        pref = f"{int(acc_idx)}|{str(symbol).upper()}|"
        for k in list(mem.keys()):
            if str(k).startswith(pref):
                keys.append(k)
    else:
        keys.append(_adaptive_add_mem_key(acc_idx, symbol, long_side))
    changed = False
    for k in keys:
        if k in mem:
            mem.pop(k, None)
            changed = True
    if changed:
        _adaptive_add_mem_flush()

# Thread pool for non-blocking exchange order placement.
# All SL/TP order API calls are submitted here so the exit manager
# loop is never blocked by slow Binance API responses.
_ORDER_PLACEMENT_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="ord_place")


def _ladder_parse_steps(raw, default_steps: int = 10) -> int:
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        steps = int(default_steps)
    if steps < 1:
        steps = 1
    if steps > 50:
        steps = 50
    return steps


def _ladder_build_pct_levels(range_raw, steps_raw, fallback_start: float, fallback_end: float) -> list[float]:
    try:
        if isinstance(range_raw, (list, tuple)) and len(range_raw) >= 2:
            start_v = float(range_raw[0])
            end_v = float(range_raw[1])
        else:
            start_v = float(fallback_start)
            end_v = float(fallback_end)
    except (TypeError, ValueError):
        start_v = float(fallback_start)
        end_v = float(fallback_end)

    if start_v <= 0:
        start_v = float(fallback_start) if fallback_start > 0 else 1.0
    if end_v <= 0:
        end_v = float(fallback_end) if fallback_end > 0 else start_v

    steps = _ladder_parse_steps(steps_raw, default_steps=10)
    if steps == 1:
        return [float(start_v)]

    delta = (float(end_v) - float(start_v)) / float(steps - 1)
    levels: list[float] = []
    for i in range(steps):
        levels.append(float(start_v) + float(i) * float(delta))
    return levels


def _ladder_nudge_next_tick(symbol: str, base_adj_price: float, direction: int, ref_price: float) -> float:
    """Move one-or-more ticks away from base_adj_price in required direction."""
    if direction == 0:
        return base_adj_price
    try:
        p = float(base_adj_price)
    except (TypeError, ValueError):
        return base_adj_price
    step = max(abs(float(ref_price)) * 1e-6, 1e-12)
    raw = p
    for _ in range(64):
        raw = raw + (step if direction > 0 else -step)
        try:
            adj = float(adjust_price(symbol, raw))
        except Exception:
            adj = p
        if direction > 0 and adj > p:
            return adj
        if direction < 0 and adj < p:
            return adj
        step *= 2.0
    return p


def _ladder_build_price_levels(
    entry_price: float,
    pct_levels: list[float],
    long_side: bool,
    is_tp: bool,
    symbol: Optional[str] = None,
) -> list[float]:
    prices: list[float] = []
    if entry_price <= 0 or not pct_levels:
        return prices

    # Expected monotonic direction for the ladder prices.
    # TP: long increases, short decreases
    # SL: long decreases, short increases
    increasing = (is_tp and long_side) or ((not is_tp) and (not long_side))
    for pct in pct_levels:
        try:
            pv = float(pct)
        except (TypeError, ValueError):
            continue
        if pv <= 0:
            continue
        if is_tp:
            if long_side:
                raw_price = entry_price * (1.0 + pv / 100.0)
            else:
                raw_price = entry_price * (1.0 - pv / 100.0)
        else:
            if long_side:
                raw_price = entry_price * (1.0 - pv / 100.0)
            else:
                raw_price = entry_price * (1.0 + pv / 100.0)

        if symbol:
            try:
                adj_price = float(adjust_price(symbol, raw_price))
            except Exception:
                adj_price = float(raw_price)
        else:
            adj_price = float(raw_price)

        if prices:
            prev = float(prices[-1])
            # If rounding collapses multiple ladder levels to the same price,
            # nudge by at least one tick so each level is distinct.
            if abs(adj_price - prev) < 1e-15:
                if symbol:
                    adj_price = _ladder_nudge_next_tick(
                        str(symbol),
                        prev,
                        1 if increasing else -1,
                        entry_price,
                    )
            # Enforce monotonic ordering after rounding.
            if increasing and adj_price <= prev:
                if symbol:
                    adj_price = _ladder_nudge_next_tick(str(symbol), prev, 1, entry_price)
                if adj_price <= prev:
                    continue
            if (not increasing) and adj_price >= prev:
                if symbol:
                    adj_price = _ladder_nudge_next_tick(str(symbol), prev, -1, entry_price)
                if adj_price >= prev:
                    continue

        prices.append(adj_price)
    return prices


def _ladder_shift_last_towards_entry(levels: list[float], entry_price: float, long_side: bool, is_tp: bool, shifts: int = 1) -> list[float]:
    if not isinstance(levels, list) or len(levels) == 0:
        return levels
    if shifts <= 0:
        return levels

    out = list(levels)
    for _ in range(int(shifts)):
        if len(out) >= 2:
            step = abs(float(out[-1]) - float(out[-2]))
        else:
            step = abs(float(entry_price) - float(out[-1])) / 10.0
        if step <= 0:
            continue

        # Direction դեպի entry.
        if is_tp:
            move = -step if long_side else step
        else:
            move = step if long_side else -step

        new_last = float(out[-1]) + float(move)
        if len(out) >= 2:
            prev = float(out[-2])
            if out[-1] >= out[-2]:
                if new_last < prev:
                    new_last = prev
            else:
                if new_last > prev:
                    new_last = prev
        out[-1] = new_last
    return out


def _constrain_adaptive_add_prices(
    symbol: str,
    add_prices: list[float],
    entry_price: float,
    sl_price: float,
    long_side: bool,
) -> list[float]:
    """Clamp adaptive add ladder so it cannot sit beyond stop-loss."""
    if not isinstance(add_prices, list) or not add_prices:
        return []
    try:
        ep = float(entry_price)
        sp = float(sl_price)
    except (TypeError, ValueError):
        return [float(x) for x in add_prices if isinstance(x, (int, float))]
    if ep <= 0 or sp <= 0:
        return [float(x) for x in add_prices if isinstance(x, (int, float))]

    out: list[float] = []
    if long_side:
        lo = sp * 1.002
        hi = ep * 0.9995
        for px in add_prices:
            try:
                p = float(px)
            except (TypeError, ValueError):
                continue
            if p <= lo or p >= ep:
                continue
            if p > hi:
                p = hi
            try:
                p = float(adjust_price(symbol, p))
            except Exception:
                pass
            if p > lo and p < ep:
                out.append(p)
    else:
        lo = ep * 1.0005
        hi = sp * 0.998
        for px in add_prices:
            try:
                p = float(px)
            except (TypeError, ValueError):
                continue
            if p <= ep or p >= hi:
                continue
            if p < lo:
                p = lo
            try:
                p = float(adjust_price(symbol, p))
            except Exception:
                pass
            if p > ep and p < hi:
                out.append(p)

    if not out:
        # single safe midpoint fallback inside (entry, stop) band
        mid = (ep + sp) / 2.0
        try:
            mid = float(adjust_price(symbol, mid))
        except Exception:
            pass
        if long_side and (mid > sp * 1.002) and (mid < ep):
            return [mid]
        if (not long_side) and (mid > ep) and (mid < sp * 0.998):
            return [mid]
        return []

    uniq = sorted(set(round(float(x), 8) for x in out), reverse=(not long_side))
    return [float(x) for x in uniq]


def _place_initial_exchange_orders(
    acc_idx,
    symbol,
    state,
    long_side,
    sl_price,
    tp_prices,
    num_legs,
    initial_amt,
    position_side,
    sl_prices=None,
):
    """Place SL + TP orders in parallel. Called from background thread pool."""
    try:
        close_side = "SELL" if long_side else "BUY"

        # Submit SL and all TPs concurrently within this thread.
        sl_futures = []
        sl_split_mode = isinstance(sl_prices, (list, tuple)) and len(sl_prices) > 0
        sl_leg_count = len(sl_prices) if sl_split_mode else 1
        with ThreadPoolExecutor(max_workers=max(2, num_legs + sl_leg_count), thread_name_prefix="ord_inner") as inner:
            if sl_split_mode:
                if sl_leg_count == 1:
                    sl_futures.append((0, inner.submit(
                        place_exchange_stop_order,
                        acc_idx, symbol, close_side, float(sl_prices[0]),
                        "STOP_MARKET", None, True, position_side,
                    )))
                else:
                    sl_leg_qty = initial_amt / float(sl_leg_count) if sl_leg_count > 0 else initial_amt
                    for i, sl_px in enumerate(list(sl_prices)[:sl_leg_count]):
                        q = (initial_amt - sl_leg_qty * i) if i == sl_leg_count - 1 else sl_leg_qty
                        sl_futures.append((i, inner.submit(
                            place_exchange_stop_order,
                            acc_idx, symbol, close_side, sl_px,
                            "STOP_MARKET", q, False, position_side,
                        )))
            else:
                sl_futures.append((0, inner.submit(
                    place_exchange_stop_order,
                    acc_idx, symbol, close_side, sl_price,
                    "STOP_MARKET", None, True, position_side,
                )))

            tp_futures = []
            leg_qty = initial_amt / num_legs if num_legs > 0 else initial_amt
            for i, tp_px in enumerate(tp_prices[:num_legs]):
                q = (initial_amt - leg_qty * i) if i == num_legs - 1 else leg_qty
                tp_futures.append((i, float(tp_px), float(q), inner.submit(
                    place_exchange_stop_order,
                    acc_idx, symbol, close_side, tp_px,
                    "TAKE_PROFIT_MARKET", q, False, position_side,
                )))

        sl_orders = []
        for i, fut in sl_futures:
            sl_info = fut.result()
            if sl_info:
                sl_orders.append(sl_info)
            else:
                if sl_split_mode:
                    log(f"[EXIT MANAGER] WARNING: SL{i+1} order NOT placed for {symbol} account {acc_idx}")
                else:
                    log(f"[EXIT MANAGER] WARNING: SL order NOT placed for {symbol} account {acc_idx}")
        if sl_orders:
            state["exchange_sl_orders"] = sl_orders
            # Keep backward compatibility for code paths that expect single SL handle.
            state["exchange_sl_order"] = sl_orders[0]
        else:
            state["exchange_sl_orders"] = []

        tp_orders = []
        for i, tp_px, tp_q, fut in tp_futures:
            tp_info = fut.result()
            if tp_info:
                if isinstance(tp_info, dict):
                    row = dict(tp_info)
                else:
                    row = {"id": None, "algo": False}
                row["price"] = float(tp_px)
                row["qty"] = abs(float(tp_q))
                tp_orders.append(row)
            else:
                log(f"[EXIT MANAGER] WARNING: TP{i+1} order NOT placed for {symbol} account {acc_idx}")
        if tp_orders:
            state["exchange_tp_orders"] = tp_orders
        log(
            f"[EXIT MANAGER] Exchange orders for {symbol} account {acc_idx}: "
            f"SL={len(sl_orders)}/{sl_leg_count} TP={len(tp_orders)}/{num_legs}"
        )
    except Exception as e:
        log(f"[EXIT MANAGER] ERROR placing exchange orders for {symbol} account {acc_idx}: {e}")


def _update_exchange_sl_async(acc_idx, symbol, state, long_side, new_sl_price, position_side, reason):
    """Cancel old exchange SL and place new one. Called from background thread pool."""
    try:
        old_sl = state.get("exchange_sl_order")
        if old_sl and isinstance(old_sl, dict):
            cancel_order_by_id(acc_idx, symbol, int(old_sl["id"]), is_algo=bool(old_sl.get("algo")))
        close_side = "SELL" if long_side else "BUY"
        new_sl = place_exchange_stop_order(
            acc_idx, symbol, close_side, new_sl_price,
            order_type="STOP_MARKET", close_position=True,
            position_side=position_side,
        )
        if new_sl:
            state["exchange_sl_order"] = new_sl
    except Exception as e:
        log(f"[EXIT MANAGER] ERROR updating exchange SL ({reason}) for {symbol} account {acc_idx}: {e}")
    finally:
        state.pop("_sl_update_pending", None)


def _cancel_exchange_side_orders(
    state,
    acc_idx,
    symbol,
    side_kind: str,
    cancel_recovery_tp: bool = True,
) -> None:
    if side_kind == "sl":
        orders = state.get("exchange_sl_orders") or []
        if not isinstance(orders, list):
            orders = []
        for order in orders:
            if not isinstance(order, dict):
                continue
            try:
                cancel_order_by_id(acc_idx, symbol, int(order["id"]), is_algo=bool(order.get("algo")))
            except Exception:
                continue
        state["exchange_sl_orders"] = []
        state["exchange_sl_order"] = None
        return

    orders = state.get("exchange_tp_orders") or []
    if not isinstance(orders, list):
        orders = []
    for order in orders:
        if not isinstance(order, dict):
            continue
        try:
            cancel_order_by_id(acc_idx, symbol, int(order["id"]), is_algo=bool(order.get("algo")))
        except Exception:
            continue
    state["exchange_tp_orders"] = []
    # Adaptive recovery TP (add-only) is tracked separately.
    # Do NOT cancel it during regular TP-ladder rebuilds.
    if cancel_recovery_tp:
        rec = state.get("adaptive_recovery_tp_order")
        if isinstance(rec, dict):
            try:
                cancel_order_by_id(acc_idx, symbol, int(rec["id"]), is_algo=bool(rec.get("algo")))
            except Exception:
                pass
        state["adaptive_recovery_tp_order"] = None


def _cancel_all_open_tp_orders_on_exchange(
    acc_idx: int,
    symbol: str,
    long_side: bool,
    keep_order: dict | None = None,
) -> None:
    """Best-effort cleanup of residual TP orders on exchange.

    Keeps only the optional add-only recovery TP (if provided) and cancels all
    other TP orders for this symbol+close-side, including orphan orders not
    tracked in local state.
    """
    try:
        clients_local = _get_clients()
        if not clients_local or acc_idx < 0 or acc_idx >= len(clients_local):
            return
        client = clients_local[acc_idx]
        close_side = "SELL" if long_side else "BUY"
        keep_id = None
        keep_algo = None
        keep_px = 0.0
        keep_qty = 0.0
        if isinstance(keep_order, dict):
            keep_id = keep_order.get("id")
            keep_algo = bool(keep_order.get("algo"))
            try:
                keep_px = float(keep_order.get("price") or 0.0)
            except (TypeError, ValueError):
                keep_px = 0.0
            try:
                keep_qty = abs(float(keep_order.get("qty") or 0.0))
            except (TypeError, ValueError):
                keep_qty = 0.0

        # Regular open orders
        try:
            rows = client.open_orders(symbol=symbol)
            rows = rows if isinstance(rows, list) else []
        except Exception:
            rows = []
        for o in rows:
            if not isinstance(o, dict):
                continue
            try:
                typ = str(o.get("type") or "").upper()
                side = str(o.get("side") or "").upper()
            except Exception:
                continue
            if side != close_side:
                continue
            if typ not in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
                continue
            oid = o.get("orderId")
            if oid is None:
                continue
            try:
                oid_i = int(oid)
            except Exception:
                continue
            if keep_id is not None and (not keep_algo) and oid_i == int(keep_id):
                continue
            # Fallback keeper by qty/price similarity (in case order-id or
            # algo-flag changed after reconnect/reload).
            try:
                opx = float(o.get("stopPrice") or o.get("price") or 0.0)
                oq = abs(float(o.get("origQty") or o.get("quantity") or 0.0))
                p_ok = keep_px > 0 and abs(opx - keep_px) / max(abs(keep_px), 1e-9) <= 0.003
                q_ok = keep_qty > 0 and abs(oq - keep_qty) / max(keep_qty, 1e-9) <= 0.20
                if p_ok and q_ok:
                    continue
            except Exception:
                pass
            try:
                cancel_order_by_id(acc_idx, symbol, oid_i, is_algo=False)
            except Exception:
                continue

        # Algo open orders
        try:
            algo = client.get_algo_open_orders(symbol=symbol)
            algo_rows = algo if isinstance(algo, list) else (
                algo.get("orders") or [] if isinstance(algo, dict) else []
            )
        except Exception:
            algo_rows = []
        for o in algo_rows:
            if not isinstance(o, dict):
                continue
            try:
                side = str(o.get("side") or "").upper()
                typ = str(o.get("type") or "").upper()
            except Exception:
                continue
            if side != close_side:
                continue
            # Some algo orders return empty `type`; treat non-close-position
            # conditional close-side algo orders as TP-like.
            is_tp_like = typ in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET")
            if (not is_tp_like) and str(o.get("algoType") or "").upper() == "CONDITIONAL":
                cp = str(o.get("closePosition") or "").lower() == "true"
                is_tp_like = not cp
            if not is_tp_like:
                continue
            aid = o.get("algoId")
            if aid is None:
                continue
            try:
                aid_i = int(aid)
            except Exception:
                continue
            if keep_id is not None and bool(keep_algo) and aid_i == int(keep_id):
                continue
            try:
                opx = float(o.get("triggerPrice") or o.get("stopPrice") or o.get("price") or 0.0)
                oq = abs(float(o.get("quantity") or o.get("qty") or 0.0))
                p_ok = keep_px > 0 and abs(opx - keep_px) / max(abs(keep_px), 1e-9) <= 0.003
                q_ok = keep_qty > 0 and abs(oq - keep_qty) / max(keep_qty, 1e-9) <= 0.20
                if p_ok and q_ok:
                    continue
            except Exception:
                pass
            try:
                cancel_order_by_id(acc_idx, symbol, aid_i, is_algo=True)
            except Exception:
                continue
    except Exception:
        return


def _is_tracked_order_still_open(acc_idx: int, symbol: str, tracked: dict) -> bool:
    """Check whether tracked order still exists on exchange."""
    if not isinstance(tracked, dict):
        return False
    oid = tracked.get("id")
    if oid is None:
        return False
    is_algo = bool(tracked.get("algo"))
    try:
        oid_i = int(oid)
    except Exception:
        return False
    try:
        clients_local = _get_clients()
        if not clients_local or acc_idx < 0 or acc_idx >= len(clients_local):
            return False
        client = clients_local[acc_idx]
        if is_algo:
            rows_raw = client.get_algo_open_orders(symbol=symbol)
            rows = rows_raw if isinstance(rows_raw, list) else (
                rows_raw.get("orders") or [] if isinstance(rows_raw, dict) else []
            )
            for o in rows:
                if not isinstance(o, dict):
                    continue
                try:
                    if int(o.get("algoId")) == oid_i:
                        return True
                except Exception:
                    continue
            return False
        rows = client.open_orders(symbol=symbol)
        rows = rows if isinstance(rows, list) else []
        for o in rows:
            if not isinstance(o, dict):
                continue
            try:
                if int(o.get("orderId")) == oid_i:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _count_open_protection_orders_on_exchange(acc_idx: int, symbol: str, long_side: bool) -> dict:
    """Return current close-side TP/SL counts from exchange."""
    out = {"tp_count": 0, "sl_count": 0}
    try:
        clients_local = _get_clients()
        if not clients_local or acc_idx < 0 or acc_idx >= len(clients_local):
            return out
        client = clients_local[acc_idx]
        close_side = "SELL" if long_side else "BUY"

        def _consume_row(row: dict) -> None:
            if not isinstance(row, dict):
                return
            side = str(row.get("side") or "").upper()
            if side != close_side:
                return
            typ = str(row.get("type") or "").upper()
            cp = str(row.get("closePosition") or "").lower() == "true"
            is_tp_like = typ in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET")
            is_sl_like = typ in ("STOP", "STOP_MARKET") or cp
            if (not is_tp_like) and str(row.get("algoType") or "").upper() == "CONDITIONAL":
                is_tp_like = not cp
            if is_tp_like:
                out["tp_count"] = int(out["tp_count"]) + 1
            elif is_sl_like:
                out["sl_count"] = int(out["sl_count"]) + 1

        try:
            regular = client.open_orders(symbol=symbol)
            regular = regular if isinstance(regular, list) else []
            for r in regular:
                _consume_row(r)
        except Exception:
            pass

        try:
            algo_raw = client.get_algo_open_orders(symbol=symbol)
            algo_rows = algo_raw if isinstance(algo_raw, list) else (
                algo_raw.get("orders") or [] if isinstance(algo_raw, dict) else []
            )
            for r in algo_rows:
                _consume_row(r)
        except Exception:
            pass
    except Exception:
        return out
    return out


def _sync_adaptive_recovery_tp_async(
    acc_idx: int,
    symbol: str,
    state: dict,
    long_side: bool,
    position_side,
    entry_price: float,
    current_price: float,
    current_abs_qty: float,
    base_core_qty: float,
) -> None:
    """Ensure add-only recovery TP exists near base entry; fallback close on target reach."""
    pending_key = "_adaptive_recovery_sync_pending"
    try:
        now_ts = time.time()
        last_sync = float(state.get("_adaptive_recovery_last_sync_ts") or 0.0)
        if (now_ts - last_sync) < 1.0:
            return
        state["_adaptive_recovery_last_sync_ts"] = now_ts

        try:
            # Prefer live position qty from state (updated every exit manager cycle)
            # over the stale parameter captured at thread-submission time.
            # This prevents placing a new recovery TP after the add fill is already closed.
            _live_abs = state.get("adaptive_last_abs_qty")
            cur_abs = abs(float(_live_abs)) if _live_abs is not None else abs(float(current_abs_qty))
        except (TypeError, ValueError):
            cur_abs = 0.0
        try:
            core_qty = max(0.0, float(base_core_qty))
        except (TypeError, ValueError):
            core_qty = 0.0
        # Recovery TP must always manage ONLY the added quantity.
        # It must NOT expand because base TP levels were partially consumed.
        add_qty = max(0.0, cur_abs - core_qty)
        if add_qty <= 0.0:
            rec = state.get("adaptive_recovery_tp_order")
            if isinstance(rec, dict):
                try:
                    cancel_order_by_id(acc_idx, symbol, int(rec["id"]), is_algo=bool(rec.get("algo")))
                except Exception:
                    pass
            state["adaptive_recovery_tp_order"] = None
            return

        try:
            base_entry = float(state.get("adaptive_base_entry_price") or entry_price or 0.0)
        except (TypeError, ValueError):
            base_entry = 0.0
        if base_entry <= 0:
            base_entry = float(entry_price) if entry_price and entry_price > 0 else 0.0
        if base_entry <= 0:
            return

        # Recovery TP for added qty must sit as close as possible to the original
        # base entry — just enough to cover the round-trip taker fee, nothing more.
        # BUG FIX: old formula used funding_buffer_pct=0.01 which stacked with
        # taker_fee*200 to produce ~1% above entry. Correct: fee-only buffer.
        # LONG cap: +0.30% | SHORT cap: +0.15%  (safety ceiling, not a target)
        taker_fee_rate = _safe_float(state.get("adaptive_recovery_taker_fee_rate"), 0.0004)
        # Round-trip fee (open + close taker): e.g. 0.0004 * 2 * 100 = 0.08%
        fee_pct = taker_fee_rate * 200.0
        fixed_recovery_pct = min(0.30, max(0.0, fee_pct)) if long_side else min(0.15, max(0.0, fee_pct))
        if long_side:
            target_px = base_entry * (1.0 + fixed_recovery_pct / 100.0)
        else:
            target_px = base_entry * (1.0 - fixed_recovery_pct / 100.0)
        try:
            target_px = float(adjust_price(symbol, target_px))
        except Exception:
            target_px = float(target_px)
        if target_px <= 0:
            return

        # Keep a single add-only recovery TP order aligned to latest add_qty.
        rec = state.get("adaptive_recovery_tp_order")
        keep_existing = False
        if isinstance(rec, dict):
            try:
                old_qty = abs(float(rec.get("qty") or 0.0))
                old_px = float(rec.get("price") or 0.0)
                if old_qty > 0 and old_px > 0:
                    q_diff = abs(old_qty - add_qty) / max(add_qty, 1e-9)
                    p_diff = abs(old_px - target_px) / max(abs(target_px), 1e-9)
                    if q_diff <= 0.12 and p_diff <= 0.002:
                        keep_existing = True  # Trust state — no API call needed
            except Exception:
                keep_existing = False
            # Periodic verification only (every 60s) — not every cycle.
            # Avoids constant proxy API calls while still catching external cancels.
            if keep_existing:
                import threading as _thr_chk
                _last_ok = float(state.get("_recovery_tp_verified_ts") or 0.0)
                if (now_ts - _last_ok) >= 60.0:
                    _chk_result: list = [False]
                    _chk_done = _thr_chk.Event()
                    def _chk_worker(_r=rec):
                        _chk_result[0] = _is_tracked_order_still_open(acc_idx, symbol, _r)
                        _chk_done.set()
                    _thr_chk.Thread(target=_chk_worker, daemon=True).start()
                    _chk_done.wait(timeout=5.0)
                    if not _chk_done.is_set() or not _chk_result[0]:
                        # Order gone (filled or externally cancelled).
                        # Guard: let next cycle confirm with live position data.
                        keep_existing = False
                        state["adaptive_recovery_tp_order"] = None
                        state["_adaptive_recovery_last_sync_ts"] = time.time() + 2.0
                        return
                    state["_recovery_tp_verified_ts"] = now_ts
        if not keep_existing and isinstance(rec, dict):
            try:
                cancel_order_by_id(acc_idx, symbol, int(rec["id"]), is_algo=bool(rec.get("algo")))
            except Exception:
                pass
            state["adaptive_recovery_tp_order"] = None

        if not keep_existing:
            close_side = "SELL" if long_side else "BUY"
            try:
                import threading as _thr_place
                _place_result: list = [None]
                _place_done = _thr_place.Event()
                def _place_worker():
                    try:
                        _place_result[0] = place_exchange_stop_order(
                            acc_idx,
                            symbol,
                            close_side,
                            target_px,
                            order_type="TAKE_PROFIT_MARKET",
                            quantity=add_qty,
                            close_position=False,
                            position_side=position_side,
                        )
                    except Exception:
                        pass
                    finally:
                        _place_done.set()
                _thr_place.Thread(target=_place_worker, daemon=True).start()
                _place_done.wait(timeout=8.0)
                info = _place_result[0] if _place_done.is_set() else None
                if info:
                    state["adaptive_recovery_tp_order"] = {
                        "id": info.get("id"),
                        "algo": bool(info.get("algo")),
                        "price": float(target_px),
                        "qty": float(add_qty),
                    }
                    log(
                        f"[ADAPTIVE][RECOVERY][TP] account={acc_idx} symbol={symbol} "
                        f"price={target_px:.8f} qty={add_qty:.8f} core_qty={core_qty:.8f} "
                        f"rule_pct={fixed_recovery_pct:.2f}"
                    )
                    return
                elif not _place_done.is_set():
                    # Proxy hung — background thread still running; block retry for 30s
                    # so the background thread has time to complete without duplicates.
                    state["_adaptive_recovery_last_sync_ts"] = time.time() + 30.0
                    log(
                        f"[ADAPTIVE][RECOVERY][TP] Timeout (>8s) placing TP for {symbol} "
                        f"account={acc_idx} — retrying in ~30s"
                    )
            except Exception:
                pass

        # If TP order cannot be placed but target is already reached, close only add qty.
        reached = (float(current_price) >= float(target_px)) if long_side else (float(current_price) <= float(target_px))
        if reached and add_qty > 0:
            signed_close = float(add_qty) if long_side else -float(add_qty)
            ok = close_position_market(
                acc_idx,
                symbol,
                signed_close,
                force_full_close=False,
                position_side=position_side,
            )
            if ok:
                log(
                    f"[ADAPTIVE][RECOVERY][FALLBACK_CLOSE] account={acc_idx} symbol={symbol} "
                    f"target={target_px:.8f} qty={add_qty:.8f} current={float(current_price):.8f}"
                )
    finally:
        state.pop(pending_key, None)


def _cancel_exchange_add_orders(state, acc_idx, symbol) -> None:
    orders = state.get("exchange_add_orders") or []
    if not isinstance(orders, list):
        orders = []
    cancelled = 0
    failed = 0
    for order in orders:
        if not isinstance(order, dict):
            continue
        try:
            cancel_order_by_id(acc_idx, symbol, int(order["id"]), is_algo=bool(order.get("algo")))
            cancelled += 1
        except Exception:
            failed += 1
            continue
    state["exchange_add_orders"] = []
    try:
        log(
            f"[ADAPTIVE][ADD][CANCEL] account={acc_idx} symbol={symbol} "
            f"requested={len(orders)} cancelled={cancelled} failed={failed}"
        )
    except Exception:
        pass


def _rebuild_adaptive_add_orders_async(
    acc_idx: int,
    symbol: str,
    state: dict,
    long_side: bool,
    position_side,
    entry_price: float,
    hit_count: int = 0,
) -> None:
    pending_key = "_adaptive_add_rebuild_pending"
    try:
        add_prices = state.get("adaptive_add_prices") or []
        if not isinstance(add_prices, list) or not add_prices:
            _cancel_exchange_add_orders(state, acc_idx, symbol)
            return
        try:
            sl_px = float(state.get("adaptive_sl_price") or 0.0)
        except (TypeError, ValueError):
            sl_px = 0.0
        try:
            ep_ref = float(entry_price)
        except (TypeError, ValueError):
            ep_ref = 0.0
        # If protective SL already moved past/through entry, add-limits are invalid:
        # they can end up logically "after stop". Force-cancel add grid.
        if sl_px > 0 and ep_ref > 0:
            sl_beyond_entry = (long_side and sl_px >= ep_ref * 0.9995) or ((not long_side) and sl_px <= ep_ref * 1.0005)
            if sl_beyond_entry:
                state["adaptive_add_prices"] = []
                _cancel_exchange_add_orders(state, acc_idx, symbol)
                log(
                    f"[ADAPTIVE][ADD][REBUILD][SKIP] account={acc_idx} symbol={symbol} "
                    f"reason=sl_beyond_entry sl={sl_px:.8f} entry={ep_ref:.8f}"
                )
                return

        add_prices = _constrain_adaptive_add_prices(
            symbol=symbol,
            add_prices=[float(x) for x in add_prices if isinstance(x, (int, float))],
            entry_price=ep_ref,
            sl_price=sl_px if sl_px > 0 else ep_ref,
            long_side=bool(long_side),
        )
        state["adaptive_add_prices"] = list(add_prices)
        if not add_prices:
            _cancel_exchange_add_orders(state, acc_idx, symbol)
            log(
                f"[ADAPTIVE][ADD][REBUILD][SKIP] account={acc_idx} symbol={symbol} "
                "all add levels invalid vs stop-loss band"
            )
            return

        try:
            base_qty = float(state.get("initial_amt") or 0.0)
        except (TypeError, ValueError):
            base_qty = 0.0
        if base_qty <= 0:
            return
        try:
            add_mult = float(state.get("adaptive_add_total_multiplier") or 0.0)
        except (TypeError, ValueError):
            add_mult = 0.0
        if add_mult <= 0:
            return

        # Target add grid total quantity is base_qty * multiplier.
        total_add_qty = base_qty * add_mult
        if total_add_qty <= 0:
            return

        clients_local = _get_clients()
        if acc_idx < 0 or acc_idx >= len(clients_local):
            return
        client = clients_local[acc_idx]
        order_side = "BUY" if long_side else "SELL"

        before_count = len(state.get("exchange_add_orders") or [])
        removed = 0
        if int(hit_count) > 0:
            # Per TP hit, attempt to remove farthest two add orders first.
            existing_orders = state.get("exchange_add_orders") or []
            if not isinstance(existing_orders, list):
                existing_orders = []
            to_remove = max(0, int(hit_count)) * 2
            while removed < to_remove and existing_orders:
                order = existing_orders.pop()
                removed += 1
                if not isinstance(order, dict):
                    continue
                try:
                    cancel_order_by_id(acc_idx, symbol, int(order["id"]), is_algo=bool(order.get("algo")))
                except Exception:
                    continue
            state["exchange_add_orders"] = existing_orders
        _cancel_exchange_add_orders(state, acc_idx, symbol)

        desired_count = len(add_prices)
        if desired_count <= 0:
            return
        leg_qty = total_add_qty / float(desired_count)
        placed = []
        skipped_invalid = 0
        place_errors = 0
        for i, px in enumerate(add_prices):
            q = (total_add_qty - leg_qty * i) if i == (desired_count - 1) else leg_qty
            try:
                qf = float(adjust_quantity(symbol, q, float(px)))
                pf = float(adjust_price(symbol, float(px)))
            except Exception:
                skipped_invalid += 1
                continue
            if qf <= 0 or pf <= 0:
                skipped_invalid += 1
                continue
            try:
                from data.symbol_blocklist import is_symbol_blocked as _isb_add
                if _isb_add(str(symbol).upper()):
                    log(f"[ADAPTIVE][ADD][BLOCKED] {symbol} is MQA-blocked — skipping add orders")
                    break
            except Exception:
                pass
            try:
                resp = client.place_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=qf,
                    order_type="LIMIT",
                    reduce_only=False,
                    price=pf,
                    time_in_force="GTC",
                    position_side=position_side,
                )
            except Exception:
                place_errors += 1
                continue
            if isinstance(resp, dict):
                oid = resp.get("orderId")
                if oid is not None:
                    try:
                        oid = int(oid)
                    except Exception:
                        pass
                    placed.append({"id": oid, "algo": False, "price": pf, "qty": qf})
        state["exchange_add_orders"] = placed

        try:
            log(
                f"[ADAPTIVE][ADD][REBUILD] account={acc_idx} symbol={symbol} "
                f"before={before_count} hit_count={int(hit_count)} removed={removed} "
                f"placed={len(placed)}/{desired_count} invalid={skipped_invalid} errors={place_errors} "
                f"total_add_qty={total_add_qty:.8f}"
            )
        except Exception:
            pass
    finally:
        state.pop(pending_key, None)


def _rebuild_ladder_side_orders_async(
    acc_idx: int,
    symbol: str,
    state: dict,
    long_side: bool,
    position_side,
    current_abs_qty: float,
    side_kind: str,
    hit_count: int = 1,
) -> None:
    pending_key = "_ladder_sl_rebuild_pending" if side_kind == "sl" else "_ladder_tp_rebuild_pending"
    try:
        try:
            qty_total = float(current_abs_qty)
            if side_kind == "tp":
                q_override = state.get("ladder_tp_qty_total")
                if q_override is not None:
                    qty_total = float(q_override)
                # Keep TP ladder and recovery TP non-overlapping by capping
                # ladder TP quantity to remaining size after add-only recovery TP.
                rec = state.get("adaptive_recovery_tp_order")
                rec_qty = 0.0
                if isinstance(rec, dict):
                    try:
                        rec_qty = abs(float(rec.get("qty") or 0.0))
                    except (TypeError, ValueError):
                        rec_qty = 0.0
                # Hard reserve: extra qty must stay exclusively for recovery TP.
                # Even if recovery order is temporarily missing/stale, do not
                # re-distribute extra quantity into regular TP ladder.
                core_qty = 0.0
                try:
                    core_qty = max(0.0, abs(float(state.get("initial_amt") or 0.0)))
                except (TypeError, ValueError):
                    core_qty = 0.0
                expected_extra_qty = max(0.0, float(current_abs_qty) - core_qty)
                reserve_qty = max(rec_qty, expected_extra_qty)
                qty_cap = max(0.0, float(current_abs_qty) - reserve_qty)
                if qty_total > qty_cap:
                    qty_total = qty_cap
        except (TypeError, ValueError):
            qty_total = 0.0
        if qty_total <= 0:
            return

        if side_kind == "sl":
            prices = state.get("ladder_sl_prices") or []
            order_type = "STOP_MARKET"
        else:
            prices = state.get("ladder_tp_prices") or []
            order_type = "TAKE_PROFIT_MARKET"

        if not isinstance(prices, list) or not prices:
            return

        # Stage 1 (mandatory order): cancel farthest two orders per hit
        # before placing the new "near-first" ladder level.
        existing_orders_key = "exchange_sl_orders" if side_kind == "sl" else "exchange_tp_orders"
        existing_orders = state.get(existing_orders_key) or []
        if not isinstance(existing_orders, list):
            existing_orders = []
        before_count = len(existing_orders)
        if side_kind == "tp" and int(hit_count) <= 0:
            # Anti-churn: if ladder TP orders already match target price/qty,
            # keep them intact and skip cancel/re-place cycle.
            try:
                desired_count_probe = max(1, len(prices))
                leg_qty_probe = qty_total / float(desired_count_probe)
                expected = []
                for i_probe, px_probe in enumerate(prices):
                    q_probe = (
                        (qty_total - leg_qty_probe * i_probe)
                        if i_probe == (desired_count_probe - 1)
                        else leg_qty_probe
                    )
                    expected.append((float(px_probe), abs(float(q_probe))))
                current = []
                for o in existing_orders:
                    if not isinstance(o, dict):
                        continue
                    p_cur = float(o.get("price") or 0.0)
                    q_cur = abs(float(o.get("qty") or o.get("quantity") or o.get("origQty") or 0.0))
                    if p_cur > 0 and q_cur > 0:
                        current.append((p_cur, q_cur))
                if len(current) == len(expected) and len(expected) > 0:
                    expected_s = sorted(expected, key=lambda x: x[0])
                    current_s = sorted(current, key=lambda x: x[0])
                    same = True
                    for (epx, eqt), (cpx, cqt) in zip(expected_s, current_s):
                        pd = abs(cpx - epx) / max(abs(epx), 1e-9)
                        qd = abs(cqt - eqt) / max(abs(eqt), 1e-9)
                        if pd > 0.0015 or qd > 0.15:
                            same = False
                            break
                    if same:
                        return
            except Exception:
                pass
        # Per-hit policy: remove only the farthest ONE order.
        # (count trend: 10 -> 9 -> 8 ...)
        remove_per_hit = 1
        to_remove = max(0, int(hit_count)) * int(remove_per_hit)
        removed = 0
        while removed < to_remove and existing_orders:
            order = existing_orders.pop()
            removed += 1
            if not isinstance(order, dict):
                continue
            try:
                cancel_order_by_id(acc_idx, symbol, int(order["id"]), is_algo=bool(order.get("algo")))
            except Exception:
                continue
        state[existing_orders_key] = existing_orders

        # Stage 2: clean residual side orders to rebuild deterministic ladder
        # quantities and prevent allocation mismatches on partial position size.
        # Keep add-only recovery TP alive while rebuilding TP ladder.
        _cancel_exchange_side_orders(
            state,
            acc_idx,
            symbol,
            side_kind,
            cancel_recovery_tp=(side_kind != "tp"),
        )
        if side_kind == "tp":
            _cancel_all_open_tp_orders_on_exchange(
                acc_idx=acc_idx,
                symbol=symbol,
                long_side=long_side,
                keep_order=state.get("adaptive_recovery_tp_order"),
            )

        close_side = "SELL" if long_side else "BUY"
        desired_count = max(1, len(prices))
        leg_qty = qty_total / float(desired_count)

        placed = []
        for i, px in enumerate(prices):
            q = (qty_total - leg_qty * i) if i == (desired_count - 1) else leg_qty
            use_close_position = (side_kind == "sl" and desired_count == 1)
            try:
                qf = float(q)
            except (TypeError, ValueError):
                qf = 0.0
            if (not use_close_position) and qf <= 0:
                continue
            info = place_exchange_stop_order(
                acc_idx,
                symbol,
                close_side,
                float(px),
                order_type=order_type,
                quantity=None if use_close_position else qf,
                close_position=bool(use_close_position),
                position_side=position_side,
            )
            if info:
                if isinstance(info, dict):
                    row = dict(info)
                else:
                    row = {"id": None, "algo": False}
                row["price"] = float(px)
                row["qty"] = 0.0 if use_close_position else abs(float(qf))
                placed.append(row)

        if side_kind == "sl":
            state["exchange_sl_orders"] = placed
            state["exchange_sl_order"] = placed[0] if placed else None
            # Emergency fallback: never leave a position without SL due to
            # immediate-trigger rejects on partial/rebuild transitions.
            if not placed:
                try:
                    cur_px = float(state.get("_last_price") or 0.0)
                except (TypeError, ValueError):
                    cur_px = 0.0
                if cur_px > 0:
                    close_side = "SELL" if long_side else "BUY"
                    emergency_px = cur_px * (0.997 if long_side else 1.003)
                    try:
                        emergency_px = float(adjust_price(symbol, emergency_px))
                    except Exception:
                        pass
                    info = place_exchange_stop_order(
                        acc_idx,
                        symbol,
                        close_side,
                        emergency_px,
                        order_type="STOP_MARKET",
                        quantity=None,
                        close_position=True,
                        position_side=position_side,
                    )
                    if info:
                        state["exchange_sl_orders"] = [info]
                        state["exchange_sl_order"] = info
                        placed = [info]
                        log(
                            f"[LADDER][REBUILD][SL-FALLBACK] account={acc_idx} symbol={symbol} "
                            f"placed closePosition STOP at {emergency_px:.8f}"
                        )
        else:
            state["exchange_tp_orders"] = placed
        after_count = len(placed)

        try:
            log(
                f"[LADDER][REBUILD] account={acc_idx} symbol={symbol} side={side_kind.upper()} "
                f"hits={int(hit_count)} removed={int(to_remove)} placed={len(placed)}/{desired_count} qty={qty_total:.8f}"
            )
            log(
                f"[LADDER][REBUILD][DETAIL] account={acc_idx} symbol={symbol} side={side_kind.upper()} "
                f"before_count={before_count} -> removed={int(to_remove)} -> placed={len(placed)} -> after_count={after_count}"
            )
        except Exception:
            pass
    finally:
        state.pop(pending_key, None)


def _ladder_restep_after_hit(levels: list[float], hits: int) -> list[float]:
    """Per hit: delete last, move new tail near first (net -1 level)."""
    if not isinstance(levels, list):
        return levels
    out = list(levels)
    if hits <= 0:
        return out

    for _ in range(int(hits)):
        if len(out) <= 1:
            break
        try:
            first = float(out[0])
            second = float(out[1]) if len(out) > 1 else float(out[0])
        except Exception:
            break
        step = abs(second - first)
        if step <= 0:
            step = abs(first) * 1e-6 if abs(first) > 0 else 1e-12
        direction = 1.0 if second >= first else -1.0
        near_first = first + direction * step

        # 1) remove only the farthest level
        out.pop()
        if not out:
            break

        # 2) move tail (previous penultimate) close to the first level
        tail_idx = len(out) - 1
        out[tail_idx] = near_first

        # keep monotonic ordering (needed after rounding/collapse cases)
        out = sorted(out, reverse=(direction < 0))
    return out


def _adaptive_pull_sl_towards_entry(
    symbol: str,
    sl_price: float,
    entry_price: float,
    long_side: bool,
    steps: int = 1,
) -> float:
    """Move adaptive hard-SL closer to entry in small steps."""
    try:
        cur = float(sl_price)
        entry = float(entry_price)
    except Exception:
        return sl_price
    if cur <= 0 or entry <= 0:
        return sl_price

    # Per step: pull ~15% of remaining distance toward entry.
    k = 0.15
    n = max(1, int(steps))
    for _ in range(n):
        if long_side:
            dist = entry - cur
            if dist <= 0:
                break
            cur = cur + dist * k
            try:
                cur = float(adjust_price(symbol, cur))
            except Exception:
                pass
            if cur >= entry:
                cur = entry * (1.0 - 1e-6)
        else:
            dist = cur - entry
            if dist <= 0:
                break
            cur = cur - dist * k
            try:
                cur = float(adjust_price(symbol, cur))
            except Exception:
                pass
            if cur <= entry:
                cur = entry * (1.0 + 1e-6)
    return cur

def _cancel_exchange_tp_for_crossed_levels(state, acc_idx, symbol, from_level, to_level):
    """Cancel exchange TP orders for levels [from_level, to_level) to prevent
    double execution when the exit manager also sends a manual close."""
    tp_orders = state.get("exchange_tp_orders")
    if not tp_orders or not isinstance(tp_orders, list):
        return
    for i in range(from_level, min(to_level, len(tp_orders))):
        order = tp_orders[i]
        if order and isinstance(order, dict):
            try:
                cancel_order_by_id(acc_idx, symbol, int(order["id"]), is_algo=bool(order.get("algo")))
            except Exception:
                pass
            tp_orders[i] = None


def _verify_and_retry_exchange_orders(acc_idx, symbol, state, long_side, position_side):
    """Verify exchange SL order still exists on Binance. Re-place if missing.

    NOTE: We intentionally do NOT re-place TP orders here.  If a TP order
    disappears (filled or expired), the exit-manager's grace-period logic
    will detect the TP crossing and handle it.  Re-placing TPs led to
    DUPLICATE orders on Binance, causing premature full position closes.
    """
    try:
        clients = _get_clients()
        if not clients or acc_idx < 0 or acc_idx >= len(clients):
            return
        client = clients[acc_idx]

        sl_order = state.get("exchange_sl_order")
        if not sl_order or not isinstance(sl_order, dict):
            return

        sl_id = int(sl_order["id"])

        # Fetch open orders from Binance (regular + algo)
        open_ids = set()
        query_ok = False
        try:
            regular = client.open_orders(symbol=symbol)
            if isinstance(regular, list):
                query_ok = True
                for o in regular:
                    try:
                        open_ids.add(int(o.get("orderId")))
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            from execution.binance_futures import _ACCOUNT_USES_ALGO
            if _ACCOUNT_USES_ALGO.get(acc_idx, False):
                algo = client.get_algo_open_orders(symbol=symbol)
                algo_list = algo if isinstance(algo, list) else (algo.get("orders") or [] if isinstance(algo, dict) else [])
                for o in algo_list:
                    if isinstance(o, dict):
                        try:
                            open_ids.add(int(o.get("algoId")))
                        except Exception:
                            pass
                query_ok = True
        except Exception:
            pass

        # If we couldn't query ANY open orders, bail out — don't assume
        # orders are missing just because the API call failed.
        if not query_ok:
            log(f"[ORDER VERIFY] Could not query open orders for {symbol} account {acc_idx} — skipping")
            return

        # SL still open on Binance — nothing to do.
        if sl_id in open_ids:
            return

        # SL not found — try to re-place it.
        sl_price = state.get("sl_price")
        if not sl_price or sl_price <= 0:
            return

        close_side = "SELL" if long_side else "BUY"
        try:
            new_sl = place_exchange_stop_order(
                acc_idx, symbol, close_side, sl_price,
                order_type="STOP_MARKET", close_position=True,
                position_side=position_side,
            )
            if new_sl:
                state["exchange_sl_order"] = new_sl
                log(f"[ORDER VERIFY] Re-placed missing SL for {symbol} account {acc_idx}: {new_sl}")
            else:
                log(f"[ORDER VERIFY] WARNING: Failed to re-place SL for {symbol} account {acc_idx}")
        except Exception as e:
            err_str = str(e)
            if "-4130" in err_str:
                # SL already exists on Binance (just with a different ID
                # than we tracked, e.g. after a BE move).  This is fine.
                log(f"[ORDER VERIFY] SL already exists on Binance for {symbol} account {acc_idx} (tracked id {sl_id} stale)")
            else:
                log(f"[ORDER VERIFY] ERROR re-placing SL for {symbol} account {acc_idx}: {e}")

    except Exception as e:
        log(f"[ORDER VERIFY] ERROR verifying orders for {symbol} account {acc_idx}: {e}")
    finally:
        state.pop("_verify_pending", None)


_ENTRY_TS_CACHE = {}

_MARK_PRICE_CACHE = {}
_MARK_PRICE_CACHE_TTL_SEC = 2.0

_OPEN_POS_SNAPSHOT_CACHE = {"ts": 0.0, "data": []}
_OPEN_POS_SNAPSHOT_LOCK = threading.Lock()
_OPEN_POS_SNAPSHOT_TTL_SEC = 1.0


def _fmt_duration_hm(seconds: float) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    if s < 0:
        s = 0.0
    total_min = int(round(s / 60.0))
    h = total_min // 60
    m = total_min % 60
    if h > 0 and m > 0:
        return f"{h} ժամ {m} րոպե"
    if h > 0:
        return f"{h} ժամ"
    return f"{m} րոպե"


def _get_last_entry_ts_from_history(account_index: int, symbol: str) -> Optional[float]:
    try:
        acc = int(account_index)
    except Exception:
        return None
    sym_u = str(symbol).upper()
    now = time.time()
    key = (acc, sym_u)
    try:
        cached = _ENTRY_TS_CACHE.get(key)
        if isinstance(cached, tuple) and len(cached) == 2:
            cached_at, ts = cached
            try:
                if (now - float(cached_at)) < 300.0:
                    return float(ts) if ts and float(ts) > 0 else None
            except Exception:
                pass
    except Exception:
        pass

    try:
        path = os.path.join(_ROOT_DIR, "data", "trade_history", f"account_{acc}.csv")
        if not os.path.exists(path):
            _ENTRY_TS_CACHE[key] = (now, 0.0)
            return None
    except Exception:
        return None

    last_ts = 0.0
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row or len(row) < 5:
                    continue
                if row[0] == "timestamp_ms":
                    continue
                try:
                    event = str(row[2] or "").upper()
                    sym = str(row[3] or "").upper()
                except Exception:
                    continue
                if sym != sym_u or event != "ENTRY":
                    continue
                try:
                    ts_ms = int(float(row[0]))
                    ts_s = float(ts_ms) / 1000.0
                    if ts_s > last_ts:
                        last_ts = ts_s
                except Exception:
                    continue
    except Exception:
        last_ts = 0.0

    try:
        _ENTRY_TS_CACHE[key] = (now, float(last_ts))
    except Exception:
        pass
    return float(last_ts) if last_ts and last_ts > 0 else None

# Per-symbol auto risk context used when accounts are configured with
# auto_sl_tp in their settings. Populated at signal time and consumed by
# the exit manager for dynamic SL/TP distances.
_AUTO_RISK_CONTEXT = {}

# Per-symbol AI SL/TP context produced by the SL/TP NN for signals.
# Key: (symbol, direction_str) where direction_str is "LONG" or "SHORT".
_AI_SLTP_CONTEXT = {}

_PENDING_ENTRIES = []
_PENDING_ENTRIES_LOCK = threading.Lock()
_LAST_SIGNAL_CANDLE_SEEN = {}
_LAST_SIGNAL_CANDLE_LOCK = threading.Lock()
_LAST_SIGNAL_COOLDOWN_SEEN = {}
_LAST_SIGNAL_COOLDOWN_LOCK = threading.Lock()
_SIGNAL_REPEAT_COOLDOWN_SEC = 90.0
_PROFILE_SYMBOL_COOLDOWN_SEEN = {}
_PROFILE_SYMBOL_COOLDOWN_LOCK = threading.Lock()

# Vardan webhook — tracks last sent action per symbol to avoid duplicate fires
_VARDAN_WEBHOOK_URL = "https://sofast.am/webhook/vardan"
_VARDAN_WEBHOOK_TOKEN = "zvuoIhgQ_RVOxoX5m0hCZAeSKV07S70AawKM0xVB53E"
_VARDAN_LAST_SENT: dict = {}
_VARDAN_LAST_SENT_LOCK = threading.Lock()


def _get_open_positions_snapshot_cached():
    now = time.time()
    try:
        cached_ts = float(_OPEN_POS_SNAPSHOT_CACHE.get("ts") or 0.0)
        cached_data = _OPEN_POS_SNAPSHOT_CACHE.get("data")
    except Exception:
        cached_ts = 0.0
        cached_data = None

    try:
        ttl = float(_OPEN_POS_SNAPSHOT_TTL_SEC)
    except Exception:
        ttl = 1.0

    if cached_data is not None and cached_ts > 0 and (now - cached_ts) <= ttl:
        try:
            return list(cached_data)
        except Exception:
            return cached_data

    acquired = False
    try:
        acquired = _OPEN_POS_SNAPSHOT_LOCK.acquire(blocking=False)
    except Exception:
        acquired = False

    if not acquired:
        # Best-effort: return stale cache instead of blocking symbol processing.
        if cached_data is not None:
            try:
                return list(cached_data)
            except Exception:
                return cached_data
        return []

    try:
        now = time.time()
        try:
            cached_ts = float(_OPEN_POS_SNAPSHOT_CACHE.get("ts") or 0.0)
            cached_data = _OPEN_POS_SNAPSHOT_CACHE.get("data")
        except Exception:
            cached_ts = 0.0
            cached_data = None
        if cached_data is not None and cached_ts > 0 and (now - cached_ts) <= ttl:
            try:
                return list(cached_data)
            except Exception:
                return cached_data

        try:
            data = get_open_positions_snapshot()
        except Exception:
            data = []

        try:
            _OPEN_POS_SNAPSHOT_CACHE["ts"] = time.time()
            _OPEN_POS_SNAPSHOT_CACHE["data"] = list(data) if isinstance(data, list) else data
        except Exception:
            pass
        return data
    finally:
        try:
            _OPEN_POS_SNAPSHOT_LOCK.release()
        except Exception:
            pass

# Pending signal follow-ups (1h volatility & 5m activity) per signal.
# Filled when a new signal է գրանցվում, օգտագործվում է
# run_signal_followup_manager()-ի կողմից ֆոնային լուպում.
_PENDING_SIGNAL_FOLLOWUPS = []
_PENDING_SIGNAL_FOLLOWUPS_LOCK = threading.Lock()


_VOL_1H_CACHE = {}
_VOL_1H_CACHE_TTL = 120.0

_VOL_24H_CACHE = {}
_VOL_24H_CACHE_TTL = 30.0


def _futures_1h_volatility(symbol: str):
    now = time.time()
    cached = _VOL_1H_CACHE.get(symbol)
    if cached is not None:
        c_ts, c_val = cached
        if (now - c_ts) < _VOL_1H_CACHE_TTL:
            return c_val

    try:
        candles = load_klines(symbol, "1h", limit=1)
    except Exception as e:
        try:
            log(f"[VOL_1H] Failed to load 1h kline for {symbol}: {e}")
        except Exception:
            pass
        return None

    if not candles:
        return None

    c = candles[0]
    try:
        high_price = float(c.get("high"))
        low_price = float(c.get("low"))
    except (TypeError, ValueError):
        return None

    if low_price <= 0 or high_price <= 0 or high_price < low_price:
        return None

    percent_move = ((high_price - low_price) / low_price) * 100.0
    result = round(percent_move, 2)
    _VOL_1H_CACHE[symbol] = (now, result)
    return result


def _futures_5m_activity(symbol: str):
    try:
        candles = load_klines(symbol, "5m", limit=30)
    except Exception as e:
        try:
            log(f"[ACT_5M] Failed to load 5m klines for {symbol}: {e}")
        except Exception:
            pass
        return None

    if not candles or len(candles) < 2:
        return None

    last_candle = candles[-1]
    hist_candles = candles[:-1]

    try:
        last_trades = last_candle.get("trades")
    except AttributeError:
        last_trades = None

    if not isinstance(last_trades, (int, float)):
        return None

    hist_values = []
    for c in hist_candles:
        t = c.get("trades") if isinstance(c, dict) else None
        if isinstance(t, (int, float)) and t > 0:
            hist_values.append(float(t))

    if not hist_values:
        return None

    avg_trades = sum(hist_values) / len(hist_values)
    if avg_trades <= 0:
        return None

    ratio = float(last_trades) / avg_trades

    if ratio < 0.5:
        status = "VERY LOW"
    elif ratio < 0.8:
        status = "LOW"
    elif ratio <= 1.2:
        status = "NORMAL"
    elif ratio <= 1.8:
        status = "HIGH"
    else:
        status = "VERY HIGH"

    return {
        "last_trades": int(last_trades),
        "avg_trades": avg_trades,
        "ratio": ratio,
        "status": status,
    }


def _futures_24h_quote_volume_usdt(symbol: str):
    sym = str(symbol or "").upper()
    if not sym:
        return None

    now = time.time()
    cached = _VOL_24H_CACHE.get(sym)
    if cached is not None:
        c_ts, c_val = cached
        if (now - c_ts) < _VOL_24H_CACHE_TTL:
            return c_val

    try:
        clients = _get_clients() or []
        if not clients:
            return None
        client = clients[0]
        data = client._request("GET", "/fapi/v1/ticker/24hr", {"symbol": sym})
        qv = data.get("quoteVolume") if isinstance(data, dict) else None
        qv_val = float(qv)
        if qv_val < 0:
            return None
        _VOL_24H_CACHE[sym] = (now, qv_val)
        return qv_val
    except Exception as e:
        try:
            log(f"[VOL_24H] Failed to fetch 24h quote volume for {sym}: {e}")
        except Exception:
            pass
        return None


def _format_usdt_volume(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.2f}K"
    return f"{v:.2f}"


def _passes_24h_volume_gate(symbol: str):
    threshold = float(MIN_24H_QUOTE_VOLUME_USDT)
    if threshold <= 0:
        return True, _futures_24h_quote_volume_usdt(symbol), threshold
    vol_24h = _futures_24h_quote_volume_usdt(symbol)
    if vol_24h is None:
        return False, None, threshold
    return bool(float(vol_24h) >= threshold), float(vol_24h), threshold

# Canonical signal SL/TP config (for Telegram + analytics)
signals_cfg = config.get("signals", {})
try:
    SIGNAL_SL_PCT = float(signals_cfg.get("sl_pct", 2.0))
except (TypeError, ValueError):
    SIGNAL_SL_PCT = 2.0

raw_signal_tp_pcts = signals_cfg.get("tp_pcts", [1.0, 2.0, 3.0])
SIGNAL_TP_PCTS = []
if isinstance(raw_signal_tp_pcts, (list, tuple)):
    for v in raw_signal_tp_pcts:
        try:
            pv = float(v)
            if pv > 0:
                SIGNAL_TP_PCTS.append(pv)
        except (TypeError, ValueError):
            continue
if not SIGNAL_TP_PCTS:
    SIGNAL_TP_PCTS = [1.0, 2.0, 3.0]

# Per-account overrides (config/binance_accounts.yaml -> accounts[0].settings),
# starting from the canonical signal config but allowing different risk per
# Binance account for actual orders.
ACCOUNT_SL_PCT = SIGNAL_SL_PCT
ACCOUNT_TP_PCTS = list(SIGNAL_TP_PCTS)
ACCOUNT_FIXED_NOTIONAL_USD = None
ACCOUNT_ENTRY_OFFSET_PCT = 0.0
ACCOUNT_ENTRY_TIMEOUT_MIN = 0

_ACCOUNTS_CFG_PATH = os.path.join("config", "binance_accounts.yaml")
_ACCOUNTS_CFG_MTIME_MAIN = None
_ACCOUNTS_CFG_CACHE_MAIN = None

_MARKET_REGIME_CFG_PATH = os.path.join("config", "market_regime.yaml")
_MARKET_REGIME_CFG_MTIME_MAIN = None
_MARKET_REGIME_CFG_CACHE_MAIN = None
_MARKET_REGIME_PROCESS: Optional[subprocess.Popen] = None
_MARKET_REGIME_LOG_HANDLE = None
_MARKET_REGIME_LOCK = threading.Lock()
_MARKET_REGIME_EXTERNAL_PID: Optional[int] = None
_MARKET_REGIME_SKIP_NOTIFY_CACHE: dict[tuple, float] = {}

# ── Market Quality Analyzer sidecar ──────────────────────────────────────────
_MQA_PROCESS: Optional[subprocess.Popen] = None
_MQA_LOG_HANDLE = None
_MQA_LOCK = threading.Lock()


def _start_market_quality_analyzer_if_needed():
    """Start market_quality_analyzer.py as an independent sidecar process.

    Observation-only by default (BLOCKING_ENABLED=False in the script).
    To activate trade blocking, set BLOCKING_ENABLED=True inside
    market_quality_analyzer.py and restart.
    """
    global _MQA_PROCESS, _MQA_LOG_HANDLE
    with _MQA_LOCK:
        if _MQA_PROCESS is not None and _MQA_PROCESS.poll() is None:
            return  # already running
        script_path = os.path.join(_ROOT_DIR, "market_quality_analyzer.py")
        if not os.path.exists(script_path):
            try:
                log("[MQA] market_quality_analyzer.py not found; skipping")
            except Exception:
                pass
            return
        try:
            logs_dir = os.path.join(_ROOT_DIR, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(logs_dir, "market_quality_analyzer.log")
            if _MQA_LOG_HANDLE is None:
                _MQA_LOG_HANDLE = open(log_path, "a", encoding="utf-8")
            python_exe = sys.executable or "python3"
            _MQA_PROCESS = subprocess.Popen(
                [python_exe, script_path],
                cwd=_ROOT_DIR,
                stdout=_MQA_LOG_HANDLE,
                stderr=_MQA_LOG_HANDLE,
            )
            try:
                log(f"[MQA] Market Quality Analyzer started (pid={_MQA_PROCESS.pid})")
            except Exception:
                pass
        except Exception as e:
            try:
                log(f"[MQA] Failed to start: {e}")
            except Exception:
                pass


def _mqa_watchdog_loop():
    """Restart market_quality_analyzer if it dies."""
    while True:
        try:
            _start_market_quality_analyzer_if_needed()
        except Exception:
            pass
        time.sleep(30.0)


def _market_quality_line(symbol: str) -> Optional[str]:
    """Read market quality verdict for symbol from cache. Returns a single info line or None."""
    try:
        cache_path = os.path.join(_ROOT_DIR, "data", "market_quality_cache.json")
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        entry = cache.get(str(symbol))
        if not entry or not isinstance(entry, dict):
            return None
        verdict = str(entry.get("verdict") or "")
        score   = entry.get("score")
        fib     = entry.get("fib_score")
        mdl     = entry.get("model_score")
        ob      = entry.get("ob_score")
        if not verdict or score is None:
            return None
        icon = {"GOOD": "✅", "SUSPECT": "⚠️", "MANIPULATED": "🚫"}.get(verdict, "❓")
        return (
            f"MQ: {icon}{verdict} score={score:.0f}/100"
            f" (fib={fib:.0f} mdl={mdl:.0f} book={ob:.0f})"
        )
    except Exception:
        return None
# ─────────────────────────────────────────────────────────────────────────────


def _get_accounts_cfg_live():
    """Read binance_accounts.yaml with lightweight mtime cache."""
    global _ACCOUNTS_CFG_MTIME_MAIN, _ACCOUNTS_CFG_CACHE_MAIN
    try:
        mtime = os.path.getmtime(_ACCOUNTS_CFG_PATH)
    except Exception:
        mtime = None

    if (
        mtime is not None
        and _ACCOUNTS_CFG_MTIME_MAIN is not None
        and _ACCOUNTS_CFG_CACHE_MAIN is not None
        and float(mtime) == float(_ACCOUNTS_CFG_MTIME_MAIN)
    ):
        return _ACCOUNTS_CFG_CACHE_MAIN

    try:
        if YAML is not None:
            with open(_ACCOUNTS_CFG_PATH) as f_acc:
                cfg = _yaml.load(f_acc)
        else:
            with open(_ACCOUNTS_CFG_PATH) as f_acc:
                cfg = _pyyaml.safe_load(f_acc)
    except Exception:
        cfg = None

    _ACCOUNTS_CFG_CACHE_MAIN = cfg if isinstance(cfg, dict) else None
    _ACCOUNTS_CFG_MTIME_MAIN = mtime
    return _ACCOUNTS_CFG_CACHE_MAIN


def _get_market_regime_cfg_live():
    """Read config/market_regime.yaml with lightweight mtime cache."""
    global _MARKET_REGIME_CFG_MTIME_MAIN, _MARKET_REGIME_CFG_CACHE_MAIN
    try:
        mtime = os.path.getmtime(_MARKET_REGIME_CFG_PATH)
    except Exception:
        mtime = None

    if (
        mtime is not None
        and _MARKET_REGIME_CFG_MTIME_MAIN is not None
        and _MARKET_REGIME_CFG_CACHE_MAIN is not None
        and float(mtime) == float(_MARKET_REGIME_CFG_MTIME_MAIN)
    ):
        return _MARKET_REGIME_CFG_CACHE_MAIN

    try:
        if YAML is not None:
            with open(_MARKET_REGIME_CFG_PATH) as f_mr:
                cfg = _yaml.load(f_mr)
        else:
            with open(_MARKET_REGIME_CFG_PATH) as f_mr:
                cfg = _pyyaml.safe_load(f_mr)
    except Exception:
        cfg = None

    _MARKET_REGIME_CFG_CACHE_MAIN = cfg if isinstance(cfg, dict) else None
    _MARKET_REGIME_CFG_MTIME_MAIN = mtime
    return _MARKET_REGIME_CFG_CACHE_MAIN


def _get_market_regime_snapshot() -> dict:
    """Read latest market regime state (best-effort)."""
    cfg = _get_market_regime_cfg_live()
    data_path = os.path.join("data", "market_regime_state.json")
    try:
        if isinstance(cfg, dict):
            rt = cfg.get("runtime")
            if isinstance(rt, dict) and rt.get("data_path"):
                data_path = str(rt.get("data_path"))
    except Exception:
        pass

    path_abs = data_path if os.path.isabs(data_path) else os.path.join(_ROOT_DIR, data_path)
    try:
        with open(path_abs, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _market_regime_human_line() -> tuple[str, str]:
    """Return short human-readable market regime line and raw regime."""
    st = _get_market_regime_snapshot()
    regime = str(st.get("last_regime") or "UNKNOWN").upper().strip()
    score = st.get("last_score")
    try:
        score_txt = f"{float(score):.3f}"
    except Exception:
        score_txt = "-"

    if regime == "ACTIVE":
        return f"🟢 Շուկան՝ ԿԱՆԱՉ (ACTIVE), score={score_txt}", regime
    if regime == "NEUTRAL":
        return f"🟡 Շուկան՝ ԴԵՂԻՆ (NEUTRAL), score={score_txt}", regime
    if regime == "QUIET":
        return f"🔴 Շուկան՝ ԿԱՐՄԻՐ (QUIET), score={score_txt}", regime
    return f"⚪ Շուկան՝ անհայտ, score={score_txt}", regime


def _start_market_regime_service_if_needed():
    """Run isolated market_regime.py as a sidecar process.

    This service is signal-only and does NOT touch trading/entry/exit logic.
    """
    global _MARKET_REGIME_PROCESS, _MARKET_REGIME_LOG_HANDLE, _MARKET_REGIME_EXTERNAL_PID
    cfg = _get_market_regime_cfg_live()
    enabled = True
    if isinstance(cfg, dict):
        enabled = bool(cfg.get("enabled", True))

    with _MARKET_REGIME_LOCK:
        # Clean up orphan/legacy regime processes to avoid duplicate messages.
        try:
            keep_pids = set()
            if _MARKET_REGIME_PROCESS is not None:
                try:
                    if _MARKET_REGIME_PROCESS.poll() is None:
                        keep_pids.add(int(_MARKET_REGIME_PROCESS.pid))
                except Exception:
                    pass
            if _MARKET_REGIME_EXTERNAL_PID is not None:
                try:
                    keep_pids.add(int(_MARKET_REGIME_EXTERNAL_PID))
                except Exception:
                    pass

            for p in os.listdir("/proc"):
                if not p.isdigit():
                    continue
                pid_i = int(p)
                if pid_i in keep_pids or pid_i == os.getpid():
                    continue
                try:
                    with open(f"/proc/{p}/cmdline", "rb") as f_cmd:
                        cmd = f_cmd.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
                except Exception:
                    continue
                if "market_regime.py" not in cmd:
                    continue
                try:
                    os.kill(pid_i, signal.SIGTERM)
                except Exception:
                    continue
        except Exception:
            pass

        is_running = (
            _MARKET_REGIME_PROCESS is not None
            and _MARKET_REGIME_PROCESS.poll() is None
        )

        if not enabled:
            if is_running:
                try:
                    _MARKET_REGIME_PROCESS.terminate()
                except Exception:
                    pass
            _MARKET_REGIME_PROCESS = None
            return

        if is_running:
            return

        script_path = os.path.join(_ROOT_DIR, "market_regime.py")
        if not os.path.exists(script_path):
            try:
                log("[MARKET REGIME] market_regime.py not found; skipping start")
            except Exception:
                pass
            return

        # If another healthy regime process already owns the shared pid file,
        # do not start a duplicate local child.
        try:
            runtime_cfg = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
            pid_path = runtime_cfg.get("pid_path") or os.path.join("data", "market_regime.pid")
            pid_abs = pid_path if os.path.isabs(str(pid_path)) else os.path.join(_ROOT_DIR, str(pid_path))
            ext_pid = None
            if os.path.exists(pid_abs):
                raw = ""
                try:
                    with open(pid_abs, "r", encoding="utf-8") as f_pid:
                        raw = (f_pid.read() or "").strip()
                except Exception:
                    raw = ""
                if raw:
                    try:
                        ext_pid = int(raw)
                    except Exception:
                        ext_pid = None
            if ext_pid and ext_pid > 0:
                alive = False
                try:
                    os.kill(int(ext_pid), 0)
                    alive = True
                except Exception:
                    alive = False
                if alive:
                    _MARKET_REGIME_EXTERNAL_PID = int(ext_pid)
                    return
        except Exception:
            pass

        try:
            logs_dir = os.path.join(_ROOT_DIR, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(logs_dir, "market_regime.log")
            if _MARKET_REGIME_LOG_HANDLE is None:
                _MARKET_REGIME_LOG_HANDLE = open(log_path, "a", encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("MARKET_REGIME_CONFIG", _MARKET_REGIME_CFG_PATH)
            _MARKET_REGIME_PROCESS = subprocess.Popen(
                [sys.executable, script_path],
                cwd=_ROOT_DIR,
                env=env,
                stdout=_MARKET_REGIME_LOG_HANDLE,
                stderr=_MARKET_REGIME_LOG_HANDLE,
            )
            # If process exits immediately (e.g. singleton already running),
            # do not report successful start and avoid watchdog restart spam.
            time.sleep(0.25)
            if _MARKET_REGIME_PROCESS.poll() is not None:
                _MARKET_REGIME_PROCESS = None
                return
            try:
                pid_val = int(_MARKET_REGIME_PROCESS.pid)
            except Exception:
                pid_val = -1
            log(f"[MARKET REGIME] Sidecar started (pid={pid_val})")
        except Exception as e:
            _MARKET_REGIME_PROCESS = None
            try:
                log(f"[MARKET REGIME] Failed to start sidecar: {e}")
            except Exception:
                pass


def _market_regime_watchdog_loop():
    """Ensure market_regime sidecar follows config enable/disable changes."""
    while True:
        try:
            _start_market_regime_service_if_needed()
        except Exception as e:
            try:
                log(f"[MARKET REGIME] Watchdog loop error: {e}")
            except Exception:
                pass
        time.sleep(20.0)


accounts_cfg = _get_accounts_cfg_live()

if isinstance(accounts_cfg, dict):
    accounts_list = accounts_cfg.get("accounts") or []
    if isinstance(accounts_list, list) and accounts_list:
        first_acc = accounts_list[0]
        if isinstance(first_acc, dict):
            settings = first_acc.get("settings") or {}
            if isinstance(settings, dict):
                # tp_mode override (1,2,3,4)
                try:
                    tp_mode_val = int(settings.get("tp_mode", TP_MODE))
                    if tp_mode_val in (1, 2, 3, 4):
                        TP_MODE = tp_mode_val
                except (TypeError, ValueError):
                    pass

                # Stop-loss distance in percent
                try:
                    sl_pct_val = settings.get("sl_pct", ACCOUNT_SL_PCT)
                    sl_pct_val = float(sl_pct_val)
                    if sl_pct_val > 0:
                        ACCOUNT_SL_PCT = sl_pct_val
                except (TypeError, ValueError):
                    pass

                # TP distances in percent list
                raw_tp_pcts = settings.get("tp_pcts", ACCOUNT_TP_PCTS)
                if isinstance(raw_tp_pcts, (list, tuple)):
                    cleaned = []
                    for v in raw_tp_pcts:
                        try:
                            pv = float(v)
                            if pv > 0:
                                cleaned.append(pv)
                        except (TypeError, ValueError):
                            continue
                    if cleaned:
                        ACCOUNT_TP_PCTS = cleaned

                # Fixed notional per trade in USDT (optional)
                try:
                    fixed_notional_val = settings.get("fixed_notional_usd")
                    if fixed_notional_val is not None:
                        ACCOUNT_FIXED_NOTIONAL_USD = float(fixed_notional_val)
                except (TypeError, ValueError):
                    ACCOUNT_FIXED_NOTIONAL_USD = None

                # Optional dynamic entry configuration (offset + timeout)
                try:
                    entry_offset_val = settings.get("entry_offset_pct")
                    if entry_offset_val is not None:
                        ACCOUNT_ENTRY_OFFSET_PCT = float(entry_offset_val)
                except (TypeError, ValueError):
                    ACCOUNT_ENTRY_OFFSET_PCT = 0.0

                try:
                    entry_timeout_val = settings.get("entry_timeout_min")
                    if entry_timeout_val is not None:
                        ACCOUNT_ENTRY_TIMEOUT_MIN = int(entry_timeout_val)
                except (TypeError, ValueError):
                    ACCOUNT_ENTRY_TIMEOUT_MIN = 0

filters_cfg = config.get("filters", {})
MIN_5M_MOVE_PCT = float(filters_cfg.get("min_5m_move_pct", 2.0))
MIN_1M_NOTIONAL_USD = float(filters_cfg.get("min_1m_notional_usd", 20000.0))
MIN_24H_QUOTE_VOLUME_USDT = float(filters_cfg.get("min_24h_quote_volume_usdt", 20000000.0))
MIN_1H_MOVE_PCT = float(filters_cfg.get("min_1h_move_pct", 0.0))
MIN_5M_TRADES = int(filters_cfg.get("min_5m_trades", 0))

signal_repeat_guard_cfg = config.get("signal_repeat_guard") or {}
if not isinstance(signal_repeat_guard_cfg, dict):
    signal_repeat_guard_cfg = {}
SMALL_SIGNAL_SYMBOL_COOLDOWN_ENABLED = bool(signal_repeat_guard_cfg.get("small_enabled", False))
try:
    SMALL_SIGNAL_SYMBOL_COOLDOWN_SEC = max(
        0.0,
        float(signal_repeat_guard_cfg.get("small_cooldown_minutes", 30.0)) * 60.0,
    )
except Exception:
    SMALL_SIGNAL_SYMBOL_COOLDOWN_SEC = 30.0 * 60.0

auto_ml_cfg = config.get("auto_ml", {})
AUTO_ML_ENABLED = bool(auto_ml_cfg.get("enabled", False))
AUTO_ML_INTERVAL_MINUTES = float(auto_ml_cfg.get("interval_minutes", 10.0))
AUTO_ML_INTERVAL_SECONDS = int(AUTO_ML_INTERVAL_MINUTES * 60)

news_guard_cfg = config.get("news_guard") or {}
NEWS_GUARD_ENABLED = bool(news_guard_cfg.get("enabled", False))

update_cfg = config.get("update") or {}
UPDATE_BASE_URL = update_cfg.get("base_url") or None
UPDATE_INCLUDE_CONFIG = bool(update_cfg.get("include_config", False))
UPDATE_INCLUDE_DATA = bool(update_cfg.get("include_data", False))

symbols_cfg = config.get("symbols")
if isinstance(symbols_cfg, list):
    SYMBOLS = symbols_cfg
elif symbols_cfg == "ALL_FUTURES" or symbols_cfg is None:
    SYMBOLS = get_all_usdt_futures()
else:
    SYMBOLS = [symbols_cfg]

# Optional secondary (LARGE) signal profile configuration.
SECONDARY_SIGNALS_CFG_PATH = os.path.join("config", "secondary_signals.yaml")
SECONDARY_SIGNALS_CFG = {}
try:
    if os.path.exists(SECONDARY_SIGNALS_CFG_PATH):
        if YAML is not None:
            with open(SECONDARY_SIGNALS_CFG_PATH) as f_sec:
                SECONDARY_SIGNALS_CFG = _yaml.load(f_sec) or {}
        else:
            with open(SECONDARY_SIGNALS_CFG_PATH) as f_sec:
                SECONDARY_SIGNALS_CFG = _pyyaml.safe_load(f_sec) or {}
except Exception:
    SECONDARY_SIGNALS_CFG = {}
if not isinstance(SECONDARY_SIGNALS_CFG, dict):
    SECONDARY_SIGNALS_CFG = {}

SECONDARY_SIGNALS_ENABLED = bool(SECONDARY_SIGNALS_CFG.get("enabled", False))
SECONDARY_PROFILE_NAME = str(SECONDARY_SIGNALS_CFG.get("profile_name") or "LARGE").strip().upper()
_secondary_symbols_raw = SECONDARY_SIGNALS_CFG.get("symbols") or []
if isinstance(_secondary_symbols_raw, list):
    SECONDARY_SYMBOLS = [str(s).upper().strip() for s in _secondary_symbols_raw if str(s).strip()]
else:
    SECONDARY_SYMBOLS = []
_secondary_filters = SECONDARY_SIGNALS_CFG.get("filters") or {}
if not isinstance(_secondary_filters, dict):
    _secondary_filters = {}
SECONDARY_MIN_5M_MOVE_PCT = float(_secondary_filters.get("min_5m_move_pct", 0.6))
SECONDARY_MIN_1M_NOTIONAL_USD = float(_secondary_filters.get("min_1m_notional_usd", 150000.0))
SECONDARY_USE_1H_MOVE_FILTER = bool(_secondary_filters.get("use_1h_move_filter", False))
SECONDARY_MIN_1H_MOVE_PCT = float(_secondary_filters.get("min_1h_move_pct", 1.5))
_secondary_telegram = SECONDARY_SIGNALS_CFG.get("telegram") or {}
if not isinstance(_secondary_telegram, dict):
    _secondary_telegram = {}
SECONDARY_USE_MAIN_CHANNEL = bool(_secondary_telegram.get("use_main_channel", True))
SECONDARY_TELEGRAM_TOKEN = _secondary_telegram.get("token") or None
SECONDARY_TELEGRAM_CHAT_ID = _secondary_telegram.get("chat_id") or None
SECONDARY_TELEGRAM_PREFIX = str(_secondary_telegram.get("prefix") or "[LARGE]").strip()
SECONDARY_START_SUBSCRIBERS_PATH = os.path.join(_ROOT_DIR, "data", "secondary_telegram_start_subscribers.json")
_secondary_signal_levels = SECONDARY_SIGNALS_CFG.get("signal_levels") or {}
if not isinstance(_secondary_signal_levels, dict):
    _secondary_signal_levels = {}
try:
    SECONDARY_SIGNAL_SL_PCT = float(_secondary_signal_levels.get("sl_pct", SIGNAL_SL_PCT))
except Exception:
    SECONDARY_SIGNAL_SL_PCT = SIGNAL_SL_PCT
SECONDARY_SIGNAL_TP_PCTS = []
_secondary_tp_raw = _secondary_signal_levels.get("tp_pcts", SIGNAL_TP_PCTS)
if isinstance(_secondary_tp_raw, (list, tuple)):
    for _v in _secondary_tp_raw:
        try:
            _pv = float(_v)
            if _pv > 0:
                SECONDARY_SIGNAL_TP_PCTS.append(_pv)
        except Exception:
            continue
if not SECONDARY_SIGNAL_TP_PCTS:
    SECONDARY_SIGNAL_TP_PCTS = list(SIGNAL_TP_PCTS)
_secondary_repeat_guard = SECONDARY_SIGNALS_CFG.get("signal_repeat_guard") or {}
if not isinstance(_secondary_repeat_guard, dict):
    _secondary_repeat_guard = {}
SECONDARY_SIGNAL_SYMBOL_COOLDOWN_ENABLED = bool(_secondary_repeat_guard.get("enabled", False))
try:
    SECONDARY_SIGNAL_SYMBOL_COOLDOWN_SEC = max(
        0.0,
        float(_secondary_repeat_guard.get("cooldown_minutes", 30.0)) * 60.0,
    )
except Exception:
    SECONDARY_SIGNAL_SYMBOL_COOLDOWN_SEC = 30.0 * 60.0


def _resolve_static_chat_id(raw_chat_id) -> int | None:
    try:
        if raw_chat_id is None:
            return None
        sval = str(raw_chat_id).strip()
        if sval in ("", "0", "-"):
            return None
        return int(sval)
    except Exception:
        return None


def _get_symbol_repeat_guard_for_profile(profile_label: str) -> tuple[bool, float]:
    p = str(profile_label or "").strip().lower()
    if p == "large":
        return SECONDARY_SIGNAL_SYMBOL_COOLDOWN_ENABLED, SECONDARY_SIGNAL_SYMBOL_COOLDOWN_SEC
    return SMALL_SIGNAL_SYMBOL_COOLDOWN_ENABLED, SMALL_SIGNAL_SYMBOL_COOLDOWN_SEC


def _load_secondary_start_chat_ids() -> list[int]:
    ids: list[int] = []
    try:
        if not os.path.exists(SECONDARY_START_SUBSCRIBERS_PATH):
            return []
        with open(SECONDARY_START_SUBSCRIBERS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_ids = payload.get("chat_ids") if isinstance(payload, dict) else []
        if not isinstance(raw_ids, list):
            return []
        seen = set()
        for x in raw_ids:
            try:
                cid = int(x)
            except Exception:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
    except Exception:
        return []
    return ids


def _coerce_bool_cfg(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    try:
        v = str(value).strip().lower()
    except Exception:
        return bool(default)
    if v in ("1", "true", "yes", "on", "enabled"):
        return True
    if v in ("0", "false", "no", "off", "disabled"):
        return False
    return bool(default)


def _coerce_float_cfg(value, default):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_base_initial_qty(account_index: int, symbol: str, entry_price: float, current_abs_qty: float) -> float:
    """Estimate core/base position size for restart-safe state recovery.

    On restarts, open position may already include scale-in adds.
    We recover base size from account fixed-notional config when available.
    """
    try:
        cur_abs = abs(float(current_abs_qty))
    except (TypeError, ValueError):
        return 0.0
    if cur_abs <= 0:
        return 0.0
    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        ep = 0.0
    if ep <= 0:
        return cur_abs

    notional = None
    try:
        cfg_live = _get_accounts_cfg_live() or {}
        accs = cfg_live.get("accounts") or []
        if isinstance(accs, list) and 0 <= int(account_index) < len(accs):
            acc = accs[int(account_index)]
            if isinstance(acc, dict):
                st = acc.get("settings") or {}
                if isinstance(st, dict):
                    raw = st.get("fixed_notional_usd")
                    if raw is not None:
                        notional = float(raw)
                    if (notional is None or notional <= 0.0):
                        ftype = str(st.get("fixed_notional_type") or "").upper()
                        fval = st.get("fixed_notional_value")
                        if fval is not None and ftype == "USDT":
                            notional = float(fval)
                        if (notional is None or notional <= 0.0):
                            # For restart-safe adaptive recovery we still need
                            # base USDT even when room_sizing_enabled is off.
                            rs_base = st.get("room_sizing_base_usdt")
                            if rs_base is not None:
                                notional = float(rs_base)
    except Exception:
        notional = None

    if notional is None or notional <= 0.0:
        return cur_abs

    try:
        q = float(adjust_quantity(symbol, float(notional) / ep, ep))
    except Exception:
        q = 0.0
    if q <= 0.0:
        return cur_abs
    # Base qty must never exceed current open qty.
    q = min(cur_abs, q)
    # If recovered base is unrealistically tiny, fall back to current.
    if q < cur_abs * 0.05:
        return cur_abs
    return q


def _resolve_signal_message_settings(account_index, account_settings: dict | None) -> dict:
    """Resolve signal message display settings (global + per-account override)."""
    resolved = {
        "include_level_age": True,
        "level_age_max_minutes": None,
        "include_leverage": True,
        "include_confidence": True,
        "include_last_1h_range": True,
        "include_last_5m_trades": True,
        "include_signal_source": True,
    }

    cfg_live = _get_accounts_cfg_live()
    global_block = {}
    if isinstance(cfg_live, dict):
        g = cfg_live.get("global_settings") or {}
        if isinstance(g, dict):
            sm = g.get("signal_message") or {}
            if isinstance(sm, dict):
                global_block = sm

    acct_block = {}
    settings_local = account_settings if isinstance(account_settings, dict) else {}
    if isinstance(settings_local, dict):
        sm = settings_local.get("signal_message") or {}
        if isinstance(sm, dict):
            acct_block = sm

    # Backward-compatible flat account keys (if user prefers simple keys).
    legacy_block = {}
    if isinstance(settings_local, dict):
        for k in (
            "include_level_age",
            "level_age_max_minutes",
            "include_leverage",
            "include_confidence",
            "include_last_1h_range",
            "include_last_5m_trades",
            "include_signal_source",
        ):
            legacy_key = f"signal_{k}"
            if legacy_key in settings_local:
                legacy_block[k] = settings_local.get(legacy_key)

    merged = dict(resolved)
    merged.update(global_block)
    merged.update(acct_block)
    merged.update(legacy_block)

    resolved["include_level_age"] = _coerce_bool_cfg(merged.get("include_level_age"), True)
    lam = _coerce_float_cfg(merged.get("level_age_max_minutes"), None)
    if lam is not None and lam <= 0:
        lam = None
    resolved["level_age_max_minutes"] = lam
    resolved["include_leverage"] = _coerce_bool_cfg(merged.get("include_leverage"), True)
    resolved["include_confidence"] = _coerce_bool_cfg(merged.get("include_confidence"), True)
    resolved["include_last_1h_range"] = _coerce_bool_cfg(merged.get("include_last_1h_range"), True)
    resolved["include_last_5m_trades"] = _coerce_bool_cfg(merged.get("include_last_5m_trades"), True)
    resolved["include_signal_source"] = _coerce_bool_cfg(merged.get("include_signal_source"), True)
    return resolved


def _resolve_negative_alert_mode() -> str:
    """Global policy for negative/skip Telegram messages: all|critical_only|off."""
    cfg_live = _get_accounts_cfg_live()
    mode = "all"
    try:
        if isinstance(cfg_live, dict):
            g = cfg_live.get("global_settings") or {}
            if isinstance(g, dict):
                block = g.get("signal_notifications") or {}
                if isinstance(block, dict):
                    raw = block.get("negative_alerts_mode")
                    if raw is not None:
                        mode = str(raw).strip().lower()
    except Exception:
        mode = "all"
    if mode in ("off", "disabled", "none"):
        return "off"
    if mode in ("critical", "critical_only", "only_critical"):
        return "critical_only"
    return "all"


def _is_critical_negative_reason(reason_text: str) -> bool:
    try:
        txt = str(reason_text or "").strip().lower()
    except Exception:
        return False
    if not txt:
        return False
    critical_markers = (
        "error",
        "exception",
        "failed",
        "timeout",
        "traceback",
        "internal",
        "unavailable",
        "429",
        "5xx",
    )
    return any(m in txt for m in critical_markers)

# Telegram setup (env overrides config)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or config.get("telegram_token", None)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or config.get("telegram_chat_id", None)

control_bot_cfg = config.get("control_bot") or {}
CONTROL_BOT_TOKEN = control_bot_cfg.get("token") or None

accounts_bot_cfg = config.get("accounts_bot") or {}
ACCOUNTS_BOT_TOKEN = accounts_bot_cfg.get("token") or None

reports_bot_cfg = config.get("reports_bot") or {}
REPORTS_BOT_TOKEN = reports_bot_cfg.get("token") or None

analytics_bot_cfg = config.get("analytics_bot") or {}
ANALYTICS_BOT_TOKEN = analytics_bot_cfg.get("token") or None

# Optional hard-coded overrides for client builds. In the admin/dev environment
# these stay as None and we fall back to config/env values. The instance
# factory patches them per-client before obfuscation so that license
# notifications can still reach the admin even though client trading.yaml does
# not contain Telegram tokens or chat IDs.
_LICENSE_TOKEN_OVERRIDE = None  # PATCH_LICENSE_TOKEN
_LICENSE_CHAT_ID_OVERRIDE = None  # PATCH_LICENSE_CHAT_ID

try:
    _ADMIN_LICENSE_TOKEN = _LICENSE_TOKEN_OVERRIDE or CONTROL_BOT_TOKEN or TELEGRAM_TOKEN
    _ADMIN_LICENSE_CHAT_ID = None
    _raw_chat_id = _LICENSE_CHAT_ID_OVERRIDE or TELEGRAM_CHAT_ID
    if _raw_chat_id is not None:
        try:
            _ADMIN_LICENSE_CHAT_ID = int(_raw_chat_id)
        except (TypeError, ValueError):
            _ADMIN_LICENSE_CHAT_ID = None

    _ALLOWED_TO_RUN = register_startup(
        "main.py",
        _ADMIN_LICENSE_TOKEN,
        _ADMIN_LICENSE_CHAT_ID,
        license_base_url=UPDATE_BASE_URL,
    )
    if not _ALLOWED_TO_RUN:
        try:
            log(
                f"[LICENSE] Instance {INSTANCE_ID} is blocked or deleted; main.py will not start trading."
            )
        except Exception:
            pass
        raise SystemExit(1)
except Exception:
    _ALLOWED_TO_RUN = True

try:
    if os.environ.get("LICENSE_STATUS") == "paused":
        TRADING_ENABLED = False
except Exception:
    pass

# Initialize ML engine
ml_engine = InferenceEngine()


_NEWS_GUARD_SEND_FAIL_CACHE = {}
_ENTRY_OPENED_BROADCAST_ENABLED = (
    str(os.getenv("ENTRY_OPENED_BROADCAST_ENABLED", "0")).strip().lower()
    in ("1", "true", "yes", "on")
)


def _broadcast_entry_opened_message(text: str) -> None:
    # Keep this notification channel disabled by default.
    # It created noisy "ENTRY OPENED" Telegram messages across multiple bots.
    if not _ENTRY_OPENED_BROADCAST_ENABLED:
        return

    if not isinstance(text, str) or not text.strip():
        return

    try:
        subs = get_subscribers(None, token=TELEGRAM_TOKEN)
    except Exception:
        subs = []

    targets = list(subs) if subs else []

    # Exclude admin chat_id (configured telegram_chat_id)
    try:
        admin_cid = int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID is not None else None
    except Exception:
        admin_cid = None
    if admin_cid is not None:
        try:
            targets = [cid for cid in targets if str(cid) != str(admin_cid)]
        except Exception:
            pass

    if not targets:
        return

    tokens = []
    for t in (TELEGRAM_TOKEN, ACCOUNTS_BOT_TOKEN, REPORTS_BOT_TOKEN, ANALYTICS_BOT_TOKEN):
        if t:
            tokens.append(str(t))
    # De-dup while preserving order
    seen = set()
    tokens = [x for x in tokens if not (x in seen or seen.add(x))]

    if not tokens:
        return

    def _send_with_token(tok: str) -> None:
        for cid in targets:
            try:
                send_telegram(text, tok, int(cid))
            except Exception:
                continue

    for tok in tokens:
        try:
            threading.Thread(target=_send_with_token, args=(tok,), daemon=True).start()
        except Exception:
            continue


def _broadcast_news_guard_message(text: str, pretty_text: str | None = None) -> None:
    """Send NEWS GUARD status to logs, signal subscribers, and control bot.

    - Always logs via monitoring.logger.log (which itself forwards to the log bot).
    - If TELEGRAM_TOKEN is configured, sends to all signal subscribers.
    - If CONTROL_BOT_TOKEN is configured, sends to the control bot chat.
    """

    # 1) Local log (also goes to log bot via monitoring/logger.py)
    try:
        log(text)
    except Exception:
        pass

    chat_text = pretty_text if isinstance(pretty_text, str) and pretty_text.strip() else text

    # 2) Broadcast to main signal subscribers
    if TELEGRAM_TOKEN:
        static_chat_id = _resolve_static_chat_id(TELEGRAM_CHAT_ID)
        try:
            subscribers = get_subscribers(static_chat_id, token=TELEGRAM_TOKEN)
        except Exception as e:
            try:
                log(f"[NEWS GUARD] Failed to load Telegram subscribers: {e}")
            except Exception:
                pass
            subscribers = []

        targets = list(subscribers)
        if not targets and static_chat_id is not None:
            targets = [static_chat_id]
        if not targets:
            return
        for chat_id in targets:
            try:
                send_telegram(chat_text, TELEGRAM_TOKEN, chat_id)
            except Exception as te:
                # If a user blocked the bot (403 Forbidden), remove them from
                # the subscriber list so we don't spam logs forever.
                err = str(te)
                if "403" in err and "blocked" in err.lower():
                    try:
                        if remove_subscriber(chat_id, token=TELEGRAM_TOKEN):
                            log(f"[NEWS GUARD] Removed blocked subscriber {chat_id}")
                    except Exception:
                        pass
                    continue

                # Suppress repeated noise for the same chat_id (best-effort)
                try:
                    now_ts = time.time()
                    last_ts = float(_NEWS_GUARD_SEND_FAIL_CACHE.get(int(chat_id)) or 0.0)
                    if last_ts and (now_ts - last_ts) < 3600.0:
                        continue
                    _NEWS_GUARD_SEND_FAIL_CACHE[int(chat_id)] = now_ts
                except Exception:
                    pass
                try:
                    log(f"[NEWS GUARD] Failed to send news status to {chat_id}: {te}")
                except Exception:
                    pass

    # 3) Send to control bot chat as well (uses same chat_id by design)
    if CONTROL_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram(chat_text, CONTROL_BOT_TOKEN, TELEGRAM_CHAT_ID)
        except Exception as te:
            try:
                log(f"[NEWS GUARD] Failed to send news status to control bot: {te}")
            except Exception:
                pass


def _normalize_signal_source(raw: object) -> str:
    try:
        v = str(raw).strip().lower() if raw is not None else "small"
    except Exception:
        v = "small"
    if v in ("both", "all"):
        return "both"
    if v in ("large",):
        return "large"
    return "small"


def _account_accepts_signal_profile(settings_local: dict, signal_profile: str) -> bool:
    prof = str(signal_profile).strip().lower()
    src = _normalize_signal_source(settings_local.get("signal_source") if isinstance(settings_local, dict) else "small")
    if src == "both":
        return True
    return src == prof


def _send_signal_message_for_profile(msg: str, signal_profile: str) -> None:
    prof = str(signal_profile).strip().lower()
    if prof == "large":
        if SECONDARY_USE_MAIN_CHANNEL:
            token = TELEGRAM_TOKEN
            chat_id = TELEGRAM_CHAT_ID
        else:
            token = SECONDARY_TELEGRAM_TOKEN
            chat_id = SECONDARY_TELEGRAM_CHAT_ID
        if not token:
            return
        if not SECONDARY_USE_MAIN_CHANNEL:
            targets = _load_secondary_start_chat_ids()
            if not targets:
                return
        else:
            static_chat_id = None
            try:
                if chat_id is not None and str(chat_id).strip() not in ("", "0", "-"):
                    static_chat_id = int(chat_id)
            except Exception:
                static_chat_id = None
            try:
                subscribers = get_subscribers(static_chat_id, token=token)
            except Exception:
                subscribers = []
            targets = list(subscribers)
            if not targets and static_chat_id is not None:
                targets = [static_chat_id]
            if not targets:
                return
        for cid in targets:
            try:
                send_telegram(msg, token, cid)
            except Exception:
                continue
        return

    if TELEGRAM_TOKEN:
        static_chat_id = None
        try:
            if TELEGRAM_CHAT_ID is not None and str(TELEGRAM_CHAT_ID).strip() not in ("", "0", "-"):
                static_chat_id = int(TELEGRAM_CHAT_ID)
        except Exception:
            static_chat_id = None
        try:
            subscribers = get_subscribers(static_chat_id, token=TELEGRAM_TOKEN)
        except Exception:
            subscribers = []
        targets = list(subscribers)
        if not targets and static_chat_id is not None:
            targets = [static_chat_id]
        if not targets:
            return
        for cid in targets:
            try:
                send_telegram(msg, TELEGRAM_TOKEN, cid)
            except Exception:
                continue


def format_price(price: float) -> str:
    abs_p = abs(price)
    if abs_p >= 100:
        return f"{price:.2f}"
    if abs_p >= 1:
        return f"{price:.3f}"
    if abs_p >= 0.01:
        return f"{price:.4f}"
    return f"{price:.8f}"


def _distance_pct_text(entry_price: float, level_price: float) -> str:
    """Return absolute distance from entry as percent text, e.g. '(1.25%)'."""
    try:
        ep = float(entry_price)
        lp = float(level_price)
    except (TypeError, ValueError):
        return ""
    if ep <= 0:
        return ""
    try:
        pct = abs((lp - ep) / ep) * 100.0
    except Exception:
        return ""
    if pct < 0:
        pct = 0.0
    return f" ({pct:.2f}%)"


def choose_auto_sl_tp(entry_price: float, atr_value: float, direction: str, confidence: float):
    """Derive dynamic SL/TP prices from ATR and ML confidence.

    This is used when an account is configured with auto_sl_tp. It
    scales a volatility-based baseline by the model's confidence so
    that high-confidence trades can target wider take-profits, while
    lower-confidence trades stay tighter.
    """

    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        return None, []

    if ep <= 0:
        return None, []

    try:
        atr = float(atr_value)
    except (TypeError, ValueError):
        atr = 0.0

    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0

    # Volatility as percent of price; clamp to a reasonable band so that
    # SL/TP are neither too tight nor absurdly wide on noisy symbols.
    base_vol_pct = 0.0
    if ep > 0 and atr > 0:
        base_vol_pct = abs(atr / ep) * 100.0
    base_vol_pct = max(min(base_vol_pct, 5.0), 0.3)

    # Confidence-driven multipliers: higher confidence => slightly
    # wider SL and materially wider TPs; lower confidence => tighter.
    if conf >= 0.85:
        risk_mult = 1.2
        reward_mult = 1.4
    elif conf >= 0.75:
        risk_mult = 1.0
        reward_mult = 1.2
    else:
        risk_mult = 0.8
        reward_mult = 1.0

    sl_pct = base_vol_pct * risk_mult

    # Base RR ladder; we will further scale TPs by reward_mult so that
    # high-confidence trades aim for higher multiples of risk.
    rr_levels = (1.0, 1.5, 2.0)
    tp_pcts = [sl_pct * rr * reward_mult for rr in rr_levels]

    sl_factor = abs(sl_pct) / 100.0
    tp_factors = [abs(p) / 100.0 for p in tp_pcts]

    if direction == "LONG":
        sl_price = ep * (1.0 - sl_factor)
        tps = [ep * (1.0 + f) for f in tp_factors]
    else:
        sl_price = ep * (1.0 + sl_factor)
        tps = [ep * (1.0 - f) for f in tp_factors]

    return sl_price, tps


def _execute_real_entry(
    symbol,
    signal,
    entry_price,
    confidence,
    account_index=None,
    entry_age_sec: float | None = None,
    signal_ts: float | None = None,
    signal_entry_price: float | None = None,
    market_price: float | None = None,
    signal_profile: str = "small",
):
    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        return

    if ep <= 0:
        return

    try:
        vol_ok, vol_24h, vol_thr = _passes_24h_volume_gate(symbol)
    except Exception:
        vol_ok, vol_24h, vol_thr = True, None, float(MIN_24H_QUOTE_VOLUME_USDT)
    if not vol_ok:
        try:
            log(
                f"[ENTRY][BLOCK][VOL24H] {symbol}: 24h quote volume="
                f"{_format_usdt_volume(vol_24h)} USDT < min {_format_usdt_volume(vol_thr)} USDT"
            )
        except Exception:
            pass
        return

    qty = 0.0

    # Resolve effective fixed notional for this specific account first.
    effective_notional_usd = ACCOUNT_FIXED_NOTIONAL_USD
    if account_index is not None:
        try:
            cfg_live = _get_accounts_cfg_live() or {}
            accs = cfg_live.get("accounts") or []
            if isinstance(accs, list) and 0 <= int(account_index) < len(accs):
                acc = accs[int(account_index)]
                st = acc.get("settings") if isinstance(acc, dict) else {}
                st = st if isinstance(st, dict) else {}
                mode = str(st.get("fixed_notional_type") or "").upper()
                v_usd = st.get("fixed_notional_usd")
                v_raw = st.get("fixed_notional_value")
                if v_usd is not None:
                    effective_notional_usd = float(v_usd)
                elif mode == "USDT" and v_raw is not None:
                    effective_notional_usd = float(v_raw)
        except Exception:
            pass

    # Fixed-notional sizing in USDT.
    if effective_notional_usd and ep > 0:
        try:
            min_notional = 5.0
            notional = float(effective_notional_usd)
            if notional < min_notional:
                notional = min_notional
            target_qty = notional / ep
            qty = adjust_quantity(symbol, target_qty, ep)
        except Exception as q_err:
            log(f"[SIZE] Failed to compute fixed-notional qty for {symbol}: {q_err}")
            qty = 0.0

    if not qty or qty <= 0:
        base_qty = 1e-9
        qty = adjust_quantity(symbol, base_qty, ep)

    if not qty or qty <= 0:
        return

    def _get_account_meta(idx: Optional[int]):
        name = None
        leverage = None
        settings_local = {}
        try:
            cfg_live = _get_accounts_cfg_live()
            if idx is not None and isinstance(cfg_live, dict):
                lst = cfg_live.get("accounts") or []
                if isinstance(lst, list) and 0 <= int(idx) < len(lst):
                    acc = lst[int(idx)]
                    if isinstance(acc, dict):
                        nm = acc.get("name")
                        if nm:
                            name = str(nm)
                        lev = acc.get("leverage")
                        if lev is not None:
                            try:
                                leverage = int(lev)
                            except Exception:
                                leverage = None
                        st = acc.get("settings")
                        if isinstance(st, dict):
                            settings_local = dict(st)
        except Exception:
            pass
        if not name:
            name = f"Account {idx}" if idx is not None else "Accounts"
        if leverage is None:
            try:
                leverage = int(DEFAULT_LEVERAGE)
            except Exception:
                leverage = None
        return name, leverage, settings_local

    acct_name, acct_leverage, acct_settings = _get_account_meta(account_index)

    # Bind AI SL/TP context to concrete account+symbol+direction at entry time.
    # This prevents other profiles/signals from overriding TP levels later.
    try:
        direction_str = "LONG" if str(signal).upper() == "BUY" else "SHORT"
        profile_key = (str(signal_profile).lower(), str(symbol), direction_str)
        legacy_key = (str(symbol), direction_str)
        src_ctx = _AI_SLTP_CONTEXT.get(profile_key) or _AI_SLTP_CONTEXT.get(legacy_key)
        if isinstance(src_ctx, dict) and account_index is not None:
            payload = dict(src_ctx)
            payload["bound_account_index"] = int(account_index)
            payload["bound_signal_profile"] = str(signal_profile).lower()
            payload["bound_ts"] = float(time.time())
            _AI_SLTP_CONTEXT[(int(account_index), str(symbol), direction_str)] = payload
    except Exception:
        pass

    start_ts_ms = float(time.time() * 1000.0)
    lev_ms = None
    order_ms = None
    total_ms = None
    attempts = 0
    final_err = ""

    lev_started = time.time()
    # Run leverage set in a background thread with a hard 20-second cap so that
    # a slow proxy or C++ libcurl hang never delays order placement.  The thread
    # is daemonised and continues in the background — if it eventually succeeds
    # it will still populate the per-symbol leverage cache for future trades.
    try:
        import threading as _thr_mod
        _lev_err_box: list = [None]
        _lev_done_evt = _thr_mod.Event()

        def _lev_worker():
            try:
                if account_index is not None:
                    set_leverage_for_account(symbol, int(DEFAULT_LEVERAGE), int(account_index))
                else:
                    set_leverage(symbol, int(DEFAULT_LEVERAGE))
            except Exception as _lev_exc:
                _lev_err_box[0] = _lev_exc
            finally:
                _lev_done_evt.set()

        _lev_t = _thr_mod.Thread(target=_lev_worker, daemon=True)
        _lev_t.start()
        _lev_done_evt.wait(timeout=20)  # max 20 seconds — never blocks order entry

        if not _lev_done_evt.is_set():
            log(f"[LEVERAGE] Timeout (>20s) setting leverage for {symbol} — proceeding to order")
            final_err = "leverage_timeout"
        elif _lev_err_box[0] is not None:
            _lev_e = _lev_err_box[0]
            log(f"[LEVERAGE] Failed to set leverage for {symbol}: {_lev_e}")
            try:
                final_err = str(_lev_e)
            except Exception:
                final_err = "leverage_error"
    except Exception as lev_e:
        log(f"[LEVERAGE] Failed to set leverage for {symbol}: {lev_e}")
        try:
            final_err = str(lev_e)
        except Exception:
            final_err = "leverage_error"
    try:
        lev_ms = float(time.time() - lev_started) * 1000.0
    except Exception:
        lev_ms = None

    # Entry order (MARKET) with dynamic fallbacks on notional (-4164)
    # and leverage limits (-2027).
    order_placed = False
    last_error = None
    started_ts = None

    try:
        acct_label = f"account {int(account_index)}" if account_index is not None else "account ?"
    except Exception:
        acct_label = "account ?"
    try:
        started_ts = time.time()
        log(
            f"[ORDER][TRY] {symbol} {signal} {acct_label}: entry={format_price(ep)} conf={float(confidence):.2f}"
        )
    except Exception:
        pass

    try:
        _po_result = place_order(
            symbol,
            signal,
            qty,
            entry_price=ep,
            confidence=confidence,
            target_account_index=account_index,
            signal_ts=signal_ts,
            signal_entry_price=signal_entry_price,
            market_price=market_price,
            signal_profile=signal_profile,
        )
        if _po_result is not None:
            order_placed = True
        attempts = max(attempts, 1)
    except Exception as e:
        last_error = e
        err_text = str(e)
        attempts = max(attempts, 1)

        if '"code":-4164' in err_text and ep > 0:
            min_notional = 5.5
            adj_qty = 0.0
            try:
                raw_qty = min_notional / ep
                for _ in range(3):
                    if not raw_qty or raw_qty <= 0:
                        break
                    candidate_qty = adjust_quantity(symbol, raw_qty, ep)
                    if not candidate_qty or candidate_qty <= 0:
                        break
                    notional = candidate_qty * ep
                    if notional >= min_notional:
                        adj_qty = candidate_qty
                        break
                    raw_qty *= 1.2
            except Exception as q_err:
                log(f"[SIZE] Failed to adjust quantity for {symbol} after -4164: {q_err}")
                adj_qty = 0.0

            if adj_qty and adj_qty > 0:
                try:
                    attempts += 1
                    place_order(
                        symbol,
                        signal,
                        adj_qty,
                        entry_price=ep,
                        confidence=confidence,
                        target_account_index=account_index,
                        signal_ts=signal_ts,
                        signal_entry_price=signal_entry_price,
                        market_price=market_price,
                        signal_profile=signal_profile,
                    )
                    order_placed = True
                    qty = adj_qty
                    log(f"[SIZE] Order for {symbol} retried with min-notional size qty={adj_qty}")
                except Exception as e2:
                    last_error = e2
                    log(f"[ORDER] Failed to place adjusted notional order for {symbol}: {e2}")

        if (not order_placed) and '"code":-2027' in err_text:
            base_lev = int(DEFAULT_LEVERAGE) if DEFAULT_LEVERAGE else 5
            log(
                f"[LEVERAGE] {symbol} exceeded max position at leverage {base_lev}x; "
                "trying lower leverage levels..."
            )
            for lev in range(base_lev - 1, 0, -1):
                try:
                    set_leverage(symbol, lev)
                except Exception as lev_e2:
                    log(f"[LEVERAGE] Failed to set leverage {lev}x for {symbol}: {lev_e2}")
                    continue

                try:
                    attempts += 1
                    place_order(
                        symbol,
                        signal,
                        qty,
                        entry_price=ep,
                        confidence=confidence,
                        target_account_index=account_index,
                        signal_ts=signal_ts,
                        signal_entry_price=signal_entry_price,
                        market_price=market_price,
                        signal_profile=signal_profile,
                    )
                    order_placed = True
                    log(f"[LEVERAGE] Order for {symbol} succeeded after lowering leverage to {lev}x")
                    break
                except Exception as e2:
                    last_error = e2
                    err_text2 = str(e2)
                    if '"code":-2027' in err_text2:
                        continue
                    log(f"[ORDER] Failed to place order for {symbol} at leverage {lev}x: {e2}")
                    break

        if not order_placed and '"code":-4164' not in err_text and '"code":-2027' not in err_text:
            try:
                log(
                    f"[ORDER][FAIL] {symbol} {signal} {acct_label}: {e}"
                )
            except Exception:
                pass

    if not order_placed:
        try:
            if last_error is not None and not final_err:
                final_err = str(last_error)
        except Exception:
            if not final_err:
                final_err = "order_error"
        try:
            elapsed = None
            if started_ts is not None:
                elapsed = float(time.time() - float(started_ts))
            if last_error is not None:
                if elapsed is not None:
                    log(f"[ORDER][ABORT] {symbol} {signal} {acct_label}: elapsed={elapsed:.2f}s err={last_error}")
                else:
                    log(f"[ORDER][ABORT] {symbol} {signal} {acct_label}: err={last_error}")
            else:
                if elapsed is not None:
                    log(f"[ORDER][ABORT] {symbol} {signal} {acct_label}: elapsed={elapsed:.2f}s")
                else:
                    log(f"[ORDER][ABORT] {symbol} {signal} {acct_label}")
        except Exception:
            pass
    if not order_placed:
        try:
            end_ms = float(time.time() * 1000.0)
            total_ms = float(end_ms - start_ts_ms)
        except Exception:
            total_ms = None
        try:
            if started_ts is not None:
                order_ms = float(time.time() - float(started_ts)) * 1000.0
        except Exception:
            order_ms = None
        try:
            log_entry_timing(
                symbol=str(symbol),
                side=str(signal),
                account_index=account_index,
                account_name=str(acct_name),
                signal_ts_raw=signal_ts,
                start_ts_ms=start_ts_ms,
                leverage_ms=lev_ms,
                order_ms=order_ms,
                total_ms=total_ms,
                order_attempts=int(attempts),
                result="FAIL",
                error=str(final_err or ""),
            )
        except Exception:
            pass
        # For accounts that are gated by market regime, send a clear Telegram
        # explanation when entry is blocked so operator sees the exact reason.
        try:
            regime_gate_enabled = bool(acct_settings.get("market_regime_gate_enabled"))
        except Exception:
            regime_gate_enabled = False
        try:
            err_txt = str(final_err or last_error or "")
        except Exception:
            err_txt = ""
        if regime_gate_enabled and (
            "market_regime=" in err_txt
            or "market_regime_state_" in err_txt
            or "blocked by market_regime" in err_txt
        ):
            try:
                regime_line, _ = _market_regime_human_line()
                skip_reason = err_txt
                key = (
                    str(symbol).upper(),
                    str(signal).upper(),
                    int(account_index) if account_index is not None else -1,
                    str(skip_reason),
                )
                now_ts = time.time()
                last_ts = float(_MARKET_REGIME_SKIP_NOTIFY_CACHE.get(key) or 0.0)
                # Anti-spam: same reason per symbol/account at most once/120s.
                if now_ts - last_ts >= 120.0:
                    neg_mode = _resolve_negative_alert_mode()
                    if neg_mode == "off":
                        return
                    if neg_mode == "critical_only" and not _is_critical_negative_reason(skip_reason):
                        return
                    _MARKET_REGIME_SKIP_NOTIFY_CACHE[key] = now_ts
                    msg_lines = [
                        f"#{symbol.replace('USDT', '/USDT')} — {('Long🟢' if signal == 'BUY' else 'Short🔴')}",
                        f"Account: {acct_name}",
                        "Գործարքը ՉԲԱՑՎԵՑ",
                        f"Պատճառ: {skip_reason}",
                        regime_line,
                    ]
                    explain_msg = "\n".join(msg_lines)
                    if TELEGRAM_TOKEN:
                        static_chat_id = _resolve_static_chat_id(TELEGRAM_CHAT_ID)
                        subscribers = get_subscribers(static_chat_id, token=TELEGRAM_TOKEN)
                        targets = list(subscribers)
                        if not targets and static_chat_id is not None:
                            targets = [static_chat_id]
                        for chat_id in targets:
                            try:
                                send_telegram(explain_msg, TELEGRAM_TOKEN, chat_id)
                            except Exception:
                                continue
            except Exception:
                pass
        return

    try:
        if started_ts is not None:
            elapsed = float(time.time() - float(started_ts))
            log(f"[ORDER][DONE] {symbol} {signal} {acct_label}: elapsed={elapsed:.2f}s")
    except Exception:
        pass

    try:
        opened_ts_ms = int(time.time() * 1000.0)
    except Exception:
        opened_ts_ms = None
    try:
        opened_iso = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(opened_ts_ms / 1000.0))
            if opened_ts_ms is not None
            else ""
        )
    except Exception:
        opened_iso = ""

    sig_ms = None
    if signal_ts is not None:
        try:
            raw = float(signal_ts)
            if raw > 1e12:
                sig_ms = raw
            elif raw > 1e9:
                sig_ms = raw * 1000.0
        except Exception:
            sig_ms = None
    delay_s = None
    if sig_ms is not None and opened_ts_ms is not None:
        try:
            delay_s = (float(opened_ts_ms) - float(sig_ms)) / 1000.0
        except Exception:
            delay_s = None

    try:
        parts = []
        parts.append("ENTRY OPENED")
        parts.append(f"Symbol: {symbol}")
        parts.append(f"Side: {signal}")
        if account_index is not None:
            parts.append(f"Account: {acct_name} (#{int(account_index)})")
        else:
            parts.append(f"Account: {acct_name}")
        if opened_iso:
            parts.append(f"Opened: {opened_iso}")
        if delay_s is not None:
            parts.append(f"Delay from signal: {delay_s:.3f}s")
        try:
            parts.append(f"Entry: {format_price(ep)}")
        except Exception:
            parts.append(f"Entry: {ep}")
        parts.append(f"Qty: {qty}")
        if acct_leverage is not None:
            parts.append(f"Leverage: x{acct_leverage}")
        if lev_ms is not None:
            parts.append(f"Leverage time: {lev_ms:.0f}ms")
        if order_ms is not None:
            parts.append(f"Order time: {order_ms:.0f}ms")
        if attempts:
            parts.append(f"Attempts: {int(attempts)}")
        entry_broadcast = "\n".join(parts)
        _broadcast_entry_opened_message(entry_broadcast)
    except Exception:
        pass

    try:
        end_ms = float(time.time() * 1000.0)
        total_ms = float(end_ms - start_ts_ms)
    except Exception:
        total_ms = None
    try:
        if started_ts is not None:
            order_ms = float(time.time() - float(started_ts)) * 1000.0
    except Exception:
        order_ms = None
    try:
        log_entry_timing(
            symbol=str(symbol),
            side=str(signal),
            account_index=account_index,
            account_name=str(acct_name),
            signal_ts_raw=signal_ts,
            start_ts_ms=start_ts_ms,
            leverage_ms=lev_ms,
            order_ms=order_ms,
            total_ms=total_ms,
            order_attempts=int(attempts or 1),
            result="OK",
            error="",
        )
    except Exception:
        pass

    order_msg = f"Placed {signal} order for {symbol}, qty={qty}, confidence={confidence:.2f}"
    log(order_msg)

    def _fmt_price(v):
        try:
            return format_price(v)
        except Exception:
            try:
                return f"{float(v):.8f}"
            except Exception:
                return "N/A"

    direction = "LONG" if signal == "BUY" else "SHORT"
    sl_line = "-"
    tp_lines: list[str] = []
    tp_mode_val = acct_settings.get("tp_mode", TP_MODE)
    try:
        tp_mode_val = int(tp_mode_val)
    except Exception:
        tp_mode_val = TP_MODE

    auto_sl_flag = acct_settings.get("auto_sl_tp")
    sl_pct_cfg = acct_settings.get("sl_pct", ACCOUNT_SL_PCT)
    tp_pcts_cfg = acct_settings.get("tp_pcts", ACCOUNT_TP_PCTS)
    ladder_enabled = bool(acct_settings.get("ladder_sl_tp_enabled"))
    adaptive_enabled = bool(acct_settings.get("adaptive_reentry_ladder_enabled"))
    profile_lower = str(signal_profile).lower()
    if profile_lower == "large":
        adaptive_enabled = adaptive_enabled and bool(acct_settings.get("adaptive_reentry_large_enabled", False))
    else:
        adaptive_enabled = adaptive_enabled and bool(acct_settings.get("adaptive_reentry_small_enabled", False))
    ladder_sl_range = acct_settings.get("ladder_sl_range_pct")
    ladder_sl_steps = acct_settings.get("ladder_sl_steps")
    ladder_tp_range = acct_settings.get("ladder_tp_range_pct")
    ladder_tp_steps = acct_settings.get("ladder_tp_steps")
    auto_sl_effective = bool(auto_sl_flag) or (
        isinstance(sl_pct_cfg, str) and sl_pct_cfg.strip().lower() == "auto"
    )

    sl_pct_use = None
    tp_pcts_use = None
    if auto_sl_effective:
        try:
            ctx = None
            if account_index is not None:
                ctx = _AI_SLTP_CONTEXT.get((int(account_index), symbol, direction))
            if not isinstance(ctx, dict):
                ctx = _AI_SLTP_CONTEXT.get((str(signal_profile).lower(), symbol, direction))
            if not isinstance(ctx, dict):
                ctx = _AI_SLTP_CONTEXT.get((symbol, direction))
        except Exception:
            ctx = None
        if isinstance(ctx, dict):
            sl_pct_use = ctx.get("sl_pct")
            tp_pcts_use = ctx.get("tp_pcts")

    if sl_pct_use is None:
        try:
            sl_pct_use = float(sl_pct_cfg)
        except Exception:
            sl_pct_use = None
    if tp_pcts_use is None:
        if isinstance(tp_pcts_cfg, (list, tuple)):
            cleaned = []
            for x in tp_pcts_cfg:
                try:
                    xv = float(x)
                    if xv > 0:
                        cleaned.append(xv)
                except Exception:
                    continue
            tp_pcts_use = cleaned
        else:
            tp_pcts_use = []

    if adaptive_enabled:
        try:
            sl_pct_adaptive = float(acct_settings.get("adaptive_reentry_sl_pct", 5.0))
            # When auto SL/TP is active, prefer fresh AI context over static
            # adaptive defaults (5% SL / 1-4% TP range).
            if auto_sl_effective and sl_pct_use is not None:
                sl_pct_adaptive = float(sl_pct_use)
            if sl_pct_adaptive > 0:
                sl_price = ep * (1.0 - sl_pct_adaptive / 100.0) if direction == "LONG" else ep * (1.0 + sl_pct_adaptive / 100.0)
                sl_price = adjust_price(symbol, sl_price)
                sl_line = f"{_fmt_price(sl_price)} ({abs(float(sl_pct_adaptive)):.2f}%)"
        except Exception:
            pass
    elif ladder_enabled:
        try:
            sl_levels = _ladder_build_pct_levels(
                ladder_sl_range,
                ladder_sl_steps,
                fallback_start=float(sl_pct_use) if sl_pct_use else 2.0,
                fallback_end=float(sl_pct_use) if sl_pct_use else 2.0,
            )
            if sl_levels:
                sl_line = (
                    f"Ladder {sl_levels[0]:.2f}% -> {sl_levels[-1]:.2f}% "
                    f"(steps={_ladder_parse_steps(ladder_sl_steps, default_steps=len(sl_levels))})"
                )
        except Exception:
            pass
    else:
        try:
            if sl_pct_use is not None and float(sl_pct_use) > 0:
                sl_price = ep * (1.0 - float(sl_pct_use) / 100.0) if direction == "LONG" else ep * (1.0 + float(sl_pct_use) / 100.0)
                sl_price = adjust_price(symbol, sl_price)
                sl_line = f"{_fmt_price(sl_price)} ({abs(float(sl_pct_use)):.2f}%)"
        except Exception:
            pass

    if adaptive_enabled:
        try:
            if auto_sl_effective and isinstance(tp_pcts_use, list) and tp_pcts_use:
                tp_pct_levels = [float(v) for v in tp_pcts_use if _safe_float(v, 0.0) > 0][:3]
            else:
                tp_pct_levels = _ladder_build_pct_levels(
                    acct_settings.get("adaptive_reentry_tp_range_pct"),
                    acct_settings.get("adaptive_reentry_tp_steps"),
                    fallback_start=2.0,
                    fallback_end=4.0,
                )
            if auto_sl_effective and tp_pct_levels:
                base_floor = 1.5
                if isinstance(tp_pcts_use, list) and tp_pcts_use:
                    try:
                        base_floor = max(base_floor, float(tp_pcts_use[0]) * 0.85)
                    except Exception:
                        pass
                tp_pct_levels = [max(float(v), float(base_floor)) for v in tp_pct_levels]
                if len(tp_pct_levels) >= 2 and tp_pct_levels[1] < tp_pct_levels[0] * 1.35:
                    tp_pct_levels[1] = tp_pct_levels[0] * 1.35
            tp_prices = _ladder_build_price_levels(
                ep,
                tp_pct_levels,
                direction == "LONG",
                True,
                symbol=symbol,
            )
            for i, (tp_price, tp_pct) in enumerate(zip(tp_prices, tp_pct_levels), start=1):
                tp_lines.append(f"TP{i}: {_fmt_price(tp_price)} ({abs(float(tp_pct)):.2f}%)")
        except Exception:
            pass
    elif ladder_enabled:
        try:
            tp_levels = _ladder_build_pct_levels(
                ladder_tp_range,
                ladder_tp_steps,
                fallback_start=float(tp_pcts_use[0]) if isinstance(tp_pcts_use, list) and tp_pcts_use else 2.0,
                fallback_end=float(tp_pcts_use[-1]) if isinstance(tp_pcts_use, list) and tp_pcts_use else 4.0,
            )
            if tp_levels:
                tp_lines.append(
                    f"Ladder TP range: {tp_levels[0]:.2f}% -> {tp_levels[-1]:.2f}% "
                    f"(steps={_ladder_parse_steps(ladder_tp_steps, default_steps=len(tp_levels))})"
                )
        except Exception:
            pass
    else:
        try:
            tps = []
            max_tps = min(3, len(tp_pcts_use) if isinstance(tp_pcts_use, list) else 0)
            for i in range(max_tps):
                pct = float(tp_pcts_use[i])
                if pct <= 0:
                    continue
                tp_price = ep * (1.0 + pct / 100.0) if direction == "LONG" else ep * (1.0 - pct / 100.0)
                tps.append((adjust_price(symbol, tp_price), pct))
            for i, tp_pack in enumerate(tps[:3], start=1):
                tp_price, tp_pct = tp_pack
                tp_lines.append(f"TP{i}: {_fmt_price(tp_price)} ({abs(float(tp_pct)):.2f}%)")
        except Exception:
            pass

    base_symbol = symbol.replace("USDT", "/USDT") if symbol.endswith("USDT") else symbol
    side_txt = "Long🟢" if signal == "BUY" else "Short🔴"
    tp_mode_txt = str(tp_mode_val)
    if tp_mode_val == 1:
        tp_mode_txt = "1 (TP1)"
    elif tp_mode_val == 2:
        tp_mode_txt = "2 (TP1+TP2)"
    elif tp_mode_val == 3:
        tp_mode_txt = "3 (TP1+TP2+TP3)"
    elif tp_mode_val == 4:
        tp_mode_txt = "4 (Split)"

    try:
        conf_pct = float(confidence) * 100.0
        conf_str = f"{conf_pct:.1f}%"
    except Exception:
        conf_str = "-"
    signal_msg_cfg = _resolve_signal_message_settings(account_index, acct_settings)

    try:
        opened_line = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    except Exception:
        opened_line = None

    profile_tag = SECONDARY_TELEGRAM_PREFIX if str(signal_profile).lower() == "large" else ""
    pretty_lines = [
        f"{profile_tag + ' ' if profile_tag else ''}#{base_symbol} — {side_txt}",
        "",
        f"Account: {acct_name}",
        f"Time: {opened_line}" if opened_line else "",
        f"Entry: {_fmt_price(ep)}",
        f"Stop Loss: {sl_line}",
    ]

    try:
        pretty_lines = [x for x in pretty_lines if x != ""]
    except Exception:
        pass

    try:
        assign = room_sizing_pop_assignment(account_index, symbol, signal)
    except Exception:
        assign = None
    if isinstance(assign, dict):
        try:
            r_i = int(assign.get("room_index")) + 1
            s_i = int(assign.get("slot_index")) + 1
            b_i = int(assign.get("batch_id") or 0)
            pretty_lines.append("")
            pretty_lines.append(f"Room: {r_i} / Slot: {s_i} / Batch: {b_i}")
        except Exception:
            pass
    if tp_lines:
        pretty_lines.append("")
        pretty_lines.append("Targets:")
        for tl in tp_lines:
            pretty_lines.append(f"  {tl}")
    pretty_lines.append("")
    if entry_age_sec is not None:
        try:
            s = float(entry_age_sec)
            if s >= 0:
                max_m = signal_msg_cfg.get("level_age_max_minutes")
                if signal_msg_cfg.get("include_level_age", True) and (
                    max_m is None or s <= float(max_m) * 60.0
                ):
                    mm = int(s // 60)
                    ss = int(s % 60)
                    pretty_lines.append(f"Level age: {mm}m {ss}s")
        except Exception:
            pass
    if signal_msg_cfg.get("include_leverage", True) and acct_leverage is not None:
        pretty_lines.append(f"Leverage: x{acct_leverage}")
    pretty_lines.append(f"TP mode: {tp_mode_txt}")
    try:
        actual_notional = float(ep) * float(qty)
        target_notional = None
        if isinstance(assign, dict) and assign.get("notional_usdt") is not None:
            target_notional = float(assign.get("notional_usdt"))
        if target_notional is not None:
            pretty_lines.append(
                f"Notional: {actual_notional:.2f} USDT (target {target_notional:.2f})"
            )
        else:
            pretty_lines.append(f"Notional: {actual_notional:.2f} USDT")
    except Exception:
        pass
    pretty_lines.append(f"Qty (coin): {qty}")
    if signal_msg_cfg.get("include_confidence", True):
        pretty_lines.append(f"Confidence: {conf_str}")
    if signal_msg_cfg.get("include_last_1h_range", True):
        try:
            vol_1h = _futures_1h_volatility(symbol)
            if vol_1h is not None:
                pretty_lines.append(f"Last 1h range: {float(vol_1h):.2f}%")
        except Exception:
            pass
    if signal_msg_cfg.get("include_last_5m_trades", True):
        try:
            act_5m = _futures_5m_activity(symbol)
            if isinstance(act_5m, dict):
                lt = act_5m.get("last_trades")
                at = act_5m.get("avg_trades")
                st = act_5m.get("status")
                if lt is not None and at is not None and st:
                    pretty_lines.append(f"Last 5m trades: {int(lt)} (avg {float(at):.0f}) - {st}")
        except Exception:
            pass
    if signal_msg_cfg.get("include_signal_source", True):
        pretty_lines.append(f"Signal source: {str(signal_profile).lower()}")
    try:
        regime_gate_enabled = bool(acct_settings.get("market_regime_gate_enabled"))
    except Exception:
        regime_gate_enabled = False
    if regime_gate_enabled:
        try:
            regime_line, _ = _market_regime_human_line()
            pretty_lines.append(regime_line)
        except Exception:
            pass
    try:
        mq_line = _market_quality_line(symbol)
        if mq_line:
            pretty_lines.append(mq_line)
    except Exception:
        pass

    pretty_msg = "\n".join(pretty_lines)

    _send_signal_message_for_profile(pretty_msg, str(signal_profile).lower())


def _execute_multi_entry_level(
    symbol,
    signal,
    entry_price,
    confidence,
    level_index,
    account_index=None,
    entry_age_sec: float | None = None,
    signal_ts: float | None = None,
    signal_entry_price: float | None = None,
    market_price: float | None = None,
    signal_profile: str = "small",
):
    """Execute a single dynamic ladder level across all accounts.

    - Multi-entry enabled accounts will size this level from their
      per-level `notional_usd` via `multi_entry_level_index` inside
      `place_order`.
    - Non-multi-entry accounts should keep behaving like a normal
      single-entry order, so we must always pass a non-null base
      quantity to `place_order`.
    """

    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        return

    if ep <= 0:
        return

    try:
        vol_ok, vol_24h, vol_thr = _passes_24h_volume_gate(symbol)
    except Exception:
        vol_ok, vol_24h, vol_thr = True, None, float(MIN_24H_QUOTE_VOLUME_USDT)
    if not vol_ok:
        try:
            log(
                f"[ENTRY][BLOCK][VOL24H] {symbol} multi-entry level {level_index}: "
                f"24h quote volume={_format_usdt_volume(vol_24h)} USDT < min {_format_usdt_volume(vol_thr)} USDT"
            )
        except Exception:
            pass
        return

    qty = 0.0

    effective_notional_usd = ACCOUNT_FIXED_NOTIONAL_USD
    if account_index is not None:
        try:
            cfg_live = _get_accounts_cfg_live() or {}
            accs = cfg_live.get("accounts") or []
            if isinstance(accs, list) and 0 <= int(account_index) < len(accs):
                acc = accs[int(account_index)]
                st = acc.get("settings") if isinstance(acc, dict) else {}
                st = st if isinstance(st, dict) else {}
                mode = str(st.get("fixed_notional_type") or "").upper()
                v_usd = st.get("fixed_notional_usd")
                v_raw = st.get("fixed_notional_value")
                if v_usd is not None:
                    effective_notional_usd = float(v_usd)
                elif mode == "USDT" and v_raw is not None:
                    effective_notional_usd = float(v_raw)
        except Exception:
            pass

    # Use the same fixed-notional sizing logic as _execute_real_entry so
    # that non-multi-entry accounts receive a valid base quantity.
    if effective_notional_usd and ep > 0:
        try:
            min_notional = 5.0
            notional = float(effective_notional_usd)
            if notional < min_notional:
                notional = min_notional
            target_qty = notional / ep
            qty = adjust_quantity(symbol, target_qty, ep)
        except Exception as q_err:
            log(
                f"[SIZE] Failed to compute fixed-notional qty for {symbol} "
                f"(multi-entry level {level_index}): {q_err}"
            )
            qty = 0.0

    if not qty or qty <= 0:
        base_qty = 1e-9
        qty = adjust_quantity(symbol, base_qty, ep)

    if not qty or qty <= 0:
        return

    try:
        set_leverage(symbol, int(DEFAULT_LEVERAGE))
    except Exception as lev_e:
        log(f"[LEVERAGE] Failed to set leverage for {symbol}: {lev_e}")

    order_placed = False

    try:
        _po_result2 = place_order(
            symbol,
            signal,
            qty,
            entry_price=ep,
            confidence=confidence,
            multi_entry_level_index=level_index,
            target_account_index=account_index,
            signal_ts=signal_ts,
            signal_entry_price=signal_entry_price,
            market_price=market_price,
            signal_profile=signal_profile,
        )
        if _po_result2 is not None:
            order_placed = True
    except Exception as e:
        err_text = str(e)

        if '"code":-4164' in err_text and ep > 0:
            # MIN_NOTIONAL violation: retry with Binance minimum notional.
            min_notional = 5.5
            adj_qty = 0.0
            try:
                raw_qty = min_notional / ep
                for _ in range(3):
                    if not raw_qty or raw_qty <= 0:
                        break
                    candidate_qty = adjust_quantity(symbol, raw_qty, ep)
                    if not candidate_qty or candidate_qty <= 0:
                        break
                    notional = candidate_qty * ep
                    if notional >= min_notional:
                        adj_qty = candidate_qty
                        break
                    raw_qty *= 1.2
            except Exception as q_err:
                log(
                    f"[SIZE] Failed to adjust quantity for {symbol} after -4164 "
                    f"(multi-entry level {level_index}): {q_err}"
                )
                adj_qty = 0.0

            if adj_qty and adj_qty > 0:
                try:
                    place_order(
                        symbol,
                        signal,
                        adj_qty,
                        entry_price=ep,
                        confidence=confidence,
                        multi_entry_level_index=level_index,
                        target_account_index=account_index,
                        signal_ts=signal_ts,
                        signal_entry_price=signal_entry_price,
                        market_price=market_price,
                        signal_profile=signal_profile,
                    )
                    order_placed = True
                    qty = adj_qty
                    log(
                        f"[SIZE] Order for {symbol} retried with min-notional size "
                        f"qty={adj_qty} (multi-entry level {level_index})"
                    )
                except Exception as e2:
                    log(
                        f"[ORDER] Failed to place adjusted notional order for {symbol} "
                        f"(multi-entry level {level_index}): {e2}"
                    )

        if (not order_placed) and '"code":-2027' in err_text:
            base_lev = int(DEFAULT_LEVERAGE) if DEFAULT_LEVERAGE else 5
            log(
                f"[LEVERAGE] {symbol} exceeded max position at leverage {base_lev}x; "
                f"trying lower leverage levels for multi-entry level {level_index}..."
            )
            for lev in range(base_lev - 1, 0, -1):
                try:
                    set_leverage(symbol, lev)
                except Exception as lev_e2:
                    log(
                        f"[LEVERAGE] Failed to set leverage {lev}x for {symbol} "
                        f"(multi-entry level {level_index}): {lev_e2}"
                    )
                    continue

                try:
                    place_order(
                        symbol,
                        signal,
                        qty,
                        entry_price=ep,
                        confidence=confidence,
                        multi_entry_level_index=level_index,
                        target_account_index=account_index,
                        signal_ts=signal_ts,
                        signal_entry_price=signal_entry_price,
                        market_price=market_price,
                        signal_profile=signal_profile,
                    )
                    order_placed = True
                    log(
                        f"[LEVERAGE] Order for {symbol} succeeded after lowering "
                        f"leverage to {lev}x (multi-entry level {level_index})"
                    )
                    break
                except Exception as e2:
                    err_text2 = str(e2)
                    if '"code":-2027' in err_text2:
                        continue
                    log(
                        f"[ORDER] Failed to place order for {symbol} at leverage {lev}x "
                        f"(multi-entry level {level_index}): {e2}"
                    )
                    break

        if not order_placed and '"code":-4164' not in err_text and '"code":-2027' not in err_text:
            log(
                f"[ORDER] Error placing multi-entry level {level_index} for {symbol}: {e}"
            )

    if not order_placed:
        return

    order_msg = f"Placed {signal} multi-entry level {level_index} for {symbol}, confidence={confidence:.2f}"
    log(order_msg)

    acct_name = f"Account {account_index}" if account_index is not None else "Accounts"
    acct_leverage = None
    acct_settings = {}
    try:
        cfg_live = _get_accounts_cfg_live()
        if account_index is not None and isinstance(cfg_live, dict):
            lst = cfg_live.get("accounts") or []
            if isinstance(lst, list) and 0 <= int(account_index) < len(lst):
                acc = lst[int(account_index)]
                if isinstance(acc, dict):
                    nm = acc.get("name")
                    if nm:
                        acct_name = str(nm)
                    lev = acc.get("leverage")
                    if lev is not None:
                        try:
                            acct_leverage = int(lev)
                        except Exception:
                            acct_leverage = None
                    st = acc.get("settings")
                    if isinstance(st, dict):
                        acct_settings = dict(st)
    except Exception:
        pass
    if acct_leverage is None:
        try:
            acct_leverage = int(DEFAULT_LEVERAGE)
        except Exception:
            acct_leverage = None

    def _fmt_price(v):
        try:
            return format_price(v)
        except Exception:
            try:
                return f"{float(v):.8f}"
            except Exception:
                return "N/A"

    direction = "LONG" if signal == "BUY" else "SHORT"
    sl_line = "-"
    tp_lines: list[str] = []
    tp_mode_val = acct_settings.get("tp_mode", TP_MODE)
    try:
        tp_mode_val = int(tp_mode_val)
    except Exception:
        tp_mode_val = TP_MODE

    sl_pct_cfg = acct_settings.get("sl_pct", ACCOUNT_SL_PCT)
    tp_pcts_cfg = acct_settings.get("tp_pcts", ACCOUNT_TP_PCTS)
    adaptive_enabled = bool(acct_settings.get("adaptive_reentry_ladder_enabled"))
    profile_lower = str(signal_profile).lower()
    if profile_lower == "large":
        adaptive_enabled = adaptive_enabled and bool(acct_settings.get("adaptive_reentry_large_enabled", False))
    else:
        adaptive_enabled = adaptive_enabled and bool(acct_settings.get("adaptive_reentry_small_enabled", False))
    try:
        sl_pct_use = float(sl_pct_cfg)
    except Exception:
        sl_pct_use = None

    tp_pcts_use = []
    if isinstance(tp_pcts_cfg, (list, tuple)):
        for x in tp_pcts_cfg:
            try:
                xv = float(x)
                if xv > 0:
                    tp_pcts_use.append(xv)
            except Exception:
                continue
    # Pull fresh AI context for accurate SL/TP preview in Telegram.
    try:
        ai_ctx = None
        if account_index is not None:
            ai_ctx = _AI_SLTP_CONTEXT.get((int(account_index), str(symbol), str(direction)))
        if not isinstance(ai_ctx, dict):
            ai_ctx = _AI_SLTP_CONTEXT.get((str(signal_profile).lower(), str(symbol), str(direction)))
        if not isinstance(ai_ctx, dict):
            ai_ctx = _AI_SLTP_CONTEXT.get((str(symbol), str(direction)))
        if isinstance(ai_ctx, dict):
            ai_ts = _safe_float(ai_ctx.get("ts"), 0.0)
            if ai_ts > 0 and (time.time() - ai_ts) <= 6 * 3600.0:
                ai_sl = _safe_float(ai_ctx.get("sl_pct"), 0.0)
                if ai_sl > 0:
                    sl_pct_use = ai_sl
                ai_tps = ai_ctx.get("tp_pcts")
                if isinstance(ai_tps, (list, tuple)):
                    parsed = []
                    for v in ai_tps:
                        pv = _safe_float(v, 0.0)
                        if pv > 0:
                            parsed.append(pv)
                    if parsed:
                        tp_pcts_use = parsed[:3]
    except Exception:
        pass

    if adaptive_enabled:
        try:
            sl_pct_adaptive = float(acct_settings.get("adaptive_reentry_sl_pct", 5.0))
            if sl_pct_use is not None:
                sl_pct_adaptive = float(sl_pct_use)
            if sl_pct_adaptive > 0:
                sl_price = ep * (1.0 - sl_pct_adaptive / 100.0) if direction == "LONG" else ep * (1.0 + sl_pct_adaptive / 100.0)
                sl_price = adjust_price(symbol, sl_price)
                sl_line = f"{_fmt_price(sl_price)} ({abs(float(sl_pct_adaptive)):.2f}%)"
        except Exception:
            pass
        try:
            if tp_pcts_use:
                tp_pct_levels = [float(v) for v in tp_pcts_use if _safe_float(v, 0.0) > 0][:3]
            else:
                tp_pct_levels = _ladder_build_pct_levels(
                    acct_settings.get("adaptive_reentry_tp_range_pct"),
                    acct_settings.get("adaptive_reentry_tp_steps"),
                    fallback_start=2.0,
                    fallback_end=4.0,
                )
            if tp_pct_levels:
                base_floor = 1.5
                if tp_pcts_use:
                    try:
                        base_floor = max(base_floor, float(tp_pcts_use[0]) * 0.85)
                    except Exception:
                        pass
                tp_pct_levels = [max(float(v), float(base_floor)) for v in tp_pct_levels]
                if len(tp_pct_levels) >= 2 and tp_pct_levels[1] < tp_pct_levels[0] * 1.35:
                    tp_pct_levels[1] = tp_pct_levels[0] * 1.35
            tp_prices = _ladder_build_price_levels(
                ep,
                tp_pct_levels,
                direction == "LONG",
                True,
                symbol=symbol,
            )
            for i, (tp_price, tp_pct) in enumerate(zip(tp_prices, tp_pct_levels), start=1):
                tp_lines.append(f"TP{i}: {_fmt_price(tp_price)} ({abs(float(tp_pct)):.2f}%)")
        except Exception:
            pass
    else:
        try:
            if sl_pct_use is not None and float(sl_pct_use) > 0:
                sl_price = ep * (1.0 - float(sl_pct_use) / 100.0) if direction == "LONG" else ep * (1.0 + float(sl_pct_use) / 100.0)
                sl_price = adjust_price(symbol, sl_price)
                sl_line = f"{_fmt_price(sl_price)} ({abs(float(sl_pct_use)):.2f}%)"
        except Exception:
            pass

        try:
            tps = []
            max_tps = min(3, len(tp_pcts_use))
            for i in range(max_tps):
                pct = float(tp_pcts_use[i])
                if pct <= 0:
                    continue
                tp_price = ep * (1.0 + pct / 100.0) if direction == "LONG" else ep * (1.0 - pct / 100.0)
                tps.append((adjust_price(symbol, tp_price), pct))
            for i, tp_pack in enumerate(tps[:3], start=1):
                tp_price, tp_pct = tp_pack
                tp_lines.append(f"TP{i}: {_fmt_price(tp_price)} ({abs(float(tp_pct)):.2f}%)")
        except Exception:
            pass

    base_symbol = symbol.replace("USDT", "/USDT") if symbol.endswith("USDT") else symbol
    side_txt = "Long🟢" if signal == "BUY" else "Short🔴"
    tp_mode_txt = str(tp_mode_val)
    if tp_mode_val == 1:
        tp_mode_txt = "1 (TP1)"
    elif tp_mode_val == 2:
        tp_mode_txt = "2 (TP2)"
    elif tp_mode_val == 3:
        tp_mode_txt = "3 (TP3)"

    try:
        conf_pct = float(confidence) * 100.0
        conf_str = f"{conf_pct:.1f}%"
    except Exception:
        conf_str = "-"
    signal_msg_cfg = _resolve_signal_message_settings(account_index, acct_settings)

    age_str = None
    age_sec = None
    if entry_age_sec is not None:
        try:
            age_sec = float(entry_age_sec)
            if age_sec >= 0:
                mm = int(age_sec // 60)
                ss = int(age_sec % 60)
                age_str = f"{mm}m {ss}s"
        except Exception:
            age_str = None
            age_sec = None

    try:
        opened_line = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    except Exception:
        opened_line = None

    profile_tag = SECONDARY_TELEGRAM_PREFIX if str(signal_profile).lower() == "large" else ""
    pretty_lines = [
        f"{profile_tag + ' ' if profile_tag else ''}#{base_symbol} — {side_txt}",
        "",
        f"Account: {acct_name}",
        f"Time: {opened_line}" if opened_line else "",
        f"Entry: {_fmt_price(ep)}",
        f"Stop Loss: {sl_line}",
    ]

    try:
        pretty_lines = [x for x in pretty_lines if x != ""]
    except Exception:
        pass

    try:
        assign = room_sizing_pop_assignment(account_index, symbol, signal)
    except Exception:
        assign = None
    if isinstance(assign, dict):
        try:
            r_i = int(assign.get("room_index")) + 1
            s_i = int(assign.get("slot_index")) + 1
            b_i = int(assign.get("batch_id") or 0)
            pretty_lines.append("")
            pretty_lines.append(f"Room: {r_i} / Slot: {s_i} / Batch: {b_i}")
        except Exception:
            pass
    if tp_lines:
        pretty_lines.append("")
        pretty_lines.append("Targets:")
        for tl in tp_lines:
            pretty_lines.append(f"  {tl}")
    pretty_lines.append("")
    pretty_lines.append(f"Entry level: {int(level_index) + 1}")
    if age_str and signal_msg_cfg.get("include_level_age", True):
        allow_age = True
        try:
            max_m = signal_msg_cfg.get("level_age_max_minutes")
            if max_m is not None and age_sec is not None:
                allow_age = float(age_sec) <= float(max_m) * 60.0
        except Exception:
            allow_age = True
        if allow_age:
            pretty_lines.append(f"Level age: {age_str}")
    if signal_msg_cfg.get("include_leverage", True) and acct_leverage is not None:
        pretty_lines.append(f"Leverage: x{acct_leverage}")
    pretty_lines.append(f"TP mode: {tp_mode_txt}")
    try:
        actual_notional = float(ep) * float(qty)
        target_notional = None
        if isinstance(assign, dict) and assign.get("notional_usdt") is not None:
            target_notional = float(assign.get("notional_usdt"))
        if target_notional is not None:
            pretty_lines.append(
                f"Notional: {actual_notional:.2f} USDT (target {target_notional:.2f})"
            )
        else:
            pretty_lines.append(f"Notional: {actual_notional:.2f} USDT")
    except Exception:
        pass
    pretty_lines.append(f"Qty (coin): {qty}")
    if signal_msg_cfg.get("include_confidence", True):
        pretty_lines.append(f"Confidence: {conf_str}")
    if signal_msg_cfg.get("include_last_1h_range", True):
        try:
            vol_1h = _futures_1h_volatility(symbol)
            if vol_1h is not None:
                pretty_lines.append(f"Last 1h range: {float(vol_1h):.2f}%")
        except Exception:
            pass
    if signal_msg_cfg.get("include_last_5m_trades", True):
        try:
            act_5m = _futures_5m_activity(symbol)
            if isinstance(act_5m, dict):
                lt = act_5m.get("last_trades")
                at = act_5m.get("avg_trades")
                st = act_5m.get("status")
                if lt is not None and at is not None and st:
                    pretty_lines.append(f"Last 5m trades: {int(lt)} (avg {float(at):.0f}) - {st}")
        except Exception:
            pass
    if signal_msg_cfg.get("include_signal_source", True):
        pretty_lines.append(f"Signal source: {str(signal_profile).lower()}")
    try:
        regime_gate_enabled = bool(acct_settings.get("market_regime_gate_enabled"))
    except Exception:
        regime_gate_enabled = False
    if regime_gate_enabled:
        try:
            regime_line, _ = _market_regime_human_line()
            pretty_lines.append(regime_line)
        except Exception:
            pass
    try:
        mq_line = _market_quality_line(symbol)
        if mq_line:
            pretty_lines.append(mq_line)
    except Exception:
        pass

    pretty_msg = "\n".join(pretty_lines)

    _send_signal_message_for_profile(pretty_msg, str(signal_profile).lower())


def _schedule_pending_entry(
    symbol, signal, entry_price, confidence, atr_value, do_real, do_paper, signal_profile="small"
):
    if not DYNAMIC_ENTRY_ENABLED:
        return False

    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        return False

    if ep <= 0:
        return False

    if signal not in ("BUY", "SELL"):
        return False

    created_ts = time.time()
    try:
        conf_val = float(confidence)
    except (TypeError, ValueError):
        conf_val = 0.0

    try:
        atr_val = float(atr_value) if atr_value is not None else 0.0
    except (TypeError, ValueError):
        atr_val = 0.0

    # If we don't have a structured accounts config, fall back to the
    # original single-offset behaviour driven by ACCOUNT_ENTRY_OFFSET_PCT.
    accounts_list = []
    cfg_live = _get_accounts_cfg_live()
    if isinstance(cfg_live, dict):
        accounts_list = cfg_live.get("accounts") or []

    if not isinstance(accounts_list, list) or not accounts_list:
        if ACCOUNT_ENTRY_OFFSET_PCT <= 0.0 or ACCOUNT_ENTRY_TIMEOUT_MIN <= 0:
            return False

        with _PENDING_ENTRIES_LOCK:
            for p in _PENDING_ENTRIES:
                if (
                    p.get("symbol") == symbol
                    and p.get("signal") == signal
                    and str(p.get("signal_profile") or "small").lower() == str(signal_profile).lower()
                ):
                    return True

            _PENDING_ENTRIES.append(
                {
                    "symbol": symbol,
                    "signal": signal,
                    "entry_price": ep,
                    "confidence": conf_val,
                    "atr": atr_val,
                    "created_ts": created_ts,
                    "do_real": bool(do_real),
                    "do_paper": bool(do_paper),
                    "offset_pct": float(ACCOUNT_ENTRY_OFFSET_PCT),
                    "timeout_min": int(ACCOUNT_ENTRY_TIMEOUT_MIN),
                    "signal_profile": str(signal_profile).lower(),
                }
            )

        try:
            log(
                f"[ENTRY] Scheduled dynamic entry for {symbol} {signal} at offset {ACCOUNT_ENTRY_OFFSET_PCT:.2f}% "
                f"with timeout {ACCOUNT_ENTRY_TIMEOUT_MIN} min (base_price={format_price(ep)})"
            )
        except Exception:
            pass

        return True

    # Per-account dynamic entries: each account uses its own
    # entry_offset_pct / entry_timeout_min from binance_accounts.yaml.
    pending_specs = []
    immediate_real_accounts = []

    for idx, acc in enumerate(accounts_list):
        if not isinstance(acc, dict):
            continue

        flag = acc.get("trade_enabled")
        if flag is False:
            continue

        settings_local = acc.get("settings") or {}
        if not isinstance(settings_local, dict):
            settings_local = {}
        if not _account_accepts_signal_profile(settings_local, signal_profile):
            continue

        multi_entry_enabled = bool(settings_local.get("multi_entry_enabled"))
        levels = settings_local.get("multi_entry_levels") or []

        try:
            entry_offset_val = float(settings_local.get("entry_offset_pct", 0.0))
        except (TypeError, ValueError):
            entry_offset_val = 0.0

        try:
            timeout_raw = settings_local.get("entry_timeout_min")
            if timeout_raw is not None:
                # Explicit per-account timeout always wins.
                timeout_min = int(timeout_raw)
            else:
                # If this account has distance-based SL/TP enabled but neither
                # per-account nor global timeout is configured, apply a safe
                # default of 5 minutes so that distance-based entry offsets
                # (Samo-style) actually schedule dynamic entries instead of
                # falling back to immediate fills.
                if bool(settings_local.get("distance_sl_tp_enabled")) and ACCOUNT_ENTRY_TIMEOUT_MIN <= 0:
                    timeout_min = 5
                else:
                    timeout_min = int(ACCOUNT_ENTRY_TIMEOUT_MIN)
        except (TypeError, ValueError):
            if bool(settings_local.get("distance_sl_tp_enabled")) and ACCOUNT_ENTRY_TIMEOUT_MIN <= 0:
                timeout_min = 5
            else:
                timeout_min = int(ACCOUNT_ENTRY_TIMEOUT_MIN)
        if timeout_min < 0:
            timeout_min = 0

        # Optional per-account distance-based SL/TP / entry-offset overrides
        # driven by last 1h HIGH/LOW range. This only affects accounts that
        # have distance_sl_tp_enabled configured in binance_accounts.yaml.
        # We reuse the same 1h volatility helper used for Telegram messages.
        dist_sl_tp_use_auto = None
        dist_sl_tp_sl_pct = None
        dist_sl_tp_tp_pcts = None
        dist_entry_offset = None
        vol_1h = None
        try:
            vol_1h = _futures_1h_volatility(symbol)
        except Exception:
            vol_1h = None

        if vol_1h is not None:
            try:
                use_auto, sl_pct_override, tp_pcts_override, entry_offset_override = compute_distance_based_sl_tp(
                    idx, float(vol_1h)
                )
            except Exception:
                use_auto = False
                sl_pct_override = None
                tp_pcts_override = None
                entry_offset_override = None

            # entry_offset_override only influences dynamic entry scheduling;
            # SL/TP overrides will be consumed later by the exit manager via
            # per-account settings/state and do not change this function's
            # behaviour directly beyond entry offset.
            dist_sl_tp_use_auto = use_auto
            if entry_offset_override is not None:
                try:
                    dist_entry_offset = float(entry_offset_override)
                except (TypeError, ValueError):
                    dist_entry_offset = None
            # Note: sl_pct_override / tp_pcts_override are intentionally not
            # injected into settings_local here to avoid mutating global
            # account config; they are used later when computing exits.

        # Multi-entry ladder for this account (e.g. Gurgen).
        if multi_entry_enabled and isinstance(levels, list) and levels:
            for lvl_idx, lvl in enumerate(levels):
                if not isinstance(lvl, dict):
                    continue
                offset_raw = lvl.get("offset_pct")
                try:
                    lvl_offset = float(offset_raw) if offset_raw is not None else 0.0
                except (TypeError, ValueError):
                    lvl_offset = 0.0

                if entry_offset_val and entry_offset_val != 0.0:
                    offset_val = entry_offset_val + lvl_offset
                else:
                    offset_val = lvl_offset

                pending_specs.append(
                    {
                        "account_index": idx,
                        "level_index": lvl_idx,
                        "offset_pct": offset_val,
                        "timeout_min": timeout_min,
                        "do_paper": bool(do_paper) and lvl_idx == 0,
                    }
                )
            continue

        # Non-multi-entry account: schedule a single dynamic entry if both
        # offset and timeout are positive; otherwise fall back to immediate
        # entry for that account only. When distance-based config provides an
        # explicit entry_offset_pct, it shadows the static entry_offset_pct
        # for this account only.
        eff_offset = entry_offset_val
        if dist_entry_offset is not None:
            try:
                eff_offset = float(dist_entry_offset)
            except (TypeError, ValueError):
                eff_offset = entry_offset_val

        if eff_offset > 0.0 and timeout_min > 0:
            pending_specs.append(
                {
                    "account_index": idx,
                    "level_index": None,
                    "offset_pct": eff_offset,
                    "timeout_min": timeout_min,
                    "do_paper": bool(do_paper),
                }
            )
        else:
            if do_real:
                immediate_real_accounts.append(idx)

    try:
        if do_real or do_paper:
            log(
                f"[ENTRY][PLAN][PROFILE: {str(signal_profile).lower()}] {symbol} {signal}: do_real={bool(do_real)} do_paper={bool(do_paper)} "
                f"immediate_real={len(immediate_real_accounts)} pending={len(pending_specs)}"
            )
    except Exception:
        pass

    scheduled_any = False

    # Append pending entries, avoiding duplicates for the same
    # (symbol, signal, account_index, level_index).
    if pending_specs:
        has_existing_for_symbol_signal = False
        with _PENDING_ENTRIES_LOCK:
            existing_keys = set()
            for p in _PENDING_ENTRIES:
                if p.get("symbol") != symbol:
                    continue
                if p.get("signal") != signal:
                    continue
                if str(p.get("signal_profile") or "small").lower() != str(signal_profile).lower():
                    continue
                has_existing_for_symbol_signal = True
                acc_raw = p.get("account_index")
                lvl_raw = p.get("level_index")
                try:
                    acc_key = int(acc_raw) if acc_raw is not None else -1
                except (TypeError, ValueError):
                    acc_key = -1
                try:
                    lvl_key = int(lvl_raw) if lvl_raw is not None else -1
                except (TypeError, ValueError):
                    lvl_key = -1
                existing_keys.add((acc_key, lvl_key))

            for spec in pending_specs:
                acc_idx = spec["account_index"]
                lvl_idx = spec["level_index"]
                key = (acc_idx, -1 if lvl_idx is None else int(lvl_idx))
                if key in existing_keys:
                    continue

                _PENDING_ENTRIES.append(
                    {
                        "symbol": symbol,
                        "signal": signal,
                        "entry_price": ep,
                        "confidence": conf_val,
                        "atr": atr_val,
                        "created_ts": created_ts,
                        "do_real": bool(do_real),
                        "do_paper": spec["do_paper"],
                        "account_index": acc_idx,
                        "level_index": lvl_idx,
                        "offset_pct": float(spec["offset_pct"]),
                        "timeout_min": int(spec["timeout_min"]),
                        "signal_profile": str(signal_profile).lower(),
                    }
                )
                existing_keys.add(key)
                scheduled_any = True

            # If everything was already scheduled for this (symbol, signal),
            # treat it as scheduled to prevent falling back to immediate entry.
            if not scheduled_any and has_existing_for_symbol_signal and not immediate_real_accounts:
                return True

    # Execute immediate entries for accounts that do not use dynamic entry.
    mp_now = None
    try:
        mp_now = get_last_price(symbol)
    except Exception:
        mp_now = None
    if immediate_real_accounts:
        max_acc_workers = min(len(immediate_real_accounts), 8)
        with ThreadPoolExecutor(max_workers=max_acc_workers) as acc_exec:
            futs = []
            for acc_idx in immediate_real_accounts:
                futs.append(
                    acc_exec.submit(
                        _execute_real_entry,
                        symbol,
                        signal,
                        entry_price,
                        confidence,
                        acc_idx,
                        None,
                        created_ts,
                        ep,
                        mp_now,
                        signal_profile,
                    )
                )
            for acc_idx, f in zip(immediate_real_accounts, futs):
                try:
                    f.result()
                except Exception as oe:
                    log(
                        f"[ORDER] Error executing immediate entry for {symbol} on account {acc_idx}: {oe}"
                    )

    if scheduled_any:
        try:
            log(
                f"[ENTRY] Scheduled dynamic entries for {symbol} {signal} "
                f"on {len(pending_specs)} account slot(s) (base_price={format_price(ep)})"
            )
        except Exception:
            pass

    return scheduled_any or bool(immediate_real_accounts)


def _passes_activity_filter(
    candles,
    min_5m_move_pct=None,
    min_1m_notional_usd=None,
    use_1h_move_filter: bool = False,
    min_1h_move_pct: float = 1.5,
    require_1h_move_pct: float = 0.0,
    min_5m_trades: int = 0,
):
    if len(candles) < 5:
        return False

    if use_1h_move_filter:
        if len(candles) < 60:
            return False
        window = candles[-60:]
        open_ref = window[0]["open"]
        close_ref = window[-1]["close"]
        if open_ref <= 0:
            return False
        move_pct = abs(close_ref - open_ref) / open_ref * 100.0
        min_move = float(min_1h_move_pct)
    else:
        last5 = candles[-5:]
        open5 = last5[0]["open"]
        close5 = last5[-1]["close"]
        if open5 <= 0:
            return False
        move_pct = abs(close5 - open5) / open5 * 100.0
        min_move = MIN_5M_MOVE_PCT if min_5m_move_pct is None else float(min_5m_move_pct)

    last = candles[-1]
    notional = last["close"] * last["volume"]

    min_notional = MIN_1M_NOTIONAL_USD if min_1m_notional_usd is None else float(min_1m_notional_usd)

    if move_pct < min_move:
        return False
    if notional < min_notional:
        return False

    # Additional 1h move check (runs alongside 5m, not instead of it)
    if require_1h_move_pct > 0:
        if len(candles) < 60:
            return False
        window = candles[-60:]
        open_ref = window[0]["open"]
        close_ref = window[-1]["close"]
        if open_ref <= 0:
            return False
        move_1h_pct = abs(close_ref - open_ref) / open_ref * 100.0
        if move_1h_pct < require_1h_move_pct:
            return False

    # 5m trades count check
    if min_5m_trades > 0:
        last_trades = candles[-1].get("trades")
        if not isinstance(last_trades, (int, float)) or int(last_trades) < min_5m_trades:
            return False

    return True


def _fire_vardan_webhook(symbol: str, action: str, confidence: float) -> None:
    """Send signal to sofast.am webhook in a daemon thread (fire-and-forget).

    Deduplicates: the same (symbol, action) pair is sent only once until the
    direction changes, so consecutive candles with the same signal fire once.
    Routes through a working proxy to bypass Cloudflare IP block on the server.
    """
    with _VARDAN_LAST_SENT_LOCK:
        if _VARDAN_LAST_SENT.get(symbol) == action:
            return
        _VARDAN_LAST_SENT[symbol] = action

    def _post() -> None:
        import requests as _req
        payload = {
            "symbol": symbol,
            "action": action,
            "confidence": round(float(confidence), 4),
            "message": "Vardans Signal",
        }
        headers = {
            "Authorization": f"Bearer {_VARDAN_WEBHOOK_TOKEN}",
            "Content-Type": "application/json",
        }
        # Try up to 3 different proxies; sofast.am blocks the server IP via Cloudflare
        proxies_to_try = []
        working = get_working_proxies()
        import random as _rand
        sample = _rand.sample(working, min(3, len(working))) if working else []
        proxies_to_try = sample or [get_random_proxy()]
        for proxy in proxies_to_try:
            try:
                proxy_dict = proxy if isinstance(proxy, dict) else None
                if proxy_dict is None:
                    from config.proxies import build_proxy_from_string
                    proxy_dict = build_proxy_from_string(proxy)
                r = _req.post(
                    _VARDAN_WEBHOOK_URL,
                    json=payload,
                    headers=headers,
                    proxies=proxy_dict,
                    timeout=8,
                )
                if r.status_code == 200:
                    return
            except Exception:
                continue

    threading.Thread(target=_post, daemon=True).start()


def _process_symbol(symbol, signal_profile="small"):
    signal = "NO_TRADE"
    confidence = 0.0
    entry_price = None
    signal_telegram_sent = False
    profile = str(signal_profile).strip().lower()
    profile_label = "small" if profile != "large" else "large"
    profile_tag = SECONDARY_TELEGRAM_PREFIX if profile_label == "large" else ""
    filter_move = MIN_5M_MOVE_PCT
    filter_notional = MIN_1M_NOTIONAL_USD
    use_1h_move_filter = False
    min_1h_move_pct = 1.5
    require_1h_move_pct = MIN_1H_MOVE_PCT
    min_5m_trades = MIN_5M_TRADES
    candle_limit = 50
    if profile_label == "large":
        filter_notional = SECONDARY_MIN_1M_NOTIONAL_USD
        use_1h_move_filter = bool(SECONDARY_USE_1H_MOVE_FILTER)
        min_1h_move_pct = float(SECONDARY_MIN_1H_MOVE_PCT)
        if use_1h_move_filter:
            candle_limit = 70

    try:
        if is_delisting_symbol(symbol):
            log(f"[DELIST] Skipping {symbol}: currently in Binance delisting announcements")
            return
        # is_symbol_blocked check removed: trade is blocked in place_order independently.
        # Signal (Telegram commission) must always pass through per architecture requirement.

        candles = fetch_candles(symbol, interval=INTERVAL, limit=candle_limit)
        if not candles or len(candles) < 20:
            return

        if not _passes_activity_filter(
            candles,
            min_5m_move_pct=filter_move,
            min_1m_notional_usd=filter_notional,
            use_1h_move_filter=use_1h_move_filter,
            min_1h_move_pct=min_1h_move_pct,
            require_1h_move_pct=require_1h_move_pct,
            min_5m_trades=min_5m_trades,
        ):
            return

        # Build feature set once (ATR, Fibonacci, structure, etc.)
        features = build_features(candles)
        atr_value = features.get("atr", 0.0)
        fib = features.get("fibonacci")
        entry_price = candles[-1]["close"]

        # Skip trade if risk engine disallows
        if not allow_trade():
            log(f"Skipping {symbol} due to risk limits")
            return

        # Generate combined signal (rule + ML)
        signal, confidence = generate_signal(candles, features=features)

        # Persist per-symbol auto-risk context so that, if some accounts are
        # configured with auto_sl_tp, the exit manager can later derive
        # dynamic SL/TP distances from ATR and model confidence.
        _AUTO_RISK_CONTEXT[symbol] = {
            "atr": atr_value,
            "confidence": confidence,
        }

        # Only log meaningful signals (BUY/SELL), skip NO_TRADE
        if signal != "NO_TRADE":
            log(f"[PROFILE: {profile_label}] {symbol} signal: {signal} (confidence {confidence:.2f})")
            _fire_vardan_webhook(symbol, "long" if signal == "BUY" else "short", confidence)

            # --- Professional analytics: ATR-based SL/TP ---
            # sl/tps -> canonical signal config (for Telegram + analytics)
            # sl_order/tps_order -> per-account config (for actual orders)
            sl = None
            tps_str = ""
            tps = []
            sl_order = None
            tps_order = []
            ai_trade_allowed = True
            ai_market_state = "ACTIVE"

            # Use explicit percent-based config for SL/TP even if ATR is missing
            # or very small. compute_sl_tp will fall back to percent mode when
            # sl_pct / tp_pcts are provided.
            direction_for_sl = "LONG" if signal == "BUY" else "SHORT"
            if profile_label == "large":
                profile_signal_sl_pct = float(SECONDARY_SIGNAL_SL_PCT)
                profile_signal_tp_pcts = list(SECONDARY_SIGNAL_TP_PCTS)
            else:
                profile_signal_sl_pct = float(SIGNAL_SL_PCT)
                profile_signal_tp_pcts = list(SIGNAL_TP_PCTS)

            # Canonical SL/TP for signals / analytics (independent of account)
            sl_sig_raw, tps_sig_raw = compute_sl_tp(
                entry_price,
                atr_value,
                direction_for_sl,
                sl_pct=profile_signal_sl_pct,
                tp_pcts=profile_signal_tp_pcts,
            )

            # Apply AI-based adjustment layer for Telegram-facing SL/TP and,
            # when enabled per-account, for live exit management as well.
            try:
                # Extended market context for AutoStopLossTakeProfit AI engine.
                closes = [float(c.get("close") or 0.0) for c in candles if isinstance(c, dict)]
                last_c = candles[-1] if candles else {}
                try:
                    o = float(last_c.get("open") or 0.0)
                    h = float(last_c.get("high") or 0.0)
                    l = float(last_c.get("low") or 0.0)
                    c = float(last_c.get("close") or 0.0)
                    v = float(last_c.get("volume") or 0.0)
                except Exception:
                    o, h, l, c, v = 0.0, 0.0, 0.0, 0.0, 0.0

                candle_range_pct = abs(h - l) / c * 100.0 if c > 0 else 0.0
                ranges = []
                for cc in candles[-21:-1]:
                    try:
                        hh = float(cc.get("high") or 0.0)
                        ll = float(cc.get("low") or 0.0)
                        cl = float(cc.get("close") or 0.0)
                    except Exception:
                        continue
                    if cl > 0:
                        ranges.append(abs(hh - ll) / cl * 100.0)
                avg_prev_range = (sum(ranges) / float(len(ranges))) if ranges else candle_range_pct
                range_expansion = (candle_range_pct / avg_prev_range) if avg_prev_range > 1e-12 else 1.0

                def _ret_std(win: int) -> float:
                    if len(closes) < win + 1:
                        return 0.0
                    rs = []
                    sub = closes[-(win + 1):]
                    for i in range(1, len(sub)):
                        p0 = float(sub[i - 1])
                        p1 = float(sub[i])
                        if p0 > 0:
                            rs.append((p1 - p0) / p0 * 100.0)
                    if not rs:
                        return 0.0
                    mu = sum(rs) / float(len(rs))
                    var = sum((x - mu) ** 2 for x in rs) / float(max(1, len(rs)))
                    return var ** 0.5

                vol_1m = abs((closes[-1] - closes[-2]) / closes[-2] * 100.0) if len(closes) >= 2 and closes[-2] > 0 else 0.0
                vol_5m = _ret_std(5)
                vol_1h = _ret_std(60)
                struct_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
                structure_level = float(struct_map.get(str(features.get("structure") or "NEUTRAL"), 0.0))
                trend_strength = min(
                    1.0,
                    (
                        abs(float(features.get("momentum") or 0.0)) * 0.4
                        + abs(float(features.get("acceleration") or 0.0)) * 0.3
                        + abs((c - o) / o) * 100.0 * 0.3 if o > 0 else 0.0
                    ),
                )

                ai_rec = suggest_sl_tp(
                    symbol=symbol,
                    side=direction_for_sl,
                    entry_price=entry_price,
                    base_sl_pct=profile_signal_sl_pct,
                    base_tp_pcts=profile_signal_tp_pcts,
                    features={
                        "atr": float(atr_value) if atr_value is not None else 0.0,
                        "confidence": float(confidence),
                        "signal_strength": float(confidence),
                        "last_price": float(entry_price),
                        "candle_range_pct": float(candle_range_pct),
                        "trend_strength": float(trend_strength),
                        "structure_level": float(structure_level),
                        "volatility_1m": float(vol_1m),
                        "volatility_5m": float(vol_5m),
                        "volatility_1h": float(vol_1h),
                        "range_expansion": float(range_expansion),
                        "volume": float(v),
                        "rsi": float(features.get("rsi") or 50.0),
                        "momentum": float(features.get("momentum") or 0.0),
                        "acceleration": float(features.get("acceleration") or 0.0),
                    },
                )
                sig_sl_pct = float(ai_rec.get("sl_pct", profile_signal_sl_pct))
                sig_tp_pcts = ai_rec.get("tp_pcts", profile_signal_tp_pcts) or profile_signal_tp_pcts
                ai_trade_allowed = bool(ai_rec.get("trade_allowed", True))
                ai_market_state = str(ai_rec.get("market_state", "ACTIVE")).upper()
                if (not ai_trade_allowed) and SLTP_AI_ENFORCE_TRADE_BLOCK:
                    log(
                        f"[SLTP-AI][SKIP][PROFILE: {profile_label}] {symbol} {signal}: "
                        f"market_state={ai_market_state} trade_allowed=false"
                    )
                    return
                if (not ai_trade_allowed) and (not SLTP_AI_ENFORCE_TRADE_BLOCK):
                    log(
                        f"[SLTP-AI][ADVISORY][PROFILE: {profile_label}] {symbol} {signal}: "
                        f"market_state={ai_market_state} trade_allowed=false (advisory-only)"
                    )

                # Persist AI SL/TP percents per symbol + direction so that the
                # exit manager can later apply the same AI distances for
                # accounts that have auto_sl_tp enabled.
                try:
                    ctx_payload = {
                        "sl_pct": float(sig_sl_pct),
                        "tp_pcts": list(sig_tp_pcts),
                        "ts": float(time.time()),
                        "market_state": str(ai_market_state),
                        "source": "sltp_nn",
                        "profile": str(profile_label),
                    }
                    # Profile-scoped key prevents cross-profile leakage
                    # (e.g. large-profile 0.8/1.2/1.8 overriding small profile).
                    _AI_SLTP_CONTEXT[(str(profile_label), symbol, direction_for_sl)] = dict(ctx_payload)
                    # Backward compatibility fallback key.
                    _AI_SLTP_CONTEXT[(symbol, direction_for_sl)] = dict(ctx_payload)
                except Exception:
                    pass
            except Exception:
                sig_sl_pct = profile_signal_sl_pct
                sig_tp_pcts = profile_signal_tp_pcts

            # Recompute canonical SL/TP prices for Telegram using the
            # AI-adjusted percent distances, while keeping account-level
            # SL/TP (sl_order/tps_order) unchanged for actual orders.
            sl_sig_raw, tps_sig_raw = compute_sl_tp(
                entry_price,
                atr_value,
                direction_for_sl,
                sl_pct=sig_sl_pct,
                tp_pcts=sig_tp_pcts,
            )

            # Align SL/TP prices to Binance tickSize so that Telegram and
            # logging always show executable levels.
            sl = adjust_price(symbol, sl_sig_raw) if sl_sig_raw is not None else None
            if sl is not None and sl <= 0:
                sl = None

            tps = []
            for tp in tps_sig_raw:
                adj_tp = adjust_price(symbol, tp)
                if adj_tp and adj_tp > 0:
                    tps.append(adj_tp)

            # Per-account SL/TP for actual order management
            sl_ord_raw, tps_ord_raw = compute_sl_tp(
                entry_price,
                atr_value,
                direction_for_sl,
                sl_pct=ACCOUNT_SL_PCT,
                tp_pcts=ACCOUNT_TP_PCTS,
            )

            sl_order = adjust_price(symbol, sl_ord_raw) if sl_ord_raw is not None else None
            if sl_order is not None and sl_order <= 0:
                sl_order = None

            tps_order = []
            for tp in tps_ord_raw:
                adj_tp = adjust_price(symbol, tp)
                if adj_tp and adj_tp > 0:
                    tps_order.append(adj_tp)

            if sl is not None or tps:
                tps_str = ", ".join(format_price(tp) for tp in tps)
                log(
                    f"[PROFILE: {profile_label}] {symbol} ATR SL={format_price(sl) if sl is not None else 'N/A'}, TP levels={tps_str} (entry={format_price(entry_price)}, ATR={format_price(atr_value)})"
                )

            # --- Professional analytics: Fibonacci key levels ---
            fib_levels_str = ""
            if fib and isinstance(fib, dict) and "levels" in fib:
                levels = fib["levels"]
                key_keys = [k for k in ("0.382", "0.5", "0.618") if k in levels]
                if key_keys:
                    fib_levels_str = ", ".join(
                        f"{k}={format_price(levels[k])}" for k in key_keys
                    )
                    log(
                        f"[PROFILE: {profile_label}] {symbol} Fibo ({fib.get('direction')} swing): {fib_levels_str}"
                    )

            last_candle = candles[-1]

            sig_ts_raw = last_candle.get("close_time") or last_candle.get("open_time")
            try:
                sig_ts = int(float(sig_ts_raw))
            except (TypeError, ValueError):
                sig_ts = None

            if sig_ts is not None and sig_ts > 0 and signal in ("BUY", "SELL"):
                try:
                    with _LAST_SIGNAL_CANDLE_LOCK:
                        prev = _LAST_SIGNAL_CANDLE_SEEN.get((profile_label, symbol))
                        if isinstance(prev, tuple) and len(prev) == 2:
                            prev_sig, prev_ts = prev
                            if prev_sig == signal and prev_ts == sig_ts:
                                return
                        _LAST_SIGNAL_CANDLE_SEEN[(profile_label, symbol)] = (signal, sig_ts)
                except Exception:
                    pass

            if signal in ("BUY", "SELL"):
                try:
                    sym_key = str(symbol).upper()
                except Exception:
                    sym_key = symbol
                try:
                    now_wall = time.time()
                    with _LAST_SIGNAL_COOLDOWN_LOCK:
                        last_wall = _LAST_SIGNAL_COOLDOWN_SEEN.get((profile_label, sym_key, signal))
                        if last_wall is not None:
                            try:
                                if (float(now_wall) - float(last_wall)) < float(_SIGNAL_REPEAT_COOLDOWN_SEC):
                                    return
                            except Exception:
                                pass
                        _LAST_SIGNAL_COOLDOWN_SEEN[(profile_label, sym_key, signal)] = float(now_wall)
                except Exception:
                    pass

                # Optional per-profile symbol cooldown (flag-driven):
                # if enabled, suppress any repeated signal on the same symbol
                # (regardless of BUY/SELL direction) for configured minutes.
                try:
                    guard_enabled, guard_cooldown_sec = _get_symbol_repeat_guard_for_profile(profile_label)
                except Exception:
                    guard_enabled, guard_cooldown_sec = False, 0.0
                if guard_enabled and float(guard_cooldown_sec) > 0.0:
                    try:
                        now_wall = time.time()
                        with _PROFILE_SYMBOL_COOLDOWN_LOCK:
                            last_wall = _PROFILE_SYMBOL_COOLDOWN_SEEN.get((profile_label, sym_key))
                            if last_wall is not None:
                                try:
                                    if (float(now_wall) - float(last_wall)) < float(guard_cooldown_sec):
                                        return
                                except Exception:
                                    pass
                            _PROFILE_SYMBOL_COOLDOWN_SEEN[(profile_label, sym_key)] = float(now_wall)
                    except Exception:
                        pass

            def _post_signal_tasks():
                vol_1h = None
                act_5m = None
                vol_24h = None
                try:
                    vol_1h = _futures_1h_volatility(symbol)
                except Exception:
                    vol_1h = None

                try:
                    act_5m = _futures_5m_activity(symbol)
                except Exception:
                    act_5m = None
                try:
                    vol_24h = _futures_24h_quote_volume_usdt(symbol)
                except Exception:
                    vol_24h = None

                try:
                    log_signal(
                        symbol=symbol,
                        direction=signal,
                        entry=entry_price,
                        sl=sl,
                        tps=tps,
                        confidence=confidence,
                        features=features,
                        last_candle=last_candle,
                        interval=INTERVAL,
                    )
                except Exception as e:
                    log(f"Signal log failed for {symbol}: {e}")

                try:
                    log_signal_details(
                        symbol=symbol,
                        direction=signal,
                        interval=INTERVAL,
                        entry=entry_price,
                        sl=sl,
                        tps=tps,
                        confidence=confidence,
                        last_candle=last_candle,
                        vol_1h=vol_1h,
                        act_5m=act_5m,
                    )
                except Exception as e:
                    log(f"Signal details log failed for {symbol}: {e}")

                base_symbol = (
                    symbol.replace("USDT", "/USDT") if symbol.endswith("USDT") else symbol
                )
                direction_text = "Long🟢" if signal == "BUY" else "Short🔴"
                sig_time_line = None
                lines = [
                    f"{profile_tag + ' ' if profile_tag else ''}#{base_symbol} - {direction_text}",
                    "",
                ]

                if sig_ts is not None and sig_ts > 0:
                    try:
                        sig_time_line = time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(int(sig_ts) / 1000.0),
                        )
                    except Exception:
                        sig_time_line = None
                if not sig_time_line:
                    try:
                        sig_time_line = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    except Exception:
                        sig_time_line = None

                if sig_time_line:
                    lines.append(f"Time: {sig_time_line}")
                    lines.append("")

                try:
                    lines.append(f"Entry: {format_price(entry_price)}")
                except Exception:
                    lines.append(f"Entry: {entry_price}")

                if (
                    DYNAMIC_ENTRY_ENABLED
                    and ACCOUNT_ENTRY_OFFSET_PCT
                    and ACCOUNT_ENTRY_OFFSET_PCT != 0.0
                    and ACCOUNT_ENTRY_TIMEOUT_MIN
                    and ACCOUNT_ENTRY_TIMEOUT_MIN > 0
                    and signal in ("BUY", "SELL")
                ):
                    try:
                        ep_dyn = float(entry_price)
                    except (TypeError, ValueError):
                        ep_dyn = None
                    if ep_dyn and ep_dyn > 0:
                        offset_frac = abs(ACCOUNT_ENTRY_OFFSET_PCT) / 100.0
                        if signal == "BUY":
                            target_price = ep_dyn * (1.0 - offset_frac)
                        else:
                            target_price = ep_dyn * (1.0 + offset_frac)
                        try:
                            lines.append(
                                "Dynamic entry: "
                                f"{ACCOUNT_ENTRY_OFFSET_PCT:.2f}% offset "
                                f"=> target {format_price(target_price)} "
                                f"(timeout {ACCOUNT_ENTRY_TIMEOUT_MIN}m)"
                            )
                        except Exception:
                            pass

                if sl is not None:
                    try:
                        lines.append(
                            f"Stop Loss: {format_price(sl)}{_distance_pct_text(entry_price, sl)}"
                        )
                    except Exception:
                        lines.append(f"Stop Loss: {sl}")

                lines.append("")
                if tps:
                    try:
                        if len(tps) >= 1:
                            lines.append(
                                f"Target 1: {format_price(tps[0])}{_distance_pct_text(entry_price, tps[0])}"
                            )
                        if len(tps) >= 2:
                            lines.append(
                                f"Target 2: {format_price(tps[1])}{_distance_pct_text(entry_price, tps[1])}"
                            )
                        if len(tps) >= 3:
                            lines.append(
                                f"Target 3: {format_price(tps[2])}{_distance_pct_text(entry_price, tps[2])}"
                            )
                    except Exception:
                        pass

                lines.append("")
                try:
                    lines.append(f"Leverage: x{DEFAULT_LEVERAGE}")
                except Exception:
                    pass

                try:
                    conf_pct = float(confidence) * 100.0
                    lines.append(f"Confidence: {conf_pct:.1f}%")
                except (TypeError, ValueError):
                    pass

                if vol_1h is not None:
                    try:
                        lines.append(f"Last 1h range: {vol_1h:.2f}%")
                    except Exception:
                        pass
                if act_5m is not None:
                    try:
                        lt = act_5m.get("last_trades")
                        at = act_5m.get("avg_trades")
                        st = act_5m.get("status")
                        if lt is not None and at is not None and st:
                            lines.append(
                                f"Last 5m trades: {int(lt)} (avg {at:.0f}) - {st}"
                            )
                    except Exception:
                        pass
                try:
                    thr_txt = _format_usdt_volume(MIN_24H_QUOTE_VOLUME_USDT)
                    vol_txt = _format_usdt_volume(vol_24h)
                    gate_ok = (
                        True
                        if float(MIN_24H_QUOTE_VOLUME_USDT) <= 0
                        else (vol_24h is not None and float(vol_24h) >= float(MIN_24H_QUOTE_VOLUME_USDT))
                    )
                    status = "✅" if gate_ok else "❌"
                    lines.append(f"Last 24h volume: {vol_txt} USDT (min {thr_txt}) {status}")
                except Exception:
                    pass

                # Coin tradability status from AutoStopLossTakeProfit AI engine.
                try:
                    worth_txt = "Արժի աշխատել ✅" if bool(ai_trade_allowed) else "Չի արժի աշխատել ❌"
                    lines.append(f"Coin վիճակ: {str(ai_market_state).upper()} | {worth_txt}")
                except Exception:
                    pass

                # MQ (Market Quality) verdict — shown in every signal so subscribers know coin quality
                try:
                    _mqa = _get_mqa_info(symbol)
                    _mqa_verdict = _mqa.get("verdict")
                    _mqa_score = _mqa.get("score")
                    if _mqa_verdict:
                        if _mqa_verdict == "GOOD":
                            _mqa_icon = "✅"
                            _mqa_trade = "Trade: OPEN"
                        elif _mqa_verdict == "SUSPECT":
                            _mqa_icon = "⚠️"
                            _mqa_trade = "Trade: BLOCKED"
                        elif _mqa_verdict == "MANIPULATED":
                            _mqa_icon = "🚫"
                            _mqa_trade = "Trade: BLOCKED"
                        else:
                            _mqa_icon = "❓"
                            _mqa_trade = ""
                        _score_txt = f" score={_mqa_score}/100" if _mqa_score is not None else ""
                        lines.append(f"MQ: {_mqa_icon}{_mqa_verdict}{_score_txt} | {_mqa_trade}")
                    else:
                        lines.append("MQ: ❓NOT ANALYZED | Trade: OPEN")
                except Exception:
                    pass

                msg = "\n".join(lines)

                if sig_ts is not None:
                    try:
                        _schedule_signal_followup(symbol, signal, INTERVAL, sig_ts)
                    except Exception as se:
                        try:
                            log(f"[FOLLOWUP] Failed to schedule follow-up for {symbol}: {se}")
                        except Exception:
                            pass

                if profile_label == "small" and TELEGRAM_TOKEN:
                    static_chat_id = None
                    try:
                        if TELEGRAM_CHAT_ID is not None and str(TELEGRAM_CHAT_ID).strip() not in ("", "0", "-"):
                            static_chat_id = int(TELEGRAM_CHAT_ID)
                    except Exception:
                        static_chat_id = None
                    try:
                        subscribers = get_subscribers(static_chat_id, token=TELEGRAM_TOKEN)
                    except Exception:
                        subscribers = []
                    targets = list(subscribers)
                    if not targets and static_chat_id is not None:
                        targets = [static_chat_id]
                    for chat_id in targets:
                        try:
                            msg_id = send_telegram(msg, TELEGRAM_TOKEN, chat_id)
                            if sig_ts is not None and msg_id is not None:
                                record_signal_message(symbol, sig_ts, chat_id, msg_id)
                        except Exception:
                            ""
                else:
                    _send_signal_message_for_profile(msg, profile_label)

            try:
                signal_telegram_sent = True
                threading.Thread(target=_post_signal_tasks, daemon=True).start()
            except Exception:
                pass

        # Dynamic or immediate entry handling for real and paper trades.
        do_paper = signal != "NO_TRADE"
        do_real = TRADING_ENABLED and signal != "NO_TRADE" and confidence > 0.62
        if do_real:
            try:
                vol_ok, vol_24h, vol_thr = _passes_24h_volume_gate(symbol)
            except Exception:
                vol_ok, vol_24h, vol_thr = True, None, float(MIN_24H_QUOTE_VOLUME_USDT)
            if not vol_ok:
                do_real = False
                try:
                    log(
                        f"[ENTRY][SKIP][VOL24H][PROFILE: {profile_label}] {symbol} {signal}: "
                        f"24h quote volume={_format_usdt_volume(vol_24h)} USDT < min {_format_usdt_volume(vol_thr)} USDT"
                    )
                except Exception:
                    pass

        mp_now = None
        try:
            mp_now = get_last_price(symbol)
        except Exception:
            mp_now = None

        if do_paper or do_real:
            scheduled = _schedule_pending_entry(
                symbol,
                signal,
                entry_price,
                confidence,
                atr_value,
                do_real,
                do_paper,
                profile_label,
            )

            if not scheduled:
                if do_paper:
                    try:
                        open_paper_trades(symbol, signal, entry_price, confidence, atr_value)
                    except Exception as pe:
                        log(f"[PAPER] Failed to open paper trades for {symbol}: {pe}")

                if do_real:
                    try:
                        try:
                            positions_snapshot = _get_open_positions_snapshot_cached()
                        except Exception:
                            positions_snapshot = []

                        sym_u = None
                        try:
                            sym_u = str(symbol).upper()
                        except Exception:
                            sym_u = symbol

                        def _has_open_pos(acc_idx: int) -> bool:
                            if not positions_snapshot or sym_u is None:
                                return False
                            for p in positions_snapshot:
                                if not isinstance(p, dict):
                                    continue
                                try:
                                    if int(p.get("account_index")) != int(acc_idx):
                                        continue
                                except Exception:
                                    continue
                                try:
                                    if str(p.get("symbol") or "").upper() != sym_u:
                                        continue
                                except Exception:
                                    continue
                                amt = p.get("position_amt")
                                try:
                                    if float(amt) != 0.0:
                                        return True
                                except Exception:
                                    continue
                            return False

                        # In multi-account mode, open the trade per account so
                        # Telegram notifications can include the correct account name.
                        cfg_live = _get_accounts_cfg_live()
                        if isinstance(cfg_live, dict) and str(cfg_live.get("mode") or "").lower() == "multi":
                            lst = cfg_live.get("accounts") or []
                            if isinstance(lst, list) and lst:
                                active = []
                                for idx, acc in enumerate(lst):
                                    if not isinstance(acc, dict):
                                        continue
                                    if acc.get("trade_enabled") is False:
                                        continue
                                    settings_local = acc.get("settings") or {}
                                    if not isinstance(settings_local, dict):
                                        settings_local = {}
                                    if not _account_accepts_signal_profile(settings_local, profile_label):
                                        continue
                                    try:
                                        if _has_open_pos(int(idx)):
                                            continue
                                    except Exception:
                                        pass
                                    active.append(idx)

                                if active:
                                    max_acc_workers = min(len(active), 8)
                                    with ThreadPoolExecutor(max_workers=max_acc_workers) as acc_exec:
                                        futs = []
                                        for idx in active:
                                            futs.append(
                                                acc_exec.submit(
                                                    _execute_real_entry,
                                                    symbol,
                                                    signal,
                                                    entry_price,
                                                    confidence,
                                                    idx,
                                                    None,
                                                    sig_ts,
                                                    entry_price,
                                                    mp_now,
                                                    profile_label,
                                                )
                                            )
                                        for idx, f in zip(active, futs):
                                            try:
                                                f.result()
                                            except Exception as oe2:
                                                log(
                                                    f"[ORDER] Error executing real entry for {symbol} on account {idx}: {oe2}"
                                                )
                            else:
                                if not _has_open_pos(0):
                                    _execute_real_entry(
                                        symbol,
                                        signal,
                                        entry_price,
                                        confidence,
                                        signal_ts=sig_ts,
                                        signal_entry_price=entry_price,
                                        market_price=mp_now,
                                        signal_profile=profile_label,
                                    )
                        else:
                            if not _has_open_pos(0):
                                _execute_real_entry(
                                    symbol,
                                    signal,
                                    entry_price,
                                    confidence,
                                    signal_ts=sig_ts,
                                    signal_entry_price=entry_price,
                                    market_price=mp_now,
                                    signal_profile=profile_label,
                                )
                    except Exception as oe:
                        log(f"[ORDER] Error executing real entry for {symbol}: {oe}")

    except Exception as e:
        log(f"Error processing {symbol}: {e}")
    finally:
        # Fallback: if we computed a BUY/SELL signal but the main body raised
        # an exception before sending the standard Telegram alert, send a
        # normal-looking signal message (entry, optional SL/TP, leverage)
        # without exposing internal error details.
        if signal != "NO_TRADE" and not signal_telegram_sent:
            try:
                base_symbol = (
                    symbol.replace("USDT", "/USDT") if symbol.endswith("USDT") else symbol
                )
                direction_text = "Long🟢" if signal == "BUY" else "Short🔴"
                price_str = format_price(entry_price) if entry_price else "N/A"

                lines = [
                    f"{profile_tag + ' ' if profile_tag else ''}#{base_symbol} - {direction_text}",
                    "",
                    f"Entry: {price_str}",
                ]

                if (
                    DYNAMIC_ENTRY_ENABLED
                    and ACCOUNT_ENTRY_OFFSET_PCT
                    and ACCOUNT_ENTRY_OFFSET_PCT != 0.0
                    and ACCOUNT_ENTRY_TIMEOUT_MIN
                    and ACCOUNT_ENTRY_TIMEOUT_MIN > 0
                    and signal in ("BUY", "SELL")
                ):
                    try:
                        ep_dyn = float(entry_price)
                    except (TypeError, ValueError):
                        ep_dyn = None

                    if ep_dyn and ep_dyn > 0:
                        offset_frac = abs(ACCOUNT_ENTRY_OFFSET_PCT) / 100.0
                        if signal == "BUY":
                            target_price = ep_dyn * (1.0 - offset_frac)
                        else:
                            target_price = ep_dyn * (1.0 + offset_frac)

                        lines.append(
                            "Dynamic entry: "
                            f"{ACCOUNT_ENTRY_OFFSET_PCT:.2f}% offset "
                            f"=> target {format_price(target_price)} "
                            f"(timeout {ACCOUNT_ENTRY_TIMEOUT_MIN}m)"
                        )

                # Try to include SL/TP levels and leverage if they are available
                try:
                    if "sl" in locals() and sl is not None:
                        lines.append(
                            f"Stop Loss: {format_price(sl)}{_distance_pct_text(entry_price, sl)}"
                        )

                    # Blank line before targets
                    lines.append("")

                    if "tps" in locals() and tps:
                        if len(tps) >= 1:
                            lines.append(
                                f"Target 1: {format_price(tps[0])}{_distance_pct_text(entry_price, tps[0])}"
                            )
                        if len(tps) >= 2:
                            lines.append(
                                f"Target 2: {format_price(tps[1])}{_distance_pct_text(entry_price, tps[1])}"
                            )
                        if len(tps) >= 3:
                            lines.append(
                                f"Target 3: {format_price(tps[2])}{_distance_pct_text(entry_price, tps[2])}"
                            )

                    lines.append("")
                    lines.append(f"Leverage: x{DEFAULT_LEVERAGE}")

                    # Include ML confidence as a percentage in fallback alerts.
                    try:
                        conf_pct = float(confidence) * 100.0
                        lines.append(f"Confidence: {conf_pct:.1f}%")
                    except (TypeError, ValueError):
                        # If confidence is missing or not numeric, skip this line.
                        pass

                    # Append last 1h volatility (HIGH/LOW range as percent), if available.
                    try:
                        vol_1h = _futures_1h_volatility(symbol)
                        if vol_1h is not None:
                            lines.append(f"Last 1h range: {vol_1h:.2f}%")
                    except Exception:
                        # Never break fallback message because of volatility helper.
                        pass

                    # Append 5m trade activity (number of trades vs historical 5m average), if available.
                    try:
                        act_5m = _futures_5m_activity(symbol)
                        if act_5m is not None:
                            lt = act_5m.get("last_trades")
                            at = act_5m.get("avg_trades")
                            st = act_5m.get("status")
                            if lt is not None and at is not None and st:
                                lines.append(
                                    f"Last 5m trades: {int(lt)} (avg {at:.0f}) - {st}"
                                )
                    except Exception:
                        # Fail silently on any errors.
                        pass
                    try:
                        vol_24h = _futures_24h_quote_volume_usdt(symbol)
                        thr_txt = _format_usdt_volume(MIN_24H_QUOTE_VOLUME_USDT)
                        vol_txt = _format_usdt_volume(vol_24h)
                        gate_ok = (
                            True
                            if float(MIN_24H_QUOTE_VOLUME_USDT) <= 0
                            else (vol_24h is not None and float(vol_24h) >= float(MIN_24H_QUOTE_VOLUME_USDT))
                        )
                        status = "✅" if gate_ok else "❌"
                        lines.append(f"Last 24h volume: {vol_txt} USDT (min {thr_txt}) {status}")
                    except Exception:
                        pass
                    try:
                        _ms = str(locals().get("ai_market_state", "ACTIVE")).upper()
                        _ta = bool(locals().get("ai_trade_allowed", True))
                        _wt = "Արժի աշխատել ✅" if _ta else "Չի արժի աշխատել ❌"
                        lines.append(f"Coin վիճակ: {_ms} | {_wt}")
                    except Exception:
                        pass
                except Exception:
                    # If anything goes wrong while enriching the fallback
                    # message, just send the basic entry-only version.
                    pass

                msg = "\n".join(lines)

                _send_signal_message_for_profile(msg, profile_label)
            except Exception as te2:
                log(f"Failed to send fallback Telegram signal for {symbol}: {te2}")


def _check_manual_exits():
    """Check all open positions and apply SL/TP per account & symbol.

    tp_mode semantics (per account, from config/binance_accounts.yaml::settings.tp_mode):
      - 1: single full exit at TP1
      - 2: two equal legs at TP1 and TP2 (if TP2 configured)
      - 3: three equal legs at TP1, TP2, TP3 (if configured)

    We implement this by keeping an initial position size per (account, symbol)
    and, depending on how many TP levels price has crossed, compute a target
    remaining fraction and close only the excess. SL always closes the full
    remaining position.
    """

    timeout_enabled = bool(
        POSITION_TIMEOUT_ENABLED and POSITION_TIMEOUT_MAX_HOURS and POSITION_TIMEOUT_MAX_HOURS > 0.0
    )
    if not AUTO_SL_TP_ENABLED and not timeout_enabled:
        return

    positions_raw = get_open_positions_snapshot()
    if not positions_raw:
        try:
            now_empty = time.time()
        except Exception:
            now_empty = 0.0

        # Don't immediately clear state on transient API hiccups.
        try:
            last_ok = float(_EXIT_LAST_NONEMPTY_POS_TS or 0.0)
        except Exception:
            last_ok = 0.0

        if _EXIT_POS_STATE and last_ok and now_empty and (now_empty - last_ok) <= 15.0:
            try:
                log(
                    "[EXIT MANAGER] Positions snapshot empty; keeping exit state briefly to avoid TP/SL state reset"
                )
            except Exception:
                pass
            return

        if _EXIT_POS_STATE:
            # Hard cleanup: if snapshots stay empty beyond grace window,
            # cancel any leftover open orders before state drop.
            for k in list(_EXIT_POS_STATE.keys()):
                try:
                    cancel_symbol_open_orders(int(k[0]), str(k[1]))
                except Exception:
                    pass
                try:
                    _adaptive_add_mem_remove(int(k[0]), str(k[1]), None)
                except Exception:
                    pass
            _EXIT_POS_STATE.clear()
        return

    # Hard rule: if account is trade_enabled=false, bot must not control it.
    # We intentionally ignore such positions in EXIT manager logic so manual
    # trading on disabled accounts is never auto-closed by this process.
    positions = []
    disabled_keys = set()
    for pos in positions_raw:
        symbol = pos.get("symbol")
        if not symbol:
            continue
        try:
            acc_idx = int(pos.get("account_index"))
        except (TypeError, ValueError):
            continue
        if bool(pos.get("trade_enabled", True)):
            positions.append(pos)
        else:
            disabled_keys.add((acc_idx, symbol))

    if disabled_keys:
        dropped = 0
        for k in list(_EXIT_POS_STATE.keys()):
            if k in disabled_keys:
                _EXIT_POS_STATE.pop(k, None)
                try:
                    _adaptive_add_mem_remove(int(k[0]), str(k[1]), None)
                except Exception:
                    pass
                dropped += 1
        if dropped > 0:
            try:
                log(
                    f"[EXIT MANAGER] Skipped disabled accounts: removed {dropped} state entries; "
                    f"manual positions on trade_enabled=false accounts are not managed"
                )
            except Exception:
                pass

    if not positions:
        return

    try:
        _EXIT_LAST_NONEMPTY_POS_TS = float(time.time())
    except Exception:
        _EXIT_LAST_NONEMPTY_POS_TS = 0.0

    now = time.time()

    # Determine which (account, symbol) keys are currently active.
    active_keys = set()
    for pos in positions:
        symbol = pos.get("symbol")
        idx_raw = pos.get("account_index")
        if not symbol:
            continue
        try:
            acc_idx = int(idx_raw)
        except (TypeError, ValueError):
            continue
        active_keys.add((acc_idx, symbol))

    # Drop any stale state entries for positions that no longer exist.
    stale_keys = [k for k in _EXIT_POS_STATE.keys() if k not in active_keys]
    for k in stale_keys:
        try:
            # Fire cancel in background — don't block exit manager loop.
            import threading as _thr_stale
            _acc_k, _sym_k = int(k[0]), str(k[1])
            _thr_stale.Thread(
                target=cancel_symbol_open_orders,
                args=(_acc_k, _sym_k),
                daemon=True,
            ).start()
        except Exception:
            pass
        _EXIT_POS_STATE.pop(k, None)
        try:
            _adaptive_add_mem_remove(int(k[0]), str(k[1]), None)
        except Exception:
            pass

    # Fetch latest prices for all symbols that have open positions.
    #
    # IMPORTANT: WebSocket price stream stores kline close prices and may miss
    # intra-candle spikes (wicks). This can cause TP2 hits to be visible in
    # Binance UI while the bot does not close.
    #
    # Use Binance Futures mark price (premiumIndex) as the preferred source,
    # with a small TTL cache to avoid excessive REST calls.
    symbols = set(sym for (_, sym) in active_keys)
    prices = {}
    if AUTO_SL_TP_ENABLED:
        now_ts = time.time()
        missing = []
        for sym in symbols:
            mp = None
            try:
                cached = _MARK_PRICE_CACHE.get(sym)
                if isinstance(cached, dict):
                    ts = float(cached.get("ts") or 0.0)
                    val = cached.get("price")
                    if val is not None and (now_ts - ts) <= _MARK_PRICE_CACHE_TTL_SEC:
                        mp = float(val)
            except Exception:
                mp = None

            if mp is not None and mp > 0:
                prices[sym] = mp
            else:
                missing.append(sym)

        if missing:
            mp_map = {}
            try:
                mp_map = get_public_mark_prices(missing, account_index=0)
            except Exception:
                mp_map = {}

            if mp_map:
                try:
                    now2 = time.time()
                except Exception:
                    now2 = now_ts

                for sym in missing:
                    try:
                        sym_u = str(sym).upper()
                    except Exception:
                        sym_u = None
                    if not sym_u:
                        continue
                    v = mp_map.get(sym_u)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv <= 0:
                        continue
                    prices[sym] = fv
                    try:
                        _MARK_PRICE_CACHE[sym] = {"ts": float(now2), "price": float(fv)}
                    except Exception:
                        pass

        # Best-effort fallback to WS price if mark price is unavailable.
        for sym in symbols:
            if sym in prices:
                continue
            try:
                ws_px = get_last_price(sym)
            except Exception:
                ws_px = None
            if ws_px is not None and ws_px > 0:
                prices[sym] = float(ws_px)

    # Optional per-symbol 1h volatility map for distance-based SL/TP rules.
    vol_1h_map = {}
    if AUTO_SL_TP_ENABLED:
        for sym in symbols:
            try:
                v = _futures_1h_volatility(sym)
            except Exception:
                v = None
            if v is not None:
                vol_1h_map[sym] = v

    # Combined PnL close: if an account has multiple open positions and their
    # total unrealised PnL exceeds a fixed % of wallet balance (from config
    # combined_pnl_close_pct), close ALL positions on that account.
    # Uses actual remaining position_amt from Binance (after partial TP closes).
    _combined_closed_accounts = set()
    if AUTO_SL_TP_ENABLED and prices:
        _acc_positions = {}  # acc_idx -> list of pos dicts
        for pos in positions:
            try:
                ai = int(pos.get("account_index"))
            except (TypeError, ValueError):
                continue
            if not bool(pos.get("combined_pnl_close_enabled")):
                continue
            _acc_positions.setdefault(ai, []).append(pos)

        for ai, acc_poss in _acc_positions.items():
            if len(acc_poss) < 2:
                continue

            try:
                threshold_pct = float(acc_poss[0].get("combined_pnl_close_pct") or 3.0)
            except (TypeError, ValueError):
                threshold_pct = 3.0

            total_upnl = 0.0
            total_notional = 0.0
            for p in acc_poss:
                sym = p.get("symbol")
                if sym not in prices:
                    continue
                try:
                    amt = float(p.get("position_amt") or 0)
                    ep = float(p.get("entry_price") or 0)
                except (TypeError, ValueError):
                    continue
                if amt == 0 or ep <= 0:
                    continue
                cpx = prices[sym]
                total_notional += cpx * abs(amt)
                if amt > 0:
                    total_upnl += (cpx - ep) * amt
                else:
                    total_upnl += (ep - cpx) * abs(amt)

            if total_upnl <= 0:
                continue

            try:
                balance = get_account_wallet_balance(ai)
            except Exception:
                balance = 0.0
            if balance <= 0:
                continue

            pnl_pct_of_balance = (total_upnl / balance) * 100.0
            if pnl_pct_of_balance < threshold_pct:
                continue

            # Threshold met — close ALL positions on this account.
            closed_count = 0
            for p in acc_poss:
                sym = p.get("symbol")
                try:
                    amt = float(p.get("position_amt") or 0)
                except (TypeError, ValueError):
                    continue
                if amt == 0:
                    continue
                ps = p.get("position_side")
                try:
                    submitted = close_position_market(
                        ai, sym, amt, force_full_close=True, position_side=ps,
                    )
                    if submitted:
                        closed_count += 1
                except Exception:
                    pass
                try:
                    cancel_symbol_open_orders(ai, sym)
                except Exception:
                    pass
                _EXIT_POS_STATE.pop((ai, sym), None)

            _combined_closed_accounts.add(ai)
            try:
                log(
                    f"[COMBINED PNL CLOSE] account {ai}: {len(acc_poss)} positions, "
                    f"notional=${total_notional:.2f}, uPnL={total_upnl:.2f} USDT "
                    f"({pnl_pct_of_balance:.2f}% of balance) >= threshold {threshold_pct:.1f}% "
                    f"— closed {closed_count} positions"
                )
            except Exception:
                pass

    for pos in positions:
        symbol = pos.get("symbol")
        idx_raw = pos.get("account_index")
        try:
            acc_idx = int(idx_raw)
        except (TypeError, ValueError):
            continue

        # Skip accounts already fully closed by combined PnL logic above.
        if acc_idx in _combined_closed_accounts:
            continue

        position_side = pos.get("position_side")

        key = (acc_idx, symbol)
        # Ensure we track opened_ts for timeout even when SL/TP is disabled
        # or a live price is unavailable.
        state = _EXIT_POS_STATE.get(key)
        if state is None or not isinstance(state, dict):
            state = {}
            _EXIT_POS_STATE[key] = state

        opened_ts = None
        try:
            opened_ts = float(state.get("opened_ts")) if state.get("opened_ts") is not None else None
        except (TypeError, ValueError):
            opened_ts = None

        opened_src = None
        try:
            opened_src = str(state.get("opened_src")) if state.get("opened_src") is not None else None
        except Exception:
            opened_src = None

        hist_ts = None
        try:
            hist_ts = _get_last_entry_ts_from_history(acc_idx, symbol)
        except Exception:
            hist_ts = None

        exch_ts = None
        try:
            ut_ms = pos.get("update_time_ms")
            if ut_ms is not None:
                exch_ts = float(int(ut_ms)) / 1000.0
        except Exception:
            exch_ts = None

        if opened_ts is None:
            if hist_ts is not None and hist_ts > 0:
                opened_ts = float(hist_ts)
                opened_src = "history"
            elif exch_ts is not None and exch_ts > 0:
                opened_ts = float(exch_ts)
                opened_src = "exchange"
            else:
                opened_ts = now
                opened_src = "now"
        else:
            if (
                (opened_src is None or str(opened_src).lower() != "history")
                and hist_ts is not None
                and hist_ts > 0
                and float(hist_ts) < float(opened_ts) - 60.0
            ):
                opened_ts = float(hist_ts)
                opened_src = "history"

        state["opened_ts"] = opened_ts
        if opened_src is not None:
            state["opened_src"] = str(opened_src)
        if opened_src is not None and state.get("opened_src_logged") != str(opened_src):
            try:
                state["opened_src_logged"] = str(opened_src)
                timeout_sec = float(POSITION_TIMEOUT_MAX_HOURS) * 3600.0 if timeout_enabled else 0.0
                age_sec = float(now) - float(opened_ts) if opened_ts is not None else 0.0
                rem_sec = max(0.0, timeout_sec - age_sec) if timeout_sec > 0 else 0.0
                log(
                    f"[EXIT MANAGER] Track {symbol} on account {acc_idx}: opened_src={opened_src} "
                    f"age={_fmt_duration_hm(age_sec)} remaining={_fmt_duration_hm(rem_sec)}"
                )
            except Exception:
                pass

        if timeout_enabled:
            try:
                elapsed_sec = float(now) - float(opened_ts)
            except Exception:
                elapsed_sec = 0.0
            timeout_sec = float(POSITION_TIMEOUT_MAX_HOURS) * 3600.0
            if elapsed_sec >= timeout_sec:
                det_ms = None
                sub_ms = None
                try:
                    try:
                        det_ms = float(time.time() * 1000.0)
                    except Exception:
                        det_ms = None
                    submitted = close_position_market(
                        acc_idx,
                        symbol,
                        float(pos.get("position_amt") or 0.0),
                        force_full_close=True,
                        position_side=position_side,
                    )
                    if not submitted:
                        log(
                            f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by TIMEOUT: close request not submitted"
                        )
                        continue
                    try:
                        sub_ms = float(time.time() * 1000.0)
                    except Exception:
                        sub_ms = None
                    try:
                        log_exit_timing(
                            symbol=str(symbol),
                            account_index=acc_idx,
                            side="BUY" if float(pos.get("position_amt") or 0.0) > 0 else "SELL",
                            reason="TIMEOUT",
                            trigger_price=None,
                            hit_price=None,
                            detected_ts_ms=det_ms,
                            submit_ts_ms=sub_ms,
                            submit_delay_ms=(float(sub_ms) - float(det_ms)) if (det_ms is not None and sub_ms is not None) else None,
                            timing_tag="TIMEOUT",
                        )
                    except Exception:
                        pass
                    log(
                        f"[EXIT MANAGER] Closed {symbol} position on account {acc_idx} by TIMEOUT "
                        f"after {elapsed_sec / 3600.0:.2f}h"
                    )
                    try:
                        on_trade_exit(
                            account_index=acc_idx,
                            symbol=symbol,
                            side="BUY" if float(pos.get("position_amt") or 0.0) > 0 else "SELL",
                            entry_price=float(pos.get("entry_price") or 0.0),
                            exit_price=None,
                            pnl_pct=None,
                            reason="TIMEOUT",
                        )
                    except Exception:
                        pass
                except Exception as e:
                    log(
                        f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by TIMEOUT: {e}"
                    )
                try:
                    cancel_symbol_open_orders(acc_idx, symbol)
                except Exception:
                    pass
                _EXIT_POS_STATE.pop(key, None)
                continue

        if not AUTO_SL_TP_ENABLED:
            continue

        if symbol not in prices:
            continue

        current_price = prices[symbol]

        amt_raw = pos.get("position_amt")
        try:
            position_amt = float(amt_raw)
        except (TypeError, ValueError):
            continue

        if not position_amt:
            continue

        entry_raw = pos.get("entry_price")
        try:
            entry_price = float(entry_raw)
        except (TypeError, ValueError):
            continue

        if entry_price <= 0:
            continue

        sl_pct = pos.get("sl_pct")
        tp_pcts = pos.get("tp_pcts")
        tp_mode_raw = pos.get("tp_mode")
        auto_sl_flag = pos.get("auto_sl_tp")
        move_sl_to_entry_flag = pos.get("move_sl_to_entry_on_first_tp")
        multi_entry_enabled = bool(pos.get("multi_entry_enabled"))
        multi_entry_levels = pos.get("multi_entry_levels") or []
        position_side = pos.get("position_side")
        ladder_enabled = bool(pos.get("ladder_sl_tp_enabled"))
        ladder_sl_range_pct = pos.get("ladder_sl_range_pct")
        ladder_sl_steps = pos.get("ladder_sl_steps")
        ladder_tp_range_pct = pos.get("ladder_tp_range_pct")
        ladder_tp_steps = pos.get("ladder_tp_steps")
        adaptive_reentry_enabled = bool(pos.get("adaptive_reentry_ladder_enabled"))
        adaptive_reentry_tp_range_pct = pos.get("adaptive_reentry_tp_range_pct")
        adaptive_reentry_tp_steps = pos.get("adaptive_reentry_tp_steps")
        adaptive_reentry_add_range_pct = pos.get("adaptive_reentry_add_range_pct")
        adaptive_reentry_add_steps = pos.get("adaptive_reentry_add_steps")
        adaptive_reentry_sl_pct = pos.get("adaptive_reentry_sl_pct")
        adaptive_reentry_add_total_multiplier = pos.get("adaptive_reentry_add_total_multiplier")
        adaptive_reentry_tighten_factor = pos.get("adaptive_reentry_tighten_factor")

        # Per-account early breakeven & guaranteed-profit SL flags.
        _early_be_enabled = bool(pos.get("early_breakeven_enabled")) and (not ladder_enabled)
        _early_be_pct = None
        try:
            _v = pos.get("early_breakeven_pct")
            if _v is not None:
                _early_be_pct = float(_v)
        except (TypeError, ValueError):
            _early_be_pct = None

        _tp_lock_enabled = bool(pos.get("tp_profit_lock_enabled")) and (not ladder_enabled)
        _tp_lock_pct = None
        try:
            _v = pos.get("tp_profit_lock_pct")
            if _v is not None:
                _tp_lock_pct = float(_v)
        except (TypeError, ValueError):
            _tp_lock_pct = None

        if sl_pct is None:
            sl_pct = ACCOUNT_SL_PCT
        if not isinstance(tp_pcts, (list, tuple)) or not tp_pcts:
            tp_pcts = list(ACCOUNT_TP_PCTS)
        if tp_mode_raw is None:
            tp_mode_raw = TP_MODE

        # Normalise auto_sl_tp semantics: either explicit boolean flag or
        # a string-valued sl_pct of "auto" can enable dynamic SL/TP.
        is_sl_auto_str = isinstance(sl_pct, str) and sl_pct.strip().lower() == "auto"
        auto_sl_enabled = bool(auto_sl_flag) or is_sl_auto_str

        # Parse SL percent once so both AI and static branches can use it.
        sl_val = None
        if sl_pct is not None:
            try:
                sl_val = float(sl_pct)
            except (TypeError, ValueError):
                sl_val = None

        # Optional per-account distance-based overrides for SL/TP behaviour.
        vol_1h = vol_1h_map.get(symbol)
        dist_use_auto = None
        dist_sl_override = None
        dist_tp_override = None
        if vol_1h is not None:
            try:
                dist_use_auto, dist_sl_override, dist_tp_override, _ = compute_distance_based_sl_tp(
                    acc_idx, vol_1h
                )
            except Exception:
                dist_use_auto = None
                dist_sl_override = None
                dist_tp_override = None

        auto_sl_effective = auto_sl_enabled
        if dist_use_auto is True:
            auto_sl_effective = True
        elif dist_use_auto is False:
            auto_sl_effective = False

        if dist_sl_override is not None:
            try:
                sl_val = float(dist_sl_override)
                sl_pct = sl_val
            except (TypeError, ValueError):
                pass

        if dist_tp_override is not None:
            try:
                tp_pcts = list(dist_tp_override)
            except TypeError:
                pass

        if not auto_sl_effective:
            if sl_val is None or not tp_pcts:
                continue

        # Clean TP percentages and cap to at most 3 levels.
        tp_levels = []
        for v in tp_pcts:
            try:
                pv = float(v)
            except (TypeError, ValueError):
                continue
            if pv > 0:
                tp_levels.append(pv)
        if not tp_levels:
            continue

        try:
            tp_mode = int(tp_mode_raw) if tp_mode_raw is not None else 1
        except (TypeError, ValueError):
            tp_mode = 1
        if tp_mode < 1:
            tp_mode = 1

        # Multi-entry positions: tp_mode-ը վերաբերում է ոչ թե equal-split
        # exiting-ին, այլ ընտրում է կոնկրետ TP աստիճանը, որտեղ ամբողջ
        # position-ը կմաքրվի (1=TP1, 2=TP2, 3=TP3):
        if multi_entry_enabled and multi_entry_levels and not auto_sl_effective:
            idx_tp = tp_mode - 1
            if idx_tp < 0:
                idx_tp = 0
            if idx_tp >= len(tp_levels):
                idx_tp = len(tp_levels) - 1
            tp_levels = [tp_levels[idx_tp]]
            num_legs = 1
        else:
            # Սովորական positions-ի համար պահում ենք հին equal-split տրամաբանությունը.
            # Number of TP legs we actually use (1, 2, or 3).
            num_legs = min(tp_mode, len(tp_levels), 3) if not auto_sl_effective else tp_mode
        if num_legs <= 0:
            continue

        long_side = position_amt > 0

        # Ladder SL/TP mode (account-level plugin, isolated by flag).
        # Works only for accounts where ladder_sl_tp_enabled=true.
        if (
            adaptive_reentry_enabled
            and (
                state is None
                or state.get("initial_amt", 0.0) <= 0
                or bool(state.get("long", long_side)) != long_side
                or not state.get("adaptive_reentry_mode")
            )
        ):
            initial_amt = _resolve_base_initial_qty(acc_idx, symbol, entry_price, abs(position_amt))
            mem_row = _adaptive_add_mem_get(acc_idx, symbol, long_side)
            try:
                mem_base_qty = float(mem_row.get("base_qty") or 0.0) if isinstance(mem_row, dict) else 0.0
            except (TypeError, ValueError):
                mem_base_qty = 0.0
            if mem_base_qty > 0:
                try:
                    cur_abs_init = abs(float(position_amt))
                except (TypeError, ValueError):
                    cur_abs_init = initial_amt
                # Prefer persisted base qty on restart if plausible.
                if mem_base_qty <= (cur_abs_init * 1.10):
                    initial_amt = min(cur_abs_init, mem_base_qty)
            if initial_amt > 0:
                # Prefer fresh account-bound AI SL/TP context to avoid static
                # 1%-4% adaptive defaults when NN suggests richer levels.
                direction_str = "LONG" if long_side else "SHORT"
                ai_ctx = (
                    _AI_SLTP_CONTEXT.get((acc_idx, symbol, direction_str))
                    or _AI_SLTP_CONTEXT.get((symbol, direction_str))
                    or {}
                )
                ai_ctx_ts = _safe_float(ai_ctx.get("ts"), 0.0) if isinstance(ai_ctx, dict) else 0.0
                ai_ctx_fresh = (time.time() - ai_ctx_ts) <= 6 * 3600.0 if ai_ctx_ts > 0 else False
                try:
                    sl_pct_val = float(adaptive_reentry_sl_pct)
                except Exception:
                    sl_pct_val = 5.0
                if ai_ctx_fresh:
                    ai_sl = _safe_float(ai_ctx.get("sl_pct"), 0.0)
                    if ai_sl > 0:
                        sl_pct_val = ai_sl
                if sl_pct_val <= 0:
                    sl_pct_val = 5.0

                ai_tp_levels = []
                if ai_ctx_fresh:
                    raw_ai_tp = ai_ctx.get("tp_pcts")
                    if isinstance(raw_ai_tp, (list, tuple)):
                        for v in raw_ai_tp:
                            pv = _safe_float(v, 0.0)
                            if pv > 0:
                                ai_tp_levels.append(float(pv))
                if ai_tp_levels:
                    tp_pct_levels = ai_tp_levels[:3]
                else:
                    tp_pct_levels = _ladder_build_pct_levels(
                        adaptive_reentry_tp_range_pct,
                        adaptive_reentry_tp_steps,
                        fallback_start=1.0,
                        fallback_end=4.0,
                    )
                if auto_sl_effective and tp_pct_levels:
                    # Prevent ultra-tight 1% TP when account is AI-managed.
                    base_floor = 1.5
                    if tp_levels:
                        try:
                            base_floor = max(base_floor, float(tp_levels[0]) * 0.85)
                        except Exception:
                            pass
                    tp_pct_levels = [max(float(v), float(base_floor)) for v in tp_pct_levels]
                    if len(tp_pct_levels) >= 2 and tp_pct_levels[1] < tp_pct_levels[0] * 1.35:
                        tp_pct_levels[1] = tp_pct_levels[0] * 1.35
                    if len(tp_pct_levels) >= 3 and tp_pct_levels[2] < tp_pct_levels[1] * 1.25:
                        tp_pct_levels[2] = tp_pct_levels[1] * 1.25
                add_pct_levels = _ladder_build_pct_levels(
                    adaptive_reentry_add_range_pct,
                    adaptive_reentry_add_steps,
                    fallback_start=2.7,
                    fallback_end=4.0,
                )
                tp_prices_adaptive = _ladder_build_price_levels(
                    entry_price, tp_pct_levels, long_side, is_tp=True, symbol=symbol
                )
                add_prices_adaptive = _ladder_build_price_levels(
                    entry_price, add_pct_levels, long_side, is_tp=False, symbol=symbol
                )
                if long_side:
                    sl_price_adaptive = entry_price * (1.0 - sl_pct_val / 100.0)
                else:
                    sl_price_adaptive = entry_price * (1.0 + sl_pct_val / 100.0)
                try:
                    sl_price_adaptive = float(adjust_price(symbol, sl_price_adaptive))
                except Exception:
                    pass
                add_prices_adaptive = _constrain_adaptive_add_prices(
                    symbol=symbol,
                    add_prices=add_prices_adaptive,
                    entry_price=entry_price,
                    sl_price=sl_price_adaptive if sl_price_adaptive else entry_price,
                    long_side=long_side,
                )
                if sl_price_adaptive and tp_prices_adaptive:
                    try:
                        add_mult = float(adaptive_reentry_add_total_multiplier)
                    except Exception:
                        add_mult = 2.0
                    if add_mult <= 0:
                        add_mult = 2.0
                    try:
                        tighten_factor = float(adaptive_reentry_tighten_factor)
                    except Exception:
                        tighten_factor = 0.2
                    if tighten_factor < 0:
                        tighten_factor = 0.0
                    if tighten_factor > 0.9:
                        tighten_factor = 0.9
                    try:
                        mem_base_entry = float(mem_row.get("base_entry")) if isinstance(mem_row, dict) else 0.0
                    except (TypeError, ValueError):
                        mem_base_entry = 0.0
                    if mem_base_entry <= 0:
                        try:
                            mem_base_entry = float(entry_price)
                        except (TypeError, ValueError):
                            mem_base_entry = 0.0

                    state = {
                        "initial_amt": initial_amt,
                        "long": long_side,
                        "adaptive_reentry_mode": True,
                        "adaptive_tp_prices": list(tp_prices_adaptive),
                        "adaptive_add_prices": list(add_prices_adaptive),
                        "adaptive_sl_price": float(sl_price_adaptive),
                        "adaptive_tp_steps": _ladder_parse_steps(adaptive_reentry_tp_steps, default_steps=len(tp_prices_adaptive)),
                        "adaptive_add_steps": _ladder_parse_steps(adaptive_reentry_add_steps, default_steps=len(add_prices_adaptive)),
                        "adaptive_crossed_tp": 0,
                        "adaptive_crossed_add": 0,
                        "adaptive_last_abs_qty": float(initial_amt),
                        "adaptive_add_total_multiplier": float(add_mult),
                        "adaptive_tighten_factor": float(tighten_factor),
                        "adaptive_base_entry_price": float(mem_base_entry),
                        "adaptive_recovery_tiny_plus_pct": 0.0,
                        "opened_ts": now,
                        "realized_pnl_notional": 0.0,
                        "exited_qty": 0.0,
                        # compatibility/state visibility
                        "tp_prices": list(tp_prices_adaptive),
                        "sl_price": float(sl_price_adaptive),
                        "num_legs": len(tp_prices_adaptive),
                        "move_be": False,
                        "be_armed": False,
                    }
                    _EXIT_POS_STATE[key] = state
                    try:
                        log(
                            f"[ADAPTIVE][INIT] account={acc_idx} symbol={symbol} side={'LONG' if long_side else 'SHORT'} "
                            f"tp_steps={len(tp_prices_adaptive)} add_steps={len(add_prices_adaptive)} "
                            f"sl_pct={sl_pct_val:.2f}% ai_ctx_fresh={ai_ctx_fresh} add_mult={add_mult:.2f} "
                            f"base_qty={initial_amt:.8f} current_qty={abs(position_amt):.8f}"
                        )
                    except Exception:
                        pass
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _place_initial_exchange_orders,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            sl_price_adaptive,
                            list(tp_prices_adaptive),
                            len(tp_prices_adaptive),
                            initial_amt,
                            position_side,
                            [sl_price_adaptive],
                        )
                    except Exception as e:
                        log(f"[ADAPTIVE][INIT] ERROR submitting SL/TP orders for {symbol} account {acc_idx}: {e}")
                    if add_prices_adaptive:
                        try:
                            state["_adaptive_add_rebuild_pending"] = True
                            _ORDER_PLACEMENT_POOL.submit(
                                _rebuild_adaptive_add_orders_async,
                                acc_idx,
                                symbol,
                                state,
                                long_side,
                                position_side,
                                entry_price,
                                0,
                            )
                        except Exception:
                            state.pop("_adaptive_add_rebuild_pending", None)

        if bool(state.get("adaptive_reentry_mode")):
            adaptive_tp_prices = state.get("adaptive_tp_prices") or []
            adaptive_add_prices = state.get("adaptive_add_prices") or []
            adaptive_sl_price = state.get("adaptive_sl_price")
            if not adaptive_tp_prices or adaptive_sl_price is None:
                continue

            try:
                current_abs = abs(float(position_amt))
            except Exception:
                current_abs = 0.0
            try:
                prev_abs_qty = abs(float(state.get("adaptive_last_abs_qty", current_abs)))
            except Exception:
                prev_abs_qty = current_abs
            try:
                initial_amt_ref = abs(float(state.get("initial_amt", current_abs)))
            except Exception:
                initial_amt_ref = current_abs
            try:
                last_base_sync = float(state.get("adaptive_base_sync_ts", 0.0))
            except (TypeError, ValueError):
                last_base_sync = 0.0
            if (now - last_base_sync) >= 60.0:
                desired_base_qty = _resolve_base_initial_qty(acc_idx, symbol, entry_price, current_abs)
                if desired_base_qty > 0:
                    if abs(desired_base_qty - initial_amt_ref) > max(1e-9, initial_amt_ref * 0.01):
                        prev_base_qty = initial_amt_ref
                        state["initial_amt"] = float(desired_base_qty)
                        initial_amt_ref = float(desired_base_qty)
                        try:
                            log(
                                f"[ADAPTIVE][BASE-SYNC] account={acc_idx} symbol={symbol} "
                                f"base_qty {prev_base_qty:.8f} -> {initial_amt_ref:.8f} "
                                f"(current_qty={current_abs:.8f})"
                            )
                        except Exception:
                            pass
                state["adaptive_base_sync_ts"] = now
            state["_last_price"] = current_price
            if not state.get("adaptive_base_entry_price"):
                state["adaptive_base_entry_price"] = float(entry_price)

            tp_crossed = 0
            for px in adaptive_tp_prices:
                if long_side:
                    if current_price >= px:
                        tp_crossed += 1
                else:
                    if current_price <= px:
                        tp_crossed += 1

            add_crossed = 0
            for px in adaptive_add_prices:
                if long_side:
                    if current_price <= px:
                        add_crossed += 1
                else:
                    if current_price >= px:
                        add_crossed += 1

            try:
                prev_tp = int(state.get("adaptive_crossed_tp", 0))
            except (TypeError, ValueError):
                prev_tp = 0
            try:
                prev_add = int(state.get("adaptive_crossed_add", 0))
            except (TypeError, ValueError):
                prev_add = 0

            # Hit counters must be monotonic for a live position.
            # If price oscillates around a TP/ADD level, we should not
            # "un-hit" a level and then count it again on the next tick.
            tp_crossed = max(tp_crossed, prev_tp)
            add_crossed = max(add_crossed, prev_add)

            # Secondary TP-hit detector by realized size reduction.
            # This prevents TP1 from being re-added when exchange filled TP
            # on a wick but current price later moved back before loop check.
            try:
                tp_steps_ref = max(1, len(adaptive_tp_prices))
                leg_qty_ref = (initial_amt_ref / float(tp_steps_ref)) if initial_amt_ref > 0 else 0.0
            except Exception:
                leg_qty_ref = 0.0
            qty_tp_crossed = 0
            if leg_qty_ref > 0:
                realized_reduction = max(0.0, initial_amt_ref - current_abs)
                # Small epsilon avoids missing boundary due quantity rounding.
                qty_tp_crossed = int((realized_reduction + leg_qty_ref * 0.15) / leg_qty_ref)
                if qty_tp_crossed < 0:
                    qty_tp_crossed = 0
                if qty_tp_crossed > tp_steps_ref:
                    qty_tp_crossed = tp_steps_ref
                tp_crossed = max(tp_crossed, qty_tp_crossed)
                # If position just dropped meaningfully this loop, enforce at least one hit.
                if (prev_abs_qty - current_abs) >= (leg_qty_ref * 0.35):
                    tp_crossed = max(tp_crossed, min(tp_steps_ref, prev_tp + 1))

            # Base TP ladder should only manage initial/core size.
            base_remaining_qty = max(
                0.0,
                float(initial_amt_ref) * max(0.0, 1.0 - (float(tp_crossed) / float(max(tp_steps_ref, 1)))),
            )
            tp_qty_target = min(float(current_abs), float(base_remaining_qty))
            state["ladder_tp_qty_total"] = float(max(tp_qty_target, 0.0))
            # Persist adaptive base/add context so restart does not lose memory
            # about added quantity that must be unwound near base entry.
            try:
                last_mem_sync = float(state.get("_adaptive_mem_sync_ts") or 0.0)
            except (TypeError, ValueError):
                last_mem_sync = 0.0
            if (now - last_mem_sync) >= 60.0:
                try:
                    base_entry_mem = float(state.get("adaptive_base_entry_price") or entry_price or 0.0)
                except (TypeError, ValueError):
                    base_entry_mem = 0.0
                try:
                    extra_outstanding = max(0.0, float(current_abs) - float(initial_amt_ref))
                except (TypeError, ValueError):
                    extra_outstanding = 0.0
                _adaptive_add_mem_upsert(
                    acc_idx=acc_idx,
                    symbol=symbol,
                    long_side=bool(long_side),
                    base_qty=float(initial_amt_ref),
                    base_entry=float(base_entry_mem),
                    extra_qty=float(extra_outstanding),
                )
                state["_adaptive_mem_sync_ts"] = now
            # Recovery TP for added chunk (current_abs - base target).
            if not state.get("_adaptive_recovery_sync_pending"):
                state["_adaptive_recovery_sync_pending"] = True
                try:
                    _ORDER_PLACEMENT_POOL.submit(
                        _sync_adaptive_recovery_tp_async,
                        acc_idx,
                        symbol,
                        state,
                        long_side,
                        position_side,
                        entry_price,
                        current_price,
                        current_abs,
                        initial_amt_ref,
                    )
                except Exception:
                    state.pop("_adaptive_recovery_sync_pending", None)

            d_tp = max(0, tp_crossed - prev_tp)
            d_add = max(0, add_crossed - prev_add)
            # Consistency guard: if tracked TP total diverges from computed
            # base TP target, rebuild TP side even without new hit events.
            force_tp_rebuild = False
            try:
                last_tp_consistency = float(state.get("_tp_consistency_check_ts") or 0.0)
            except (TypeError, ValueError):
                last_tp_consistency = 0.0
            # Strict guard: verify TP presence at least every minute.
            periodic_tp_check = (now - last_tp_consistency) >= 60.0
            if periodic_tp_check:
                state["_tp_consistency_check_ts"] = now
            try:
                existing_tp = state.get("exchange_tp_orders") or []
                if not isinstance(existing_tp, list):
                    existing_tp = []
                existing_tp_qty = 0.0
                for _o in existing_tp:
                    if not isinstance(_o, dict):
                        continue
                    try:
                        existing_tp_qty += abs(
                            float(
                                _o.get("qty")
                                or _o.get("quantity")
                                or _o.get("origQty")
                                or 0.0
                            )
                        )
                    except (TypeError, ValueError):
                        continue
                target_tp_qty = float(state.get("ladder_tp_qty_total") or 0.0)
                # Ignore tiny dust drift; rebuild only on meaningful mismatch.
                if periodic_tp_check and target_tp_qty > 0:
                    ex_counts = _count_open_protection_orders_on_exchange(acc_idx, symbol, long_side)
                    ex_tp_count = int(ex_counts.get("tp_count") or 0)
                    state["_exchange_tp_count"] = ex_tp_count
                    state["_exchange_sl_count"] = int(ex_counts.get("sl_count") or 0)
                    if ex_tp_count <= 0:
                        force_tp_rebuild = True
                    if existing_tp_qty <= 0.0:
                        force_tp_rebuild = True
                    mismatch = abs(existing_tp_qty - target_tp_qty) / max(target_tp_qty, 1e-9)
                    if mismatch > 0.35:
                        force_tp_rebuild = True
                elif periodic_tp_check and existing_tp_qty > 0:
                    force_tp_rebuild = True
            except Exception:
                force_tp_rebuild = False

            if d_tp <= 0 and d_add <= 0 and not force_tp_rebuild:
                state["adaptive_last_abs_qty"] = current_abs
                continue

            # Add fill -> TP ladder: remove last one, move tail near first.
            if d_add > 0:
                shifted_tp = _ladder_shift_last_towards_entry(
                    adaptive_tp_prices,
                    entry_price,
                    long_side,
                    is_tp=True,
                    shifts=d_add,
                )
                shifted_tp = _ladder_restep_after_hit(shifted_tp, d_add)
                state["adaptive_tp_prices"] = shifted_tp
                adaptive_tp_prices = shifted_tp
                state["tp_prices"] = list(shifted_tp)
                state["num_legs"] = len(shifted_tp)

            # TP fill -> Add ladder: remove last one, move tail near first.
            if d_tp > 0:
                shifted_add = _ladder_shift_last_towards_entry(
                    adaptive_add_prices,
                    entry_price,
                    long_side,
                    is_tp=False,
                    shifts=d_tp,
                )
                shifted_add = _ladder_restep_after_hit(shifted_add, d_tp)
                shifted_add = _constrain_adaptive_add_prices(
                    symbol=symbol,
                    add_prices=shifted_add,
                    entry_price=entry_price,
                    sl_price=float(state.get("adaptive_sl_price") or 0.0) if state.get("adaptive_sl_price") is not None else entry_price,
                    long_side=long_side,
                )
                state["adaptive_add_prices"] = shifted_add
                adaptive_add_prices = shifted_add
                # TP progress also tightens hard-SL toward entry.
                try:
                    cur_sl = float(state.get("adaptive_sl_price"))
                except Exception:
                    cur_sl = None
                if cur_sl is not None:
                    new_sl = _adaptive_pull_sl_towards_entry(
                        symbol=symbol,
                        sl_price=cur_sl,
                        entry_price=entry_price,
                        long_side=long_side,
                        steps=d_tp,
                    )
                    state["adaptive_sl_price"] = float(new_sl)
                    state["sl_price"] = float(new_sl)

            state["adaptive_crossed_tp"] = tp_crossed
            state["adaptive_crossed_add"] = add_crossed

            # Rebuild TP orders on any event so quantities stay aligned to
            # current position size and fully close the remaining position.
            tp_remaining_prices = list(adaptive_tp_prices[int(tp_crossed):]) if adaptive_tp_prices else []
            if current_abs > 0 and tp_remaining_prices:
                state["ladder_tp_prices"] = tp_remaining_prices
                if not state.get("_ladder_tp_rebuild_pending"):
                    state["_ladder_tp_rebuild_pending"] = True
                    tp_rebuild_hits = d_add if d_add > 0 else d_tp
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _rebuild_ladder_side_orders_async,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            position_side,
                            current_abs,
                            "tp",
                            max(0, int(tp_rebuild_hits)),
                        )
                    except Exception:
                        state.pop("_ladder_tp_rebuild_pending", None)
            elif current_abs > 0:
                # All TP levels are already consumed; keep exchange TP side clean.
                try:
                    _cancel_exchange_side_orders(
                        state,
                        acc_idx,
                        symbol,
                        "tp",
                        cancel_recovery_tp=False,
                    )
                except Exception:
                    pass

            # Rebuild adaptive hard-SL with latest qty and tightened price.
            sl_orders_tracked = state.get("exchange_sl_orders") or []
            if not isinstance(sl_orders_tracked, list):
                sl_orders_tracked = []
            sl_missing_local = (len(sl_orders_tracked) <= 0)
            need_sl_rebuild = (d_tp > 0 or d_add > 0)
            if not need_sl_rebuild and sl_missing_local:
                try:
                    last_sl_presence = float(state.get("_sl_presence_check_ts") or 0.0)
                except (TypeError, ValueError):
                    last_sl_presence = 0.0
                if (now - last_sl_presence) >= 60.0:
                    need_sl_rebuild = True
                    state["_sl_presence_check_ts"] = now
            elif not need_sl_rebuild:
                # Even if local cache says SL exists, verify exchange presence.
                try:
                    last_sl_presence = float(state.get("_sl_presence_check_ts") or 0.0)
                except (TypeError, ValueError):
                    last_sl_presence = 0.0
                if (now - last_sl_presence) >= 60.0:
                    state["_sl_presence_check_ts"] = now
                    ex_counts = _count_open_protection_orders_on_exchange(acc_idx, symbol, long_side)
                    ex_sl_count = int(ex_counts.get("sl_count") or 0)
                    state["_exchange_sl_count"] = ex_sl_count
                    if ex_sl_count <= 0:
                        need_sl_rebuild = True

            if current_abs > 0 and state.get("adaptive_sl_price") is not None and need_sl_rebuild:
                try:
                    state["ladder_sl_prices"] = [float(state.get("adaptive_sl_price"))]
                except Exception:
                    state["ladder_sl_prices"] = []
                if state["ladder_sl_prices"] and not state.get("_ladder_sl_rebuild_pending"):
                    state["_ladder_sl_rebuild_pending"] = True
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _rebuild_ladder_side_orders_async,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            position_side,
                            current_abs,
                            "sl",
                            max(0, int(d_tp if d_tp > 0 else d_add)),
                        )
                    except Exception:
                        state.pop("_ladder_sl_rebuild_pending", None)

            need_add_rebuild = (d_tp > 0 or d_add > 0)
            if adaptive_add_prices and need_add_rebuild:
                if not state.get("_adaptive_add_rebuild_pending"):
                    state["_adaptive_add_rebuild_pending"] = True
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _rebuild_adaptive_add_orders_async,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            position_side,
                            entry_price,
                            max(0, int(d_tp if d_tp > 0 else d_add)),
                        )
                    except Exception:
                        state.pop("_adaptive_add_rebuild_pending", None)

            try:
                log(
                    f"[ADAPTIVE][STEP] account={acc_idx} symbol={symbol} "
                    f"d_tp={d_tp} d_add={d_add} tp_levels={len(adaptive_tp_prices)} add_levels={len(adaptive_add_prices)} "
                    f"sl={float(state.get('adaptive_sl_price') or 0.0):.8f} "
                    f"cross(tp={tp_crossed},add={add_crossed}) prev(tp={prev_tp},add={prev_add}) "
                    f"qty_cross_tp={qty_tp_crossed} "
                    f"entry={entry_price:.8f} current={current_price:.8f}"
                )
            except Exception:
                pass
            state["adaptive_last_abs_qty"] = current_abs
            continue

        if (
            ladder_enabled
            and (
                state is None
                or state.get("initial_amt", 0.0) <= 0
                or bool(state.get("long", long_side)) != long_side
                or not state.get("ladder_mode")
            )
        ):
            initial_amt = _resolve_base_initial_qty(acc_idx, symbol, entry_price, abs(position_amt))
            if initial_amt > 0:
                try:
                    sl_default = float(sl_val) if sl_val is not None else SIGNAL_SL_PCT
                except Exception:
                    sl_default = SIGNAL_SL_PCT
                if sl_default <= 0:
                    sl_default = SIGNAL_SL_PCT if SIGNAL_SL_PCT > 0 else 2.0

                tp_default_start = tp_levels[0] if tp_levels else 2.0
                tp_default_end = tp_levels[-1] if tp_levels else 4.0
                if tp_default_start <= 0:
                    tp_default_start = 2.0
                if tp_default_end <= 0:
                    tp_default_end = tp_default_start

                sl_pct_levels = _ladder_build_pct_levels(
                    ladder_sl_range_pct,
                    ladder_sl_steps,
                    fallback_start=sl_default,
                    fallback_end=sl_default,
                )
                tp_pct_levels = _ladder_build_pct_levels(
                    ladder_tp_range_pct,
                    ladder_tp_steps,
                    fallback_start=tp_default_start,
                    fallback_end=tp_default_end,
                )

                sl_prices = _ladder_build_price_levels(
                    entry_price, sl_pct_levels, long_side, is_tp=False, symbol=symbol
                )
                tp_prices_ladder = _ladder_build_price_levels(
                    entry_price, tp_pct_levels, long_side, is_tp=True, symbol=symbol
                )

                if sl_prices and tp_prices_ladder:
                    state = {
                        "initial_amt": initial_amt,
                        "long": long_side,
                        "ladder_mode": True,
                        "ladder_sl_steps": _ladder_parse_steps(ladder_sl_steps, default_steps=len(sl_prices)),
                        "ladder_tp_steps": _ladder_parse_steps(ladder_tp_steps, default_steps=len(tp_prices_ladder)),
                        "ladder_sl_prices": sl_prices,
                        "ladder_tp_prices": tp_prices_ladder,
                        "crossed_sl": 0,
                        "crossed_tp": 0,
                        "target_remaining_amt": initial_amt,
                        "opened_ts": now,
                        "realized_pnl_notional": 0.0,
                        "exited_qty": 0.0,
                        # keep compatibility keys for existing diagnostics.
                        "tp_prices": tp_prices_ladder,
                        "sl_price": sl_prices[0],
                        "num_legs": max(
                            _ladder_parse_steps(ladder_sl_steps, default_steps=len(sl_prices)),
                            _ladder_parse_steps(ladder_tp_steps, default_steps=len(tp_prices_ladder)),
                        ),
                        "move_be": False,
                        "be_armed": False,
                    }
                    _EXIT_POS_STATE[key] = state
                    try:
                        log(
                            f"[EXIT CONFIG][LADDER] account={acc_idx} symbol={symbol} "
                            f"side={'LONG' if long_side else 'SHORT'} "
                            f"sl_range=[{sl_pct_levels[0]:.4f}%..{sl_pct_levels[-1]:.4f}%] steps={len(sl_pct_levels)} "
                            f"tp_range=[{tp_pct_levels[0]:.4f}%..{tp_pct_levels[-1]:.4f}%] steps={len(tp_pct_levels)}"
                        )
                    except Exception:
                        pass
                    # Place exchange-side ladder orders so Binance shows
                    # the full ladder (e.g. 10 SL + 10 TP).
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _place_initial_exchange_orders,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            sl_prices[0],
                            list(tp_prices_ladder),
                            len(tp_prices_ladder),
                            initial_amt,
                            position_side,
                            list(sl_prices),
                        )
                    except Exception as e:
                        log(f"[EXIT MANAGER] ERROR submitting ladder exchange orders for {symbol} account {acc_idx}: {e}")

        # Initialise state for new positions or changed direction.
        if (
            state is None
            or state.get("initial_amt", 0.0) <= 0
            or bool(state.get("long", long_side)) != long_side
        ):
            initial_amt = _resolve_base_initial_qty(acc_idx, symbol, entry_price, abs(position_amt))

            # Compute concrete TP prices and SL price.
            if auto_sl_effective:
                # For accounts with auto_sl_tp enabled, prefer the AI SL/TP
                # layer that was used for the signal's Telegram message.
                direction_str = "LONG" if long_side else "SHORT"
                ai_ctx = (
                    _AI_SLTP_CONTEXT.get((acc_idx, symbol, direction_str))
                    or _AI_SLTP_CONTEXT.get((symbol, direction_str))
                    or {}
                )
                ai_ctx_ts = _safe_float(ai_ctx.get("ts"), 0.0) if isinstance(ai_ctx, dict) else 0.0
                # Use only fresh AI context for initial TP/SL placement.
                ai_ctx_fresh = (time.time() - ai_ctx_ts) <= 6 * 3600.0 if ai_ctx_ts > 0 else False

                # Base percentages: fall back to per-account config if AI
                # context is missing or incomplete.
                try:
                    ai_sl_pct = float(
                        ai_ctx.get("sl_pct", sl_val if sl_val is not None else SIGNAL_SL_PCT)
                    ) if ai_ctx_fresh else float(sl_val if sl_val is not None else SIGNAL_SL_PCT)
                except (TypeError, ValueError):
                    ai_sl_pct = sl_val if sl_val is not None else SIGNAL_SL_PCT

                ai_tp_raw = ai_ctx.get("tp_pcts") if (isinstance(ai_ctx, dict) and ai_ctx_fresh) else None
                ai_tp_pcts = []
                if isinstance(ai_tp_raw, (list, tuple)):
                    for pct in ai_tp_raw:
                        try:
                            pv = float(pct)
                        except (TypeError, ValueError):
                            continue
                        if pv > 0:
                            ai_tp_pcts.append(pv)

                # Ensure AI TP has enough legs for configured exit mode.
                # If NN returns only one TP, expand a ladder around it rather
                # than collapsing everything to a single 1% TP.
                needed_legs = max(1, min(int(num_legs), 3))
                if not ai_tp_pcts:
                    ai_tp_pcts = list(tp_levels[:needed_legs])
                if len(ai_tp_pcts) < needed_legs and ai_tp_pcts:
                    seed = float(ai_tp_pcts[0])
                    if seed <= 0:
                        seed = float(tp_levels[0]) if tp_levels else 1.5
                    # Keep first leg meaningful and avoid ultra-tight default 1%.
                    if tp_levels:
                        try:
                            seed = max(seed, float(tp_levels[0]) * 0.90)
                        except Exception:
                            pass
                    rr = [1.0, 1.6, 2.2]
                    expanded = []
                    for i in range(needed_legs):
                        mult = rr[i] if i < len(rr) else (rr[-1] + (i - len(rr) + 1) * 0.6)
                        expanded.append(max(seed * float(mult), 0.5))
                    ai_tp_pcts = expanded[:needed_legs]

                tp_prices = []
                max_tp_levels = min(num_legs, 3)
                for pct in ai_tp_pcts[:max_tp_levels]:
                    try:
                        pv = float(pct)
                    except (TypeError, ValueError):
                        continue
                    if pv <= 0:
                        continue
                    if long_side:
                        tp_price = entry_price * (1.0 + pv / 100.0)
                    else:
                        tp_price = entry_price * (1.0 - pv / 100.0)
                    tp_prices.append(tp_price)

                # If AI context is missing or produced no valid TP levels,
                # fall back to percent-based distances from the account
                # configuration so that exits are still protected.
                if not tp_prices or ai_sl_pct is None:
                    tp_prices = []
                    max_tp_levels = min(len(tp_levels), num_legs, 3)
                    for i in range(max_tp_levels):
                        pct = tp_levels[i]
                        if long_side:
                            tp_price = entry_price * (1.0 + pct / 100.0)
                        else:
                            tp_price = entry_price * (1.0 - pct / 100.0)
                        tp_prices.append(tp_price)

                    base_sl = sl_val if sl_val is not None else SIGNAL_SL_PCT
                    if long_side:
                        sl_price = entry_price * (1.0 - base_sl / 100.0)
                    else:
                        sl_price = entry_price * (1.0 + base_sl / 100.0)
                else:
                    if long_side:
                        sl_price = entry_price * (1.0 - ai_sl_pct / 100.0)
                    else:
                        sl_price = entry_price * (1.0 + ai_sl_pct / 100.0)

                # Recompute num_legs to match actual AI TP levels (never exceed
                # the number of concrete TP prices we generated).
                num_legs = min(max(num_legs, 1), len(tp_prices))
                try:
                    log(
                        f"[EXIT MANAGER][AI-TP] account={acc_idx} symbol={symbol} "
                        f"fresh_ctx={ai_ctx_fresh} tp_mode={tp_mode} "
                        f"tp_pcts={','.join(f'{float(x):.3f}' for x in ai_tp_pcts[:max_tp_levels])}"
                    )
                except Exception:
                    pass
            else:
                tp_prices = []
                for i in range(num_legs):
                    pct = tp_levels[i]
                    if long_side:
                        tp_price = entry_price * (1.0 + pct / 100.0)
                    else:
                        tp_price = entry_price * (1.0 - pct / 100.0)
                    tp_prices.append(tp_price)

                # Compute SL price for full exit.
                base_sl = sl_val if sl_val is not None else SIGNAL_SL_PCT
                if long_side:
                    sl_price = entry_price * (1.0 - base_sl / 100.0)
                else:
                    sl_price = entry_price * (1.0 + base_sl / 100.0)

            state = {
                "initial_amt": initial_amt,
                "long": long_side,
                "tp_prices": tp_prices,
                "sl_price": sl_price,
                "num_legs": num_legs,
                "move_be": bool(move_sl_to_entry_flag) and (not ladder_enabled),
                "be_armed": False,
                "opened_ts": now,
                "realized_pnl_notional": 0.0,
                "exited_qty": 0.0,
            }
            # Log per-account SL/TP configuration when a new position is seen.
            try:
                side_txt = "LONG" if long_side else "SHORT"
                tp_str = ", ".join(f"{p:.8f}" for p in tp_prices)
                log(
                    f"[EXIT CONFIG] account={acc_idx} symbol={symbol} side={side_txt} "
                    f"sl={sl_price:.8f} tps=[{tp_str}] num_legs={num_legs} "
                    f"move_be={bool(move_sl_to_entry_flag) and (not ladder_enabled)} "
                    f"early_be={_early_be_enabled}:{_early_be_pct} "
                    f"tp_lock={_tp_lock_enabled}:{_tp_lock_pct}"
                )
            except Exception:
                pass
            _EXIT_POS_STATE[key] = state

            # Place exchange-side SL + TP orders (dual tracking) — non-blocking.
            # Orders are placed in parallel via _ORDER_PLACEMENT_POOL so the
            # exit manager loop is not delayed by slow API responses.
            try:
                _ORDER_PLACEMENT_POOL.submit(
                    _place_initial_exchange_orders,
                    acc_idx, symbol, state, long_side, sl_price,
                    list(tp_prices), num_legs, initial_amt, position_side,
                )
            except Exception as e:
                log(f"[EXIT MANAGER] ERROR submitting exchange orders for {symbol} account {acc_idx}: {e}")

        initial_amt = state.get("initial_amt", 0.0)
        tp_prices = state.get("tp_prices") or []
        sl_price = state.get("sl_price")
        num_legs = state.get("num_legs", num_legs)
        # Always use live account setting (from current snapshot), not only
        # the value captured when this position state was first created.
        move_be = bool(move_sl_to_entry_flag) and (not ladder_enabled)
        state["move_be"] = move_be
        be_armed = bool(state.get("be_armed", False))

        try:
            realized_pnl_notional = float(state.get("realized_pnl_notional") or 0.0)
        except (TypeError, ValueError):
            realized_pnl_notional = 0.0

        try:
            exited_qty = float(state.get("exited_qty") or 0.0)
        except (TypeError, ValueError):
            exited_qty = 0.0

        def _leg_pnl_notional(qty: float, exit_px: float) -> float:
            try:
                q = float(qty)
                xp = float(exit_px)
            except Exception:
                return 0.0
            if q <= 0 or xp <= 0 or entry_price <= 0:
                return 0.0
            if long_side:
                return (xp - entry_price) * q
            return (entry_price - xp) * q

        def _final_trade_pnl_pct(remaining_qty: float, exit_px: float) -> float:
            if initial_amt <= 0 or entry_price <= 0:
                return pnl_pct
            total_notional = realized_pnl_notional + _leg_pnl_notional(remaining_qty, exit_px)
            try:
                return (total_notional / (entry_price * initial_amt)) * 100.0
            except Exception:
                return pnl_pct

        current_abs = abs(position_amt)
        if initial_amt <= 0 or current_abs <= 0:
            continue

        # Approximate PnL percentage at current price (per position).
        direction = 1.0 if long_side else -1.0
        pnl_pct = (current_price - entry_price) / entry_price * 100.0 * direction

        # 0) Time-based exit: close full position if it has been open longer
        # than the configured timeout.
        if (
            POSITION_TIMEOUT_ENABLED
            and POSITION_TIMEOUT_MAX_HOURS
            and POSITION_TIMEOUT_MAX_HOURS > 0.0
        ):
            try:
                opened_ts = float(state.get("opened_ts", now))
            except (TypeError, ValueError):
                opened_ts = now
            state["opened_ts"] = opened_ts
            elapsed_sec = now - opened_ts
            timeout_sec = POSITION_TIMEOUT_MAX_HOURS * 3600.0
            if elapsed_sec >= timeout_sec:
                try:
                    submitted = close_position_market(
                        acc_idx,
                        symbol,
                        position_amt,
                        force_full_close=True,
                        position_side=position_side,
                    )
                    if not submitted:
                        log(
                            f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by TIMEOUT: close request not submitted"
                        )
                        continue
                    log(
                        f"[EXIT MANAGER] Closed {symbol} position on account {acc_idx} "
                        f"by TIMEOUT after {elapsed_sec / 3600.0:.2f}h "
                        f"(pnl_pct={pnl_pct:.2f}%)"
                    )
                    try:
                        log_trade_exit(
                            account_index=acc_idx,
                            symbol=symbol,
                            side="SELL" if long_side else "BUY",
                            qty=abs(position_amt),
                            entry_price=entry_price,
                            exit_price=current_price,
                            pnl_pct=pnl_pct,
                            reason="TIMEOUT",
                        )
                    except Exception:
                        pass
                    try:
                        on_trade_exit(
                            account_index=acc_idx,
                            symbol=symbol,
                            side="BUY" if long_side else "SELL",
                            entry_price=entry_price,
                            exit_price=current_price,
                            pnl_pct=pnl_pct,
                            reason="TIMEOUT",
                        )
                    except Exception:
                        pass
                    try:
                        update_fibo_after_exit(
                            acc_idx,
                            _final_trade_pnl_pct(abs(position_amt), current_price),
                        )
                    except Exception:
                        pass
                    if pnl_pct < 0:
                        try:
                            record_symbol_loss(acc_idx, symbol)
                        except Exception:
                            pass
                except Exception as e:
                    log(
                        f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by TIMEOUT: {e}"
                    )
                try:
                    cancel_symbol_open_orders(acc_idx, symbol)
                except Exception:
                    pass
                _EXIT_POS_STATE.pop(key, None)
                continue

        # Ladder SL/TP plugin path (isolated per-account).
        if bool(state.get("ladder_mode")):
            ladder_tp_prices = state.get("ladder_tp_prices") or []
            ladder_sl_prices = state.get("ladder_sl_prices") or []
            if not ladder_tp_prices or not ladder_sl_prices:
                continue

            tp_crossed = 0
            for px in ladder_tp_prices:
                if long_side:
                    if current_price >= px:
                        tp_crossed += 1
                else:
                    if current_price <= px:
                        tp_crossed += 1

            sl_crossed = 0
            for px in ladder_sl_prices:
                if long_side:
                    if current_price <= px:
                        sl_crossed += 1
                else:
                    if current_price >= px:
                        sl_crossed += 1

            try:
                prev_tp = int(state.get("crossed_tp", 0))
            except (TypeError, ValueError):
                prev_tp = 0
            try:
                prev_sl = int(state.get("crossed_sl", 0))
            except (TypeError, ValueError):
                prev_sl = 0

            d_tp = max(0, tp_crossed - prev_tp)
            d_sl = max(0, sl_crossed - prev_sl)
            if d_tp <= 0 and d_sl <= 0:
                continue

            # Opposite-grid shift on hit:
            # TP hit -> shift SL ladder one step toward entry.
            # SL hit -> shift TP ladder one step toward entry.
            if d_tp > 0:
                shifted_sl = _ladder_shift_last_towards_entry(
                    ladder_sl_prices,
                    entry_price,
                    long_side,
                    is_tp=False,
                    shifts=d_tp,
                )
                shifted_sl = _ladder_restep_after_hit(shifted_sl, d_tp)
                state["ladder_sl_prices"] = shifted_sl
                ladder_sl_prices = shifted_sl
                try:
                    log(
                        f"[LADDER] {symbol} account {acc_idx}: TP advanced by {d_tp}, "
                        f"SL ladder shifted toward entry"
                    )
                except Exception:
                    pass
                if not state.get("_ladder_sl_rebuild_pending"):
                    state["_ladder_sl_rebuild_pending"] = True
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _rebuild_ladder_side_orders_async,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            position_side,
                            current_abs,
                            "sl",
                        d_tp,
                        )
                    except Exception:
                        state.pop("_ladder_sl_rebuild_pending", None)
            if d_sl > 0:
                shifted_tp = _ladder_shift_last_towards_entry(
                    ladder_tp_prices,
                    entry_price,
                    long_side,
                    is_tp=True,
                    shifts=d_sl,
                )
                shifted_tp = _ladder_restep_after_hit(shifted_tp, d_sl)
                state["ladder_tp_prices"] = shifted_tp
                ladder_tp_prices = shifted_tp
                try:
                    log(
                        f"[LADDER] {symbol} account {acc_idx}: SL advanced by {d_sl}, "
                        f"TP ladder shifted toward entry"
                    )
                except Exception:
                    pass
                if not state.get("_ladder_tp_rebuild_pending"):
                    state["_ladder_tp_rebuild_pending"] = True
                    try:
                        _ORDER_PLACEMENT_POOL.submit(
                            _rebuild_ladder_side_orders_async,
                            acc_idx,
                            symbol,
                            state,
                            long_side,
                            position_side,
                            current_abs,
                            "tp",
                        d_sl,
                        )
                    except Exception:
                        state.pop("_ladder_tp_rebuild_pending", None)

            tp_steps = _ladder_parse_steps(state.get("ladder_tp_steps"), default_steps=max(1, len(ladder_tp_prices)))
            sl_steps = _ladder_parse_steps(state.get("ladder_sl_steps"), default_steps=max(1, len(ladder_sl_prices)))
            if tp_crossed > tp_steps:
                tp_crossed = tp_steps
            if sl_crossed > sl_steps:
                sl_crossed = sl_steps

            tp_target_remaining = initial_amt * max(0.0, 1.0 - (float(tp_crossed) / float(tp_steps)))
            sl_target_remaining = initial_amt * max(0.0, 1.0 - (float(sl_crossed) / float(sl_steps)))
            try:
                old_target_remaining = float(state.get("target_remaining_amt", initial_amt))
            except (TypeError, ValueError):
                old_target_remaining = initial_amt

            target_remaining_amt = min(old_target_remaining, tp_target_remaining, sl_target_remaining)
            if target_remaining_amt < 0:
                target_remaining_amt = 0.0

            qty_to_close = current_abs - target_remaining_amt
            if qty_to_close <= 0:
                state["crossed_tp"] = tp_crossed
                state["crossed_sl"] = sl_crossed
                state["target_remaining_amt"] = target_remaining_amt
                continue

            step_size, min_qty = get_symbol_lot_constraints(symbol)

            def _ceil_to_step_ladder(q: float) -> float:
                if step_size and step_size > 0:
                    try:
                        steps = int(q / step_size)
                        if steps * step_size < q:
                            steps += 1
                        return float(steps) * step_size
                    except Exception:
                        return q
                return q

            force_full_close = target_remaining_amt <= 0
            dust_threshold = 0.0
            if step_size and step_size > 0:
                dust_threshold = step_size * 1.001
            if min_qty and min_qty > 0:
                if dust_threshold <= 0 or min_qty > dust_threshold:
                    dust_threshold = min_qty * 1.001

            if dust_threshold > 0 and target_remaining_amt <= dust_threshold:
                qty_to_close = current_abs
                target_remaining_amt = 0.0
                force_full_close = True
            else:
                qty_to_close = _ceil_to_step_ladder(qty_to_close)
                if qty_to_close > current_abs:
                    qty_to_close = current_abs

            signed_close_amt = qty_to_close if long_side else -qty_to_close
            reason = "LADDER_TP" if d_tp >= d_sl else "LADDER_SL"

            try:
                submitted = close_position_market(
                    acc_idx,
                    symbol,
                    signed_close_amt,
                    force_full_close=bool(force_full_close),
                    position_side=position_side,
                )
                if not submitted:
                    continue

                state["crossed_tp"] = tp_crossed
                state["crossed_sl"] = sl_crossed
                state["target_remaining_amt"] = target_remaining_amt

                try:
                    realized_pnl_notional = float(state.get("realized_pnl_notional") or 0.0)
                except (TypeError, ValueError):
                    realized_pnl_notional = 0.0
                try:
                    exited_qty = float(state.get("exited_qty") or 0.0)
                except (TypeError, ValueError):
                    exited_qty = 0.0
                realized_pnl_notional = realized_pnl_notional + _leg_pnl_notional(qty_to_close, current_price)
                exited_qty = exited_qty + qty_to_close
                state["realized_pnl_notional"] = realized_pnl_notional
                state["exited_qty"] = exited_qty

                log(
                    f"[EXIT MANAGER] Ladder close {symbol} account {acc_idx} "
                    f"reason={reason} qty={qty_to_close:.8f} "
                    f"(tp={tp_crossed}/{tp_steps}, sl={sl_crossed}/{sl_steps}, pnl={pnl_pct:.2f}%)"
                )
                try:
                    log_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="SELL" if long_side else "BUY",
                        qty=qty_to_close,
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason=reason,
                    )
                except Exception:
                    pass
            except Exception as e:
                log(f"[EXIT MANAGER] Ladder close failed for {symbol} account {acc_idx}: {e}")
                continue

            if target_remaining_amt <= 0:
                try:
                    update_fibo_after_exit(
                        acc_idx,
                        _final_trade_pnl_pct(0.0, current_price),
                    )
                except Exception:
                    pass
                try:
                    on_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="BUY" if long_side else "SELL",
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason=reason,
                    )
                except Exception:
                    pass
                if pnl_pct < 0:
                    try:
                        record_symbol_loss(acc_idx, symbol)
                    except Exception:
                        pass
                try:
                    cancel_symbol_open_orders(acc_idx, symbol)
                except Exception:
                    pass
                _EXIT_POS_STATE.pop(key, None)
            continue

        if not tp_prices or sl_price is None:
            continue

        # Periodic order verification: check that exchange SL/TP orders
        # still exist on Binance every ~30 seconds. Re-place any missing.
        _ORDER_VERIFY_INTERVAL = 30.0
        try:
            last_verify = float(state.get("_last_verify_ts") or 0.0)
        except (TypeError, ValueError):
            last_verify = 0.0
        if (
            now - last_verify >= _ORDER_VERIFY_INTERVAL
            and state.get("exchange_sl_order") is not None
            and not state.get("_verify_pending")
        ):
            state["_last_verify_ts"] = now
            state["_verify_pending"] = True
            try:
                _ORDER_PLACEMENT_POOL.submit(
                    _verify_and_retry_exchange_orders,
                    acc_idx, symbol, state, long_side, position_side,
                )
            except Exception:
                state.pop("_verify_pending", None)

        # 1) Stop-loss: always close the full remaining position.
        sl_hit = (long_side and current_price <= sl_price) or (
            (not long_side) and current_price >= sl_price
        )
        if sl_hit:
            det_ms = None
            sub_ms = None
            try:
                try:
                    det_ms = float(time.time() * 1000.0)
                except Exception:
                    det_ms = None
                submitted = close_position_market(
                    acc_idx,
                    symbol,
                    position_amt,
                    force_full_close=True,
                    position_side=position_side,
                )
                if not submitted:
                    log(
                        f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by SL: close request not submitted"
                    )
                    try:
                        cancel_symbol_open_orders(acc_idx, symbol)
                    except Exception:
                        pass
                    _EXIT_POS_STATE.pop(key, None)
                    continue
                try:
                    sub_ms = float(time.time() * 1000.0)
                except Exception:
                    sub_ms = None
                try:
                    log_exit_timing(
                        symbol=str(symbol),
                        account_index=acc_idx,
                        side="SELL" if long_side else "BUY",
                        reason="SL",
                        trigger_price=float(sl_price) if sl_price is not None else None,
                        hit_price=float(current_price) if current_price is not None else None,
                        detected_ts_ms=det_ms,
                        submit_ts_ms=sub_ms,
                        submit_delay_ms=(float(sub_ms) - float(det_ms)) if (det_ms is not None and sub_ms is not None) else None,
                        timing_tag="SL",
                    )
                except Exception:
                    pass
                log(
                    f"[EXIT MANAGER] Closed {symbol} position on account {acc_idx} "
                    f"by SL at price {current_price:.8f} (pnl_pct={pnl_pct:.2f}%)"
                )
                try:
                    log_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="SELL" if long_side else "BUY",
                        qty=abs(position_amt),
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason="SL",
                    )
                except Exception:
                    pass
                try:
                    on_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="BUY" if long_side else "SELL",
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason="SL",
                    )
                except Exception:
                    pass
                if pnl_pct < 0:
                    try:
                        record_symbol_loss(acc_idx, symbol)
                    except Exception:
                        pass
                # Update per-account Fibonacci staking state (if enabled for
                # this Binance account) based on realised PnL of the full
                # position exit.
                try:
                    update_fibo_after_exit(
                        acc_idx,
                        _final_trade_pnl_pct(abs(position_amt), current_price),
                    )
                except Exception:
                    pass
            except Exception as e:
                log(
                    f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by SL: {e}"
                )
            try:
                cancel_symbol_open_orders(acc_idx, symbol)
            except Exception:
                pass
            _EXIT_POS_STATE.pop(key, None)
            continue

        # 2) Take-profit: compute how many TP levels price has crossed.
        crossed = 0
        for tp_price in tp_prices:
            if long_side:
                if current_price >= tp_price:
                    crossed += 1
            else:
                if current_price <= tp_price:
                    crossed += 1

        # Feature: Early breakeven — move SL to entry when PnL >= threshold
        # but first TP has not been reached yet.  Per-account flag.
        if (
            _early_be_enabled
            and _early_be_pct is not None
            and _early_be_pct > 0
            and crossed == 0
            and pnl_pct >= _early_be_pct
            and not state.get("early_be_done")
        ):
            state["sl_price"] = entry_price
            sl_price = entry_price
            state["early_be_done"] = True
            try:
                log(
                    f"[EXIT MANAGER] Early breakeven: {symbol} account {acc_idx} "
                    f"PnL {pnl_pct:.2f}% >= {_early_be_pct}% — SL moved to entry {entry_price:.8f}"
                )
            except Exception:
                pass
            # Update exchange SL order to entry price — non-blocking.
            if not state.get("_sl_update_pending"):
                state["_sl_update_pending"] = True
                try:
                    _ORDER_PLACEMENT_POOL.submit(
                        _update_exchange_sl_async,
                        acc_idx, symbol, state, long_side, entry_price, position_side, "early BE",
                    )
                except Exception:
                    state.pop("_sl_update_pending", None)

        if crossed <= 0:
            continue

        crossed = min(crossed, num_legs)

        if multi_entry_enabled and multi_entry_levels and pnl_pct > 0.0:
            det_ms = None
            sub_ms = None
            try:
                try:
                    det_ms = float(time.time() * 1000.0)
                except Exception:
                    det_ms = None
                submitted = close_position_market(
                    acc_idx,
                    symbol,
                    position_amt,
                    force_full_close=True,
                    position_side=position_side,
                )
                if not submitted:
                    log(
                        f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by TP: close request not submitted"
                    )
                    continue
                try:
                    sub_ms = float(time.time() * 1000.0)
                except Exception:
                    sub_ms = None
                try:
                    tp_trig = None
                    try:
                        if crossed > 0 and crossed - 1 < len(tp_prices):
                            tp_trig = float(tp_prices[crossed - 1])
                    except Exception:
                        tp_trig = None
                    log_exit_timing(
                        symbol=str(symbol),
                        account_index=acc_idx,
                        side="SELL" if long_side else "BUY",
                        reason=f"TP_FULL_{crossed}",
                        trigger_price=tp_trig,
                        hit_price=float(current_price) if current_price is not None else None,
                        detected_ts_ms=det_ms,
                        submit_ts_ms=sub_ms,
                        submit_delay_ms=(float(sub_ms) - float(det_ms)) if (det_ms is not None and sub_ms is not None) else None,
                        timing_tag="TP_FULL",
                    )
                except Exception:
                    pass
                log(
                    f"[EXIT MANAGER] Closed {symbol} position on account {acc_idx} "
                    f"by TP at price {current_price:.8f} (pnl_pct={pnl_pct:.2f}%, crossed {crossed} level(s))"
                )
                try:
                    log_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="SELL" if long_side else "BUY",
                        qty=abs(position_amt),
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason=f"TP_FULL_{crossed}",
                    )
                except Exception:
                    pass
                try:
                    on_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="BUY" if long_side else "SELL",
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason=f"TP_FULL_{crossed}",
                    )
                except Exception:
                    pass
                try:
                    update_fibo_after_exit(
                        acc_idx,
                        _final_trade_pnl_pct(abs(position_amt), current_price),
                    )
                except Exception:
                    pass
                try:
                    with _PENDING_ENTRIES_LOCK:
                        _PENDING_ENTRIES[:] = [
                            e for e in _PENDING_ENTRIES if e.get("symbol") != symbol
                        ]
                except Exception:
                    pass
            except Exception as e:
                log(
                    f"[EXIT MANAGER] Failed to close {symbol} position on account {acc_idx} by TP: {e}"
                )
            try:
                cancel_symbol_open_orders(acc_idx, symbol)
            except Exception:
                pass
            _EXIT_POS_STATE.pop(key, None)
            continue

        # Feature: Guaranteed-profit SL lock — on first TP hit, move SL to
        # entry + tp_profit_lock_pct% in the trade direction.  Per-account flag.
        if (
            _tp_lock_enabled
            and _tp_lock_pct is not None
            and _tp_lock_pct > 0
            and crossed >= 1
            and not state.get("profit_lock_done")
        ):
            if long_side:
                lock_sl = entry_price * (1.0 + _tp_lock_pct / 100.0)
            else:
                lock_sl = entry_price * (1.0 - _tp_lock_pct / 100.0)
            state["sl_price"] = lock_sl
            sl_price = lock_sl
            state["profit_lock_done"] = True
            state["be_armed"] = True
            be_armed = True
            try:
                log(
                    f"[EXIT MANAGER] Profit lock: {symbol} account {acc_idx} "
                    f"TP1 crossed — SL locked at {lock_sl:.8f} "
                    f"(entry {'+' if long_side else '-'}{_tp_lock_pct}%)"
                )
            except Exception:
                pass
            # Update exchange SL order to profit-lock price — non-blocking.
            if not state.get("_sl_update_pending"):
                state["_sl_update_pending"] = True
                try:
                    _ORDER_PLACEMENT_POOL.submit(
                        _update_exchange_sl_async,
                        acc_idx, symbol, state, long_side, lock_sl, position_side, "profit lock",
                    )
                except Exception:
                    state.pop("_sl_update_pending", None)

        # Optionally move SL to breakeven (entry price) once first TP is hit,
        # but only if there are at least 2 TP levels configured.
        # Skipped if profit lock already set a higher SL above.
        if (
            move_be
            and not be_armed
            and len(tp_prices) > 1
            and crossed >= 1
        ):
            sl_price = entry_price
            state["sl_price"] = sl_price
            state["be_armed"] = True
            be_armed = True
            # Update exchange SL order to entry price — non-blocking.
            if not state.get("_sl_update_pending"):
                state["_sl_update_pending"] = True
                try:
                    _ORDER_PLACEMENT_POOL.submit(
                        _update_exchange_sl_async,
                        acc_idx, symbol, state, long_side, entry_price, position_side, "move BE",
                    )
                except Exception:
                    state.pop("_sl_update_pending", None)

        # Avoid duplicate partial closes for the same TP level. We only
        # execute a new partial close when the number of crossed TP levels
        # has increased beyond what we've already processed for this
        # position.
        try:
            prev_crossed = int(state.get("crossed", 0))
        except (TypeError, ValueError):
            prev_crossed = 0
        if crossed <= prev_crossed:
            # Crossing already fully handled — clear any stale grace timer.
            state.pop("_tp_grace_ts", None)
            state.pop("_tp_grace_crossed", None)
            continue

        # Equal-split legs: after k crossed levels, we want (num_legs - k)/num_legs
        # of the original position remaining.
        target_remaining_frac = float(num_legs - crossed) / float(num_legs)
        target_remaining_amt = initial_amt * target_remaining_frac

        step_size, min_qty = get_symbol_lot_constraints(symbol)

        def _ceil_to_step(q: float) -> float:
            if step_size and step_size > 0:
                try:
                    steps = int(q / step_size)
                    if steps * step_size < q:
                        steps += 1
                    return float(steps) * step_size
                except Exception:
                    return q
            return q

        # If we already have less than or equal to target remaining, the
        # exchange TP order already handled this level — just update state.
        if current_abs <= target_remaining_amt * 1.0001:
            state["crossed"] = crossed
            state.pop("_tp_grace_ts", None)
            state.pop("_tp_grace_crossed", None)
            try:
                log(
                    f"[EXIT MANAGER] TP{crossed} for {symbol} account {acc_idx}: "
                    f"position already at target ({current_abs:.6f} <= {target_remaining_amt:.6f}) "
                    f"— exchange TP order handled it"
                )
            except Exception:
                pass
            continue

        # ---- GRACE PERIOD (5 seconds) ----
        # We detected TP crossing but the position is NOT yet reduced.
        # The exchange TP order on Binance should handle it — give it
        # 5 seconds before we intervene with a manual close.
        _TP_GRACE_SECONDS = 5.0
        grace_ts = state.get("_tp_grace_ts")
        grace_crossed = state.get("_tp_grace_crossed", 0)

        if grace_ts is None or grace_crossed != crossed:
            # First time we see this crossing — start the grace timer.
            state["_tp_grace_ts"] = now
            state["_tp_grace_crossed"] = crossed
            try:
                log(
                    f"[EXIT MANAGER] TP{crossed} crossed for {symbol} account {acc_idx}: "
                    f"waiting {_TP_GRACE_SECONDS}s for exchange TP order to fill "
                    f"(current={current_abs:.6f}, target={target_remaining_amt:.6f})"
                )
            except Exception:
                pass
            continue

        # Grace timer is running — check if enough time has passed.
        try:
            elapsed_grace = float(now) - float(grace_ts)
        except (TypeError, ValueError):
            elapsed_grace = 0.0
        if elapsed_grace < _TP_GRACE_SECONDS:
            # Still waiting — do nothing, let exchange TP handle it.
            continue

        # Grace period expired and position is STILL not reduced.
        # The exchange TP order did NOT fire — close manually.
        state.pop("_tp_grace_ts", None)
        state.pop("_tp_grace_crossed", None)
        try:
            log(
                f"[EXIT MANAGER] TP{crossed} for {symbol} account {acc_idx}: "
                f"grace period expired ({elapsed_grace:.1f}s), exchange TP did NOT close — closing manually "
                f"(current={current_abs:.6f}, target={target_remaining_amt:.6f})"
            )
        except Exception:
            pass

        qty_to_close = current_abs - target_remaining_amt
        if qty_to_close <= 0:
            continue

        # Cancel exchange TP orders for levels we are about to close
        # manually. This prevents DOUBLE EXECUTION.
        try:
            _cancel_exchange_tp_for_crossed_levels(state, acc_idx, symbol, prev_crossed, crossed)
        except Exception:
            pass

        force_full_close = target_remaining_amt <= 0
        dust_threshold = 0.0
        if step_size and step_size > 0:
            dust_threshold = step_size * 1.001
        if min_qty and min_qty > 0:
            if dust_threshold <= 0 or min_qty > dust_threshold:
                dust_threshold = min_qty * 1.001

        if dust_threshold > 0 and target_remaining_amt <= dust_threshold:
            qty_to_close = current_abs
            target_remaining_amt = 0.0
            force_full_close = True
        else:
            # Round up the close amount so reduce-only market order doesn't
            # leave extra remainder due to stepSize flooring.
            qty_to_close = _ceil_to_step(qty_to_close)
            if qty_to_close > current_abs:
                qty_to_close = current_abs

        signed_close_amt = qty_to_close if long_side else -qty_to_close

        try:
            submitted = close_position_market(
                acc_idx,
                symbol,
                signed_close_amt,
                force_full_close=bool(force_full_close),
                position_side=position_side,
            )
            if not submitted:
                try:
                    log(
                        f"[EXIT MANAGER] Failed to close partial {symbol} position on account {acc_idx}: close request not submitted"
                    )
                except Exception:
                    pass
                state["crossed"] = prev_crossed
                continue

            state["crossed"] = crossed
            try:
                det_ms = float(time.time() * 1000.0)
            except Exception:
                det_ms = None
            try:
                tp_trig = None
                try:
                    if crossed > 0 and crossed - 1 < len(tp_prices):
                        tp_trig = float(tp_prices[crossed - 1])
                except Exception:
                    tp_trig = None
                log_exit_timing(
                    symbol=str(symbol),
                    account_index=acc_idx,
                    side="SELL" if long_side else "BUY",
                    reason=f"TP_{crossed}",
                    trigger_price=tp_trig,
                    hit_price=float(current_price) if current_price is not None else None,
                    detected_ts_ms=det_ms,
                    submit_ts_ms=det_ms,
                    submit_delay_ms=0.0,
                    timing_tag="TP_PARTIAL",
                )
            except Exception:
                pass
            log(
                f"[EXIT MANAGER] Closed partial {symbol} position on account {acc_idx} "
                f"by TP (crossed {crossed} level(s)) at price {current_price:.8f} "
                f"(pnl_pct={pnl_pct:.2f}%)"
            )
        except Exception as e:
            log(
                f"[EXIT MANAGER] Failed to close partial {symbol} position on account {acc_idx}: {e}"
            )
        else:
            try:
                realized_pnl_notional = realized_pnl_notional + _leg_pnl_notional(qty_to_close, current_price)
                exited_qty = exited_qty + qty_to_close
                state["realized_pnl_notional"] = realized_pnl_notional
                state["exited_qty"] = exited_qty
                log_trade_exit(
                    account_index=acc_idx,
                    symbol=symbol,
                    side="SELL" if long_side else "BUY",
                    qty=qty_to_close,
                    entry_price=entry_price,
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                    reason=f"TP_{crossed}",
                )
            except Exception:
                pass
            # If this TP leg closes the full position (no remaining target
            # amount), treat it as the final trade result for per-account
            # Fibonacci staking.
            if target_remaining_amt <= 0:
                try:
                    update_fibo_after_exit(
                        acc_idx,
                        _final_trade_pnl_pct(0.0, current_price),
                    )
                except Exception:
                    pass
                try:
                    on_trade_exit(
                        account_index=acc_idx,
                        symbol=symbol,
                        side="BUY" if long_side else "SELL",
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        reason=f"TP_{crossed}",
                    )
                except Exception:
                    pass
                if pnl_pct > 0.0:
                    try:
                        with _PENDING_ENTRIES_LOCK:
                            _PENDING_ENTRIES[:] = [
                                e for e in _PENDING_ENTRIES if e.get("symbol") != symbol
                            ]
                    except Exception:
                        pass


def run_exit_manager():
    last_room_sync = 0.0
    if AITM_MASTER_ENABLED:
        log("[EXIT MANAGER] ANALYSIS ONLY MODE (AITM master enabled)")
    while True:
        loop_start = time.time()
        try:
            _check_manual_exits()
            try:
                check_paper_exits()
            except Exception as pe:
                # log(f"[PAPER EXIT] loop error: {pe}")
                ""
            try:
                now = time.time()
                if (now - float(last_room_sync)) >= 15.0:
                    room_sizing_sync_all_accounts()
                    last_room_sync = now
            except Exception:
                pass
        except Exception as e:
            log(f"[EXIT MANAGER] loop error: {e}")
        try:
            elapsed = time.time() - float(loop_start)
        except Exception:
            elapsed = 0.0
        try:
            sleep_s = float(EXIT_MANAGER_INTERVAL_SECONDS) - float(elapsed)
        except Exception:
            sleep_s = EXIT_MANAGER_INTERVAL_SECONDS
        if sleep_s < 0:
            sleep_s = 0.0
        time.sleep(sleep_s)


def _schedule_signal_followup(symbol: str, direction: str, interval: str, ts_ms: int) -> None:
    """Register 1h volatility + 5m activity follow-ups for a new signal.

    - ts_ms: Unix timestamp (ms) of the signal candle (same as in signal_details_log).
    - scheduled_ts: when to recompute 1h volatility (now + 1h).
    - activity_next_ts: next 5m activity check time (now + 60s).
    - activity_end_ts: stop 5m checks after 1 hour from now.
    """

    if not symbol or direction not in ("BUY", "SELL"):
        return

    try:
        ts_key = int(ts_ms)
    except (TypeError, ValueError):
        return

    now = time.time()
    one_hour = 3600.0

    item = {
        "symbol": symbol,
        "direction": direction,
        "ts_ms": ts_key,
        "scheduled_ts": now + one_hour,      # 1h volatility follow-up
        "activity_next_ts": now + 60.0,      # first 5m activity re-check in 1 minute
        "activity_end_ts": now + one_hour,   # stop activity follow-ups after 1h
    }

    with _PENDING_SIGNAL_FOLLOWUPS_LOCK:
        replaced = False
        for i, existing in enumerate(_PENDING_SIGNAL_FOLLOWUPS):
            try:
                ex_ts = int(existing.get("ts_ms") or 0)
            except (TypeError, ValueError):
                ex_ts = 0

            if (
                existing.get("symbol") == symbol
                and existing.get("direction") == direction
                and ex_ts == ts_key
            ):
                _PENDING_SIGNAL_FOLLOWUPS[i] = item
                replaced = True
                break

        if not replaced:
            _PENDING_SIGNAL_FOLLOWUPS.append(item)


def _check_signal_followups():
    with _PENDING_SIGNAL_FOLLOWUPS_LOCK:
        items = list(_PENDING_SIGNAL_FOLLOWUPS)

    if not items:
        return

    now = time.time()
    new_items = []

    for item in items:
        symbol = item.get("symbol")
        direction = item.get("direction")
        ts_ms = item.get("ts_ms")
        sched = item.get("scheduled_ts")

        if not symbol or ts_ms is None:
            continue

        act_next = item.get("activity_next_ts")
        act_end = item.get("activity_end_ts")

        act_next_ts = None
        act_end_ts = None
        if act_next is not None:
            try:
                act_next_ts = float(act_next)
            except (TypeError, ValueError):
                act_next_ts = None
        if act_end is not None:
            try:
                act_end_ts = float(act_end)
            except (TypeError, ValueError):
                act_end_ts = None

        status = None

        if (
            act_end_ts is not None
            and now <= act_end_ts
            and act_next_ts is not None
            and now >= act_next_ts
        ):
            try:
                act = _futures_5m_activity(symbol)
            except Exception as e:
                act = None
                # try:
                #     log(f"[ACTIVITY_FOLLOWUP] Failed to compute 5m activity for {symbol}: {e}")
                # except Exception:
                #     pass

            if isinstance(act, dict):
                last_trades = act.get("last_trades")
                avg_trades = act.get("avg_trades")
                status = act.get("status")
                # try:
                #     log(
                #         f"[ACTIVITY_FOLLOWUP] {symbol} 5m activity: "
                #         f"last={last_trades}, avg={avg_trades}, status={status}"
                #     )
                # except Exception:
                #     pass

                try:
                    check_ts_ms = int(now * 1000.0)
                except Exception:
                    check_ts_ms = None

                if check_ts_ms is not None:
                    try:
                        update_signal_activity(
                            symbol,
                            direction,
                            ts_ms,
                            check_ts_ms,
                            last_trades,
                            avg_trades,
                            status,
                        )
                    except Exception as e_upd:
                        try:
                            # log(
                            #     f"[ACTIVITY_FOLLOWUP] Failed to update activity log for {symbol}: {e_upd}"
                            # )
                            ""
                        except Exception:
                            pass

            if isinstance(status, str) and status == "NORMAL":
                item["activity_next_ts"] = None
                item["activity_end_ts"] = None
            elif act_end_ts is not None and now + 60.0 <= act_end_ts:
                item["activity_next_ts"] = now + 60.0
            else:
                item["activity_next_ts"] = None

        # 1h volatility follow-up
        sched_ts = None
        if sched is not None:
            try:
                sched_ts = float(sched)
            except (TypeError, ValueError):
                sched_ts = None

        if sched_ts is not None and now >= sched_ts:
            try:
                vol_after = _futures_1h_volatility(symbol)
            except Exception as e:
                vol_after = None
                try:
                    log(f"[FOLLOWUP] Failed to compute 1h volatility for {symbol}: {e}")
                except Exception:
                    pass

            followup_ts_ms = int(now * 1000.0)

            try:
                update_signal_followup(symbol, direction, ts_ms, followup_ts_ms, vol_after)
            except Exception as e:
                try:
                    log(f"[FOLLOWUP] Failed to update signal follow-up for {symbol}: {e}")
                except Exception:
                    pass

            item["scheduled_ts"] = None

        keep = False

        if item.get("scheduled_ts") is not None:
            keep = True
        else:
            if act_end_ts is not None and now <= act_end_ts and item.get("activity_next_ts") is not None:
                keep = True

        if keep:
            new_items.append(item)

    with _PENDING_SIGNAL_FOLLOWUPS_LOCK:
        _PENDING_SIGNAL_FOLLOWUPS[:] = new_items


def run_signal_followup_manager():
    while True:
        try:
            _check_signal_followups()
        except Exception as e:
            try:
                log(f"[FOLLOWUP] manager loop error: {e}")
            except Exception:
                pass
        time.sleep(60.0)


def _check_pending_entries():
    if not DYNAMIC_ENTRY_ENABLED:
        return

    with _PENDING_ENTRIES_LOCK:
        entries = list(_PENDING_ENTRIES)

    if not entries:
        return

    now = time.time()

    symbols = {e.get("symbol") for e in entries if e.get("symbol")}
    prices = {}
    for sym in symbols:
        price = get_last_price(sym)
        if price is None or price <= 0:
            continue
        prices[sym] = price

    positions_snapshot = []
    try:
        positions_snapshot = _get_open_positions_snapshot_cached()
    except Exception:
        positions_snapshot = []
    open_pos_keys = set()
    open_pos_any = set()
    if positions_snapshot:
        for p in positions_snapshot:
            if not isinstance(p, dict):
                continue
            sym = p.get("symbol")
            if not sym:
                continue
            amt = p.get("position_amt")
            try:
                if float(amt) == 0.0:
                    continue
            except Exception:
                continue
            try:
                acc = int(p.get("account_index"))
            except Exception:
                continue
            try:
                sym_u = str(sym).upper()
            except Exception:
                sym_u = sym
            open_pos_keys.add((acc, sym_u))
            open_pos_any.add(sym_u)

    new_entries = []

    for e in entries:
        symbol = e.get("symbol")
        signal = e.get("signal")
        base_ep = e.get("entry_price")
        conf = e.get("confidence", 0.0)
        atr_val = e.get("atr", 0.0)
        created_ts = e.get("created_ts", now)
        do_real = bool(e.get("do_real"))
        do_paper = bool(e.get("do_paper"))
        signal_profile = str(e.get("signal_profile") or "small").lower()

        if not symbol or signal not in ("BUY", "SELL"):
            continue
        try:
            base_ep_val = float(base_ep)
        except (TypeError, ValueError):
            continue
        if base_ep_val <= 0:
            continue

        timeout_raw = e.get("timeout_min")
        try:
            timeout_min = float(timeout_raw) if timeout_raw is not None else float(ACCOUNT_ENTRY_TIMEOUT_MIN)
        except (TypeError, ValueError):
            timeout_min = float(ACCOUNT_ENTRY_TIMEOUT_MIN)
        if timeout_min < 0.0:
            timeout_min = 0.0
        timeout_sec = timeout_min * 60.0 if timeout_min > 0.0 else 0.0

        age = now - float(created_ts)
        expired = timeout_sec > 0.0 and age >= timeout_sec

        price = prices.get(symbol)
        triggered = False
        target_price = None

        offset_pct = e.get("offset_pct")
        try:
            offset_pct_val = float(offset_pct) if offset_pct is not None else float(ACCOUNT_ENTRY_OFFSET_PCT)
        except (TypeError, ValueError):
            offset_pct_val = 0.0
        offset = abs(offset_pct_val) / 100.0 if offset_pct_val else 0.0

        if price is not None:
            if offset > 0.0:
                if signal == "BUY":
                    target_price = base_ep_val * (1.0 - offset)
                    if price <= target_price:
                        triggered = True
                else:
                    target_price = base_ep_val * (1.0 + offset)
                    if price >= target_price:
                        triggered = True
            else:
                # 0% offset -> անմիջապես trigger, օգտագործվում է
                # multi-entry առաջին մակարդակի համար (օր. offset_pct=0.0).
                target_price = base_ep_val
                triggered = True

        if triggered:
            exec_price = price if price is not None and price > 0 else base_ep_val
            level_idx = e.get("level_index")
            acc_raw = e.get("account_index")
            try:
                acc_idx = int(acc_raw) if acc_raw is not None else None
            except (TypeError, ValueError):
                acc_idx = None

            sym_u = None
            try:
                sym_u = str(symbol).upper()
            except Exception:
                sym_u = symbol

            if do_real and sym_u:
                if level_idx is None:
                    if acc_idx is not None:
                        if (int(acc_idx), sym_u) in open_pos_keys:
                            continue
                    else:
                        if sym_u in open_pos_any:
                            continue
                try:
                    vol_ok, vol_24h, vol_thr = _passes_24h_volume_gate(symbol)
                except Exception:
                    vol_ok, vol_24h, vol_thr = True, None, float(MIN_24H_QUOTE_VOLUME_USDT)
                if not vol_ok:
                    try:
                        log(
                            f"[ENTRY][WAIT][VOL24H] {symbol} {signal}: "
                            f"24h quote volume={_format_usdt_volume(vol_24h)} USDT < min {_format_usdt_volume(vol_thr)} USDT; "
                            f"dynamic entry kept pending"
                        )
                    except Exception:
                        pass
                    new_entries.append(e)
                    continue

            try:
                if do_real:
                    if level_idx is not None:
                        _execute_multi_entry_level(
                            symbol,
                            signal,
                            exec_price,
                            conf,
                            level_idx,
                            acc_idx,
                            entry_age_sec=float(age) if age is not None else None,
                            signal_ts=created_ts,
                            signal_entry_price=base_ep_val,
                            market_price=price,
                            signal_profile=signal_profile,
                        )
                    else:
                        _execute_real_entry(
                            symbol,
                            signal,
                            exec_price,
                            conf,
                            account_index=acc_idx,
                            entry_age_sec=float(age) if age is not None else None,
                            signal_ts=created_ts,
                            signal_entry_price=base_ep_val,
                            market_price=price,
                            signal_profile=signal_profile,
                        )
                if do_paper:
                    # Միայն առաջին մակարդակի համար բացենք paper trade,
                    # որպեսզի չբազմապատկենք simulated positions-ը.
                    if level_idx is None or level_idx == 0:
                        open_paper_trades(symbol, signal, exec_price, conf, atr_val)
            except Exception as e_exec:
                try:
                    log(f"[ENTRY] Dynamic entry execution error for {symbol}: {e_exec}")
                except Exception:
                    pass

            try:
                log(
                    f"[ENTRY] Dynamic entry triggered for {symbol} {signal} at "
                    f"price {format_price(exec_price)} (base={format_price(base_ep_val)}, "
                    f"offset={offset_pct_val:.2f}%, profile={signal_profile})"
                )
            except Exception:
                pass
            continue

        if expired:
            try:
                log(
                    f"[ENTRY] Dynamic entry timed out for {symbol} {signal} after "
                    f"{timeout_min:.0f} min; skipping trade."
                )
            except Exception:
                pass
            continue

        new_entries.append(e)

    with _PENDING_ENTRIES_LOCK:
        _PENDING_ENTRIES[:] = new_entries


def run_entry_manager():
    while True:
        try:
            _check_pending_entries()
        except Exception as e:
            try:
                log(f"[ENTRY] manager loop error: {e}")
            except Exception:
                pass
        time.sleep(ENTRY_MANAGER_INTERVAL_SECONDS)


def _validate_dual_profile_readiness() -> None:
    try:
        if not SECONDARY_SIGNALS_ENABLED:
            log("[PROFILE: large] disabled")
            return
        if not SECONDARY_SYMBOLS:
            log("[PROFILE: large][WARN] enabled but symbols list is empty")
        if not SECONDARY_USE_MAIN_CHANNEL and not SECONDARY_TELEGRAM_TOKEN:
            log("[PROFILE: large][WARN] dedicated channel selected but token is empty")
        static_chat_id = _resolve_static_chat_id(SECONDARY_TELEGRAM_CHAT_ID)
        if not SECONDARY_USE_MAIN_CHANNEL and static_chat_id is None:
            log("[PROFILE: large] using public subscriber mode (/start) with no static chat_id")
    except Exception:
        pass


def main_loop():
    """
    Main trading loop
    """
    log("Starting main trading loop...")
    _validate_dual_profile_readiness()

    try:
        if AITM_MASTER_ENABLED:
            _skip_non_aitm_trade_action("Startup cleanup trade actions")
        else:
            startup_cleanup_accounts()
    except Exception as e:
        try:
            log(f"[STARTUP CLEANUP] Failed: {e}")
        except Exception:
            pass

    try:
        stats_thread = threading.Thread(target=run_stats_bot, daemon=True)
        stats_thread.start()
        log("[STATS BOT] Background analytics bot started")
    except Exception as e:
        log(f"[STATS BOT] Failed to start analytics bot: {e}")

    try:
        accounts_thread = threading.Thread(target=run_accounts_bot, daemon=True)
        accounts_thread.start()
        log("[ACCOUNTS BOT] Background accounts bot started")
    except Exception as e:
        log(f"[ACCOUNTS BOT] Failed to start accounts bot: {e}")

    try:
        reports_thread = threading.Thread(target=run_account_report_bot, daemon=True)
        reports_thread.start()
        log("[REPORTS BOT] Background account report bot started")
    except Exception as e:
        log(f"[REPORTS BOT] Failed to start account report bot: {e}")

    # Backtest Control Bot — starts/stops with main engine as a subsystem.
    # Sole control interface for the backtesting system via Telegram.
    _backtest_bot = None
    try:
        from backtest.production.telegram_bot.bot import BacktestControlBot
        _backtest_bot = BacktestControlBot()
        _backtest_bot.start()
        log("[BACKTEST BOT] Background backtest control bot started")
    except Exception as e:
        log(f"[BACKTEST BOT] Failed to start backtest control bot: {e}")

    try:
        ws_started = bool(start_price_stream(SYMBOLS, INTERVAL))
        if ws_started:
            log(
                f"[WS PRICE] Background Binance price stream started for {len(SYMBOLS)} symbols at interval {INTERVAL}"
            )
        else:
            log("[WS PRICE] Background Binance price stream not started")
    except Exception as e:
        log(f"[WS PRICE] Failed to start price stream: {e}")

    try:
        exit_thread = threading.Thread(target=run_exit_manager, daemon=True)
        exit_thread.start()
        log("[EXIT MANAGER] Background exit manager started")
    except Exception as e:
        log(f"[EXIT MANAGER] Failed to start exit manager: {e}")

    if DYNAMIC_ENTRY_ENABLED:
        try:
            entry_thread = threading.Thread(target=run_entry_manager, daemon=True)
            entry_thread.start()
            log("[ENTRY MANAGER] Background entry manager started")
        except Exception as e:
            log(f"[ENTRY MANAGER] Failed to start entry manager: {e}")

    try:
        followup_thread = threading.Thread(target=run_signal_followup_manager, daemon=True)
        followup_thread.start()
        log("[FOLLOWUP] Background signal follow-up manager started")
    except Exception as e:
        log(f"[FOLLOWUP] Failed to start signal follow-up manager: {e}")

    # Isolated AI-driven trade management (open positions only).
    # This module is account-flag controlled and does not alter entry/signal logic.
    try:
        ai_tm_thread = threading.Thread(
            target=run_ai_trade_manager,
            args=(_get_accounts_cfg_live,),
            daemon=True,
        )
        ai_tm_thread.start()
        log("[AI-TM] Background AI trade manager started")
    except Exception as e:
        log(f"[AI-TM] Failed to start AI trade manager: {e}")

    # Standalone /start collector side-thread (optional, non-blocking).
    try:
        from monitoring import telegram_start_collector as _tg_start_collector

        start_collector_thread = threading.Thread(target=_tg_start_collector.main, daemon=True)
        start_collector_thread.start()
        log("[START COLLECTOR] telegram_start_collector started")
    except Exception as e:
        log(f"[START COLLECTOR] Failed to start telegram_start_collector: {e}")

    # Secondary token /start collector side-thread (optional, non-blocking).
    try:
        from monitoring import secondary_telegram_start_collector as _secondary_tg_start_collector

        secondary_start_collector_thread = threading.Thread(
            target=_secondary_tg_start_collector.main, daemon=True
        )
        secondary_start_collector_thread.start()
        log("[START COLLECTOR][SECONDARY] secondary_telegram_start_collector started")
    except Exception as e:
        log(f"[START COLLECTOR][SECONDARY] Failed to start secondary collector: {e}")

    # Isolated Market Regime sidecar (signal-only, no trade impact).
    try:
        _start_market_regime_service_if_needed()
        market_regime_thread = threading.Thread(target=_market_regime_watchdog_loop, daemon=True)
        market_regime_thread.start()
        log("[MARKET REGIME] Sidecar watchdog started")
    except Exception as e:
        log(f"[MARKET REGIME] Failed to start sidecar watchdog: {e}")

    # Market Quality Analyzer sidecar (observation-only, blocking disabled by default).
    try:
        _start_market_quality_analyzer_if_needed()
        mqa_thread = threading.Thread(target=_mqa_watchdog_loop, daemon=True)
        mqa_thread.start()
        log("[MQA] Market Quality Analyzer sidecar watchdog started")
    except Exception as e:
        log(f"[MQA] Failed to start sidecar watchdog: {e}")

    profile_symbol_count = len(SYMBOLS)
    if SECONDARY_SIGNALS_ENABLED and SECONDARY_SYMBOLS:
        profile_symbol_count += len(SECONDARY_SYMBOLS)
        try:
            log(
                f"[PROFILE: large] enabled symbols={len(SECONDARY_SYMBOLS)} "
                f"mode=core-logic-parity-with-small prefix={SECONDARY_TELEGRAM_PREFIX}"
            )
        except Exception:
            pass
    max_workers = min(32, profile_symbol_count) or 1
    # Force first AUTO_ML run immediately on startup so that
    # label_signals + training execute without waiting a full interval.
    last_auto_ml_run = time.time() - AUTO_ML_INTERVAL_SECONDS
    sltp_train_future = None
    sltp_train_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sltp_train")

    news_was_active = False
    last_us_news_summary_date = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            # Daily summary of high-impact US macro events for the
            # [00:00, next day 12:00] UTC window, sent once per calendar day.
            try:
                today_utc = datetime.now(timezone.utc).date()
            except Exception:
                today_utc = None

            if today_utc is not None and today_utc != last_us_news_summary_date:
                us_events = []
                try:
                    us_events = get_us_high_impact_events_for_today_window()
                except Exception as e:
                    try:
                        log(f"[NEWS GUARD] Failed to build US news summary: {e}")
                    except Exception:
                        pass

                if us_events:
                    try:
                        lines = [
                            "[NEWS GUARD] Today's US high-impact economic events (00:00 – next day 12:00 UTC):"
                        ]
                        pretty_lines = [
                            "US high-impact economic events (UTC)"
                        ]
                        try:
                            sorted_events = sorted(
                                us_events,
                                key=lambda ev: ev.get("time") or datetime.now(timezone.utc),
                            )
                        except Exception:
                            sorted_events = us_events

                        for ev in sorted_events:
                            when = ev.get("time")
                            if hasattr(when, "isoformat"):
                                when_str = when.isoformat()  # type: ignore[call-arg]
                            else:
                                when_str = str(when)
                            try:
                                when_pretty = (
                                    when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                                    if hasattr(when, "astimezone")
                                    else str(when)
                                )
                            except Exception:
                                when_pretty = str(when)
                            event_name = ev.get("event", "")
                            actual = ev.get("actual")
                            forecast = ev.get("forecast")
                            previous = ev.get("previous")

                            details = []
                            if actual not in (None, ""):
                                details.append(f"Actual: {actual}")
                            if forecast not in (None, ""):
                                details.append(f"Forecast: {forecast}")
                            if previous not in (None, ""):
                                details.append(f"Previous: {previous}")

                            if details:
                                lines.append(
                                    f"- {when_str} – {event_name} (" + " | ".join(details) + ")"
                                )
                                pretty_lines.append(
                                    f"- {when_pretty} — {event_name} (" + " | ".join(details) + ")"
                                )
                            else:
                                lines.append(f"- {when_str} – {event_name}")
                                pretty_lines.append(f"- {when_pretty} — {event_name}")

                        msg = "\n".join(lines)
                        pretty_msg = "\n".join(pretty_lines)
                        _broadcast_news_guard_message(msg, pretty_text=pretty_msg)
                    except Exception:
                        # Never let news summary failures break trading loop.
                        pass

                last_us_news_summary_date = today_utc

            active_events = []
            try:
                active_events = get_active_news_events()
            except Exception as e:
                log(f"[NEWS GUARD] Error while checking news window: {e}")

            news_active = bool(active_events)

            # NEWS GUARD նոտիվիկացիաները միշտ ուղարկվում են, որպեսզի
            # դու Telegram-ում տեսնես, որ բարձր ազդեցության նյուզ կա,
            # բայց արդյոք թրեյդինգն ավտոմատ կկանգնի, որոշվում է
            # NEWS_GUARD_ENABLED config flag-ով։

            if news_active and not news_was_active:
                ev = active_events[0]
                try:
                    when = ev.get("time")
                    if hasattr(when, "isoformat"):
                        when_str = when.isoformat()  # type: ignore[call-arg]
                    else:
                        when_str = str(when)
                    try:
                        when_pretty = (
                            when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                            if hasattr(when, "astimezone")
                            else str(when)
                        )
                    except Exception:
                        when_pretty = str(when)
                    country = ev.get("country", "")
                    event_name = ev.get("event", "")
                    if NEWS_GUARD_ENABLED:
                        _broadcast_news_guard_message(
                            f"[NEWS GUARD] High-impact news detected: {country} - {event_name} at {when_str}. "
                            f"Trading will be PAUSED automatically (news_guard.enabled = true)."
                            ,
                            pretty_text=(
                                "High-impact news detected\n"
                                f"- {country} — {event_name}\n"
                                f"- Time: {when_pretty}\n"
                                "Trading: PAUSED"
                            ),
                        )
                    else:
                        _broadcast_news_guard_message(
                            f"[NEWS GUARD] High-impact news detected: {country} - {event_name} at {when_str}. "
                            f"Trading is NOT auto-paused (news_guard.enabled = false)."
                            ,
                            pretty_text=(
                                "High-impact news detected\n"
                                f"- {country} — {event_name}\n"
                                f"- Time: {when_pretty}\n"
                                "Trading: NOT paused"
                            ),
                        )
                except Exception:
                    if NEWS_GUARD_ENABLED:
                        _broadcast_news_guard_message(
                            "[NEWS GUARD] High-impact news window detected. Trading will be PAUSED automatically."
                            ,
                            pretty_text=(
                                "High-impact news window detected\n"
                                "Trading: PAUSED"
                            ),
                        )
                    else:
                        _broadcast_news_guard_message(
                            "[NEWS GUARD] High-impact news window detected. Trading is NOT auto-paused (disabled in config)."
                            ,
                            pretty_text=(
                                "High-impact news window detected\n"
                                "Trading: NOT paused"
                            ),
                        )

            if not news_active and news_was_active:
                if NEWS_GUARD_ENABLED:
                    _broadcast_news_guard_message(
                        "[NEWS GUARD] High-impact news window ended. Trading is RESUMED automatically."
                        ,
                        pretty_text=(
                            "High-impact news window ended\n"
                            "Trading: RESUMED"
                        ),
                    )
                else:
                    _broadcast_news_guard_message(
                        "[NEWS GUARD] High-impact news window ended. Trading was never auto-paused (disabled in config)."
                        ,
                        pretty_text=(
                            "High-impact news window ended\n"
                            "Trading: NOT paused"
                        ),
                    )

            news_was_active = news_active

            # Եթե NEWS_GUARD_ENABLED=True և հիմա high-impact նյուզի window ենք,
            # ապա ընդհանրապես չենք մտնում _process_symbol, այսինքն
            # նոր positions չենք բացում news-ի ընթացքում։
            if NEWS_GUARD_ENABLED and news_active:
                _sleep_until_next_candle(INTERVAL, buffer_sec=2.0)
                continue

            # Periodic self-learning ML loop (label + train + reload model)
            if AUTO_ML_ENABLED:
                now = time.time()
                if now - last_auto_ml_run >= AUTO_ML_INTERVAL_SECONDS:
                    try:
                        from ml.auto_ml import run_auto_ml
                        # log("[AUTO_ML] Running label_signals + train_from_signal_log...")
                        run_auto_ml()
                        # log("[AUTO_ML] Completed auto ML update")
                        # Non-blocking SL/TP AI training/update.
                        try:
                            if sltp_train_future is not None and not sltp_train_future.done():
                                log("[SLTP-NN TRAIN] previous training still running; skip this cycle")
                            else:
                                def _run_sltp_train_task():
                                    try:
                                        run_sl_tp_nn_training()
                                    except Exception:
                                        pass
                                sltp_train_future = sltp_train_pool.submit(_run_sltp_train_task)
                                log("[SLTP-NN TRAIN] started async training/update task")
                        except Exception as se:
                            log(f"[SLTP-NN TRAIN] Async training dispatch failed: {se}")
                    except Exception as e:
                        ""
                        # log(f"[AUTO_ML] Auto ML update failed: {e}")
                    finally:
                        last_auto_ml_run = now

            # Refresh Telegram subscribers (handles new /start commands)
            if TELEGRAM_TOKEN:
                try:
                    update_subscribers(
                        TELEGRAM_TOKEN,
                        instance_id=INSTANCE_ID,
                        update_base_url=UPDATE_BASE_URL,
                        include_config=UPDATE_INCLUDE_CONFIG,
                        include_data=UPDATE_INCLUDE_DATA,
                    )
                except Exception as e:
                    log(f"Telegram subscribers update failed: {e}")
            if (
                SECONDARY_SIGNALS_ENABLED
                and SECONDARY_TELEGRAM_TOKEN
                and str(SECONDARY_TELEGRAM_TOKEN) != str(TELEGRAM_TOKEN)
            ):
                try:
                    update_subscribers(
                        SECONDARY_TELEGRAM_TOKEN,
                        instance_id=INSTANCE_ID,
                        update_base_url=UPDATE_BASE_URL,
                        include_config=UPDATE_INCLUDE_CONFIG,
                        include_data=UPDATE_INCLUDE_DATA,
                    )
                except Exception as e:
                    log(f"Secondary Telegram subscribers update failed: {e}")

            futures = [executor.submit(_process_symbol, symbol, "small") for symbol in SYMBOLS]
            if SECONDARY_SIGNALS_ENABLED and SECONDARY_SYMBOLS:
                futures.extend(
                    executor.submit(_process_symbol, symbol, "large") for symbol in SECONDARY_SYMBOLS
                )
            for f in futures:
                # Propagate any exception from worker threads
                f.result()

            # Sleep until next candle boundary so signals fire within
            # ~2 s of each candle close instead of drifting.
            _sleep_until_next_candle(INTERVAL, buffer_sec=2.0)


def calculate_qty(capital, confidence):
    """
    Simple position sizing: proportional to capital and confidence
    """
    risk_factor = 0.01  # default risk per trade
    return max(0.001, capital * risk_factor * confidence)


def _sleep_until_next_candle(interval: str, buffer_sec: float = 2.0) -> None:
    """Sleep until the next candle boundary + buffer.

    Instead of sleeping a fixed duration (which drifts), compute the exact
    moment the current candle closes and sleep until then.  With buffer_sec=2
    the bot wakes ~2 s after candle close, giving Binance time to finalise
    the candle while keeping signal latency under 3 s.
    """
    try:
        iv_sec = float(interval_to_seconds(interval))
    except Exception:
        iv_sec = 60.0
    if iv_sec <= 0:
        iv_sec = 60.0

    now = time.time()
    # Next candle boundary = ceil(now / iv_sec) * iv_sec
    next_boundary = (int(now / iv_sec) + 1) * iv_sec
    sleep_s = (next_boundary - now) + buffer_sec
    # Clamp: never sleep less than 1 s or more than iv_sec + buffer
    if sleep_s < 1.0:
        sleep_s = 1.0
    elif sleep_s > iv_sec + buffer_sec:
        sleep_s = iv_sec + buffer_sec
    time.sleep(sleep_s)


def interval_to_seconds(interval):
    """
    Convert interval string to seconds
    """
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    return 60  # default 1m


if __name__ == "__main__":
    # Offline SL/TP NN training from accumulated sl_tp_nn_log.csv.
    # This runs once on startup and updates sl_tp_nn_weights.json so that
    # Telegram-facing AI SL/TP տոկոսները լինեն առավել ադապտացված պատմական
    # արդյունքներին, առանց ազդելու core exit manager-ի վրա.
    try:
        run_sl_tp_nn_training()
    except Exception as e:
        try:
            log(f"[SLTP-NN TRAIN] Offline training skipped due to error: {e}")
        except Exception:
            pass

    main_loop()
