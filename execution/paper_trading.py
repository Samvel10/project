from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ruamel.yaml import YAML

from data.ws_price_stream import get_last_price
from monitoring.logger import log
from monitoring.paper_trade_history import log_paper_entry, log_paper_exit

_YAML = YAML(typ="safe")
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "paper_accounts.yaml"

_PAPER_CONFIG_LOADED = False
_PAPER_ENABLED = False
_PAPER_ACCOUNTS: List[Dict[str, Any]] = []
_PAPER_POSITIONS: Dict[Tuple[int, str], Dict[str, Any]] = {}


def _load_paper_config() -> None:
    global _PAPER_CONFIG_LOADED, _PAPER_ENABLED, _PAPER_ACCOUNTS
    if _PAPER_CONFIG_LOADED:
        return

    cfg: Dict[str, Any] = {}
    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8") as f:
                raw = _YAML.load(f)
                if isinstance(raw, dict):
                    cfg = raw
        except Exception:
            cfg = {}

    _PAPER_ENABLED = bool(cfg.get("enabled", False))

    accounts_cfg = cfg.get("accounts") or []
    accounts: List[Dict[str, Any]] = []
    if isinstance(accounts_cfg, list):
        for acc in accounts_cfg:
            if not isinstance(acc, dict):
                continue
            trade_enabled = acc.get("trade_enabled")
            fixed_notional = acc.get("fixed_notional_usd", 0)
            settings = acc.get("settings") or {}
            if not isinstance(settings, dict):
                settings = {}
            accounts.append(
                {
                    "name": acc.get("name") or "",
                    "trade_enabled": False if trade_enabled is False else True,
                    "fixed_notional_usd": fixed_notional,
                    "settings": settings,
                }
            )

    _PAPER_ACCOUNTS = accounts
    _PAPER_CONFIG_LOADED = True


def paper_trading_enabled() -> bool:
    _load_paper_config()
    return bool(_PAPER_ENABLED and _PAPER_ACCOUNTS)


def _get_account_settings(index: int) -> Tuple[Dict[str, Any], Dict[str, Any]] | Tuple[None, None]:
    if index < 0 or index >= len(_PAPER_ACCOUNTS):
        return None, None
    acc = _PAPER_ACCOUNTS[index]
    settings = acc.get("settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    return acc, settings


def _parse_confidence_threshold(raw: Any) -> float | None:
    """Parse a per-account confidence threshold into [0,1] or return None.

    Accepts values like 0.7 or 70 (interpreted as 70%).
    """

    if raw in (None, ""):
        # Default to 70% if not explicitly configured in YAML
        return 0.7
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 1.0:
        val = val / 100.0
    if val < 0.0:
        val = 0.0
    if val > 1.0:
        val = 1.0
    return val


def _auto_sl_tp_from_atr_conf(
    entry_price: float, atr_value: float, long_side: bool, confidence: float
):
    """Derive dynamic SL/TP prices for paper trading from ATR and confidence."""

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

    base_vol_pct = 0.0
    if ep > 0 and atr > 0:
        base_vol_pct = abs(atr / ep) * 100.0
    base_vol_pct = max(min(base_vol_pct, 5.0), 0.3)

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
    rr_levels = (1.0, 1.5, 2.0)
    tp_pcts = [sl_pct * rr * reward_mult for rr in rr_levels]

    sl_factor = abs(sl_pct) / 100.0
    tp_factors = [abs(p) / 100.0 for p in tp_pcts]

    if long_side:
        sl_price = ep * (1.0 - sl_factor)
        tps = [ep * (1.0 + f) for f in tp_factors]
    else:
        sl_price = ep * (1.0 + sl_factor)
        tps = [ep * (1.0 - f) for f in tp_factors]

    return sl_price, tps


def open_paper_trades(
    symbol: str,
    signal: str,
    entry_price: float,
    confidence: float,
    atr_value: float | None = None,
) -> None:
    """Open simulated trades on all enabled paper accounts for a given signal.

    This does not send any orders to Binance. It only records internal
    positions that will be managed by check_paper_exits() using WebSocket
    prices.
    """

    _load_paper_config()
    if not _PAPER_ENABLED or not _PAPER_ACCOUNTS:
        return

    if signal not in ("BUY", "SELL"):
        return

    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        return
    if ep <= 0:
        return

    long_side = signal == "BUY"

    for idx, acc in enumerate(_PAPER_ACCOUNTS):
        if not acc.get("trade_enabled", True):
            continue

        acc_obj, settings = _get_account_settings(idx)
        if acc_obj is None:
            continue

        # Optional per-paper-account confidence threshold.
        conf_thr = _parse_confidence_threshold(settings.get("confidence"))
        if conf_thr is not None and confidence < conf_thr:
            # Do not open trades on this paper account if signal is weaker.
            continue

        sl_pct = settings.get("sl_pct")
        tp_pcts = settings.get("tp_pcts") or []
        tp_mode_raw = settings.get("tp_mode")
        auto_sl_flag = settings.get("auto_sl_tp")

        is_sl_auto_str = isinstance(sl_pct, str) and sl_pct.strip().lower() == "auto"
        auto_sl_enabled = bool(auto_sl_flag) or is_sl_auto_str

        tp_levels: List[float] = []
        if not auto_sl_enabled:
            try:
                sl_val = float(sl_pct)
            except (TypeError, ValueError):
                continue

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

        # Number of TP legs we actually use for paper exits. For dynamic
        # auto_sl_enabled accounts we start from tp_mode and later clamp it
        # to the number of concrete TP prices we generate, so that
        # tp_mode=1 truly means a single TP target.
        num_legs = tp_mode if auto_sl_enabled else min(tp_mode, len(tp_levels), 3)
        if num_legs <= 0:
            continue

        fixed_notional = acc_obj.get("fixed_notional_usd") or 0
        try:
            notional = float(fixed_notional)
        except (TypeError, ValueError):
            notional = 0.0
        if notional <= 0:
            notional = 10.0  # default simulated size

        qty = notional / ep
        if qty <= 0:
            continue

        # Compute concrete TP prices for each leg.
        if auto_sl_enabled:
            sl_dyn, tps_dyn = _auto_sl_tp_from_atr_conf(
                ep, atr_value or 0.0, long_side, confidence
            )

            tp_prices = []
            if tps_dyn:
                # Limit dynamic TP levels to the configured number of legs so
                # that tp_mode controls how many TP targets the paper
                # position uses.
                for tp_price in tps_dyn[:num_legs]:
                    try:
                        tp_val = float(tp_price)
                    except (TypeError, ValueError):
                        continue
                    if tp_val > 0:
                        tp_prices.append(tp_val)

            if not tp_prices or sl_dyn is None or sl_dyn <= 0:
                # Fallback to a simple 2,3,4% ladder if dynamic computation
                # fails, so that paper exits are still well-defined. Respect
                # num_legs so that tp_mode=1 yields a single TP.
                tp_prices = []
                fallback_pcts = (2.0, 3.0, 4.0)
                for pct in fallback_pcts[:num_legs]:
                    if long_side:
                        tp_price = ep * (1.0 + pct / 100.0)
                    else:
                        tp_price = ep * (1.0 - pct / 100.0)
                    tp_prices.append(tp_price)

                if long_side:
                    sl_price = ep * (1.0 - 2.0 / 100.0)
                else:
                    sl_price = ep * (1.0 + 2.0 / 100.0)
            else:
                sl_price = float(sl_dyn)
                # Never use more TP legs than the dynamic prices we have.
                num_legs = min(max(num_legs, 1), len(tp_prices))
        else:
            tp_prices = []
            for i in range(num_legs):
                pct = tp_levels[i]
                if long_side:
                    tp_price = ep * (1.0 + pct / 100.0)
                else:
                    tp_price = ep * (1.0 - pct / 100.0)
                tp_prices.append(tp_price)

            # Compute SL price for full exit.
            if long_side:
                sl_price = ep * (1.0 - sl_val / 100.0)
            else:
                sl_price = ep * (1.0 + sl_val / 100.0)

        key = (idx, symbol)
        _PAPER_POSITIONS[key] = {
            "initial_qty": abs(qty),
            "remaining_qty": abs(qty),
            "long": long_side,
            "tp_prices": tp_prices,
            "sl_price": sl_price,
            "num_legs": num_legs,
            "entry_price": ep,
        }

        try:
            log_paper_entry(
                account_index=idx,
                symbol=symbol,
                side=signal,
                qty=abs(qty),
                entry_price=ep,
                reason="ENTRY",
            )
        except Exception:
            pass


def check_paper_exits() -> None:
    """Apply SL/TP logic to all open paper positions using WebSocket prices."""

    if not paper_trading_enabled():
        return

    if not _PAPER_POSITIONS:
        return

    # Collect symbols that currently have open paper positions.
    symbols = set(sym for (_acc, sym) in _PAPER_POSITIONS.keys())
    prices: Dict[str, float] = {}
    for sym in symbols:
        price = get_last_price(sym)
        if price is None or price <= 0:
            continue
        prices[sym] = price

    if not prices:
        return

    keys = list(_PAPER_POSITIONS.keys())
    for key in keys:
        acc_idx, symbol = key
        pos = _PAPER_POSITIONS.get(key)
        if not pos:
            continue

        if symbol not in prices:
            continue

        current_price = prices[symbol]
        long_side = bool(pos.get("long", True))
        try:
            entry_price = float(pos.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        if entry_price <= 0:
            continue

        try:
            initial_qty = float(pos.get("initial_qty") or 0.0)
            remaining_qty = float(pos.get("remaining_qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if initial_qty <= 0 or remaining_qty <= 0:
            _PAPER_POSITIONS.pop(key, None)
            continue

        tp_prices: List[float] = pos.get("tp_prices") or []
        sl_price = pos.get("sl_price")
        num_legs = int(pos.get("num_legs") or 0)

        if not tp_prices or sl_price is None or num_legs <= 0:
            continue

        direction = 1.0 if long_side else -1.0
        pnl_pct = (current_price - entry_price) / entry_price * 100.0 * direction

        # 1) Stop-loss: always close the full remaining position.
        sl_hit = (long_side and current_price <= sl_price) or (
            (not long_side) and current_price >= sl_price
        )
        if sl_hit:
            try:
                log_paper_exit(
                    account_index=acc_idx,
                    symbol=symbol,
                    side="SELL" if long_side else "BUY",
                    qty=remaining_qty,
                    entry_price=entry_price,
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                    reason="SL",
                )
            except Exception:
                pass

            _PAPER_POSITIONS.pop(key, None)
            # try:
            #     log(
            #         f"[PAPER EXIT] Closed {symbol} paper position on account {acc_idx} "
            #         f"by SL at price {current_price:.8f} (pnl_pct={pnl_pct:.2f}%)"
            #     )
            # except Exception:
            #     pass
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

        if crossed <= 0:
            continue

        crossed = min(crossed, num_legs)

        target_remaining_frac = float(num_legs - crossed) / float(num_legs)
        target_remaining_qty = initial_qty * target_remaining_frac

        if remaining_qty <= target_remaining_qty * 1.0001:
            continue

        qty_to_close = remaining_qty - target_remaining_qty
        if qty_to_close <= 0:
            continue

        new_remaining = remaining_qty - qty_to_close
        if new_remaining <= 0:
            _PAPER_POSITIONS.pop(key, None)
        else:
            pos["remaining_qty"] = new_remaining

        try:
            log_paper_exit(
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

        # try:
        #     log(
        #         f"[PAPER EXIT] Closed partial {symbol} paper position on account {acc_idx} "
        #         f"by TP (crossed {crossed} level(s)) at price {current_price:.8f} "
        #         f"(pnl_pct={pnl_pct:.2f}%)"
        #     )
        # except Exception:
        #     pass
