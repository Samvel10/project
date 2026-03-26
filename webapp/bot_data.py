"""
Helpers to read live bot data without importing main.py.
All functions are read-only and safe to call from the web process.
"""
from __future__ import annotations
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CFG  = ROOT / "config"
BINANCE_FAPI = "https://fapi.binance.com"
_TIMEOUT = 8


# ─── config readers ───────────────────────────────────────────────────────────

def read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_trading_config() -> dict:
    return read_yaml(CFG / "trading.yaml")


def get_accounts_config() -> dict:
    return read_yaml(CFG / "binance_accounts.yaml")


def get_market_regime_config() -> dict:
    return read_yaml(CFG / "market_regime.yaml")


def get_all_configs() -> dict:
    return {
        "trading":           read_yaml(CFG / "trading.yaml"),
        "accounts":          read_yaml(CFG / "binance_accounts.yaml"),
        "market_regime":     read_yaml(CFG / "market_regime.yaml"),
        "execution":         read_yaml(CFG / "execution.yaml"),
        "risk":              read_yaml(CFG / "risk.yaml"),
        "ml":                read_yaml(CFG / "ml.yaml"),
        "secondary_signals": read_yaml(CFG / "secondary_signals.yaml"),
        "symbols":           read_yaml(CFG / "symbols.yaml"),
    }


# ─── runtime state ────────────────────────────────────────────────────────────

def get_market_regime_state() -> dict:
    try:
        p = DATA / "market_regime_state.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def get_market_quality_cache() -> dict:
    try:
        p = DATA / "market_quality_cache.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def get_symbol_blocklist() -> list:
    try:
        p = DATA / "symbol_blocklist.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return list(data.get("symbols", []))
    except Exception:
        pass
    return []


def get_main_process_state() -> dict:
    try:
        p = DATA / "main_process_state.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# ─── signal log ───────────────────────────────────────────────────────────────

def _get_large_symbols() -> set:
    """Read the large-profile symbol list from secondary_signals.yaml."""
    try:
        cfg = read_yaml(CFG / "secondary_signals.yaml")
        syms = cfg.get("symbols") or []
        return {str(s).upper().strip() for s in syms if s}
    except Exception:
        return set()


_SIGNAL_CSV_FIELDS = [
    "timestamp_ms", "symbol", "direction", "interval", "entry", "sl",
    "tp1", "tp2", "tp3", "confidence", "rsi", "momentum", "acceleration",
    "volatility", "atr", "structure", "range_type", "fib_direction", "label",
    "outcome_type", "pnl_pct", "hold_minutes", "sl_hit", "tp1_hit", "tp2_hit",
    "tp3_hit", "sl_alerted", "tp1_alerted", "tp2_alerted", "tp3_alerted",
    "tp1_sent", "tp2_sent", "tp3_sent",
]


def get_signal_log(limit: int = 200) -> List[dict]:
    rows = []
    try:
        p = DATA / "signal_log.csv"
        if not p.exists():
            return []
        large_symbols = _get_large_symbols()
        with open(p, "r", encoding="utf-8") as f:
            # CSV has no header row — use explicit fieldnames
            first = f.readline().strip()
            f.seek(0)
            has_header = first.startswith("timestamp_ms")
            reader = csv.DictReader(
                f,
                fieldnames=None if has_header else _SIGNAL_CSV_FIELDS
            )
            for row in reader:
                d = dict(row)
                # Skip the header row if it exists
                if d.get("timestamp_ms") == "timestamp_ms":
                    continue
                raw_label = (d.get("label") or "").strip().lower()
                if raw_label in ("small", "large"):
                    d["category"] = raw_label
                else:
                    symbol = (d.get("symbol") or "").upper().strip()
                    d["category"] = "large" if symbol in large_symbols else "small"
                rows.append(d)
        return rows[-limit:]
    except Exception:
        return rows


# ─── Binance API helpers ───────────────────────────────────────────────────────

def _binance_get(path: str, params: dict = None) -> Any:
    resp = requests.get(f"{BINANCE_FAPI}{path}", params=params or {}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_binance_account_balance(api_key: str, api_secret: str) -> List[dict]:
    """Fetch account balance for a Binance Futures account."""
    import hmac
    import hashlib
    ts = int(time.time() * 1000)
    params = f"timestamp={ts}"
    sig = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    try:
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v2/balance",
            params={"timestamp": ts, "signature": sig},
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def get_binance_positions(api_key: str, api_secret: str) -> List[dict]:
    """Fetch open positions for a Binance Futures account."""
    import hmac
    import hashlib
    ts = int(time.time() * 1000)
    params = f"timestamp={ts}"
    sig = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    try:
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v2/positionRisk",
            params={"timestamp": ts, "signature": sig},
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [p for p in data if float(p.get("positionAmt", 0)) != 0]
    except Exception:
        return []


def get_binance_trade_history(api_key: str, api_secret: str, symbol: str, limit: int = 50) -> List[dict]:
    """Fetch recent trade history for a symbol."""
    import hmac
    import hashlib
    ts = int(time.time() * 1000)
    query = f"symbol={symbol}&limit={limit}&timestamp={ts}"
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    try:
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/userTrades",
            params={"symbol": symbol, "limit": limit, "timestamp": ts, "signature": sig},
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def get_klines(symbol: str, interval: str = "1m", limit: int = 200) -> List[dict]:
    """Fetch OHLCV candles."""
    try:
        raw = _binance_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        return [
            {
                "time":   int(r[0]) // 1000,
                "open":   float(r[1]),
                "high":   float(r[2]),
                "low":    float(r[3]),
                "close":  float(r[4]),
                "volume": float(r[5]),
            }
            for r in raw
        ]
    except Exception:
        return []


def get_all_accounts_with_state() -> List[dict]:
    """Return all accounts from config with their live state."""
    cfg = get_accounts_config()
    accounts = cfg.get("accounts") or []
    result = []
    for idx, acc in enumerate(accounts):
        settings = acc.get("settings") or {}
        # API keys are stored directly on the account object in binance_accounts.yaml
        api_key    = acc.get("api_key") or ""
        api_secret = acc.get("api_secret") or ""
        # trade_enabled and leverage are also top-level account fields
        trade_enabled = bool(acc.get("trade_enabled", False))
        leverage      = acc.get("leverage") or settings.get("leverage")
        result.append({
            "index":         idx,
            "name":          acc.get("name") or f"Account {idx+1}",
            "trade_enabled": trade_enabled,
            "notional":      settings.get("fixed_notional_usd") or settings.get("fixed_notional_value"),
            "leverage":      leverage,
            "api_key":       api_key,
            "api_secret":    api_secret,
            "proxy":         acc.get("proxy") or "",
            "aitm_enabled":  bool(settings.get("ai_dynamic_trade_management", False)),
            "signal_source": settings.get("signal_source", "small"),
        })
    return result


# ─── running processes ────────────────────────────────────────────────────────

def get_process_status() -> dict:
    status = {}
    names = {
        "main.py":                     "Trading Engine",
        "market_regime.py":            "Market Regime",
        "market_quality_analyzer.py":  "Market Quality",
        "main_control_bot.py":         "Control Bot",
    }
    try:
        for proc_file in Path("/proc").iterdir():
            if not proc_file.name.isdigit():
                continue
            try:
                cmd = (proc_file / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
                for script, label in names.items():
                    if script in cmd and label not in status:
                        status[label] = {"pid": int(proc_file.name), "running": True}
            except Exception:
                pass
    except Exception:
        pass
    for label in names.values():
        if label not in status:
            status[label] = {"pid": None, "running": False}
    return status


# ─── binance signed request helper ────────────────────────────────────────────

def _binance_signed(path: str, api_key: str, api_secret: str,
                    params: dict = None) -> Any:
    import hmac, hashlib
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in p.items())
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    p["signature"] = sig
    headers = {"X-MBX-APIKEY": api_key}
    resp = requests.get(f"{BINANCE_FAPI}{path}", params=p,
                        headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ─── 24h ticker & symbol search ───────────────────────────────────────────────

def get_binance_ticker_24h(symbol: str) -> dict:
    try:
        return _binance_get("/fapi/v1/ticker/24hr", {"symbol": symbol.upper()})
    except Exception:
        return {}


def get_binance_mark_price(symbol: str) -> dict:
    try:
        return _binance_get("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
    except Exception:
        return {}


_SYMBOLS_CACHE: List[str] = []
_SYMBOLS_CACHED_AT: float = 0.0

def get_binance_usdt_symbols() -> List[str]:
    global _SYMBOLS_CACHE, _SYMBOLS_CACHED_AT
    if _SYMBOLS_CACHE and time.time() - _SYMBOLS_CACHED_AT < 3600:
        return _SYMBOLS_CACHE
    try:
        data = _binance_get("/fapi/v1/exchangeInfo")
        syms = sorted(
            s["symbol"] for s in data.get("symbols", [])
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        )
        _SYMBOLS_CACHE = syms
        _SYMBOLS_CACHED_AT = time.time()
        return syms
    except Exception:
        return _SYMBOLS_CACHE or []


# ─── open orders ──────────────────────────────────────────────────────────────

def get_binance_open_orders(api_key: str, api_secret: str) -> List[dict]:
    try:
        return _binance_signed("/fapi/v1/openOrders", api_key, api_secret)
    except Exception:
        return []


# ─── income / transaction history ─────────────────────────────────────────────

def get_binance_income_history(api_key: str, api_secret: str,
                                income_type: str = None, limit: int = 100) -> List[dict]:
    try:
        params = {"limit": limit}
        if income_type:
            params["incomeType"] = income_type
        return _binance_signed("/fapi/v1/income", api_key, api_secret, params)
    except Exception:
        return []


# ─── all asset balances ────────────────────────────────────────────────────────

def create_listen_key(api_key: str) -> str:
    """Create a Binance Futures user data stream listen key."""
    try:
        resp = requests.post(
            f"{BINANCE_FAPI}/fapi/v1/listenKey",
            headers={"X-MBX-APIKEY": api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("listenKey", "")
    except Exception:
        return ""


def renew_listen_key(api_key: str, listen_key: str) -> bool:
    """Keep-alive a listen key (must be called every 30 min)."""
    try:
        resp = requests.put(
            f"{BINANCE_FAPI}/fapi/v1/listenKey",
            headers={"X-MBX-APIKEY": api_key},
            params={"listenKey": listen_key},
            timeout=_TIMEOUT,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_binance_all_balances(api_key: str, api_secret: str) -> List[dict]:
    try:
        data = _binance_signed("/fapi/v2/account", api_key, api_secret)
        assets = data.get("assets", [])
        return [a for a in assets
                if float(a.get("walletBalance", 0)) != 0
                or float(a.get("unrealizedProfit", 0)) != 0]
    except Exception:
        return []
