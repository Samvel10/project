import time
import hmac
import hashlib
import os
import math
import inspect
import threading
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import requests
from urllib.parse import urlencode
try:
    from ruamel.yaml import YAML  # type: ignore
except Exception:
    YAML = None
import yaml as _pyyaml

from config.proxies import get_random_proxy, build_proxy_from_string
from data.symbol_blocklist import is_symbol_blocked
from monitoring.logger import log
from monitoring.telegram import send_telegram
from monitoring.subscribers import get_subscribers
from monitoring.trade_history import log_trade_entry
from execution.sl_tp import compute_sl_tp
from config import settings as app_settings

# C++ accelerated execution primitives (optional).
# If the compiled module is available, hot-path functions (signing,
# HTTP, price/qty math) use C++ for lower latency. Otherwise the
# original Python implementations below are used — behavior is identical.
try:
    from execution import binance_fast as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    _cpp = None  # type: ignore[assignment]
    _CPP_AVAILABLE = False

# Startup confirmation — will be visible in bot logs.
try:
    if _CPP_AVAILABLE:
        log("[C++ ACCEL] binance_fast C++ module LOADED — signing, HTTP (thread-safe), price/qty math accelerated")
    else:
        log("[C++ ACCEL] binance_fast C++ module NOT found — using Python fallback (no speed loss, same behavior)")
except Exception:
    pass

BASE_URL = "https://fapi.binance.com"

# Global state for multi-account support and exchange filters
_ACCOUNTS_MODE = "env"  # "env", "single", or "multi"
_CLIENTS = None  # type: ignore[var-annotated]
_ACCOUNT_TRADE_ENABLED: list[bool] = []
_ACCOUNT_LEVERAGE_OVERRIDES: list[Optional[int]] = []
_ACCOUNT_SETTINGS: list[Dict[str, Any]] = []
_ACCOUNT_PROXIES = []  # per-account fixed proxies (dict or None)
_ACCOUNT_NAMES: list[str] = []  # human-friendly names from binance_accounts.yaml
_ACCOUNTS_CFG_MTIME: Optional[float] = None
_GLOBAL_DD_PAUSE_ENABLED_DEFAULT: bool = True
_GLOBAL_DD_PAUSE_PCT_DEFAULT: float = 20.0

# Tracks requested vs effective leverage per (account_index, symbol).
# When Binance rejects a requested leverage (-4028), we fall back to a lower
# leverage and scale notional by requested/effective so that exposure stays
# consistent.
_REQUESTED_LEVERAGE_BY_SYMBOL: Dict[Tuple[int, str], int] = {}
_EFFECTIVE_LEVERAGE_BY_SYMBOL: Dict[Tuple[int, str], int] = {}
_LEVERAGE_SET_TS_BY_SYMBOL: Dict[Tuple[int, str], float] = {}
_LEVERAGE_CACHE_TTL_SEC: float = 600.0
_SYMBOL_FILTERS = {}
_EXCHANGE_INFO_LOADED = False
_YAML = YAML(typ="safe") if YAML is not None else None
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "binance_accounts.yaml"
_EXECUTION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "execution.yaml"
_TRADE_HISTORY_DIR = Path(__file__).resolve().parents[1] / "data" / "trade_history"
_PROXY_CONFIG = None
_PROXY_CONFIG_LOADED = False
_SYMBOL_451_BLACKLIST: set[str] = set()
_MARGIN_MODE_LOGGED_ERRORS = set()
_FIBO_STATE: Dict[int, Dict[str, Any]] = {}
_FIBO_RESET_DONE: set[int] = set()
_STARTUP_CLEANUP_DONE: set[int] = set()
_DAILY_DD_STATE: Dict[int, Dict[str, Any]] = {}
_DELAY_LIMIT_STATE: Dict[int, Dict[str, Any]] = {}
_DELAY_LIMIT_LOGS: Dict[int, Dict[str, Any]] = {}
_SYMBOL_LOSS_TIMESTAMPS: Dict[Tuple[int, str], List[float]] = {}
_SYMBOL_LOSS_COOLDOWN_UNTIL: Dict[Tuple[int, str], float] = {}
_ADAPTIVE_SIZING_CACHE: Dict[int, Dict[str, Any]] = {}
_ADAPTIVE_SIZING_LOCK = threading.Lock()

_ROOM_SIZING_DIR = Path(__file__).resolve().parents[1] / "data" / "room_sizing"
_ROOM_SIZING_DIR.mkdir(parents=True, exist_ok=True)
_ROOM_SIZING_LOCK = threading.Lock()
_ROOM_SIZING_LAST_SYNC: Dict[int, float] = {}
_ROOM_LAST_ASSIGNMENT: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

_ACCOUNT_PAUSE_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "account_pause_state.json"
_ACCOUNT_PAUSE_STATE_LOCK = threading.RLock()
_ACCOUNT_PAUSE_STATE_LOADED = False
_ACCOUNT_PAUSE_STATE: Dict[str, Any] = {}
_MARKET_REGIME_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "market_regime_state.json"
_MARKET_REGIME_STATE_CACHE: Optional[Dict[str, Any]] = None
_MARKET_REGIME_STATE_MTIME: Optional[float] = None

_COMMISSION_RATE_CACHE: Dict[str, Dict[str, float]] = {}
_COMMISSION_RATE_CACHE_LOCK = threading.Lock()
_COMMISSION_RATE_CACHE_TTL_SEC = 6 * 60 * 60

_TELEGRAM_CFG_LOADED = False
_TELEGRAM_TOKEN: Optional[str] = None
_TELEGRAM_CHAT_ID: Optional[int] = None

AITM_CALLER_ID = "AITM"


def _is_aitm_master_enabled() -> bool:
    try:
        return bool(getattr(app_settings, "AITM_MASTER_ENABLED", False))
    except Exception:
        return False


def _resolve_execution_caller(caller: str) -> str:
    c = str(caller or "UNKNOWN").strip()
    if c and c != "UNKNOWN":
        return c
    try:
        frm = inspect.stack()[2]
        fn = os.path.basename(str(frm.filename or ""))
        return fn or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _execution_allowed(caller: str, action_label: str) -> bool:
    caller_id = _resolve_execution_caller(caller)
    if _is_aitm_master_enabled() and caller_id != AITM_CALLER_ID:
        # Keep AITM as sole authority for OPEN-TRADE MANAGEMENT mutations
        # (close/cancel/SL/TP edits), but still allow ENTRY path to open new
        # trades and set leverage from non-AITM callers.
        allow_non_aitm = {
            "place_order",
            "set_leverage",
            "set_leverage_for_account",
        }
        if str(action_label) in allow_non_aitm:
            log(f"[ENTRY EXECUTION] allowed non-AITM caller={caller_id} action={action_label}")
            return True
        log(f"[BLOCKED] Unauthorized execution attempt from {caller_id} ({action_label})")
        return False
    if _is_aitm_master_enabled() and caller_id == AITM_CALLER_ID:
        log(f"[AITM EXECUTION] {action_label} allowed")
    return True


def _get_account_settings(index: int) -> Dict[str, Any]:
    if index < 0 or index >= len(_ACCOUNT_SETTINGS):
        return {}
    settings = _ACCOUNT_SETTINGS[index]
    if not isinstance(settings, dict):
        return {}
    return settings


def _account_accepts_signal_profile(settings: Dict[str, Any], signal_profile: Optional[str]) -> bool:
    """Per-account signal source routing.

    signal_source:
      - small (default)
      - large
      - both
    """
    if signal_profile is None:
        return True
    if not isinstance(settings, dict):
        return str(signal_profile).lower() == "small"
    src_raw = settings.get("signal_source", "small")
    src = str(src_raw).strip().lower() if src_raw is not None else "small"
    prof = str(signal_profile).strip().lower()
    if src in ("both", "all"):
        return True
    if src in ("small", "large"):
        return src == prof
    # Backward compatibility: unknown/empty -> small
    return prof == "small"


def _resolve_time_window_hour(settings: Dict[str, Any]) -> float:
    """Resolve current hour for time-window checks.

    By default, this uses server local time (legacy behavior).
    If account settings provide `time_window_utc_offset_hours`, the current
    hour is computed from UTC + offset, independent of process timezone.
    """

    if not isinstance(settings, dict):
        now = time.localtime()
        return now.tm_hour + now.tm_min / 60.0

    raw_offset = settings.get("time_window_utc_offset_hours")
    if raw_offset in (None, ""):
        now = time.localtime()
        return now.tm_hour + now.tm_min / 60.0

    try:
        offset_h = float(raw_offset)
    except (TypeError, ValueError):
        now = time.localtime()
        return now.tm_hour + now.tm_min / 60.0

    # Keep offsets in a sane bound (e.g. -12 .. +14 timezones).
    if offset_h < -24.0 or offset_h > 24.0:
        now = time.localtime()
        return now.tm_hour + now.tm_min / 60.0

    utc_now = time.gmtime()
    utc_hour = utc_now.tm_hour + utc_now.tm_min / 60.0 + utc_now.tm_sec / 3600.0
    return (utc_hour + offset_h) % 24.0


def _leverage_key(account_index: int, symbol: str) -> Tuple[int, str]:
    try:
        idx = int(account_index)
    except Exception:
        idx = -1
    try:
        sym_u = str(symbol).upper()
    except Exception:
        sym_u = str(symbol)
    return idx, sym_u


def _get_notional_scale_for_leverage(account_index: int, symbol: str) -> float:
    """Return notional scaling factor to compensate for leverage fallback.

    If requested leverage was rejected and effective leverage is lower, scale
    notional by requested/effective (e.g., 10->5 => 2.0).
    """

    key = _leverage_key(account_index, symbol)
    req = _REQUESTED_LEVERAGE_BY_SYMBOL.get(key)
    eff = _EFFECTIVE_LEVERAGE_BY_SYMBOL.get(key)

    try:
        req_v = int(req) if req is not None else 0
    except Exception:
        req_v = 0
    try:
        eff_v = int(eff) if eff is not None else 0
    except Exception:
        eff_v = 0

    if req_v > 0 and eff_v > 0 and eff_v < req_v:
        try:
            scale = float(req_v) / float(eff_v)
        except Exception:
            return 1.0
        if scale <= 0 or not math.isfinite(scale):
            return 1.0
        # Safety cap: if something goes wrong, do not explode sizes.
        if scale > 10.0:
            scale = 10.0
        return scale
    return 1.0


def _candidate_leverages(desired: int) -> List[int]:
    try:
        d = int(desired)
    except Exception:
        d = 1
    if d <= 0:
        d = 1

    cands: List[int] = [d]
    if d > 5:
        cands.append(5)

    # Then try descending levels.
    for lev in range(min(d - 1, 4) if d > 5 else d - 1, 0, -1):
        cands.append(int(lev))
    # Ensure uniqueness while preserving order.
    out: List[int] = []
    seen = set()
    for x in cands:
        if x <= 0:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _get_cached_effective_leverage(account_index: int, symbol: str, desired: int) -> Optional[int]:
    key = _leverage_key(account_index, symbol)
    try:
        desired_v = int(desired)
    except Exception:
        desired_v = 0
    if desired_v <= 0:
        return None

    try:
        last_req = _REQUESTED_LEVERAGE_BY_SYMBOL.get(key)
        last_eff = _EFFECTIVE_LEVERAGE_BY_SYMBOL.get(key)
        last_ts = _LEVERAGE_SET_TS_BY_SYMBOL.get(key)
    except Exception:
        return None

    try:
        if last_req is None or int(last_req) != int(desired_v):
            return None
    except Exception:
        return None

    try:
        if last_eff is None or int(last_eff) <= 0:
            return None
    except Exception:
        return None

    try:
        if last_ts is None:
            return None
        age = float(time.time()) - float(last_ts)
    except Exception:
        return None

    try:
        ttl = float(_LEVERAGE_CACHE_TTL_SEC)
    except Exception:
        ttl = 0.0
    if ttl > 0 and age <= ttl:
        try:
            return int(last_eff)
        except Exception:
            return None
    return None


def _set_leverage_with_fallback(client: Any, account_index: int, symbol: str, desired: int) -> int:
    """Set leverage with automatic fallback on Binance -4028 (invalid leverage)."""

    key = _leverage_key(account_index, symbol)
    try:
        desired_v = int(desired)
    except Exception:
        desired_v = 1
    if desired_v <= 0:
        desired_v = 1

    _REQUESTED_LEVERAGE_BY_SYMBOL[key] = int(desired_v)

    last_invalid = None
    for lev in _candidate_leverages(desired_v):
        try:
            client.set_leverage(symbol, int(lev))
            _EFFECTIVE_LEVERAGE_BY_SYMBOL[key] = int(lev)
            try:
                _LEVERAGE_SET_TS_BY_SYMBOL[key] = time.time()
            except Exception:
                pass
            if int(lev) != int(desired_v):
                try:
                    log(
                        f"[LEVERAGE] {symbol} desired {desired_v}x rejected; using {lev}x instead"
                    )
                except Exception:
                    pass
            return int(lev)
        except Exception as e:
            err_text = str(e)
            # -4028: Leverage is not valid for this symbol.
            if '"code":-4028' in err_text or "Leverage" in err_text and "not valid" in err_text:
                last_invalid = e
                continue
            raise

    if last_invalid is not None:
        raise last_invalid
    raise RuntimeError("Failed to set leverage")


def _yaml_load_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            if _YAML is not None:
                data = _YAML.load(f)
            else:
                data = _pyyaml.safe_load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_account_pause_state_loaded() -> None:
    global _ACCOUNT_PAUSE_STATE_LOADED, _ACCOUNT_PAUSE_STATE
    if _ACCOUNT_PAUSE_STATE_LOADED:
        return
    with _ACCOUNT_PAUSE_STATE_LOCK:
        if _ACCOUNT_PAUSE_STATE_LOADED:
            return
        try:
            if not _ACCOUNT_PAUSE_STATE_PATH.parent.exists():
                _ACCOUNT_PAUSE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if _ACCOUNT_PAUSE_STATE_PATH.exists():
                raw = _ACCOUNT_PAUSE_STATE_PATH.read_text(encoding="utf-8")
                data = json.loads(raw) if raw.strip() else {}
                _ACCOUNT_PAUSE_STATE = data if isinstance(data, dict) else {}
            else:
                _ACCOUNT_PAUSE_STATE = {}
        except Exception:
            _ACCOUNT_PAUSE_STATE = {}
        _ACCOUNT_PAUSE_STATE_LOADED = True


def _save_account_pause_state() -> None:
    try:
        if not _ACCOUNT_PAUSE_STATE_PATH.parent.exists():
            _ACCOUNT_PAUSE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ACCOUNT_PAUSE_STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_ACCOUNT_PAUSE_STATE, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_ACCOUNT_PAUSE_STATE_PATH)
    except Exception:
        return


def _pause_node_for_index(index: int) -> Dict[str, Any]:
    _ensure_account_pause_state_loaded()
    try:
        key = str(int(index))
    except Exception:
        key = str(index)
    node = _ACCOUNT_PAUSE_STATE.get(key)
    if not isinstance(node, dict):
        node = {}
        _ACCOUNT_PAUSE_STATE[key] = node
    return node


def _is_global_drawdown_ok(index: int, client: Any) -> bool:
    """Return True if account is allowed to open new entries.

    This does not block reduce-only exits.
    """

    settings = _get_account_settings(index)
    if not isinstance(settings, dict):
        settings = {}

    try:
        enabled_raw = settings.get("global_dd_pause_enabled")
    except Exception:
        enabled_raw = None
    enabled = (
        bool(_GLOBAL_DD_PAUSE_ENABLED_DEFAULT)
        if enabled_raw is None
        else bool(enabled_raw)
    )
    if not enabled:
        return True

    limit_pct = float(_GLOBAL_DD_PAUSE_PCT_DEFAULT)
    try:
        raw = settings.get("global_dd_pause_pct")
        if raw is not None:
            limit_pct = float(raw)
    except Exception:
        limit_pct = float(_GLOBAL_DD_PAUSE_PCT_DEFAULT)
    if limit_pct <= 0:
        limit_pct = 20.0

    with _ACCOUNT_PAUSE_STATE_LOCK:
        node = _pause_node_for_index(index)
        if bool(node.get("paused")):
            return False

        baseline = node.get("baseline_balance")
        try:
            baseline_val = float(baseline) if baseline is not None else 0.0
        except Exception:
            baseline_val = 0.0

        cur = 0.0
        try:
            cur = float(_get_account_wallet_balance_usdt_for_dd(client) or 0.0)
        except Exception:
            cur = 0.0
        if cur <= 0:
            return True

        if baseline_val <= 0:
            node["baseline_balance"] = float(cur)
            node["paused"] = False
            node["updated_ts"] = time.time()
            _save_account_pause_state()
            return True

        try:
            dd_pct = (float(cur) - float(baseline_val)) / float(baseline_val) * 100.0
        except Exception:
            return True

        if dd_pct <= -float(limit_pct):
            node["paused"] = True
            node["paused_ts"] = time.time()
            node["dd_pct"] = float(dd_pct)
            node["baseline_balance"] = float(baseline_val)
            node["current_balance"] = float(cur)
            _save_account_pause_state()

            try:
                acc_name = (
                    _ACCOUNT_NAMES[index]
                    if 0 <= int(index) < len(_ACCOUNT_NAMES)
                    else f"Account {index}"
                )
            except Exception:
                acc_name = f"Account {index}"

            msg = (
                f"[DD][PAUSE] {acc_name} paused: balance drawdown {dd_pct:.2f}% "
                f"(start {baseline_val:.2f} -> now {cur:.2f} USDT). "
                "Manual admin resume required."
            )
            try:
                log(msg)
            except Exception:
                pass
            try:
                _broadcast_room_signal(msg)
            except Exception:
                pass

            return False

        node["updated_ts"] = time.time()
        _save_account_pause_state()
        return True


def get_paused_accounts_snapshot() -> List[Dict[str, Any]]:
    """Return a list of paused accounts with drawdown metadata."""

    _ensure_account_pause_state_loaded()
    out: List[Dict[str, Any]] = []
    with _ACCOUNT_PAUSE_STATE_LOCK:
        for k, v in (_ACCOUNT_PAUSE_STATE or {}).items():
            if not isinstance(v, dict):
                continue
            if not bool(v.get("paused")):
                continue
            try:
                idx = int(k)
            except Exception:
                idx = k
            try:
                name = (
                    _ACCOUNT_NAMES[int(idx)]
                    if isinstance(idx, int) and 0 <= int(idx) < len(_ACCOUNT_NAMES)
                    else f"Account {idx}"
                )
            except Exception:
                name = f"Account {idx}"
            out.append(
                {
                    "index": idx,
                    "name": name,
                    "paused": True,
                    "paused_ts": v.get("paused_ts"),
                    "dd_pct": v.get("dd_pct"),
                    "baseline_balance": v.get("baseline_balance"),
                    "current_balance": v.get("current_balance"),
                }
            )
    return out


def resume_account_from_drawdown_pause(index: int) -> bool:
    """Resume a paused account and reset baseline to current balance."""

    clients = _get_clients()
    if not clients:
        return False
    try:
        idx = int(index)
    except Exception:
        return False
    if idx < 0 or idx >= len(clients):
        return False
    client = clients[idx]

    cur = 0.0
    try:
        cur = float(_get_account_wallet_balance_usdt_for_dd(client) or 0.0)
    except Exception:
        cur = 0.0
    if cur <= 0:
        return False

    with _ACCOUNT_PAUSE_STATE_LOCK:
        node = _pause_node_for_index(idx)
        node["paused"] = False
        node["baseline_balance"] = float(cur)
        node["current_balance"] = float(cur)
        node["dd_pct"] = 0.0
        node["resumed_ts"] = time.time()
        _save_account_pause_state()

    try:
        acc_name = (
            _ACCOUNT_NAMES[idx]
            if 0 <= int(idx) < len(_ACCOUNT_NAMES)
            else f"Account {idx}"
        )
    except Exception:
        acc_name = f"Account {idx}"
    msg = f"[DD][RESUME] {acc_name} resumed by admin. New baseline={cur:.2f} USDT"
    try:
        log(msg)
    except Exception:
        pass
    try:
        _broadcast_room_signal(msg)
    except Exception:
        pass
    return True


def resume_all_accounts_from_drawdown_pause() -> Dict[str, Any]:
    clients = _get_clients()
    if not clients:
        return {"ok": False, "resumed": 0, "failed": 0}

    _ensure_account_pause_state_loaded()
    resumed = 0
    failed = 0

    for idx in range(len(clients)):
        try:
            with _ACCOUNT_PAUSE_STATE_LOCK:
                node = (_ACCOUNT_PAUSE_STATE or {}).get(str(idx)) or {}
                is_paused = bool(node.get("paused"))
        except Exception:
            is_paused = False

        if not is_paused:
            continue

        try:
            ok = resume_account_from_drawdown_pause(idx)
        except Exception:
            ok = False

        if ok:
            resumed += 1
        else:
            failed += 1

    return {"ok": True, "resumed": resumed, "failed": failed}


class BinanceFuturesClient:
    def __init__(self, api_key, api_secret, proxies=None):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self._api_secret_str = api_secret  # kept for C++ sign_params (needs raw str)
        # Optional fixed proxies dict specific to this account. When set,
        # ALL requests (public + signed) for this client will go through
        # this proxy, regardless of global proxy rotation settings.
        self._fixed_proxies = proxies
        # C++ persistent HTTP client (one per account, reuses TCP+TLS).
        # Protected by _cpp_lock because libcurl handles are NOT thread-safe
        # and multiple threads (exit manager + order pool) share this client.
        self._cpp_http = None
        self._cpp_lock = threading.Lock()
        if _CPP_AVAILABLE:
            try:
                cfg = _cpp.HttpClientConfig()
                cfg.api_key = api_key
                cfg.api_secret = api_secret
                cfg.base_url = BASE_URL
                cfg.timeout_sec = 10
                cfg.recv_window = 50000
                if isinstance(proxies, dict):
                    cfg.proxy = str(proxies.get("https") or proxies.get("http") or "")
                self._cpp_http = _cpp.HttpClient(cfg)
            except Exception:
                self._cpp_http = None

    def _sign(self, params: dict):
        if _CPP_AVAILABLE:
            try:
                param_list = [(str(k), str(v)) for k, v in params.items()]
                return _cpp.sign_params(self._api_secret_str, param_list)
            except Exception:
                pass
        # Original Python fallback — behavior unchanged.
        query = urlencode(params)
        signature = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={signature}"

    def _headers(self):
        return {"X-MBX-APIKEY": self.api_key}

    def _request(self, method, path, params=None, signed=False):
        if params is None:
            params = {}

        # C++ fast path: use persistent TLS connection if available.
        # Lock protects the libcurl handle (not thread-safe).
        # Use acquire(timeout=30) so a background leverage thread holding the
        # lock for a slow proxy POST never blocks order placement indefinitely.
        # If lock cannot be acquired in 30s, fall through to Python requests.
        if self._cpp_http is not None:
            _cpp_lock_acquired = self._cpp_lock.acquire(timeout=30)
            if _cpp_lock_acquired:
                try:
                    str_params = {str(k): str(v) for k, v in params.items()}
                    body = self._cpp_http.request(method, path, str_params, signed)
                    return json.loads(body)
                except Exception as cpp_err:
                    # Preserve 451/403 blacklist behavior from Python path.
                    cpp_err_str = str(cpp_err)
                    if ("451" in cpp_err_str or "403" in cpp_err_str):
                        try:
                            if path.startswith("/fapi/v1/order") and "symbol" in params:
                                _SYMBOL_451_BLACKLIST.add(str(params["symbol"]).upper())
                        except Exception:
                            pass
                    # Re-raise ALL errors — never fall through to Python.
                    raise Exception(cpp_err_str)
                finally:
                    self._cpp_lock.release()
            else:
                # Lock held by background thread (e.g. slow proxy leverage call).
                # Fall through to Python requests path — no C++ call was made,
                # so no duplicate-request risk. Python timeout=10s is reliable.
                log(f"[CPP_LOCK] lock contention on {path} — falling back to Python requests")

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            # Use a generous default recvWindow to tolerate latency and proxy
            # routing so that we avoid frequent -1021 timestamp errors.
            if "recvWindow" not in params:
                params["recvWindow"] = 50000
            query = self._sign(params)
        else:
            query = urlencode(params)

        url = f"{BASE_URL}{path}?{query}"

        # If this client has a fixed proxy configured, always use it for
        # all requests without rotation.
        if self._fixed_proxies is not None:
            attempts = 1
            use_trading_proxies = False
        else:
            # Proxy policy is controlled by execution.yaml / proxies.*
            # settings for accounts without a fixed proxy.
            use_trading_proxies = False
            if signed:
                # Only rotate proxies for signed requests when explicitly enabled.
                use_trading_proxies = _should_use_proxy_for_trading()

            attempts = 3
        last_err = None

        for attempt in range(attempts):
            if self._fixed_proxies is not None:
                proxies = self._fixed_proxies
            else:
                proxies = _get_proxies_for_request(signed)
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=self._headers(),
                    timeout=10,
                    proxies=proxies,
                )
            except requests.RequestException as e:
                last_err = e
                # If account has a fixed proxy and that proxy fails at
                # transport level, do one direct fallback attempt to avoid
                # total request loss due a single dead proxy endpoint.
                if self._fixed_proxies is not None:
                    try:
                        resp = requests.request(
                            method,
                            url,
                            headers=self._headers(),
                            timeout=10,
                            proxies=None,
                        )
                        if resp.status_code == 200:
                            return resp.json()
                        raise Exception(f"Binance error {resp.status_code}: {resp.text}")
                    except Exception:
                        pass
                # On proxy/connection errors, try a different proxy if allowed.
                if attempt < attempts - 1:
                    continue
                raise Exception(f"Binance request error: {e}")

            if resp.status_code == 200:
                return resp.json()

            # For trading with proxies, rotate on region/proxy-related codes.
            if resp.status_code in (451, 403):
                # If this is an order endpoint with a symbol param, remember
                # the symbol so that higher-level logic can avoid trading it.
                try:
                    if path.startswith("/fapi/v1/order") and "symbol" in params:
                        _SYMBOL_451_BLACKLIST.add(str(params["symbol"]).upper())
                except Exception:
                    pass

                if attempt < attempts - 1 and self._fixed_proxies is None:
                    last_err = Exception(f"Binance error {resp.status_code}: {resp.text}")
                    continue

            # Any other non-200: surface immediately.
            raise Exception(f"Binance error {resp.status_code}: {resp.text}")

        if last_err is not None:
            raise last_err

        raise Exception("Binance request failed after retries with trading proxies")

    # ---------- PUBLIC ----------

    def exchange_info(self):
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def mark_price(self, symbol: str):
        return self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})

    def commission_rate(self, symbol: str):
        return self._request("GET", "/fapi/v1/commissionRate", {"symbol": str(symbol).upper()}, signed=True)

    # ---------- ACCOUNT ----------

    def balance(self):
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def account(self):
        return self._request("GET", "/fapi/v2/account", signed=True)

    def position_risk(self):
        return self._request("GET", "/fapi/v2/positionRisk", signed=True)

    def open_orders(self, symbol: Optional[str] = None):
        params = {}
        if symbol:
            params["symbol"] = str(symbol).upper()
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ):
        params: Dict[str, Any] = {"symbol": str(symbol).upper()}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if orig_client_order_id is not None:
            params["origClientOrderId"] = str(orig_client_order_id)
        return self._request("DELETE", "/fapi/v1/order", params, signed=True)

    def income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = str(symbol).upper()
        if income_type:
            params["incomeType"] = str(income_type)
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if limit is not None:
            params["limit"] = int(limit)
        return self._request("GET", "/fapi/v1/income", params, signed=True)

    # ---------- ORDERS ----------

    def place_order(
        self,
        symbol,
        side,
        quantity=None,
        order_type="MARKET",
        reduce_only=False,
        price=None,
        stop_price=None,
        close_position=None,
        time_in_force=None,
        position_side: Optional[str] = None,
        working_type: Optional[str] = None,
    ):
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }

        if quantity is not None:
            params["quantity"] = quantity
        if reduce_only:
            params["reduceOnly"] = "true"
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if close_position is not None:
            params["closePosition"] = str(close_position).lower()
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if working_type is not None:
            params["workingType"] = working_type

        if position_side is not None:
            try:
                ps = str(position_side).upper().strip()
            except Exception:
                ps = ""
            if ps:
                params["positionSide"] = ps

        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_algo_order(
        self,
        symbol,
        side,
        algo_type="CONDITIONAL",
        order_type="STOP_MARKET",
        quantity=None,
        reduce_only=False,
        trigger_price=None,
        close_position=None,
        working_type="CONTRACT_PRICE",
        time_in_force=None,
        position_side: Optional[str] = None,
    ):
        params = {
            "symbol": symbol,
            "side": side,
            "algoType": algo_type,
            "type": order_type,
        }

        use_close_position = bool(close_position)
        if use_close_position:
            params["closePosition"] = "true"
        else:
            if quantity is not None:
                # Normalize: str(42.0)="42.0" fails on algo endpoint (-1111),
                # but str(42)="42" is accepted. Use int for whole-number quantities.
                _q = float(quantity)
                params["quantity"] = int(_q) if _q == int(_q) else _q
            if reduce_only:
                params["reduceOnly"] = "true"

        if trigger_price is not None:
            params["triggerPrice"] = trigger_price
        if working_type is not None:
            params["workingType"] = working_type
        if time_in_force is not None:
            params["timeInForce"] = time_in_force

        if position_side is not None:
            try:
                ps = str(position_side).upper().strip()
            except Exception:
                ps = ""
            if ps:
                params["positionSide"] = ps

        return self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

    def cancel_all_open_orders(self, symbol: str):
        """Cancel ALL open regular orders for a symbol in one API call (DELETE /fapi/v1/allOpenOrders)."""
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": str(symbol).upper()}, signed=True)

    def cancel_algo_order(self, algo_id: int):
        """Cancel an active algo/conditional order by algoId."""
        params = {"algoId": int(algo_id)}
        return self._request("DELETE", "/fapi/v1/algoOrder", params, signed=True)

    def get_algo_open_orders(self, symbol: Optional[str] = None):
        """Query all open algo/conditional orders."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = str(symbol).upper()
        return self._request("GET", "/fapi/v1/openAlgoOrders", params, signed=True)

    # ---------- LEVERAGE ----------

    def set_leverage(self, symbol: str, leverage: int):
        params = {"symbol": symbol, "leverage": int(leverage)}
        return self._request("POST", "/fapi/v1/leverage", params, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str):
        params = {"symbol": symbol, "marginType": margin_type}
        return self._request("POST", "/fapi/v1/marginType", params, signed=True)


def _ensure_room_telegram_config() -> None:
    global _TELEGRAM_CFG_LOADED, _TELEGRAM_TOKEN, _TELEGRAM_CHAT_ID
    if _TELEGRAM_CFG_LOADED:
        return

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "trading.yaml"
    try:
        cfg = _yaml_load_file(cfg_path)
    except Exception:
        cfg = {}

    token = cfg.get("telegram_token") or os.environ.get("TELEGRAM_TOKEN")
    chat_id = cfg.get("telegram_chat_id") or os.environ.get("TELEGRAM_CHAT_ID")
    try:
        chat_id_int = int(chat_id) if chat_id is not None else None
    except (TypeError, ValueError):
        chat_id_int = None

    _TELEGRAM_TOKEN = str(token) if token else None
    _TELEGRAM_CHAT_ID = chat_id_int
    _TELEGRAM_CFG_LOADED = True


def _broadcast_room_signal(text: str) -> None:
    try:
        msg = str(text)
    except Exception:
        return

    _ensure_room_telegram_config()
    if not _TELEGRAM_TOKEN or _TELEGRAM_CHAT_ID is None:
        return

    def _async_send() -> None:
        try:
            subs = []
            try:
                subs = get_subscribers(_TELEGRAM_CHAT_ID, token=_TELEGRAM_TOKEN)
            except Exception:
                subs = []
            targets = list(subs) if subs else [_TELEGRAM_CHAT_ID]
            for cid in targets:
                try:
                    send_telegram(msg, _TELEGRAM_TOKEN, int(cid))
                except Exception:
                    continue
        except Exception:
            pass

    try:
        threading.Thread(target=_async_send, daemon=True).start()
    except Exception:
        pass


def _room_state_path(account_index: int) -> Path:
    return _ROOM_SIZING_DIR / f"room_state_{int(account_index)}.json"


def _room_default_settings(settings: Dict[str, Any]) -> Tuple[int, int, float, float]:
    try:
        rooms = int(settings.get("room_sizing_rooms", settings.get("rooms_count", 5)))
    except Exception:
        rooms = 5
    try:
        slots = int(settings.get("room_sizing_slots", settings.get("slots_per_room", 10)))
    except Exception:
        slots = 10
    try:
        base = float(settings.get("room_sizing_base_usdt", settings.get("base_usdt", 5.5)))
    except Exception:
        base = 5.5
    try:
        mult = float(settings.get("room_sizing_multiplier", settings.get("multiplier", 2.0)))
    except Exception:
        mult = 2.0

    if rooms <= 0:
        rooms = 5
    if slots <= 0:
        slots = 10
    if base <= 0:
        base = 5.5
    if mult <= 0:
        mult = 2.0
    if base < 5.5:
        base = 5.5
    return rooms, slots, base, mult


def _load_room_state(account_index: int, settings: Dict[str, Any]) -> Dict[str, Any]:
    path = _room_state_path(account_index)
    rooms, slots, base, mult = _room_default_settings(settings)

    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                # Migrate persisted room-state to CURRENT config defaults so
                # target notional never uses stale historical base_usdt.
                try:
                    data["account_index"] = int(account_index)
                    data["rooms_count"] = int(rooms)
                    data["slots_per_room"] = int(slots)
                    data["base_usdt"] = float(base)
                    data["multiplier"] = float(mult)

                    rooms_list = data.get("rooms")
                    if not isinstance(rooms_list, list):
                        rooms_list = []

                    # Resize room count to current config.
                    if len(rooms_list) > int(rooms):
                        rooms_list = rooms_list[: int(rooms)]
                    while len(rooms_list) < int(rooms):
                        rooms_list.append(
                            {
                                "status": "IDLE",
                                "current_notional_usdt": float(base),
                                "batch_id": 0,
                                "batch_start_ts": None,
                                "batch_end_ts": None,
                                "slots": [None for _ in range(int(slots))],
                            }
                        )

                    for i, r in enumerate(rooms_list):
                        if not isinstance(r, dict):
                            rooms_list[i] = {
                                "status": "IDLE",
                                "current_notional_usdt": float(base),
                                "batch_id": 0,
                                "batch_start_ts": None,
                                "batch_end_ts": None,
                                "slots": [None for _ in range(int(slots))],
                            }
                            continue

                        # Keep existing notional only if >= configured base.
                        try:
                            cur_not = float(r.get("current_notional_usdt"))
                        except Exception:
                            cur_not = float(base)
                        if cur_not < float(base):
                            cur_not = float(base)
                        r["current_notional_usdt"] = float(cur_not)

                        # Resize slots list to current config.
                        room_slots = r.get("slots")
                        if not isinstance(room_slots, list):
                            room_slots = []
                        if len(room_slots) > int(slots):
                            room_slots = room_slots[: int(slots)]
                        while len(room_slots) < int(slots):
                            room_slots.append(None)
                        r["slots"] = room_slots

                    data["rooms"] = rooms_list
                except Exception:
                    # Fallback to original data if migration has an issue.
                    pass
                return data
        except Exception:
            pass

    rooms_list = []
    for _ in range(rooms):
        rooms_list.append(
            {
                "status": "IDLE",
                "current_notional_usdt": float(base),
                "batch_id": 0,
                "batch_start_ts": None,
                "batch_end_ts": None,
                "slots": [None for _ in range(slots)],
            }
        )

    return {
        "version": 1,
        "account_index": int(account_index),
        "rooms_count": int(rooms),
        "slots_per_room": int(slots),
        "base_usdt": float(base),
        "multiplier": float(mult),
        "rooms": rooms_list,
        "history": [],
    }


def _save_room_state(account_index: int, state: Dict[str, Any]) -> None:
    path = _room_state_path(account_index)
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        try:
            path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def room_sizing_reset_account(account_index: int) -> None:
    try:
        settings = _get_account_settings(int(account_index))
    except Exception:
        settings = {}
    if not isinstance(settings, dict):
        settings = {}

    rooms, slots, base, mult = _room_default_settings(settings)
    rooms_list = []
    for _ in range(int(rooms)):
        rooms_list.append(
            {
                "status": "IDLE",
                "current_notional_usdt": float(base),
                "batch_id": 0,
                "batch_start_ts": None,
                "batch_end_ts": None,
                "slots": [None for _ in range(int(slots))],
            }
        )

    state = {
        "version": 1,
        "account_index": int(account_index),
        "rooms_count": int(rooms),
        "slots_per_room": int(slots),
        "base_usdt": float(base),
        "multiplier": float(mult),
        "rooms": rooms_list,
        "history": [],
    }

    with _ROOM_SIZING_LOCK:
        try:
            _ROOM_SIZING_LAST_SYNC.pop(int(account_index), None)
        except Exception:
            pass
        try:
            keys_to_del = []
            for k in _ROOM_LAST_ASSIGNMENT.keys():
                try:
                    if isinstance(k, tuple) and k and int(k[0]) == int(account_index):
                        keys_to_del.append(k)
                except Exception:
                    continue
            for k in keys_to_del:
                try:
                    _ROOM_LAST_ASSIGNMENT.pop(k, None)
                except Exception:
                    continue
        except Exception:
            pass
        _save_room_state(int(account_index), state)


def _room_sizing_enabled_for_settings(settings: Dict[str, Any]) -> bool:
    try:
        return bool(settings.get("room_sizing_enabled") or settings.get("room_batch_enabled"))
    except Exception:
        return False


def _symbol_has_open_position(client: BinanceFuturesClient, symbol: str) -> bool:
    try:
        positions = client.position_risk()
    except Exception:
        return False
    if not isinstance(positions, list):
        return False
    sym_u = str(symbol).upper()
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        try:
            if str(pos.get("symbol") or "").upper() != sym_u:
                continue
            amt = float(pos.get("positionAmt"))
        except Exception:
            continue
        if amt != 0.0:
            return True
    return False


def room_sizing_reserve_slot(
    account_index: int,
    symbol: str,
    side: str,
    settings: Dict[str, Any],
) -> Tuple[bool, Optional[float], Optional[int], Optional[int], str]:
    if not _room_sizing_enabled_for_settings(settings):
        return False, None, None, None, "disabled"

    with _ROOM_SIZING_LOCK:
        state = _load_room_state(account_index, settings)
        rooms_list = state.get("rooms") or []
        if not isinstance(rooms_list, list) or not rooms_list:
            return True, None, None, None, "no_rooms"

        filling_idx = None
        idle_idx = None
        for i, r in enumerate(rooms_list):
            if not isinstance(r, dict):
                continue
            st = r.get("status")
            if filling_idx is None and st == "FILLING":
                filling_idx = i
            if idle_idx is None and st == "IDLE":
                idle_idx = i
        chosen = filling_idx if filling_idx is not None else idle_idx
        if chosen is None:
            _save_room_state(account_index, state)
            return True, None, None, None, "all_rooms_busy"

        room = rooms_list[int(chosen)]
        if not isinstance(room, dict):
            return True, None, None, None, "bad_room"

        if room.get("status") == "IDLE":
            room["status"] = "FILLING"
            room["batch_id"] = int(room.get("batch_id") or 0) + 1
            room["batch_start_ts"] = time.time()
            room["batch_end_ts"] = None
            slots_per_room = int(state.get("slots_per_room") or 10)
            room["slots"] = [None for _ in range(slots_per_room)]

        slots = room.get("slots")
        if not isinstance(slots, list):
            return True, None, None, None, "bad_slots"

        slot_idx = None
        for i, s in enumerate(slots):
            if s is None:
                slot_idx = i
                break
        if slot_idx is None:
            room["status"] = "LOCKED"
            _save_room_state(account_index, state)
            return True, None, int(chosen), None, "room_locked"

        notional = float(room.get("current_notional_usdt") or float(state.get("base_usdt") or 5.5))
        if notional < float(state.get("base_usdt") or 5.5):
            notional = float(state.get("base_usdt") or 5.5)
        if notional < 5.5:
            notional = 5.5

        slots[int(slot_idx)] = {
            "status": "RESERVED",
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "open_ts": time.time(),
            "close_ts": None,
            "pnl_usdt": None,
        }

        full = True
        for s in slots:
            if s is None:
                full = False
                break
        room["status"] = "LOCKED" if full else "FILLING"
        try:
            key = (int(account_index), str(symbol).upper(), str(side).upper())
            _ROOM_LAST_ASSIGNMENT[key] = {
                "ts": time.time(),
                "room_index": int(chosen),
                "slot_index": int(slot_idx),
                "batch_id": int(room.get("batch_id") or 0),
                "notional_usdt": float(notional),
                "status": str(room.get("status") or ""),
            }
        except Exception:
            pass

        _save_room_state(account_index, state)
        return True, float(notional), int(chosen), int(slot_idx), "reserved"


def room_sizing_pop_assignment(account_index: int, symbol: str, side: str) -> Optional[Dict[str, Any]]:
    try:
        key = (int(account_index), str(symbol).upper(), str(side).upper())
    except Exception:
        return None
    try:
        return _ROOM_LAST_ASSIGNMENT.pop(key, None)
    except Exception:
        return None


def room_sizing_release_slot(
    account_index: int,
    settings: Dict[str, Any],
    room_index: int,
    slot_index: int,
) -> None:
    with _ROOM_SIZING_LOCK:
        state = _load_room_state(account_index, settings)
        rooms_list = state.get("rooms")
        if not isinstance(rooms_list, list):
            return
        if room_index < 0 or room_index >= len(rooms_list):
            return
        room = rooms_list[room_index]
        if not isinstance(room, dict):
            return
        slots = room.get("slots")
        if not isinstance(slots, list):
            return
        if slot_index < 0 or slot_index >= len(slots):
            return
        slots[slot_index] = None

        any_open = False
        any_slot = False
        for s in slots:
            if s is not None:
                any_slot = True
                if isinstance(s, dict) and s.get("status") in ("RESERVED", "OPEN"):
                    any_open = True
        if not any_slot:
            room["status"] = "IDLE"
        else:
            room["status"] = "FILLING" if any_open else "FILLING"

        _save_room_state(account_index, state)


def room_sizing_commit_open(
    account_index: int,
    settings: Dict[str, Any],
    room_index: int,
    slot_index: int,
    order_response: Any,
) -> None:
    with _ROOM_SIZING_LOCK:
        state = _load_room_state(account_index, settings)
        rooms_list = state.get("rooms")
        if not isinstance(rooms_list, list):
            return
        if room_index < 0 or room_index >= len(rooms_list):
            return
        room = rooms_list[room_index]
        if not isinstance(room, dict):
            return
        slots = room.get("slots")
        if not isinstance(slots, list):
            return
        if slot_index < 0 or slot_index >= len(slots):
            return
        s = slots[slot_index]
        if not isinstance(s, dict):
            return
        s["status"] = "OPEN"
        if isinstance(order_response, dict):
            if "orderId" in order_response:
                try:
                    s["order_id"] = int(order_response.get("orderId"))
                except Exception:
                    pass
            if "clientOrderId" in order_response:
                try:
                    s["client_order_id"] = str(order_response.get("clientOrderId"))
                except Exception:
                    pass
        _save_room_state(account_index, state)


def _income_sum_for_window(
    client: BinanceFuturesClient,
    symbol: str,
    start_ts: float,
    end_ts: float,
) -> float:
    start_ms = int(max(0.0, float(start_ts) - 5.0) * 1000.0)
    end_ms = int((float(end_ts) + 5.0) * 1000.0)
    total = 0.0
    try:
        items = client.income_history(symbol=symbol, start_time_ms=start_ms, end_time_ms=end_ms, limit=1000)
    except Exception:
        items = []
    if not isinstance(items, list):
        return 0.0
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            inc = float(it.get("income"))
        except Exception:
            continue
        total += inc
    return total


def room_sizing_sync_account(account_index: int) -> None:
    clients = _get_clients()
    if not clients or account_index < 0 or account_index >= len(clients):
        return
    client = clients[account_index]

    try:
        settings = _get_account_settings(account_index)
    except Exception:
        settings = {}
    if not isinstance(settings, dict) or not _room_sizing_enabled_for_settings(settings):
        return

    cumulative_pnl_plugin = False
    try:
        raw_flag = (
            settings.get("room_sizing_cumulative_pnl_plugin")
            if isinstance(settings, dict)
            else None
        )
        if raw_flag is None:
            raw_flag = settings.get("room_sizing_cumulative_pnl")
        if raw_flag is None:
            raw_flag = settings.get("room_sizing_cumulative_loss")
        cumulative_pnl_plugin = bool(raw_flag)
    except Exception:
        cumulative_pnl_plugin = False

    recovery_formula_plugin = False
    try:
        raw_flag = (
            settings.get("room_sizing_recovery_formula_plugin")
            if isinstance(settings, dict)
            else None
        )
        if raw_flag is None:
            raw_flag = settings.get("room_sizing_recovery_plugin")
        if raw_flag is None:
            raw_flag = settings.get("room_sizing_required_capital_plugin")
        recovery_formula_plugin = bool(raw_flag)
    except Exception:
        recovery_formula_plugin = False

    now = time.time()
    last = _ROOM_SIZING_LAST_SYNC.get(account_index) or 0.0
    if now - float(last) < 15.0:
        return
    _ROOM_SIZING_LAST_SYNC[account_index] = now

    try:
        positions = client.position_risk()
    except Exception:
        return
    if not isinstance(positions, list):
        return

    open_syms: set[str] = set()
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        sym = pos.get("symbol")
        if not sym:
            continue
        try:
            amt = float(pos.get("positionAmt"))
        except Exception:
            continue
        if amt != 0.0:
            open_syms.add(str(sym).upper())

    with _ROOM_SIZING_LOCK:
        state = _load_room_state(account_index, settings)
        rooms_list = state.get("rooms")
        if not isinstance(rooms_list, list):
            return

        finalized_msgs: list[str] = []

        for r_idx, room in enumerate(rooms_list):
            if not isinstance(room, dict):
                continue
            slots = room.get("slots")
            if not isinstance(slots, list):
                continue

            changed = False

            for s_idx, s in enumerate(slots):
                if not isinstance(s, dict):
                    continue
                st = s.get("status")
                if st not in ("OPEN", "RESERVED"):
                    continue
                sym = str(s.get("symbol") or "").upper()
                if not sym:
                    continue
                if sym in open_syms:
                    continue
                s["status"] = "CLOSED"
                s["close_ts"] = time.time()
                changed = True

            if not changed:
                continue

            if room.get("status") == "LOCKED":
                all_closed = True
                for s in slots:
                    if not isinstance(s, dict):
                        all_closed = False
                        break
                    if s.get("status") != "CLOSED":
                        all_closed = False
                        break

                if all_closed:
                    room_pnl = 0.0
                    for s in slots:
                        sym = str(s.get("symbol") or "").upper()
                        try:
                            open_ts = float(s.get("open_ts") or 0.0)
                            close_ts = float(s.get("close_ts") or 0.0)
                        except Exception:
                            open_ts = 0.0
                            close_ts = 0.0
                        if not sym or open_ts <= 0 or close_ts <= 0:
                            continue
                        pnl = _income_sum_for_window(client, sym, open_ts, close_ts)
                        s["pnl_usdt"] = pnl
                        room_pnl += pnl

                    hist = state.get("history")
                    if not isinstance(hist, list):
                        hist = []

                    pnl_basis = float(room_pnl)
                    cum_pnl_room = None
                    if cumulative_pnl_plugin:
                        try:
                            prev = 0.0
                            for it in hist:
                                if not isinstance(it, dict):
                                    continue
                                try:
                                    if int(it.get("room")) != int(r_idx):
                                        continue
                                except Exception:
                                    continue
                                try:
                                    prev += float(it.get("pnl_usdt") or 0.0)
                                except Exception:
                                    continue
                            cum_pnl_room = float(prev) + float(room_pnl)
                            pnl_basis = float(cum_pnl_room)
                        except Exception:
                            pnl_basis = float(room_pnl)
                            cum_pnl_room = None

                    base_usdt = float(state.get("base_usdt") or 5.5)
                    mult = float(state.get("multiplier") or 2.0)
                    slots_n = int(state.get("slots_per_room") or 10)
                    next_notional = base_usdt

                    used_recovery_formula = False
                    if pnl_basis < 0 and recovery_formula_plugin:
                        try:
                            wins = 0
                            losses = 0
                            for s in slots:
                                if not isinstance(s, dict):
                                    continue
                                try:
                                    pv = float(s.get("pnl_usdt") or 0.0)
                                except Exception:
                                    pv = 0.0
                                if pv > 0:
                                    wins += 1
                                elif pv < 0:
                                    losses += 1

                            sl_pct = None
                            tp_pcts = None
                            if isinstance(settings, dict):
                                sl_pct = settings.get("sl_pct")
                                tp_pcts = settings.get("tp_pcts")

                            if sl_pct is None or tp_pcts is None:
                                try:
                                    cfg_path = Path(__file__).resolve().parents[1] / "config" / "trading.yaml"
                                    cfg = _yaml_load_file(cfg_path)
                                except Exception:
                                    cfg = {}
                                sigs = cfg.get("signals") if isinstance(cfg, dict) else None
                                if isinstance(sigs, dict):
                                    if sl_pct is None:
                                        sl_pct = sigs.get("sl_pct")
                                    if tp_pcts is None:
                                        tp_pcts = sigs.get("tp_pcts")

                            try:
                                sl_val = float(sl_pct) if sl_pct is not None else 2.0
                            except Exception:
                                sl_val = 2.0

                            tp_list: list[float] = []
                            if isinstance(tp_pcts, (list, tuple)):
                                for v in tp_pcts:
                                    try:
                                        fv = float(v)
                                    except Exception:
                                        continue
                                    if fv > 0:
                                        tp_list.append(fv)
                            if not tp_list:
                                tp_list = [2.0]

                            tp_index = 0
                            if isinstance(settings, dict):
                                try:
                                    tp_index = int(settings.get("room_sizing_recovery_tp_index") or 0)
                                except Exception:
                                    tp_index = 0
                            if tp_index < 0:
                                tp_index = 0
                            if tp_index >= len(tp_list):
                                tp_index = 0
                            tp_val = float(tp_list[tp_index])

                            lev_val = 1.0
                            if isinstance(settings, dict):
                                try:
                                    lev_val = float(settings.get("room_sizing_recovery_leverage") or 0.0) or 0.0
                                except Exception:
                                    lev_val = 0.0
                            if lev_val <= 0.0:
                                try:
                                    idx0 = int(account_index)
                                except Exception:
                                    idx0 = -1
                                if 0 <= idx0 < len(_ACCOUNT_LEVERAGE_OVERRIDES):
                                    try:
                                        ov = _ACCOUNT_LEVERAGE_OVERRIDES[idx0]
                                        if ov is not None:
                                            lev_val = float(ov)
                                    except Exception:
                                        lev_val = 1.0
                            if lev_val <= 0.0:
                                lev_val = 1.0

                            denom_n = float(slots_n) if slots_n > 0 else 10.0
                            win_rate = float(wins) / denom_n
                            loss_rate = float(losses) / denom_n

                            edge_rate = (win_rate * (tp_val / 100.0) - loss_rate * (sl_val / 100.0)) * float(lev_val)
                            if edge_rate > 0:
                                total_needed = abs(float(pnl_basis)) / edge_rate
                                add_per_trader = (total_needed / denom_n) * float(mult)
                                next_notional = float(base_usdt) + float(add_per_trader)
                                used_recovery_formula = True
                        except Exception:
                            used_recovery_formula = False

                    if pnl_basis < 0 and not used_recovery_formula:
                        try:
                            loss_per = abs(pnl_basis) / float(slots_n)
                        except Exception:
                            loss_per = 0.0
                        next_notional = base_usdt + loss_per * mult
                    if next_notional < base_usdt:
                        next_notional = base_usdt
                    if next_notional < 5.5:
                        next_notional = 5.5

                    room["current_notional_usdt"] = float(next_notional)
                    room["status"] = "IDLE"
                    room["batch_end_ts"] = time.time()
                    room["batch_start_ts"] = None
                    room["slots"] = [None for _ in range(slots_n)]

                    hist.append(
                        {
                            "ts": time.time(),
                            "room": int(r_idx),
                            "batch_id": int(room.get("batch_id") or 0),
                            "pnl_usdt": float(room_pnl),
                            "next_notional_usdt": float(next_notional),
                        }
                    )
                    state["history"] = hist[-200:]

                    if cumulative_pnl_plugin:
                        try:
                            acc_name_dbg = (
                                _ACCOUNT_NAMES[account_index]
                                if 0 <= int(account_index) < len(_ACCOUNT_NAMES)
                                else f"Account {account_index}"
                            )
                        except Exception:
                            acc_name_dbg = f"Account {account_index}"
                        try:
                            cum_txt = f"{float(cum_pnl_room):.4f}" if cum_pnl_room is not None else "n/a"
                            log(
                                f"[ROOM] account={acc_name_dbg} room={int(r_idx)+1} pnl_mode=cumulative "
                                f"last_batch_pnl={float(room_pnl):.4f} cum_pnl={cum_txt} basis={float(pnl_basis):.4f}"
                            )
                        except Exception:
                            pass

                    if used_recovery_formula:
                        try:
                            acc_name_dbg = (
                                _ACCOUNT_NAMES[account_index]
                                if 0 <= int(account_index) < len(_ACCOUNT_NAMES)
                                else f"Account {account_index}"
                            )
                        except Exception:
                            acc_name_dbg = f"Account {account_index}"
                        try:
                            log(
                                f"[ROOM] account={acc_name_dbg} room={int(r_idx)+1} sizing_mode=recovery "
                                f"basis_pnl={float(pnl_basis):.4f} next_notional={float(next_notional):.4f}"
                            )
                        except Exception:
                            pass

                    try:
                        acc_name = (
                            _ACCOUNT_NAMES[account_index]
                            if 0 <= int(account_index) < len(_ACCOUNT_NAMES)
                            else f"Account {account_index}"
                        )
                    except Exception:
                        acc_name = f"Account {account_index}"
                    finalized_msgs.append(
                        "\n".join(
                            [
                                f"[ROOM] {acc_name} — Room {int(r_idx)+1} finalized",
                                f"PnL: {room_pnl:.4f} USDT",
                                f"Next notional: {next_notional:.4f} USDT",
                            ]
                        )
                    )

        _save_room_state(account_index, state)

    for m in finalized_msgs:
        try:
            log(m)
        except Exception:
            pass
        try:
            _broadcast_room_signal(m)
        except Exception:
            pass


def room_sizing_sync_all_accounts() -> None:
    clients = _get_clients()
    if not clients:
        return
    for idx in range(len(clients)):
        try:
            # Proactive drawdown monitoring: if an account drops below the global
            # threshold (default 20%), pause it and broadcast alert even if no
            # new entries are attempted.
            _is_global_drawdown_ok(idx, clients[idx])
        except Exception:
            pass
        try:
            room_sizing_sync_account(idx)
        except Exception:
            continue


def _room_sizing_sync_state_files_to_current_config() -> None:
    """Rewrite all per-account room_state JSONs using current settings.

    This keeps persisted room sizing state aligned with live
    `config/binance_accounts.yaml` whenever account config is reloaded.
    """
    with _ROOM_SIZING_LOCK:
        for idx, settings in enumerate(_ACCOUNT_SETTINGS):
            if not isinstance(settings, dict):
                settings = {}
            try:
                state = _load_room_state(int(idx), settings)
                _save_room_state(int(idx), state)
            except Exception:
                continue


def _load_accounts_config():
    """Load binance_accounts.yaml if present.

    Expected structure:

    mode: "single" | "multi"
    accounts:
      - name: "acc1"
        api_key: "..."
        api_secret: "..."
    """

    if not _CONFIG_PATH.exists():
        return None

    cfg = _yaml_load_file(_CONFIG_PATH)
    if not cfg:
        return None

    if not isinstance(cfg, dict):
        return None
    return cfg


def _get_account_confidence_threshold(idx: int) -> Optional[float]:
    """Return per-account confidence threshold as a float in [0,1], or None.

    The threshold is read from the account's settings["confidence"]. Values
    greater than 1 are interpreted as percents (e.g. 70 -> 0.70).
    """

    try:
        if idx < 0 or idx >= len(_ACCOUNT_SETTINGS):
            return None
        settings = _ACCOUNT_SETTINGS[idx]
        if not isinstance(settings, dict):
            return None
        raw = settings.get("confidence")
        if raw in (None, ""):
            # Default to 70% if not explicitly configured
            return 0.7
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if val <= 0:
            return None
        # If user provides 70, interpret as 70%
        if val > 1.0:
            val = val / 100.0
        # Clamp to [0, 1]
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        return val
    except Exception:
        return None


def _is_trading_time_for_account(index: int) -> bool:
    settings = _get_account_settings(index)
    if not isinstance(settings, dict):
        return True
    flag = settings.get("time_window_enabled")
    if not flag:
        return True
    cur_hour = _resolve_time_window_hour(settings)

    # Optional multi-window support: when `time_windows` is defined for an
    # account, treat it as a list of trading intervals. Each interval can be
    # either a 2-element list/tuple [start_hour, end_hour] or a dict with
    # keys {"start_hour", "end_hour"}. Hours are interpreted in local
    # server time, consistent with the original single-window behaviour.
    windows = settings.get("time_windows")
    if isinstance(windows, list) and windows:
        any_valid = False
        for win in windows:
            if isinstance(win, dict):
                start_raw = win.get("start_hour")
                end_raw = win.get("end_hour")
            elif isinstance(win, (list, tuple)) and len(win) >= 2:
                start_raw, end_raw = win[0], win[1]
            else:
                continue

            try:
                start_h = float(start_raw)
                end_h = float(end_raw)
            except (TypeError, ValueError):
                continue

            # Allow end_h == 24.0 as a shorthand for wrapping to 00:00 next day.
            if start_h < 0 or start_h >= 24 or end_h < 0 or end_h > 24:
                continue

            any_valid = True

            if start_h == end_h:
                # Degenerate interval means full-day trading when enabled.
                return True

            eff_end = end_h
            if eff_end == 24.0:
                eff_end = 0.0

            if start_h < eff_end:
                if start_h <= cur_hour < eff_end:
                    return True
            else:
                if cur_hour >= start_h or cur_hour < eff_end:
                    return True

        # If at least one interval was valid but none matched, trading is
        # disabled for this account at the current time.
        if any_valid:
            return False
        # If all configured windows were invalid, fall back to legacy
        # single-window behaviour below.

    # Legacy single-window behaviour: use time_window_start_hour /
    # time_window_end_hour if defined.
    start_raw = settings.get("time_window_start_hour")
    end_raw = settings.get("time_window_end_hour")
    try:
        start_h = float(start_raw)
        end_h = float(end_raw)
    except (TypeError, ValueError):
        return True
    if start_h < 0 or start_h >= 24 or end_h < 0 or end_h >= 24:
        return True

    if start_h == end_h:
        return True
    if start_h < end_h:
        return start_h <= cur_hour < end_h
    return cur_hour >= start_h or cur_hour < end_h


def _get_market_regime_state_live() -> Optional[Dict[str, Any]]:
    """Read market regime state from the sidecar output file (mtime cached)."""
    global _MARKET_REGIME_STATE_CACHE, _MARKET_REGIME_STATE_MTIME
    try:
        mtime = _MARKET_REGIME_STATE_PATH.stat().st_mtime
    except Exception:
        _MARKET_REGIME_STATE_CACHE = None
        _MARKET_REGIME_STATE_MTIME = None
        return None

    if (
        _MARKET_REGIME_STATE_CACHE is not None
        and _MARKET_REGIME_STATE_MTIME is not None
        and float(_MARKET_REGIME_STATE_MTIME) == float(mtime)
    ):
        return _MARKET_REGIME_STATE_CACHE

    try:
        raw = _MARKET_REGIME_STATE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        data = None

    if isinstance(data, dict):
        _MARKET_REGIME_STATE_CACHE = data
        _MARKET_REGIME_STATE_MTIME = mtime
        return data

    _MARKET_REGIME_STATE_CACHE = None
    _MARKET_REGIME_STATE_MTIME = mtime
    return None


def _is_market_regime_gate_ok_for_account(index: int) -> Tuple[bool, str]:
    """Account-level gate: allow entries only on ACTIVE/NEUTRAL regime.

    This gate is enabled only when settings.market_regime_gate_enabled=true.
    """
    settings = _get_account_settings(index)
    if not isinstance(settings, dict):
        return True, ""

    if not bool(settings.get("market_regime_gate_enabled")):
        return True, ""

    state = _get_market_regime_state_live()
    if not isinstance(state, dict):
        return False, "market_regime_state_missing"

    regime = str(state.get("last_regime") or "").upper().strip()
    if regime not in ("ACTIVE", "NEUTRAL"):
        return False, f"market_regime={regime or 'UNKNOWN'}"

    try:
        max_age_min = float(settings.get("market_regime_max_age_minutes", 30.0))
    except Exception:
        max_age_min = 30.0
    if max_age_min <= 0:
        max_age_min = 30.0

    try:
        age_sec = time.time() - float(_MARKET_REGIME_STATE_PATH.stat().st_mtime)
    except Exception:
        age_sec = 10**9
    if age_sec > (max_age_min * 60.0):
        return False, f"market_regime_state_stale>{max_age_min:.1f}m"

    return True, f"market_regime={regime}"


def _is_entry_gate_ok_for_account(index: int) -> Tuple[bool, str]:
    """Unified entry gate for time-window OR market-regime mode.

    When market_regime_gate_enabled=true for an account:
      - time-window gating is bypassed
      - entries are allowed only if regime is ACTIVE or NEUTRAL

    Otherwise legacy time-window logic is used unchanged.
    """
    settings = _get_account_settings(index)
    if not isinstance(settings, dict):
        return True, ""

    # Prefer LIVE account config for this specific gate so that updates in
    # binance_accounts.yaml are applied immediately and do not depend on
    # possibly stale in-memory account settings.
    live_gate_enabled = None
    live_gate_max_age = None
    live_time_window_enabled = None
    try:
        cfg_live = _load_accounts_config()
        accounts_live = cfg_live.get("accounts") if isinstance(cfg_live, dict) else None
        live_settings = None
        if isinstance(accounts_live, list) and accounts_live:
            # Resolve by runtime account name first (more robust than index).
            runtime_name = None
            try:
                if 0 <= int(index) < len(_ACCOUNT_NAMES):
                    runtime_name = str(_ACCOUNT_NAMES[int(index)]).strip()
            except Exception:
                runtime_name = None

            if runtime_name:
                for acc_live in accounts_live:
                    if not isinstance(acc_live, dict):
                        continue
                    nm = acc_live.get("name")
                    if str(nm).strip() == runtime_name:
                        s_live = acc_live.get("settings")
                        if isinstance(s_live, dict):
                            live_settings = s_live
                        break

            # Fallback: by index if name-based resolution didn't match.
            if live_settings is None and 0 <= int(index) < len(accounts_live):
                acc_live = accounts_live[int(index)]
                if isinstance(acc_live, dict):
                    s_live = acc_live.get("settings")
                    if isinstance(s_live, dict):
                        live_settings = s_live

        if isinstance(live_settings, dict):
            if live_settings.get("market_regime_gate_enabled") is not None:
                live_gate_enabled = bool(live_settings.get("market_regime_gate_enabled"))
            if live_settings.get("market_regime_max_age_minutes") is not None:
                try:
                    live_gate_max_age = float(live_settings.get("market_regime_max_age_minutes"))
                except Exception:
                    live_gate_max_age = None
            if live_settings.get("time_window_enabled") is not None:
                live_time_window_enabled = bool(live_settings.get("time_window_enabled"))
    except Exception:
        pass

    gate_enabled = bool(settings.get("market_regime_gate_enabled"))
    if live_gate_enabled is not None:
        gate_enabled = bool(live_gate_enabled)
        # Inject live override so downstream gate function uses the same value.
        settings["market_regime_gate_enabled"] = gate_enabled
    if live_gate_max_age is not None:
        settings["market_regime_max_age_minutes"] = float(live_gate_max_age)
    if live_time_window_enabled is not None:
        settings["time_window_enabled"] = bool(live_time_window_enabled)

    if gate_enabled:
        return _is_market_regime_gate_ok_for_account(index)

    if not _is_trading_time_for_account(index):
        return False, "time_window"

    return True, ""


def _is_daily_drawdown_ok(index: int, client: BinanceFuturesClient) -> bool:
    global _DAILY_DD_STATE

    settings = _get_account_settings(index)
    if not isinstance(settings, dict):
        return True

    # Feature is opt-in per account.
    if not settings.get("daily_dd_limit_enabled"):
        return True

    limit_raw = settings.get("daily_dd_limit_pct")
    try:
        limit_pct = float(limit_raw)
    except (TypeError, ValueError):
        return True
    if limit_pct <= 0:
        return True

    # Use local calendar day for daily reset.
    today = time.strftime("%Y-%m-%d", time.localtime())
    state = _DAILY_DD_STATE.get(index)

    # New day or uninitialised: snapshot starting balance and allow trading.
    if not isinstance(state, dict) or state.get("date") != today:
        start_balance = _get_account_balance_usdt(client)
        if start_balance <= 0:
            _DAILY_DD_STATE[index] = {"date": today, "start_balance": None, "hit": False}
            return True
        _DAILY_DD_STATE[index] = {"date": today, "start_balance": float(start_balance), "hit": False}
        return True

    # Once the limit is hit for the day, keep blocking until the next day.
    if state.get("hit"):
        return False

    start_balance = state.get("start_balance")
    if not isinstance(start_balance, (int, float)) or start_balance <= 0:
        return True

    current_balance = _get_account_balance_usdt(client)
    if current_balance <= 0:
        return True

    try:
        dd_pct = (float(current_balance) - float(start_balance)) / float(start_balance) * 100.0
    except Exception:
        return True

    if dd_pct <= -limit_pct:
        state["hit"] = True
        _DAILY_DD_STATE[index] = state
        return False

    return True


def _compute_recent_pnl_pct_for_account(index: int, window_hours: float) -> Optional[float]:
    """Approximate recent PnL %% for the given account over a rolling window.

    Uses EXIT rows from data/trade_history/account_X.csv and sums the
    recorded pnl_pct values over the last ``window_hours`` hours.
    """

    if window_hours <= 0:
        return None

    try:
        path = _TRADE_HISTORY_DIR / f"account_{index}.csv"
    except Exception:
        return None

    if not path.exists():
        # No history yet – treat as flat PnL.
        return 0.0

    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - int(window_hours * 3600.0 * 1000.0)

    total_pnl = 0.0
    seen = False
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_raw = row.get("timestamp_ms") or ""
                    if not ts_raw:
                        continue
                    ts_ms = int(float(ts_raw))
                except (TypeError, ValueError):
                    continue

                if ts_ms < cutoff_ms:
                    continue

                event = (row.get("event") or "").upper()
                if event != "EXIT":
                    continue

                pnl_raw = row.get("pnl_pct")
                if pnl_raw in (None, ""):
                    continue
                try:
                    pnl_val = float(pnl_raw)
                except (TypeError, ValueError):
                    continue

                total_pnl += pnl_val
                seen = True
    except Exception:
        return None

    if not seen:
        return 0.0
    return total_pnl


def _get_adaptive_sizing_cfg(settings: Dict[str, Any], fallback_base: Optional[float]) -> Tuple[bool, float, float, float]:
    if not isinstance(settings, dict):
        return False, 0.0, 0.0, 0.0
    enabled = bool(settings.get("adaptive_sizing_enabled"))

    base_raw = settings.get("adaptive_sizing_base_pct")
    if base_raw is None:
        base_raw = fallback_base
    step_raw = settings.get("adaptive_sizing_step_pct")
    max_raw = settings.get("adaptive_sizing_max_pct")

    try:
        base = float(base_raw) if base_raw is not None else 1.0
    except (TypeError, ValueError):
        base = 1.0
    try:
        step = float(step_raw) if step_raw is not None else 1.0
    except (TypeError, ValueError):
        step = 1.0
    try:
        maxp = float(max_raw) if max_raw is not None else 10.0
    except (TypeError, ValueError):
        maxp = 10.0

    if base <= 0:
        base = 1.0
    if step <= 0:
        step = 1.0
    if maxp < base:
        maxp = base
    return enabled, base, step, maxp


def _read_realised_trade_pnls_from_history(index: int, max_trades: int = 200) -> List[float]:
    try:
        path = _TRADE_HISTORY_DIR / f"account_{int(index)}.csv"
    except Exception:
        return []
    if not path.exists():
        return []

    trades: List[Tuple[int, float]] = []
    open_pos: Dict[str, Dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_ms = int(float(row.get("timestamp_ms") or 0))
                except Exception:
                    continue
                event = (row.get("event") or "").upper()
                sym = (row.get("symbol") or "").upper()
                if not sym:
                    continue

                if event == "ENTRY":
                    try:
                        qty = float(row.get("qty") or 0)
                        ep = float(row.get("entry_price") or 0)
                    except Exception:
                        continue
                    if qty <= 0 or ep <= 0:
                        continue
                    open_pos[sym] = {
                        "entry_qty": float(qty),
                        "entry_price": float(ep),
                        "exited_qty": 0.0,
                        "pnl_notional": 0.0,
                    }
                    continue

                if event != "EXIT":
                    continue
                pos = open_pos.get(sym)
                if not isinstance(pos, dict):
                    continue

                try:
                    exit_qty = float(row.get("qty") or 0)
                except Exception:
                    continue
                if exit_qty <= 0:
                    continue

                pnl_raw = row.get("pnl_pct")
                if pnl_raw in (None, ""):
                    continue
                try:
                    pnl_leg = float(pnl_raw)
                except Exception:
                    continue

                entry_price = pos.get("entry_price")
                entry_qty = pos.get("entry_qty")
                if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                    continue
                if not isinstance(entry_qty, (int, float)) or entry_qty <= 0:
                    continue

                leg_notional = float(exit_qty) * float(entry_price)
                pos["pnl_notional"] = float(pos.get("pnl_notional") or 0.0) + (leg_notional * (float(pnl_leg) / 100.0))
                pos["exited_qty"] = float(pos.get("exited_qty") or 0.0) + float(exit_qty)

                if float(pos["exited_qty"]) >= float(entry_qty) * 0.999:
                    entry_notional = float(entry_qty) * float(entry_price)
                    if entry_notional > 0:
                        pnl_total = float(pos["pnl_notional"]) / entry_notional * 100.0
                        trades.append((ts_ms, float(pnl_total)))
                    open_pos.pop(sym, None)
    except Exception:
        return []

    trades.sort(key=lambda x: x[0])
    pnl_list = [p for (_, p) in trades]
    if max_trades > 0 and len(pnl_list) > max_trades:
        pnl_list = pnl_list[-max_trades:]
    return pnl_list


def _compute_loss_streak(index: int) -> int:
    pnl_list = _read_realised_trade_pnls_from_history(index)
    streak = 0
    for pnl in reversed(pnl_list):
        try:
            v = float(pnl)
        except Exception:
            break
        if v < 0.0:
            streak += 1
        else:
            break
    return max(0, int(streak))


def _get_adaptive_pct(index: int, settings: Dict[str, Any], fallback_base: Optional[float]) -> Optional[float]:
    enabled, base, step, maxp = _get_adaptive_sizing_cfg(settings, fallback_base=fallback_base)
    if not enabled:
        return None

    now = time.time()
    with _ADAPTIVE_SIZING_LOCK:
        cached = _ADAPTIVE_SIZING_CACHE.get(int(index))
        if isinstance(cached, dict):
            try:
                ts = float(cached.get("ts") or 0.0)
                pct = float(cached.get("pct") or 0.0)
            except Exception:
                ts = 0.0
                pct = 0.0
            if pct > 0 and (now - ts) < 20.0:
                return pct

    streak = _compute_loss_streak(int(index))
    pct = base + float(streak) * step
    if pct > maxp:
        pct = maxp
    if pct < base:
        pct = base

    with _ADAPTIVE_SIZING_LOCK:
        _ADAPTIVE_SIZING_CACHE[int(index)] = {"pct": float(pct), "ts": now, "streak": int(streak)}

    try:
        name = _ACCOUNT_NAMES[index] if 0 <= int(index) < len(_ACCOUNT_NAMES) else f"account_{index}"
        log(f"[SIZING] account={name} index={index} adaptive pct={pct:.2f}% (loss_streak={streak})")
    except Exception:
        pass

    return float(pct)


def _is_delay_limit_ok(index: int) -> bool:
    """Return False when per-account delay_limit conditions block trading.

    Logic (per account index):
    - If settings.delay_limit_enabled is not set/false, always return True.
    - Else, read settings.delay_limit_conditions as an ordered list of
      stages. Each stage has window_hours, max_loss_pct, cooldown_hours.
    - While in an active cooldown window, always return False.
    - Otherwise, evaluate the current stage using recent PnL over
      ``window_hours``. If PnL <= -max_loss_pct, start cooldown for
      ``cooldown_hours`` and advance to the next stage (wrapping to 0
      after the last stage). If no condition is hit, allow trading.
    """

    settings = _get_account_settings(index)
    if not isinstance(settings, dict):
        return True

    if not settings.get("delay_limit_enabled"):
        return True

    raw_conditions = settings.get("delay_limit_conditions") or []
    conditions: List[Dict[str, float]] = []
    if isinstance(raw_conditions, list):
        for c in raw_conditions:
            if not isinstance(c, dict):
                continue
            try:
                wh = float(c.get("window_hours"))
                ml = float(c.get("max_loss_pct"))
                ch = float(c.get("cooldown_hours"))
            except (TypeError, ValueError):
                continue
            if wh <= 0 or ml <= 0 or ch <= 0:
                continue
            conditions.append(
                {
                    "window_hours": wh,
                    "max_loss_pct": ml,
                    "cooldown_hours": ch,
                }
            )

    if not conditions:
        return True

    now_ts = time.time()
    state = _DELAY_LIMIT_STATE.get(index) or {}
    if not isinstance(state, dict):
        state = {}

    # Resolve human-friendly account name for logging.
    try:
        acc_name = _ACCOUNT_NAMES[index] if 0 <= index < len(_ACCOUNT_NAMES) else f"account_{index}"
    except Exception:
        acc_name = f"account_{index}"

    cooldown_until = state.get("cooldown_until")
    if isinstance(cooldown_until, (int, float)) and cooldown_until > 0:
        if now_ts < float(cooldown_until):
            # Still inside an active cooldown window for this account.
            try:
                expiry_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(cooldown_until)))
                log(
                    f"[DELAY_LIMIT] account={acc_name} index={index} still in cooldown until {expiry_iso}; "
                    f"skipping new trades."
                )
            except Exception:
                pass
            return False
        else:
            # Cooldown just expired; allow trading again and log this event.
            try:
                expiry_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(cooldown_until)))
                cur_stage = state.get("stage", 0)
                log(
                    f"[DELAY_LIMIT] account={acc_name} index={index} cooldown expired at {expiry_iso}; "
                    f"re-enabling trading (stage={cur_stage})."
                )
            except Exception:
                pass

    try:
        stage_val = int(state.get("stage", 0))
    except (TypeError, ValueError):
        stage_val = 0
    if stage_val < 0 or stage_val >= len(conditions):
        stage_val = 0

    cond = conditions[stage_val]
    pnl_pct = _compute_recent_pnl_pct_for_account(index, cond["window_hours"])
    if pnl_pct is None:
        return True

    if pnl_pct <= -cond["max_loss_pct"]:
        cooldown_seconds = cond["cooldown_hours"] * 3600.0
        next_stage = stage_val + 1
        if next_stage >= len(conditions):
            next_stage = 0

        cooldown_until_ts = now_ts + cooldown_seconds
        _DELAY_LIMIT_STATE[index] = {
            "cooldown_until": cooldown_until_ts,
            "stage": next_stage,
        }

        # Log detailed delay-limit activation so the user can see when and for how
        # long this account is blocked.
        try:
            window_h = cond["window_hours"]
            max_loss = cond["max_loss_pct"]
            cooldown_h = cond["cooldown_hours"]
            expiry_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cooldown_until_ts))
            log(
                f"[DELAY_LIMIT] account={acc_name} index={index} hit stage={stage_val} "
                f"(window={window_h:.2f}h, pnl={pnl_pct:.2f}% <= -{max_loss:.2f}%) → "
                f"cooldown={cooldown_h:.2f}h until {expiry_iso} (next_stage={next_stage})."
            )
        except Exception:
            pass

        return False

    _DELAY_LIMIT_STATE[index] = {"stage": stage_val}
    return True


def record_symbol_loss(account_index: int, symbol: str) -> None:
    """Record a losing trade timestamp for the (account, symbol) pair.

    Called by the exit manager in main.py whenever a position closes at a loss.
    Used by _is_symbol_loss_cooldown_ok to enforce per-coin cooldown.
    """
    key = (int(account_index), str(symbol).upper())
    now_ts = time.time()
    lst = _SYMBOL_LOSS_TIMESTAMPS.get(key)
    if lst is None:
        lst = []
        _SYMBOL_LOSS_TIMESTAMPS[key] = lst
    lst.append(now_ts)

    # Read account settings for this account to evaluate cooldown.
    settings = _get_account_settings(account_index)
    if not isinstance(settings, dict):
        return
    if not settings.get("loss_cooldown_enabled"):
        return

    try:
        max_losses = int(settings.get("loss_cooldown_max_losses") or 2)
    except (TypeError, ValueError):
        max_losses = 2
    try:
        window_hours = float(settings.get("loss_cooldown_window_hours") or 1.0)
    except (TypeError, ValueError):
        window_hours = 1.0
    try:
        cooldown_hours = float(settings.get("loss_cooldown_hours") or 3.0)
    except (TypeError, ValueError):
        cooldown_hours = 3.0

    if max_losses < 1:
        max_losses = 2
    if window_hours <= 0:
        window_hours = 1.0
    if cooldown_hours <= 0:
        cooldown_hours = 3.0

    # Prune old timestamps outside the window.
    cutoff = now_ts - window_hours * 3600.0
    lst[:] = [ts for ts in lst if ts >= cutoff]

    # Check if loss count in the window has reached the threshold.
    if len(lst) >= max_losses:
        cooldown_until = now_ts + cooldown_hours * 3600.0
        _SYMBOL_LOSS_COOLDOWN_UNTIL[key] = cooldown_until
        try:
            acc_name = _ACCOUNT_NAMES[account_index] if 0 <= account_index < len(_ACCOUNT_NAMES) else f"account_{account_index}"
        except Exception:
            acc_name = f"account_{account_index}"
        try:
            expiry_str = time.strftime("%H:%M:%S", time.localtime(cooldown_until))
            log(
                f"[LOSS COOLDOWN] {symbol} on {acc_name}: {len(lst)} losses in "
                f"{window_hours}h window — blocking new entries until {expiry_str} "
                f"({cooldown_hours}h cooldown)"
            )
        except Exception:
            pass


def _is_symbol_loss_cooldown_ok(account_index: int, symbol: str) -> bool:
    """Return False if this (account, symbol) is currently in a loss cooldown.

    Gating check called from place_order before opening new positions.
    """
    settings = _get_account_settings(account_index)
    if not isinstance(settings, dict):
        return True
    if not settings.get("loss_cooldown_enabled"):
        return True

    key = (int(account_index), str(symbol).upper())
    cooldown_until = _SYMBOL_LOSS_COOLDOWN_UNTIL.get(key)
    if cooldown_until is None:
        return True

    now_ts = time.time()
    if now_ts < float(cooldown_until):
        return False

    # Cooldown expired — clean up.
    _SYMBOL_LOSS_COOLDOWN_UNTIL.pop(key, None)
    return True


def _load_proxy_config():
    """Load execution.yaml for proxy-related settings (best-effort)."""

    global _PROXY_CONFIG, _PROXY_CONFIG_LOADED
    if _PROXY_CONFIG_LOADED:
        return _PROXY_CONFIG

    cfg = None
    if _EXECUTION_CONFIG_PATH.exists():
        cfg = _yaml_load_file(_EXECUTION_CONFIG_PATH)

    if not isinstance(cfg, dict):
        cfg = {}

    _PROXY_CONFIG = cfg
    _PROXY_CONFIG_LOADED = True
    return _PROXY_CONFIG


def _should_use_proxy_for_trading() -> bool:
    """Return True if proxies should be used for signed trading requests."""

    cfg = _load_proxy_config() or {}
    proxies_cfg = cfg.get("proxies") or {}
    # If proxies are explicitly disabled, never use them for trading.
    if proxies_cfg.get("enabled") is False:
        return False
    return bool(proxies_cfg.get("use_for_trading", False))


def _get_proxies_for_request(signed: bool):
    """Decide which proxies dict to use for a given request.

    When proxies.enabled is False in execution.yaml, we return an empty dict
    to ensure that no proxy (including environment-level HTTP(S)_PROXY) is
    used. Otherwise we preserve the previous behaviour: always use a random
    proxy for public endpoints, and use proxies for signed endpoints only
    when proxies.use_for_trading is True.
    """

    cfg = _load_proxy_config() or {}
    proxies_cfg = cfg.get("proxies") or {}

    # Explicitly disabled: force no proxies at all.
    if proxies_cfg.get("enabled") is False:
        return {}

    if signed:
        # Dedicated trading proxy takes absolute priority over random rotation.
        trading_proxy_str = (proxies_cfg.get("trading_proxy") or "").strip()
        if trading_proxy_str:
            return build_proxy_from_string(trading_proxy_str)
        if _should_use_proxy_for_trading():
            return get_random_proxy()
        return None

    # Public endpoints: keep using random proxies by default.
    return get_random_proxy()


def _get_clients():
    """Return a list of BinanceFuturesClient instances.

    Precedence:
    1) config/binance_accounts.yaml (single or multi account)
    2) BINANCE_API_KEY / BINANCE_API_SECRET from environment.
    """

    global _CLIENTS, _ACCOUNTS_MODE, _ACCOUNTS_CFG_MTIME, _ACCOUNT_PROXIES
    global _GLOBAL_DD_PAUSE_ENABLED_DEFAULT, _GLOBAL_DD_PAUSE_PCT_DEFAULT

    current_cfg_mtime: Optional[float] = None
    try:
        if _CONFIG_PATH.exists():
            current_cfg_mtime = _CONFIG_PATH.stat().st_mtime
    except Exception:
        current_cfg_mtime = None

    if _CLIENTS is not None:
        # Hot-reload account settings when binance_accounts.yaml changes.
        if (
            current_cfg_mtime is not None
            and _ACCOUNTS_CFG_MTIME is not None
            and current_cfg_mtime != _ACCOUNTS_CFG_MTIME
        ):
            try:
                log("[CONFIG] Detected binance_accounts.yaml change; reloading account clients/settings")
            except Exception:
                pass
            _CLIENTS = None
            _ACCOUNT_PROXIES = []
        else:
            return _CLIENTS

    clients = []
    enabled_flags = []
    leverage_overrides = []
    settings_list = []
    account_names: list[str] = []
    _ACCOUNT_PROXIES = []
    _GLOBAL_DD_PAUSE_ENABLED_DEFAULT = True
    _GLOBAL_DD_PAUSE_PCT_DEFAULT = 20.0

    cfg = _load_accounts_config()
    if cfg:
        mode = str(cfg.get("mode", "single")).lower()
        _ACCOUNTS_MODE = "multi" if mode == "multi" else "single"

        # Optional global defaults applied to all accounts when per-account
        # overrides are not set.
        g = cfg.get("global_settings")
        if isinstance(g, dict):
            try:
                if g.get("global_dd_pause_enabled") is not None:
                    _GLOBAL_DD_PAUSE_ENABLED_DEFAULT = bool(g.get("global_dd_pause_enabled"))
            except Exception:
                pass
            try:
                raw_dd = g.get("global_dd_pause_pct")
                if raw_dd is not None:
                    dd_val = float(raw_dd)
                    if dd_val > 0:
                        _GLOBAL_DD_PAUSE_PCT_DEFAULT = dd_val
            except Exception:
                pass

        accounts = cfg.get("accounts") or []
        if isinstance(accounts, list):
            for idx, acc in enumerate(accounts):
                if not isinstance(acc, dict):
                    continue
                api_key = acc.get("api_key")
                api_secret = acc.get("api_secret")
                if not api_key or not api_secret:
                    continue

                proxy_str = acc.get("proxy")
                fixed_proxies = build_proxy_from_string(proxy_str) if proxy_str else None

                client = BinanceFuturesClient(api_key, api_secret, proxies=fixed_proxies)
                clients.append(client)

                flag = acc.get("trade_enabled")
                enabled_flags.append(False if flag is False else True)

                lev = acc.get("leverage")
                if lev is None:
                    leverage_overrides.append(None)
                else:
                    try:
                        leverage_overrides.append(int(lev))
                    except (TypeError, ValueError):
                        leverage_overrides.append(None)

                settings = acc.get("settings") or {}
                if isinstance(settings, dict):
                    settings_list.append(settings)
                else:
                    settings_list.append({})

                # Human-readable account name for logging (e.g. delay-limit).
                name = acc.get("name")
                if not isinstance(name, str) or not name:
                    name = f"account_{len(account_names)}"
                account_names.append(name)

                # Keep a parallel list of per-account fixed proxies for
                # potential future use.
                _ACCOUNT_PROXIES.append(fixed_proxies)

    if not clients:
        # Fallback to environment-based single account
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "No Binance API credentials configured. "
                "Set BINANCE_API_KEY/BINANCE_API_SECRET or config/binance_accounts.yaml"
            )
        clients = [BinanceFuturesClient(api_key, api_secret)]
        _ACCOUNTS_MODE = "single"
        enabled_flags = [True]
        leverage_overrides = [None]
        settings_list = [{}]
        account_names = ["primary"]
        _ACCOUNT_PROXIES.append(None)

    global _ACCOUNT_TRADE_ENABLED, _ACCOUNT_LEVERAGE_OVERRIDES, _ACCOUNT_SETTINGS, _ACCOUNT_NAMES
    _ACCOUNT_TRADE_ENABLED = enabled_flags or [True] * len(clients)
    _ACCOUNT_LEVERAGE_OVERRIDES = leverage_overrides or [None] * len(clients)
    _ACCOUNT_SETTINGS = settings_list or [{}] * len(clients)
    _ACCOUNT_NAMES = account_names or [f"account_{i}" for i in range(len(clients))]
    try:
        _room_sizing_sync_state_files_to_current_config()
    except Exception:
        pass

    _ACCOUNTS_CFG_MTIME = current_cfg_mtime
    _CLIENTS = clients
    return _CLIENTS


def startup_cleanup_accounts() -> None:
    """Best-effort startup cleanup per account.

    Controlled by per-account settings keys:
    - startup_cancel_open_orders: bool
    - startup_flatten_positions: bool
    - fibo_staking_reset / fibo_staking_reset_once / fibo_reset: bool

    Runs at most once per process per account.
    """

    global _STARTUP_CLEANUP_DONE

    try:
        clients = _get_clients()
    except Exception:
        return
    if not clients:
        return

    for idx, client in enumerate(clients):
        if idx in _STARTUP_CLEANUP_DONE:
            continue

        try:
            settings = _get_account_settings(idx)
        except Exception:
            settings = {}

        if not isinstance(settings, dict):
            settings = {}

        cancel_flag = bool(settings.get("startup_cancel_open_orders"))
        flatten_flag = bool(settings.get("startup_flatten_positions"))
        reset_fibo_flag = bool(
            settings.get("fibo_staking_reset")
            or settings.get("fibo_staking_reset_once")
            or settings.get("fibo_reset")
            or settings.get("startup_reset_fibo")
        )

        reset_room_flag = bool(
            settings.get("startup_reset_room_sizing")
            or settings.get("startup_reset_rooms")
            or settings.get("room_sizing_reset")
        )

        if not (cancel_flag or flatten_flag or reset_fibo_flag or reset_room_flag):
            continue

        _STARTUP_CLEANUP_DONE.add(idx)

        acc_name = None
        try:
            if 0 <= idx < len(_ACCOUNT_NAMES):
                acc_name = _ACCOUNT_NAMES[idx]
        except Exception:
            acc_name = None
        name_txt = f"{acc_name}" if acc_name else f"index {idx}"

        if reset_fibo_flag:
            try:
                _FIBO_STATE[idx] = {
                    "cur_index": 0,
                    "sequence": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0],
                }
                _FIBO_RESET_DONE.add(idx)
                log(f"[STARTUP CLEANUP] Reset Fibonacci staking state for account {name_txt}")
            except Exception:
                pass

        if reset_room_flag:
            try:
                room_sizing_reset_account(idx)
                log(f"[STARTUP CLEANUP] Reset room sizing state for account {name_txt}")
            except Exception:
                pass

        if cancel_flag:
            try:
                orders = client.open_orders()
            except Exception as e:
                try:
                    log(f"[STARTUP CLEANUP] Failed to fetch open orders for account {name_txt}: {e}")
                except Exception:
                    pass
                orders = []

            if isinstance(orders, list):
                canceled = 0
                for o in orders:
                    if not isinstance(o, dict):
                        continue
                    sym = o.get("symbol")
                    oid = o.get("orderId")
                    if not sym or oid is None:
                        continue
                    try:
                        client.cancel_order(str(sym), order_id=int(oid))
                        canceled += 1
                    except Exception:
                        continue
                try:
                    log(
                        f"[STARTUP CLEANUP] Canceled {canceled} open orders for account {name_txt}"
                    )
                except Exception:
                    pass

        if flatten_flag:
            try:
                positions = client.position_risk()
            except Exception as e:
                try:
                    log(f"[STARTUP CLEANUP] Failed to fetch positions for account {name_txt}: {e}")
                except Exception:
                    pass
                positions = []

            closed = 0
            if isinstance(positions, list):
                for pos in positions:
                    if not isinstance(pos, dict):
                        continue
                    sym = pos.get("symbol")
                    amt_raw = pos.get("positionAmt")
                    try:
                        amt = float(amt_raw)
                    except (TypeError, ValueError):
                        continue
                    if not sym or amt == 0.0:
                        continue
                    try:
                        close_position_market(idx, str(sym), amt, force_full_close=True)
                        closed += 1
                    except Exception:
                        continue
            try:
                log(f"[STARTUP CLEANUP] Flatten requested; sent close for {closed} positions on account {name_txt}")
            except Exception:
                pass


def _get_distance_rules_for_account(index: int) -> Tuple[bool, list[Dict[str, Any]]]:
    """Return (enabled, rules) for per-account distance-based SL/TP.

    Rules are read from settings["distance_sl_tp_rules"]. When enabled is
    False or no valid rules are configured, the caller should ignore the
    returned list and fall back to the account's normal SL/TP behaviour.
    """

    try:
        settings = _get_account_settings(index)
    except Exception:
        return False, []

    if not isinstance(settings, dict):
        return False, []

    enabled = bool(settings.get("distance_sl_tp_enabled"))
    raw_rules = settings.get("distance_sl_tp_rules") or []
    if not enabled:
        return False, []

    if not isinstance(raw_rules, list) or not raw_rules:
        # Feature explicitly enabled but no usable rules configured: log once
        # per call and fall back to legacy behaviour.
        try:
            log(
                f"[DIST_SLTP] distance_sl_tp_enabled is true but distance_sl_tp_rules "
                f"missing or empty for account index {index}; falling back to static/AI SL/TP."
            )
        except Exception:
            pass
        return False, []

    cleaned: list[Dict[str, Any]] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        rule: Dict[str, Any] = {}
        # max_distance_pct can be None (acts as fallback bucket)
        if "max_distance_pct" in r:
            try:
                md = r["max_distance_pct"]
                rule["max_distance_pct"] = None if md is None else float(md)
            except (TypeError, ValueError):
                rule["max_distance_pct"] = None
        else:
            rule["max_distance_pct"] = None

        rule["use_auto_sl_tp"] = bool(r.get("use_auto_sl_tp"))

        # Optional explicit SL/TP and entry offset overrides.
        if "sl_pct" in r:
            try:
                rule["sl_pct"] = float(r["sl_pct"])
            except (TypeError, ValueError):
                pass
        if "tp_pcts" in r and isinstance(r["tp_pcts"], (list, tuple)):
            tps: list[float] = []
            for v in r["tp_pcts"]:
                try:
                    pv = float(v)
                except (TypeError, ValueError):
                    continue
                if pv > 0:
                    tps.append(pv)
            if tps:
                rule["tp_pcts"] = tps

        if "entry_offset_pct" in r:
            try:
                rule["entry_offset_pct"] = float(r["entry_offset_pct"])
            except (TypeError, ValueError):
                pass

        cleaned.append(rule)

    if not cleaned:
        # All rules were invalid or unusable; warn and fall back.
        try:
            log(
                f"[DIST_SLTP] No valid distance_sl_tp_rules for account index {index}; "
                f"falling back to static/AI SL/TP."
            )
        except Exception:
            pass
        return False, []

    # Preserve order as defined in config; callers will pick the first rule
    # whose max_distance_pct covers the given distance, with None acting as
    # catch-all.
    return True, cleaned


def compute_distance_based_sl_tp(
    account_index: int,
    distance_pct: float,
) -> Tuple[Optional[bool], Optional[float], Optional[list[float]], Optional[float]]:
    """Compute per-account SL/TP/entry_offset overrides for a given distance.

    Returns a tuple (use_auto_sl_tp, sl_pct, tp_pcts, entry_offset_pct).

    - If no applicable rule is found or feature is disabled, all override
      values are None and use_auto_sl_tp is False, so callers can fall back to
      existing behaviour.
    - When use_auto_sl_tp is True, callers should keep the account's existing
      auto_sl_tp/static SL/TP but may still apply entry_offset_pct if not
      None.
    - When use_auto_sl_tp is False and sl_pct/tp_pcts are provided, callers
      may override the account's SL/TP percents with these values.
    """

    try:
        dist = float(distance_pct)
    except (TypeError, ValueError):
        dist = 0.0
    if dist < 0:
        dist = 0.0

    enabled, rules = _get_distance_rules_for_account(account_index)
    if not enabled or not rules:
        # None for use_auto_sl_tp => no override; callers keep existing
        # auto_sl_tp semantics and SL/TP percents.
        return None, None, None, None

    chosen: Optional[Dict[str, Any]] = None
    fallback: Optional[Dict[str, Any]] = None

    for r in rules:
        md = r.get("max_distance_pct")
        if md is None:
            # Catch-all bucket; remember but do not select yet.
            if fallback is None:
                fallback = r
            continue
        try:
            md_val = float(md)
        except (TypeError, ValueError):
            continue
        if dist <= md_val:
            chosen = r
            break

    if chosen is None:
        chosen = fallback

    if chosen is None:
        # No matching bucket; treat as no-op override.
        return None, None, None, None

    use_auto = bool(chosen.get("use_auto_sl_tp"))
    sl_pct = None
    tp_pcts = None
    entry_offset = None

    if "sl_pct" in chosen:
        try:
            val = float(chosen["sl_pct"])
            if val > 0:
                sl_pct = val
        except (TypeError, ValueError):
            pass

    if "tp_pcts" in chosen and isinstance(chosen["tp_pcts"], (list, tuple)):
        tps: list[float] = []
        for v in chosen["tp_pcts"]:
            try:
                pv = float(v)
            except (TypeError, ValueError):
                continue
            if pv > 0:
                tps.append(pv)
        if tps:
            tp_pcts = tps

    if "entry_offset_pct" in chosen:
        try:
            entry_offset = float(chosen["entry_offset_pct"])
        except (TypeError, ValueError):
            entry_offset = None

    return use_auto, sl_pct, tp_pcts, entry_offset


def _get_fibo_notional_for_account(index: int, entry_price: float) -> float:
    """Return Fibonacci-based notional in USDT for a given account.

    The ladder is account-local and driven solely by a Fibonacci index
    (cur_index) that advances by one step on each realised losing trade and
    resets to zero on a profitable trade. This guarantees that if a trade is
    opened with a given Fibonacci notional (e.g. 5 USDT) and closes in loss,
    the next trade for that account will use the next Fibonacci notional
    (e.g. 8 USDT), then 13, 21, etc.
    """

    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        return 0.0
    if ep <= 0:
        return 0.0

    # Check per-account toggle
    settings = _get_account_settings(index)
    if not settings.get("fibo_staking_enabled"):
        return 0.0

    reset_flag = bool(
        settings.get("fibo_staking_reset")
        or settings.get("fibo_staking_reset_once")
        or settings.get("fibo_reset")
    )
    if reset_flag and index not in _FIBO_RESET_DONE:
        _FIBO_STATE[index] = {
            "cur_index": 0,
            "sequence": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0],
        }
        _FIBO_RESET_DONE.add(index)

    fibo_mult = 1.0
    try:
        raw_mult = settings.get("fibo_staking_multiplier")
        if raw_mult is None:
            raw_mult = settings.get("fibo_staking_mult")
        if raw_mult is not None:
            fibo_mult = float(raw_mult)
    except Exception:
        fibo_mult = 1.0
    if fibo_mult <= 0:
        fibo_mult = 1.0

    # Build (or reuse) per-account fibo state
    state = _FIBO_STATE.get(index)
    if not isinstance(state, dict):
        state = {
            "cur_index": 0,
            "sequence": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0],
        }

    seq = state.get("sequence")
    # If sequence is missing, empty, or based on an old schema (e.g. starting
    # from 2.0 instead of 1.0), reset it to the new base ladder.
    reset_seq = False
    if not isinstance(seq, list) or not seq:
        reset_seq = True
    else:
        try:
            first = float(seq[0])
        except Exception:
            reset_seq = True
        else:
            if first != 1.0:
                reset_seq = True

    if reset_seq:
        seq = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
        state["sequence"] = seq

    try:
        cur_index = int(state.get("cur_index", 0))
    except Exception:
        cur_index = 0
    if cur_index < 0:
        cur_index = 0

    # Extend Fibonacci sequence on demand (unbounded growth) to cover index.
    while len(seq) <= cur_index + 2:
        try:
            a = float(seq[-1])
            b = float(seq[-2])
            seq.append(a + b)
        except Exception:
            break

    if cur_index >= len(seq):
        cur_index = len(seq) - 1

    try:
        notional = float(seq[cur_index]) * fibo_mult
    except Exception:
        notional = 0.0

    if notional > 0.0 and notional < 5.5:
        notional = 5.5

    state["cur_index"] = cur_index
    state["sequence"] = seq
    state["last_notional"] = notional
    _FIBO_STATE[index] = state
    return notional


def update_fibo_after_exit(account_index: int, pnl_pct) -> None:
    """Update per-account Fibonacci loss streak after a realised trade.

    Positive pnl_pct resets the streak to zero, negative increments it by one.
    When the feature is disabled for that account, this is a no-op.
    """

    if account_index is None:
        return
    try:
        idx = int(account_index)
    except (TypeError, ValueError):
        return
    if idx < 0:
        return

    settings = _get_account_settings(idx)
    if not settings.get("fibo_staking_enabled"):
        return

    try:
        if pnl_pct is None:
            return
        val = float(pnl_pct)
    except (TypeError, ValueError):
        return

    state = _FIBO_STATE.get(idx)
    if not isinstance(state, dict):
        state = {"cur_index": 0, "sequence": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]}

    try:
        cur_index = int(state.get("cur_index", 0))
    except Exception:
        cur_index = 0
    if cur_index < 0:
        cur_index = 0

    if val > 0.0:
        # Any profitable trade resets ladder to the base (2 USDT).
        cur_index = 0
    elif val < 0.0:
        # Each losing trade advances one Fibonacci step for the next entry.
        cur_index += 1

    if cur_index < 0:
        cur_index = 0
    state["cur_index"] = cur_index
    _FIBO_STATE[idx] = state


def _get_client():
    """Backward-compatible helper: return primary client (first in list)."""

    clients = _get_clients()
    return clients[0]


def _get_account_wallet_balance_usdt_for_dd(client: BinanceFuturesClient) -> float:
    acct = None
    try:
        acct = client.account()
    except Exception:
        acct = None

    if isinstance(acct, dict):
        raw = None
        for k in ("totalWalletBalance", "totalMarginBalance", "availableBalance"):
            if k in acct and acct.get(k) is not None:
                raw = acct.get(k)
                break
        if raw is not None:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                return v

    try:
        balances = client.balance()
    except Exception:
        return 0.0
    if not isinstance(balances, list):
        return 0.0
    for item in balances:
        if not isinstance(item, dict):
            continue
        asset = item.get("asset")
        if str(asset).upper() != "USDT":
            continue
        raw_val = item.get("balance")
        if raw_val is None:
            raw_val = item.get("crossWalletBalance")
        if raw_val is None:
            raw_val = item.get("availableBalance")
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            return 0.0
        if val <= 0:
            return 0.0
        return val
    return 0.0


_WALLET_BALANCE_CACHE: Dict[int, Dict[str, Any]] = {}
_WALLET_BALANCE_CACHE_TTL = 30.0


def get_account_wallet_balance(account_index: int) -> float:
    """Return USDT wallet balance for an account (cached 30s).

    Uses 'balance' (total wallet) rather than 'availableBalance' so that
    the value includes margin locked in open positions.
    """
    now = time.time()
    cached = _WALLET_BALANCE_CACHE.get(account_index)
    if cached is not None:
        if (now - cached.get("ts", 0.0)) < _WALLET_BALANCE_CACHE_TTL:
            return cached.get("val", 0.0)

    clients = _get_clients()
    if not clients or account_index < 0 or account_index >= len(clients):
        return 0.0

    client = clients[account_index]
    try:
        balances = client.balance()
    except Exception:
        return 0.0

    if not isinstance(balances, list):
        return 0.0

    for item in balances:
        if not isinstance(item, dict):
            continue
        if str(item.get("asset", "")).upper() != "USDT":
            continue
        try:
            val = float(item.get("balance", 0.0))
        except (TypeError, ValueError):
            val = 0.0
        _WALLET_BALANCE_CACHE[account_index] = {"ts": now, "val": val}
        return val

    return 0.0


def _get_account_balance_usdt(client: BinanceFuturesClient) -> float:
    """Return available USDT balance for BALANCE_PCT sizing."""

    try:
        balances = client.balance()
    except Exception:
        return 0.0

    if not isinstance(balances, list):
        return 0.0

    for item in balances:
        if not isinstance(item, dict):
            continue
        asset = item.get("asset")
        if str(asset).upper() != "USDT":
            continue
        raw_val = item.get("availableBalance")
        if raw_val is None:
            raw_val = item.get("balance")
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            return 0.0
        if val <= 0:
            return 0.0
        return val

    return 0.0


def _compute_account_fixed_notional(
    symbol: str,
    client: BinanceFuturesClient,
    settings: Dict[str, Any],
    account_index: Optional[int] = None,
) -> float:
    """Per-account notional in USDT from settings (USDT or BALANCE_PCT).

    BALANCE_PCT semantics:
    - Take X%% of current USDT balance (availableBalance/balance)
    - Multiply by account leverage override from binance_accounts.yaml (if set)
      so that effective notional ~= balance * (pct/100) * leverage.
    """

    if not isinstance(settings, dict):
        return 0.0

    mode_raw = settings.get("fixed_notional_type") or settings.get("fixed_notional_mode")
    mode = str(mode_raw).strip().upper() if mode_raw is not None else ""

    notional_val = 0.0

    if mode == "BALANCE_PCT":
        value = settings.get("fixed_notional_value")
        if value is None:
            return 0.0
        try:
            pct = float(value)
        except (TypeError, ValueError):
            return 0.0
        if pct <= 0:
            return 0.0
        if account_index is not None:
            try:
                idx_ad = int(account_index)
            except (TypeError, ValueError):
                idx_ad = -1
            if idx_ad >= 0:
                ap = _get_adaptive_pct(idx_ad, settings, fallback_base=pct)
                if ap is not None and ap > 0:
                    pct = float(ap)
        bal = _get_account_balance_usdt(client)
        if bal <= 0:
            return 0.0
        # Base notional from balance percent
        notional_val = bal * pct / 100.0

        # If this account has a configured leverage override, scale the
        # notional by that leverage so that BALANCE_PCT represents percent of
        # balance used as *margin* at the configured leverage.
        lev_mult = 1.0
        if account_index is not None:
            try:
                idx = int(account_index)
            except (TypeError, ValueError):
                idx = -1
            if 0 <= idx < len(_ACCOUNT_LEVERAGE_OVERRIDES):
                lev_override = _ACCOUNT_LEVERAGE_OVERRIDES[idx]
                try:
                    if lev_override is not None:
                        lev_val = float(lev_override)
                        if lev_val > 0:
                            lev_mult = lev_val
                except (TypeError, ValueError):
                    pass

        notional_val = notional_val * lev_mult
    else:
        value = settings.get("fixed_notional_value")
        raw_val = value
        if raw_val is None:
            raw_val = settings.get("fixed_notional_usd")
        if raw_val is None:
            return 0.0
        try:
            notional_val = float(raw_val)
        except (TypeError, ValueError):
            return 0.0

    if notional_val <= 0:
        return 0.0

    if symbol.upper().endswith("USDT") and notional_val < 5.5:
        notional_val = 5.5

    return notional_val


def _compute_multi_entry_level_notional(
    symbol: str,
    client: BinanceFuturesClient,
    lvl: Dict[str, Any],
    account_index: Optional[int] = None,
) -> float:
    """Per-level notional in USDT (USDT or BALANCE_PCT) for multi-entry.

    BALANCE_PCT semantics mirror _compute_account_fixed_notional: X%% of
    account USDT balance multiplied by that account's leverage override when
    configured.
    """

    if not isinstance(lvl, dict):
        return 0.0

    mode_raw = lvl.get("size_mode") or lvl.get("notional_mode")
    mode = str(mode_raw).strip().upper() if mode_raw is not None else ""

    notional_val = 0.0

    if mode == "BALANCE_PCT":
        pct_raw = lvl.get("size_value") or lvl.get("size_pct") or lvl.get("notional_pct")
        if pct_raw is None:
            return 0.0
        try:
            pct = float(pct_raw)
        except (TypeError, ValueError):
            return 0.0
        if pct <= 0:
            return 0.0
        bal = _get_account_balance_usdt(client)
        if bal <= 0:
            return 0.0
        notional_val = bal * pct / 100.0

        # Apply the same leverage-aware semantics as per-account sizing so
        # that BALANCE_PCT here also represents percent of balance used as
        # margin at the configured leverage.
        lev_mult = 1.0
        if account_index is not None:
            try:
                idx = int(account_index)
            except (TypeError, ValueError):
                idx = -1
            if 0 <= idx < len(_ACCOUNT_LEVERAGE_OVERRIDES):
                lev_override = _ACCOUNT_LEVERAGE_OVERRIDES[idx]
                try:
                    if lev_override is not None:
                        lev_val = float(lev_override)
                        if lev_val > 0:
                            lev_mult = lev_val
                except (TypeError, ValueError):
                    pass

        notional_val = notional_val * lev_mult
    else:
        raw_nt = lvl.get("size_value")
        if raw_nt is None:
            raw_nt = lvl.get("notional_usd")
        if raw_nt is None:
            return 0.0
        try:
            notional_val = float(raw_nt)
        except (TypeError, ValueError):
            return 0.0

    if notional_val <= 0:
        return 0.0

    if symbol.upper().endswith("USDT") and notional_val < 5.5:
        notional_val = 5.5

    return notional_val


def _get_account_margin_mode(index: int):
    if index < 0 or index >= len(_ACCOUNT_SETTINGS):
        return None
    settings = _ACCOUNT_SETTINGS[index] or {}
    mode = settings.get("margin_mode")
    if not mode:
        return None
    mode_str = str(mode).strip().lower()
    if mode_str in ("cross", "isolated"):
        return mode_str
    return None


def _ensure_margin_mode_for_account_symbol(
    client: BinanceFuturesClient, acc_index: int, symbol: str
):
    mode = _get_account_margin_mode(acc_index)
    if not mode:
        return
    margin_type = "CROSSED" if mode == "cross" else "ISOLATED"
    try:
        client.set_margin_type(symbol, margin_type)
    except Exception as e:
        key = (acc_index, symbol)
        if key in _MARGIN_MODE_LOGGED_ERRORS:
            return
        _MARGIN_MODE_LOGGED_ERRORS.add(key)
        # try:
        #     log(
        #         f"[MARGIN] Failed to set margin mode {mode} for {symbol} on account index {acc_index}: {e}"
        #     )
        # except Exception:
        #     pass


def _ensure_exchange_info():
    """Fetch and cache exchangeInfo once for symbol filters.

    Uses the first available client. exchangeInfo is a public endpoint,
    so any account (or even missing keys) is acceptable as long as
    HTTP requests work.
    """

    global _EXCHANGE_INFO_LOADED, _SYMBOL_FILTERS
    if _EXCHANGE_INFO_LOADED:
        return

    client = _get_client()
    info = client.exchange_info()
    symbols = info.get("symbols") or []

    parsed = {}
    for sym in symbols:
        name = sym.get("symbol")
        if not name:
            continue
        filters = sym.get("filters") or []

        min_qty = None
        step_size = None
        min_notional = None
        tick_size = None

        for f in filters:
            ftype = f.get("filterType")
            if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                try:
                    if "minQty" in f:
                        min_qty = float(f["minQty"])
                    if "stepSize" in f:
                        step_size = float(f["stepSize"])
                except (TypeError, ValueError):
                    pass
            elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                # Futures APIs sometimes use "notional" field name.
                try:
                    if "minNotional" in f:
                        min_notional = float(f["minNotional"])
                    elif "notional" in f:
                        min_notional = float(f["notional"])
                except (TypeError, ValueError):
                    pass
            elif ftype in ("PRICE_FILTER", "MARKET_PRICE_FILTER"):
                try:
                    if "tickSize" in f:
                        tick_size = float(f["tickSize"])
                except (TypeError, ValueError):
                    pass

        parsed[name] = {
            "min_qty": min_qty,
            "step_size": step_size,
            "min_notional": min_notional,
            "tick_size": tick_size,
        }

    _SYMBOL_FILTERS = parsed
    _EXCHANGE_INFO_LOADED = True


def _get_symbol_filters(symbol: str):
    """Return cached filters dict for a given symbol, or None."""

    if not _EXCHANGE_INFO_LOADED:
        _ensure_exchange_info()
    return _SYMBOL_FILTERS.get(symbol)


def _get_public_mark_price(client: BinanceFuturesClient, symbol: str) -> Optional[float]:
    try:
        data = client.mark_price(str(symbol).upper())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    mp = data.get("markPrice")
    try:
        v = float(mp)
    except Exception:
        return None
    if v <= 0:
        return None
    return v


def get_public_mark_price(symbol: str, account_index: int = 0) -> Optional[float]:
    clients = _get_clients()
    if not clients:
        return None
    try:
        idx = int(account_index)
    except Exception:
        idx = 0
    if idx < 0 or idx >= len(clients):
        idx = 0
    try:
        return _get_public_mark_price(clients[idx], symbol)
    except Exception:
        return None


def get_public_mark_prices(symbols, account_index: int = 0) -> Dict[str, float]:
    clients = _get_clients()
    if not clients:
        return {}

    try:
        idx = int(account_index)
    except Exception:
        idx = 0
    if idx < 0 or idx >= len(clients):
        idx = 0

    try:
        req_syms = [str(s).upper() for s in (symbols or []) if s]
    except Exception:
        req_syms = []
    if not req_syms:
        return {}
    req_set = set(req_syms)

    try:
        url = f"{BASE_URL}/fapi/v1/premiumIndex"
        proxies = _get_proxies_for_request(False)
        resp = requests.get(url, timeout=3, proxies=proxies)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    out: Dict[str, float] = {}

    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            sym = row.get("symbol")
            if not sym:
                continue
            try:
                sym_u = str(sym).upper()
            except Exception:
                continue
            if sym_u not in req_set:
                continue
            mp = row.get("markPrice")
            try:
                v = float(mp)
            except Exception:
                continue
            if v <= 0:
                continue
            out[sym_u] = v
        return out

    if isinstance(data, dict):
        sym = data.get("symbol")
        if sym:
            try:
                sym_u = str(sym).upper()
            except Exception:
                sym_u = None
            if sym_u and sym_u in req_set:
                mp = data.get("markPrice")
                try:
                    v = float(mp)
                except Exception:
                    v = None
                if v is not None and v > 0:
                    out[sym_u] = v
        return out

    return {}


def get_symbol_lot_constraints(symbol: str) -> tuple[float, float]:
    """Return (step_size, min_qty) for a symbol from cached exchange info."""

    try:
        filters = _get_symbol_filters(symbol)
        if not filters:
            return 0.0, 0.0
        step_size = float(filters.get("step_size") or 0.0)
        min_qty = float(filters.get("min_qty") or 0.0)
        return step_size, min_qty
    except Exception:
        return 0.0, 0.0


def adjust_price(symbol: str, price: float) -> float:
    """Adjust price to satisfy Binance PRICE_FILTER tick size.

    Rounds down to the nearest tickSize multiple. If no filter is known,
    returns the price as-is.
    """

    try:
        p = float(price)
    except (TypeError, ValueError):
        return 0.0

    if p <= 0:
        return 0.0

    filters = _get_symbol_filters(symbol)
    if not filters:
        return p

    tick = filters.get("tick_size") or 0.0
    if tick and tick > 0:
        # Compute decimal places from tick_size to fix floating-point representation.
        # e.g. tick=0.001 → 3 decimals; needed because 114*0.001=0.11399999... in IEEE754,
        # which Binance rejects with -1111 "Precision is over the maximum defined".
        try:
            _tick_s = f"{tick:.10f}".rstrip("0")
            _tick_decimals = len(_tick_s.split(".")[1]) if "." in _tick_s else 0
        except Exception:
            _tick_decimals = 8
        if _CPP_AVAILABLE:
            try:
                p = _cpp.adjust_price(p, tick)
                return round(p, _tick_decimals)
            except Exception:
                pass
        p = math.floor(p / tick) * tick
        p = round(p, _tick_decimals)

    return p


def adjust_quantity(symbol: str, requested_qty: float, price: float) -> float:
    """Adjust quantity to satisfy Binance LOT_SIZE and MIN_NOTIONAL filters.

    - Rounds down by stepSize
    - Ensures qty >= minQty
    - Ensures qty * price >= minNotional (if defined)

    Returns 0.0 if a valid quantity cannot be constructed.
    """

    try:
        qty = float(requested_qty)
    except (TypeError, ValueError):
        return 0.0

    if qty <= 0 or price is None or price <= 0:
        return 0.0

    sym_u = str(symbol).upper()
    hard_min_notional = 0.0
    if sym_u.endswith("USDT"):
        hard_min_notional = 5.0

    filters = _get_symbol_filters(symbol)

    if _CPP_AVAILABLE:
        try:
            step_size = float((filters or {}).get("step_size") or 0.0)
            min_qty_f = float((filters or {}).get("min_qty") or 0.0)
            min_notional_f = float((filters or {}).get("min_notional") or 0.0)
            return _cpp.adjust_quantity(qty, price, step_size, min_qty_f, min_notional_f, hard_min_notional)
        except Exception:
            pass

    if not filters:
        if hard_min_notional > 0:
            try:
                req = hard_min_notional / float(price)
            except Exception:
                req = 0.0
            if req > 0 and qty < req:
                qty = req
        if qty <= 0:
            return 0.0
        try:
            qty = math.ceil(qty * 1e8) / 1e8
        except Exception:
            pass
        return float(f"{qty:.8f}")

    step_size = filters.get("step_size") or 0.0
    min_qty = filters.get("min_qty") or 0.0
    min_notional = filters.get("min_notional") or 0.0

    # Binance USDT futures pairs գրեթե միշտ ունեն առնվազն 5 USDT
    # minimum notional. Եթե exchangeInfo-ից եկող min_notional-ը
    # բացակայում է կամ անհավանականորեն փոքր է, օգտագործում ենք
    # 5.0 որպես ստորին սահման, որպեսզի խուսափենք -4164 error-ից։
    if hard_min_notional > 0 and (not min_notional or min_notional < hard_min_notional):
        min_notional = hard_min_notional

    # 1) Apply stepSize (floor toward zero)
    if step_size > 0:
        qty = math.floor(qty / step_size) * step_size

    # 2) Ensure >= minQty
    if min_qty > 0 and qty < min_qty:
        qty = min_qty

    # 3) Ensure notional >= minNotional
    if min_notional > 0:
        notional = qty * price
        if notional < min_notional:
            required_qty = min_notional / price
            if step_size > 0:
                qty = math.ceil(required_qty / step_size) * step_size
            else:
                qty = required_qty

    if qty <= 0:
        return 0.0

    # Round to a reasonable precision to avoid float noise
    return float(f"{qty:.8f}")


def adjust_quantity_up(symbol: str, requested_qty: float, price: float) -> float:
    try:
        qty = float(requested_qty)
    except (TypeError, ValueError):
        return 0.0

    if qty <= 0 or price is None or price <= 0:
        return 0.0

    sym_u = str(symbol).upper()
    hard_min_notional = 0.0
    if sym_u.endswith("USDT"):
        hard_min_notional = 5.0

    filters = _get_symbol_filters(symbol)
    if not filters:
        if hard_min_notional > 0:
            try:
                req = hard_min_notional / float(price)
            except Exception:
                req = 0.0
            if req > 0 and qty < req:
                qty = req
        if qty <= 0:
            return 0.0
        try:
            qty = math.ceil(qty * 1e8) / 1e8
        except Exception:
            pass
        return float(f"{qty:.8f}")

    step_size = filters.get("step_size") or 0.0
    min_qty = filters.get("min_qty") or 0.0
    min_notional = filters.get("min_notional") or 0.0

    if hard_min_notional > 0 and (not min_notional or min_notional < hard_min_notional):
        min_notional = hard_min_notional

    if step_size > 0:
        steps = int(qty / step_size)
        if float(steps) * step_size < qty:
            steps += 1
        qty = float(steps) * step_size

    if min_qty > 0 and qty < min_qty:
        qty = min_qty
        if step_size > 0:
            steps = int(qty / step_size)
            if float(steps) * step_size < qty:
                steps += 1
            qty = float(steps) * step_size

    if min_notional > 0:
        notional = qty * price
        if notional < min_notional:
            required_qty = min_notional / price
            if step_size > 0:
                steps = int(required_qty / step_size)
                if float(steps) * step_size < required_qty:
                    steps += 1
                qty = float(steps) * step_size
            else:
                qty = required_qty

    if qty <= 0:
        return 0.0

    return float(f"{qty:.8f}")


def _get_symbol_fee_rate(client: Any, symbol: str, order_type: str) -> float:
    sym_u = str(symbol).upper()
    now = time.time()

    maker = 0.0
    taker = 0.0

    with _COMMISSION_RATE_CACHE_LOCK:
        cached = _COMMISSION_RATE_CACHE.get(sym_u)
        if cached and (now - float(cached.get("ts", 0.0))) < float(_COMMISSION_RATE_CACHE_TTL_SEC):
            maker = float(cached.get("maker", 0.0))
            taker = float(cached.get("taker", 0.0))
        else:
            try:
                data = client.commission_rate(sym_u)
                if isinstance(data, dict):
                    maker = float(data.get("makerCommissionRate") or 0.0)
                    taker = float(data.get("takerCommissionRate") or 0.0)
            except Exception:
                maker = 0.0
                taker = 0.0
            _COMMISSION_RATE_CACHE[sym_u] = {"ts": float(now), "maker": float(maker), "taker": float(taker)}

    ot = str(order_type).upper()
    if ot == "LIMIT":
        return float(maker if maker > 0 else taker)
    return float(taker if taker > 0 else maker)


def adjust_close_quantity(symbol: str, requested_qty: float, round_up: bool = False) -> float:
    """Adjust close quantity to satisfy LOT_SIZE precision without minNotional.

    Used for reduce-only market exits (SL/TP). We only care about stepSize
    and minQty to avoid -1111 precision errors, and do not enforce
    MIN_NOTIONAL because the position already exists on the exchange.
    """

    try:
        qty = float(requested_qty)
    except (TypeError, ValueError):
        return 0.0

    if qty <= 0:
        return 0.0

    filters = _get_symbol_filters(symbol)
    if not filters:
        return max(0.0, qty)

    step_size = filters.get("step_size") or 0.0
    min_qty = filters.get("min_qty") or 0.0

    if _CPP_AVAILABLE:
        try:
            return _cpp.adjust_close_quantity(qty, step_size, min_qty, round_up)
        except Exception:
            pass

    # Apply stepSize
    if step_size > 0:
        if round_up:
            steps = int(qty / step_size)
            if steps * step_size < qty:
                steps += 1
            qty = float(steps) * step_size
        else:
            qty = math.floor(qty / step_size) * step_size

    # Ensure >= minQty if defined (best-effort; reduce-only will clamp to
    # current position size if this slightly overshoots).
    if min_qty > 0 and qty < min_qty:
        qty = min_qty

    if qty <= 0:
        return 0.0

    return float(f"{qty:.8f}")


def is_symbol_blacklisted(symbol: str) -> bool:
    """Return True if symbol previously triggered a 451/403 region error.

    When True, higher-level trading logic should avoid sending further
    orders for this symbol and simply skip to the next one.
    """

    try:
        return str(symbol).upper() in _SYMBOL_451_BLACKLIST
    except Exception:
        return False


def is_symbol_trade_blocked(symbol: str) -> bool:
    """Return True when symbol is blocked by operator or exchange blacklist."""
    try:
        if is_symbol_blocked(symbol):
            return True
    except Exception:
        pass
    return is_symbol_blacklisted(symbol)


def _repeat_enabled(account_index: int) -> bool:
    try:
        settings = _get_account_settings(int(account_index))
    except Exception:
        settings = {}
    if not isinstance(settings, dict):
        return False
    return bool(settings.get("repeat_signal_enabled"))


def _repeat_get_open_position_entry_price(client: Any, symbol: str, side: str) -> Optional[float]:
    try:
        positions = client.position_risk()
    except Exception:
        return None
    if not isinstance(positions, list):
        return None
    sym_u = str(symbol).upper()
    side_u = str(side).upper()
    for pos in positions:
        try:
            if str(pos.get("symbol") or "").upper() != sym_u:
                continue
            amt = float(pos.get("positionAmt"))
        except Exception:
            continue
        if not amt:
            continue
        if side_u == "BUY" and amt <= 0:
            continue
        if side_u == "SELL" and amt >= 0:
            continue
        try:
            ep = float(pos.get("entryPrice"))
        except Exception:
            ep = None
        if ep is not None and ep > 0:
            return ep
    return None


def _repeat_get_open_position_meta(client: Any, symbol: str, side: str) -> Tuple[float, float]:
    """Return (entry_price, abs_position_qty) for the matching side."""
    try:
        positions = client.position_risk()
    except Exception:
        return 0.0, 0.0
    if not isinstance(positions, list):
        return 0.0, 0.0
    sym_u = str(symbol).upper()
    side_u = str(side).upper()
    for pos in positions:
        try:
            if str(pos.get("symbol") or "").upper() != sym_u:
                continue
            amt = float(pos.get("positionAmt"))
        except Exception:
            continue
        if amt == 0.0:
            continue
        if side_u == "BUY" and amt <= 0:
            continue
        if side_u == "SELL" and amt >= 0:
            continue
        try:
            ep = float(pos.get("entryPrice"))
        except Exception:
            ep = 0.0
        return max(0.0, ep), abs(float(amt))
    return 0.0, 0.0


def _place_entry_protection_fast(
    client: Any,
    account_index: int,
    symbol: str,
    side: str,
    settings: Dict[str, Any],
    entry_price_hint: Optional[float] = None,
) -> None:
    """Fast-path initial SL/TP placement right after entry fill.

    Uses the same Binance client request path (C++ accelerated when available)
    and is designed as best-effort: failures are logged but never fail entry.
    """
    try:
        enabled_raw = settings.get("entry_fast_protection_enabled")
        enabled = True if enabled_raw is None else bool(enabled_raw)
    except Exception:
        enabled = True
    if not enabled:
        return

    side_u = str(side).upper()
    if side_u not in ("BUY", "SELL"):
        return
    close_side = "SELL" if side_u == "BUY" else "BUY"
    long_side = side_u == "BUY"

    entry_px = 0.0
    pos_qty = 0.0
    for _ in range(5):
        ep_now, q_now = _repeat_get_open_position_meta(client, symbol, side_u)
        if ep_now > 0 and q_now > 0:
            entry_px, pos_qty = ep_now, q_now
            break
        time.sleep(0.15)
    if entry_px <= 0 and entry_price_hint is not None:
        try:
            entry_px = float(entry_price_hint)
        except Exception:
            entry_px = 0.0
    if entry_px <= 0 or pos_qty <= 0:
        return

    sl_pct = 0.0
    try:
        sl_pct = float(settings.get("sl_pct") or 0.0)
    except Exception:
        sl_pct = 0.0
    if sl_pct <= 0:
        sl_pct = 2.0

    tp_pcts: List[float] = []
    raw_tp = settings.get("tp_pcts")
    if isinstance(raw_tp, (list, tuple)):
        for v in raw_tp:
            try:
                pv = float(v)
                if pv > 0:
                    tp_pcts.append(pv)
            except Exception:
                continue
    if not tp_pcts:
        tp_pcts = [1.0]

    try:
        tp_mode = int(settings.get("tp_mode", 1))
    except Exception:
        tp_mode = 1
    tp_idx = 0
    if tp_mode == 2 and len(tp_pcts) >= 2:
        tp_idx = 1
    elif tp_mode == 3 and len(tp_pcts) >= 3:
        tp_idx = 2
    tp_pct = float(tp_pcts[tp_idx] if 0 <= tp_idx < len(tp_pcts) else tp_pcts[0])

    sl_price_raw = entry_px * (1.0 - sl_pct / 100.0) if long_side else entry_px * (1.0 + sl_pct / 100.0)
    tp_price_raw = entry_px * (1.0 + tp_pct / 100.0) if long_side else entry_px * (1.0 - tp_pct / 100.0)
    sl_price = adjust_price(symbol, sl_price_raw)
    tp_price = adjust_price(symbol, tp_price_raw)
    tp_qty = adjust_close_quantity(symbol, pos_qty)
    if tp_qty <= 0:
        return

    has_sl = False
    has_tp = False
    try:
        orders = client.open_orders(symbol=str(symbol).upper())
    except Exception:
        orders = []
    if isinstance(orders, list):
        for o in orders:
            try:
                ot = str(o.get("type") or "")
                oside = str(o.get("side") or "")
                if oside != close_side:
                    continue
                if ot in ("STOP_MARKET", "STOP"):
                    has_sl = True
                elif ot in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                    has_tp = True
            except Exception:
                continue
    try:
        algo_orders = client.get_algo_open_orders(symbol=str(symbol).upper())
    except Exception:
        algo_orders = []
    if isinstance(algo_orders, list):
        for o in algo_orders:
            try:
                ot = str(o.get("type") or "")
                oside = str(o.get("side") or "")
                if oside != close_side:
                    continue
                if ot in ("STOP_MARKET", "STOP"):
                    has_sl = True
                elif ot in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                    has_tp = True
            except Exception:
                continue

    def _place_conditional(order_type: str, stop_px: float, qty: Optional[float], close_pos: bool) -> None:
        try:
            client.place_order(
                symbol=symbol,
                side=close_side,
                order_type=order_type,
                stop_price=stop_px,
                quantity=qty,
                close_position=close_pos,
                reduce_only=(not close_pos),
            )
            return
        except Exception as e1:
            e1s = str(e1)
            if '"code":-4120' not in e1s and '"code":-4130' not in e1s:
                raise
        # Binance endpoint requires algo route for this symbol/order type.
        client.place_algo_order(
            symbol=symbol,
            side=close_side,
            order_type=order_type,
            trigger_price=stop_px,
            quantity=qty,
            close_position=close_pos,
            reduce_only=(not close_pos),
            working_type="MARK_PRICE",
        )

    sl_placed = False
    try:
        if not has_sl:
            _place_conditional("STOP_MARKET", sl_price, None, True)
            sl_placed = True
        if not has_tp:
            # Always wait 500 ms before TP — whether SL was just placed or
            # was already present.  Binance rejects back-to-back conditional
            # orders on the same symbol when sent too quickly.
            time.sleep(0.5)
            _place_conditional("TAKE_PROFIT_MARKET", tp_price, tp_qty, False)
        log(
            f"[ENTRY][FAST-PROTECT] account={int(account_index)} symbol={symbol} "
            f"sl_set={not has_sl} tp_set={not has_tp} entry={entry_px:.8f} tp={tp_price:.8f} sl={sl_price:.8f}"
        )
    except Exception as e:
        log(
            f"[ENTRY][FAST-PROTECT][WARN] account={int(account_index)} symbol={symbol} err={e}"
        )


def place_order(
        symbol,
        side,
        quantity=None,
        order_type="MARKET",
        reduce_only=False,
        price=None,
        stop_price=None,
        close_position=None,
        time_in_force=None,
        entry_price=None,
        confidence: Optional[float] = None,
        multi_entry_level_index: Optional[int] = None,
        target_account_index: Optional[int] = None,
        signal_ts: Optional[float] = None,
        signal_entry_price: Optional[float] = None,
        market_price: Optional[float] = None,
        signal_profile: Optional[str] = None,
        caller: str = "UNKNOWN",
    ):
    """Place an order using one or multiple Binance accounts.

    Behaviour:
    - If config/binance_accounts.yaml exists and mode == "multi", the same
      order is sent to all configured accounts and a list of responses is
      returned.
    - Otherwise, a single primary account is used and its response is
      returned (backward compatible with previous behaviour).
    """

    if not _execution_allowed(caller, "place_order"):
        return None

    clients = _get_clients()
    if not clients:
        raise RuntimeError("No Binance clients available")

    # Optional per-call restriction: when target_account_index is provided in
    # multi-account mode, only that specific account index will be used for
    # this order. This allows the dynamic entry manager to trigger different
    # accounts at different prices.
    target_idx: Optional[int]
    if target_account_index is not None:
        try:
            target_idx = int(target_account_index)
        except (TypeError, ValueError):
            target_idx = -1
    else:
        target_idx = None

    # If this symbol is blocked (manual blocklist OR MQA SUSPECT/MANIPULATED),
    # reject new entry orders but still allow reduce-only close orders so existing
    # positions can always be exited.
    if is_symbol_blocked(symbol):
        if reduce_only or close_position:
            # Allow closing/reducing an existing position even on a blocked symbol
            pass
        else:
            import sys as _sys
            print(
                f"[ORDER][BLOCKED] {symbol} is blocked (MQA SUSPECT/MANIPULATED or manual blocklist)"
                f" — entry order rejected (caller={caller})",
                file=_sys.stderr,
                flush=True,
            )
            return None

    if is_symbol_blacklisted(symbol):
        raise Exception(
            f"Symbol {symbol} is blacklisted due to previous 451/403 Binance errors; skipping order."
        )

    if _ACCOUNTS_MODE == "multi" and len(clients) > 1:
        results = []
        last_err = None
        any_skipped = False
        for idx, client in enumerate(clients):
            if target_idx is not None and target_idx >= 0 and idx != target_idx:
                continue
            if idx < len(_ACCOUNT_TRADE_ENABLED) and not _ACCOUNT_TRADE_ENABLED[idx]:
                try:
                    acc_name = (
                        _ACCOUNT_NAMES[idx]
                        if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                        else f"Account {idx}"
                    )
                except Exception:
                    acc_name = f"Account {idx}"
                log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} trade_enabled=false")
                any_skipped = True
                continue
            try:
                settings_for_profile = _get_account_settings(idx)
            except Exception:
                settings_for_profile = {}
            if not reduce_only and not _account_accepts_signal_profile(settings_for_profile, signal_profile):
                any_skipped = True
                continue
            if not reduce_only:
                try:
                    if not _is_global_drawdown_ok(idx, client):
                        try:
                            acc_name = (
                                _ACCOUNT_NAMES[idx]
                                if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                                else f"Account {idx}"
                            )
                        except Exception:
                            acc_name = f"Account {idx}"
                        log(f"[DD][PAUSE] Skip entry for {symbol} on {acc_name}: paused by drawdown")
                        any_skipped = True
                        continue
                    
                except Exception:
                    pass
                gate_ok, gate_reason = _is_entry_gate_ok_for_account(idx)
                if not gate_ok:
                    try:
                        acc_name = (
                            _ACCOUNT_NAMES[idx]
                            if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                            else f"Account {idx}"
                        )
                    except Exception:
                        acc_name = f"Account {idx}"
                    if gate_reason == "time_window":
                        log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} blocked by time_window")
                    else:
                        log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} blocked by {gate_reason}")
                    any_skipped = True
                    if target_idx is not None and int(idx) == int(target_idx):
                        raise RuntimeError(f"{acc_name} blocked by {gate_reason}")
                    continue
                if not _is_daily_drawdown_ok(idx, client):
                    try:
                        acc_name = (
                            _ACCOUNT_NAMES[idx]
                            if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                            else f"Account {idx}"
                        )
                    except Exception:
                        acc_name = f"Account {idx}"
                    log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} blocked by daily_drawdown_limit")
                    any_skipped = True
                    continue
                if not _is_delay_limit_ok(idx):
                    try:
                        acc_name = (
                            _ACCOUNT_NAMES[idx]
                            if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                            else f"Account {idx}"
                        )
                    except Exception:
                        acc_name = f"Account {idx}"
                    log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} blocked by delay_limit")
                    any_skipped = True
                    continue
                if not _is_symbol_loss_cooldown_ok(idx, symbol):
                    try:
                        acc_name = (
                            _ACCOUNT_NAMES[idx]
                            if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                            else f"Account {idx}"
                        )
                    except Exception:
                        acc_name = f"Account {idx}"
                    log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} blocked by loss_cooldown")
                    any_skipped = True
                    continue

            try:
                settings = _get_account_settings(idx)
            except Exception:
                settings = {}
            multi_entry_enabled = bool(settings.get("multi_entry_enabled"))
            multi_entry_levels = settings.get("multi_entry_levels") or []
            apply_repeat = False
            if not reduce_only and _repeat_enabled(idx):
                if multi_entry_enabled:
                    if multi_entry_level_index is None:
                        apply_repeat = True
                    else:
                        try:
                            apply_repeat = int(multi_entry_level_index) == 0
                        except Exception:
                            apply_repeat = False
                else:
                    if multi_entry_level_index is None:
                        apply_repeat = True
                    else:
                        try:
                            apply_repeat = int(multi_entry_level_index) == 0
                        except Exception:
                            apply_repeat = False

            # Per-account confidence threshold: եթե կոնֆիգում սահմանված է
            # շեմ, ապա այն account-ի համար trade-ը կբացվի միայն այն ժամանակ,
            # երբ model-ի confidence-ը **մեծ կամ հավասար է** threshold-ին.
            # Այսինքն՝ confidence < threshold  → skip account.
            if not reduce_only and confidence is not None:
                try:
                    conf_thr = _get_account_confidence_threshold(idx)
                except Exception:
                    conf_thr = None
                if conf_thr is not None:
                    try:
                        conf_val = float(confidence)
                        thr_val = float(conf_thr)
                    except Exception:
                        conf_val = None
                        thr_val = None
                    eps = 1e-9
                    if conf_val is not None and thr_val is not None and (conf_val + eps) < thr_val:
                        try:
                            acc_name = (
                                _ACCOUNT_NAMES[idx]
                                if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                                else f"Account {idx}"
                            )
                        except Exception:
                            acc_name = f"Account {idx}"
                        try:
                            log(
                                f"[ORDER][SKIP] {symbol} {side}: {acc_name} confidence {conf_val:.6f} < threshold {thr_val:.6f}"
                            )
                        except Exception:
                            log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} confidence below threshold")
                        any_skipped = True
                        continue

            if not reduce_only and apply_repeat:
                pos_ep = _repeat_get_open_position_entry_price(client, symbol, side)
                if pos_ep is not None and pos_ep > 0:
                    try:
                        mp = float(market_price) if market_price is not None else None
                    except (TypeError, ValueError):
                        mp = None
                    if mp is None or mp <= 0:
                        try:
                            mp = float(entry_price) if entry_price is not None else None
                        except (TypeError, ValueError):
                            mp = None
                    if mp is None or mp <= 0:
                        continue
                    side_u = str(side).upper()
                    if side_u == "SELL":
                        if not (mp > pos_ep):
                            continue
                    else:
                        if not (mp < pos_ep):
                            continue
            _ensure_margin_mode_for_account_symbol(client, idx, symbol)

            # If this call comes from a specific multi-entry ladder level
            # (multi_entry_level_index is not None) but this particular
            # account does NOT have multi_entry_enabled=true, then we do not
            # want to open a separate order for every ladder level on this
            # account. Instead, non-multi-entry accounts should participate at
            # most once (on level 0). For higher levels, skip this account
            # entirely so that the additional scaling applies only to
            # accounts configured with multi_entry_enabled=true.
            if (
                not reduce_only
                and multi_entry_level_index is not None
                and not multi_entry_enabled
                and multi_entry_level_index != 0
            ):
                continue

            # Reverse signal: if account has reverse_signal=true, flip BUY<->SELL
            _eff_side = side
            if not reduce_only:
                try:
                    _acct_settings = _get_account_settings(idx)
                    if _acct_settings.get("reverse_signal"):
                        _eff_side = "SELL" if str(side).upper() == "BUY" else "BUY"
                except Exception:
                    pass

            used_multi_entry = False
            if (
                not reduce_only
                and multi_entry_enabled
                and multi_entry_levels
                and entry_price is not None
            ):
                lev_scale = _get_notional_scale_for_leverage(idx, symbol)
                try:
                    ep_val = float(entry_price)
                except (TypeError, ValueError):
                    ep_val = 0.0

                indices: List[int] = []
                if ep_val > 0:
                    if multi_entry_level_index is None:
                        indices = list(range(len(multi_entry_levels)))
                    else:
                        try:
                            lvl_idx = int(multi_entry_level_index)
                        except (TypeError, ValueError):
                            lvl_idx = -1
                        if 0 <= lvl_idx < len(multi_entry_levels):
                            indices = [lvl_idx]

                if indices:
                    used_multi_entry = True
                    for lvl_idx in indices:
                        lvl = multi_entry_levels[lvl_idx]
                        if not isinstance(lvl, dict):
                            continue

                        try:
                            notional_val = _compute_multi_entry_level_notional(
                                symbol, client, lvl, account_index=idx
                            )
                        except Exception:
                            continue
                        if notional_val <= 0:
                            continue

                        try:
                            if lev_scale != 1.0:
                                notional_val = float(notional_val) * float(lev_scale)
                        except Exception:
                            pass

                        try:
                            target_qty = notional_val / ep_val
                            lvl_qty = adjust_quantity(symbol, target_qty, ep_val)
                        except Exception:
                            continue

                        if not lvl_qty or lvl_qty <= 0:
                            continue

                        try:
                            # 500 ms between every DCA level — Binance rejects
                            # rapid back-to-back conditional orders on the same symbol.
                            time.sleep(0.5)
                            res = client.place_order(
                                symbol=symbol,
                                side=_eff_side,
                                quantity=lvl_qty,
                                order_type=order_type,
                                reduce_only=False,
                                price=price,
                                stop_price=stop_price,
                                close_position=close_position,
                                time_in_force=time_in_force,
                            )
                            results.append(res)
                            try:
                                log_trade_entry(
                                    account_index=idx,
                                    symbol=symbol,
                                    side=side,
                                    qty=lvl_qty,
                                    entry_price=entry_price,
                                )
                            except Exception:
                                pass
                        except Exception as e:
                            last_err = e
                            try:
                                log(
                                    f"[ORDER][MULTI] Failed to place multi-entry order for {symbol} on account index {idx}: {e}"
                                )
                            except Exception:
                                pass
                            continue

            if used_multi_entry:
                # Skip standard single-order/Fibonacci path for this account.
                continue

            # Per-account quantity: first try fixed-notional sizing (USDT կամ
            # բալանսի %), ապա fallback Fibonacci staking-ին, եթե միացված է։
            eff_qty = quantity
            room_used = False
            room_used_idx: Optional[int] = None
            slot_used_idx: Optional[int] = None
            room_settings = settings if isinstance(settings, dict) else {}
            if not reduce_only and entry_price is not None:
                try:
                    ep_val = float(entry_price)
                except (TypeError, ValueError):
                    ep_val = 0.0

                if ep_val > 0 and _room_sizing_enabled_for_settings(room_settings):
                    if _symbol_has_open_position(client, symbol):
                        try:
                            acc_name = (
                                _ACCOUNT_NAMES[idx]
                                if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                                else f"Account {idx}"
                            )
                        except Exception:
                            acc_name = f"Account {idx}"
                        if target_idx is not None and idx == target_idx:
                            raise RuntimeError(f"[ROOM] SKIP {symbol}: open position exists")
                        log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} room_sizing open position exists")
                        any_skipped = True
                        continue

                    ok, notional_val, room_i, slot_i, reason = room_sizing_reserve_slot(
                        idx, symbol, side, room_settings
                    )
                    if ok and notional_val and notional_val > 0 and room_i is not None and slot_i is not None:
                        try:
                            px_room = ep_val
                            try:
                                if str(order_type).upper() == "MARKET":
                                    if market_price is not None and float(market_price) > 0:
                                        px_room = float(market_price)
                                    else:
                                        mp = _get_public_mark_price(client, symbol)
                                        if mp is not None and float(mp) > 0:
                                            px_room = float(mp)
                            except Exception:
                                px_room = ep_val

                            fee_rate = 0.0
                            try:
                                fee_rate = float(_get_symbol_fee_rate(client, symbol, order_type))
                            except Exception:
                                fee_rate = 0.0
                            if fee_rate < 0:
                                fee_rate = 0.0

                            lev_scale = _get_notional_scale_for_leverage(idx, symbol)
                            target_notional = float(notional_val) * (1.0 + fee_rate)
                            try:
                                if lev_scale != 1.0:
                                    target_notional = float(target_notional) * float(lev_scale)
                            except Exception:
                                pass
                            target_qty = float(target_notional) / float(px_room)
                            cand_qty = adjust_quantity_up(symbol, target_qty, float(px_room))
                            if cand_qty and cand_qty > 0:
                                eff_qty = cand_qty
                                room_used = True
                                room_used_idx = int(room_i)
                                slot_used_idx = int(slot_i)
                                log(
                                    f"[ROOM] Account {idx} room={int(room_i)+1} slot={int(slot_i)+1} "
                                    f"symbol={symbol} notional={float(notional_val):.4f} qty={cand_qty}"
                                )
                        except Exception as e_room_qty:
                            try:
                                room_sizing_release_slot(idx, room_settings, int(room_i), int(slot_i))
                            except Exception:
                                pass
                            if target_idx is not None and idx == target_idx:
                                raise RuntimeError(f"[ROOM] Failed to size {symbol}: {e_room_qty}")
                            continue
                    else:
                        try:
                            acc_name = (
                                _ACCOUNT_NAMES[idx]
                                if 0 <= int(idx) < len(_ACCOUNT_NAMES)
                                else f"Account {idx}"
                            )
                        except Exception:
                            acc_name = f"Account {idx}"
                        if target_idx is not None and idx == target_idx:
                            raise RuntimeError(f"[ROOM] SKIP {symbol}: {reason}")
                        try:
                            log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} room_sizing reserve_slot failed: {reason}")
                        except Exception:
                            log(f"[ORDER][SKIP] {symbol} {side}: {acc_name} room_sizing reserve_slot failed")
                        any_skipped = True
                        continue

                # IMPORTANT: if we used room sizing, eff_qty must remain
                # authoritative. Do NOT override it with fixed-notional or
                # Fibonacci sizing for this account.
                if not room_used:
                    account_notional = 0.0
                    if ep_val > 0:
                        try:
                            account_notional = _compute_account_fixed_notional(
                                symbol, client, settings, account_index=idx
                            )
                        except Exception:
                            account_notional = 0.0

                    if account_notional and account_notional > 0 and ep_val > 0:
                        try:
                            lev_scale = _get_notional_scale_for_leverage(idx, symbol)
                            try:
                                if lev_scale != 1.0:
                                    account_notional = float(account_notional) * float(lev_scale)
                            except Exception:
                                pass
                            target_qty = account_notional / ep_val
                            cand_qty = adjust_quantity(symbol, target_qty, ep_val)
                            if cand_qty and cand_qty > 0:
                                eff_qty = cand_qty
                        except Exception:
                            pass
                    else:
                        try:
                            fibo_notional = _get_fibo_notional_for_account(idx, entry_price)
                        except Exception:
                            fibo_notional = 0.0
                        if fibo_notional and fibo_notional > 0 and ep_val > 0:
                            try:
                                lev_scale = _get_notional_scale_for_leverage(idx, symbol)
                                try:
                                    if lev_scale != 1.0:
                                        fibo_notional = float(fibo_notional) * float(lev_scale)
                                except Exception:
                                    pass
                                target_qty = fibo_notional / ep_val
                                cand_qty = adjust_quantity(symbol, target_qty, ep_val)
                                if cand_qty and cand_qty > 0:
                                    eff_qty = cand_qty
                            except Exception:
                                pass

            try:
                adj_qty = eff_qty
                if not reduce_only and (not room_used) and eff_qty is not None:
                    px = None
                    try:
                        if str(order_type).upper() == "MARKET":
                            if market_price is not None:
                                px = float(market_price)
                            if px is None or px <= 0:
                                px = _get_public_mark_price(client, symbol)
                            if (px is None or px <= 0) and entry_price is not None:
                                px = float(entry_price)
                        else:
                            if price is not None:
                                px = float(price)
                            if (px is None or px <= 0) and entry_price is not None:
                                px = float(entry_price)
                            if px is None or px <= 0:
                                px = _get_public_mark_price(client, symbol)
                    except Exception:
                        px = None

                    if px is not None and px > 0:
                        try:
                            adj_qty = adjust_quantity(symbol, float(eff_qty), float(px))
                        except Exception:
                            adj_qty = eff_qty

                res = client.place_order(
                    symbol=symbol,
                    side=_eff_side,
                    quantity=adj_qty,
                    order_type=order_type,
                    reduce_only=reduce_only,
                    price=price,
                    stop_price=stop_price,
                    close_position=close_position,
                    time_in_force=time_in_force,
                )
                results.append(res)
                if not reduce_only:
                    try:
                        log_trade_entry(
                            account_index=idx,
                            symbol=symbol,
                            side=side,
                            qty=adj_qty if adj_qty is not None else 0.0,
                            entry_price=entry_price,
                        )
                    except Exception:
                        pass
                    try:
                        _place_entry_protection_fast(
                            client=client,
                            account_index=int(idx),
                            symbol=symbol,
                            side=side,
                            settings=settings if isinstance(settings, dict) else {},
                            entry_price_hint=entry_price,
                        )
                    except Exception:
                        pass

                if (
                    not reduce_only
                    and room_used
                    and room_used_idx is not None
                    and slot_used_idx is not None
                ):
                    try:
                        room_sizing_commit_open(idx, room_settings, room_used_idx, slot_used_idx, res)
                    except Exception:
                        pass

            except Exception as e:  # best-effort: skip failed accounts
                last_err = e
                try:
                    log(
                        f"[ORDER][MULTI] Failed to place order for {symbol} on account index {idx}: {e}"
                    )
                except Exception:
                    pass
                if (
                    not reduce_only
                    and room_used
                    and room_used_idx is not None
                    and slot_used_idx is not None
                ):
                    try:
                        room_sizing_release_slot(idx, room_settings, room_used_idx, slot_used_idx)
                    except Exception:
                        pass
                continue

        # If at least one account succeeded, treat the overall call as success.
        if results:
            return results

        # All accounts failed
        if last_err is not None:
            raise last_err
        if any_skipped:
            raise RuntimeError(
                f"All accounts skipped placing {side} order for {symbol} due to gating checks; see [ORDER][SKIP] logs"
            )
        raise RuntimeError("Failed to place order on all Binance accounts")

    # Single-account mode
    client = clients[0]
    if not reduce_only:
        try:
            settings_0 = _get_account_settings(0)
        except Exception:
            settings_0 = {}
        if not _account_accepts_signal_profile(settings_0, signal_profile):
            raise RuntimeError(f"Trading disabled by signal_source routing for account 0 (profile={signal_profile})")

    if not reduce_only:
        if not _is_global_drawdown_ok(0, client):
            raise RuntimeError("Trading disabled: account 0 paused by global drawdown limit")
        gate_ok_0, gate_reason_0 = _is_entry_gate_ok_for_account(0)
        if not gate_ok_0:
            if gate_reason_0 == "time_window":
                raise RuntimeError("Trading disabled by time window for account 0")
            raise RuntimeError(f"Trading disabled by {gate_reason_0} for account 0")
        if not _is_daily_drawdown_ok(0, client):
            raise RuntimeError("Trading disabled by daily drawdown limit for account 0")
        if not _is_delay_limit_ok(0):
            raise RuntimeError("Trading disabled by delay limit for account 0")
        if not _is_symbol_loss_cooldown_ok(0, symbol):
            raise RuntimeError(f"Trading disabled by loss cooldown for {symbol} on account 0")

    apply_repeat = False
    try:
        settings = _get_account_settings(0)
    except Exception:
        settings = {}
    multi_entry_enabled = bool(settings.get("multi_entry_enabled"))

    if not reduce_only and _repeat_enabled(0):
        if multi_entry_enabled:
            if multi_entry_level_index is None:
                apply_repeat = True
            else:
                try:
                    apply_repeat = int(multi_entry_level_index) == 0
                except Exception:
                    apply_repeat = False
        else:
            if multi_entry_level_index is None:
                apply_repeat = True
            else:
                try:
                    apply_repeat = int(multi_entry_level_index) == 0
                except Exception:
                    apply_repeat = False

    # Single-account mode: apply confidence threshold for the primary account.
    # Նույն քաղաքականությունն է՝ trade-ը թույլատրվում է, երբ
    # confidence ≥ threshold, իսկ confidence < threshold դեպքում skip.
    if not reduce_only and confidence is not None:
        try:
            conf_thr = _get_account_confidence_threshold(0)
        except Exception:
            conf_thr = None
        if conf_thr is not None:
            try:
                conf_val = float(confidence)
                thr_val = float(conf_thr)
            except Exception:
                conf_val = None
                thr_val = None
            eps = 1e-9
            if conf_val is not None and thr_val is not None and (conf_val + eps) < thr_val:
                raise RuntimeError(
                    f"Signal confidence {conf_val:.6f} is below account 0 threshold {thr_val:.6f}; skipping order."
                )

    if not reduce_only and apply_repeat:
        pos_ep = _repeat_get_open_position_entry_price(client, symbol, side)
        if pos_ep is not None and pos_ep > 0:
            try:
                mp = float(market_price) if market_price is not None else None
            except (TypeError, ValueError):
                mp = None
            if mp is None or mp <= 0:
                try:
                    mp = float(entry_price) if entry_price is not None else None
                except (TypeError, ValueError):
                    mp = None
            if mp is None or mp <= 0:
                return
            side_u = str(side).upper()
            if side_u == "SELL":
                if not (mp > pos_ep):
                    return
            else:
                if not (mp < pos_ep):
                    return
    _ensure_margin_mode_for_account_symbol(client, 0, symbol)

    multi_entry_levels = settings.get("multi_entry_levels") or []

    used_multi_entry = False
    if (
        not reduce_only
        and multi_entry_enabled
        and multi_entry_levels
        and entry_price is not None
    ):
        lev_scale = _get_notional_scale_for_leverage(0, symbol)
        try:
            ep_val = float(entry_price)
        except (TypeError, ValueError):
            ep_val = 0.0

        indices: List[int] = []
        if ep_val > 0:
            if multi_entry_level_index is None:
                indices = list(range(len(multi_entry_levels)))
            else:
                try:
                    lvl_idx = int(multi_entry_level_index)
                except (TypeError, ValueError):
                    lvl_idx = -1
                if 0 <= lvl_idx < len(multi_entry_levels):
                    indices = [lvl_idx]

        if indices:
            used_multi_entry = True
            last_res = None
            for lvl_idx in indices:
                lvl = multi_entry_levels[lvl_idx]
                if not isinstance(lvl, dict):
                    continue

                try:
                    notional_val = _compute_multi_entry_level_notional(
                        symbol, client, lvl, account_index=0
                    )
                except Exception:
                    continue
                if notional_val <= 0:
                    continue

                try:
                    if lev_scale != 1.0:
                        notional_val = float(notional_val) * float(lev_scale)
                except Exception:
                    pass

                try:
                    target_qty = notional_val / ep_val
                    lvl_qty = adjust_quantity(symbol, target_qty, ep_val)
                except Exception:
                    continue

                if not lvl_qty or lvl_qty <= 0:
                    continue

                try:
                    last_res = client.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=lvl_qty,
                        order_type=order_type,
                        reduce_only=False,
                        price=price,
                        stop_price=stop_price,
                        close_position=close_position,
                        time_in_force=time_in_force,
                    )
                    try:
                        log_trade_entry(
                            account_index=0,
                            symbol=symbol,
                            side=side,
                            qty=lvl_qty,
                            entry_price=entry_price,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        log(
                            f"[ORDER][SINGLE] Failed to place multi-entry order for {symbol} on primary account: {e}"
                        )
                    except Exception:
                        pass
                    continue

            if used_multi_entry:
                return last_res

    # Optional per-account fixed-notional sizing (USDT կամ բալանսի %),
    # fallback Fibonacci staking for primary account.
    eff_qty = quantity
    if not reduce_only and entry_price is not None:
        try:
            ep_val = float(entry_price)
        except (TypeError, ValueError):
            ep_val = 0.0

        account_notional = 0.0
        if ep_val > 0:
            try:
                account_notional = _compute_account_fixed_notional(
                    symbol, client, settings, account_index=0
                )
            except Exception:
                account_notional = 0.0

        if account_notional and account_notional > 0 and ep_val > 0:
            try:
                lev_scale = _get_notional_scale_for_leverage(0, symbol)
                try:
                    if lev_scale != 1.0:
                        account_notional = float(account_notional) * float(lev_scale)
                except Exception:
                    pass
                target_qty = account_notional / ep_val
                cand_qty = adjust_quantity(symbol, target_qty, ep_val)
                if cand_qty and cand_qty > 0:
                    eff_qty = cand_qty
            except Exception:
                pass
        else:
            try:
                fibo_notional = _get_fibo_notional_for_account(0, entry_price)
            except Exception:
                fibo_notional = 0.0
            if fibo_notional and fibo_notional > 0 and ep_val > 0:
                try:
                    lev_scale = _get_notional_scale_for_leverage(0, symbol)
                    try:
                        if lev_scale != 1.0:
                            fibo_notional = float(fibo_notional) * float(lev_scale)
                    except Exception:
                        pass
                    target_qty = fibo_notional / ep_val
                    cand_qty = adjust_quantity(symbol, target_qty, ep_val)
                    if cand_qty and cand_qty > 0:
                        eff_qty = cand_qty
                except Exception:
                    pass

    if not reduce_only and eff_qty is not None:
        try:
            px = None
            if str(order_type).upper() == "MARKET":
                if market_price is not None:
                    px = float(market_price)
                if px is None or px <= 0:
                    try:
                        px = _get_public_mark_price(client, symbol)
                    except Exception:
                        px = None
                if (px is None or px <= 0) and entry_price is not None:
                    px = float(entry_price)
            else:
                if price is not None:
                    px = float(price)
                if (px is None or px <= 0) and entry_price is not None:
                    px = float(entry_price)
                if px is None or px <= 0:
                    try:
                        px = _get_public_mark_price(client, symbol)
                    except Exception:
                        px = None
        except Exception:
            px = None
        if px is not None and px > 0:
            try:
                cand_qty = adjust_quantity(symbol, float(eff_qty), px)
                if cand_qty and cand_qty > 0:
                    eff_qty = cand_qty
            except Exception:
                pass

    res = client.place_order(
        symbol=symbol,
        side=side,
        quantity=eff_qty,
        order_type=order_type,
        reduce_only=reduce_only,
        price=price,
        stop_price=stop_price,
        close_position=close_position,
        time_in_force=time_in_force,
    )
    if not reduce_only:
        try:
            log_trade_entry(
                account_index=0,
                symbol=symbol,
                side=side,
                qty=eff_qty if eff_qty is not None else 0.0,
                entry_price=entry_price,
            )
        except Exception:
            pass
        try:
            _place_entry_protection_fast(
                client=client,
                account_index=0,
                symbol=symbol,
                side=side,
                settings=settings if isinstance(settings, dict) else {},
                entry_price_hint=entry_price,
            )
        except Exception:
            pass
    return res


def set_leverage(symbol: str, leverage: int, caller: str = "UNKNOWN"):
    """Set leverage for a symbol across one or multiple accounts.

    This is best-effort: any errors are expected to be handled by callers.
    """

    if not _execution_allowed(caller, "set_leverage"):
        return None

    clients = _get_clients()
    if not clients:
        raise RuntimeError("No Binance clients available")

    if _ACCOUNTS_MODE == "multi" and len(clients) > 1:
        results = []
        last_err = None
        for idx, client in enumerate(clients):
            if idx < len(_ACCOUNT_TRADE_ENABLED) and not _ACCOUNT_TRADE_ENABLED[idx]:
                continue

            lev_to_use = leverage
            if idx < len(_ACCOUNT_LEVERAGE_OVERRIDES):
                override = _ACCOUNT_LEVERAGE_OVERRIDES[idx]
                if override is not None:
                    lev_to_use = int(override)

            try:
                cached_eff = _get_cached_effective_leverage(idx, symbol, int(lev_to_use))
                if cached_eff is not None:
                    eff_lev = int(cached_eff)
                else:
                    eff_lev = _set_leverage_with_fallback(client, idx, symbol, int(lev_to_use))
                results.append({"symbol": str(symbol).upper(), "account_index": int(idx), "leverage": int(eff_lev)})
            except Exception as e:  # best-effort per account
                last_err = e
                try:
                    log(
                        f"[LEVERAGE][MULTI] Failed to set leverage {lev_to_use}x for {symbol} on account index {idx}: {e}"
                    )
                except Exception:
                    pass
                continue

        if results:
            return results

        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to set leverage on all Binance accounts")

    client = clients[0]
    cached_eff = _get_cached_effective_leverage(0, symbol, int(leverage))
    if cached_eff is not None:
        eff_lev = int(cached_eff)
    else:
        eff_lev = _set_leverage_with_fallback(client, 0, symbol, int(leverage))
    return {"symbol": str(symbol).upper(), "account_index": 0, "leverage": int(eff_lev)}


def set_leverage_for_account(symbol: str, leverage: int, account_index: int, caller: str = "UNKNOWN"):
    """Set leverage for a symbol on a single account index.

    This is an optimization helper for multi-account mode so callers that
    already know the target account do not re-run leverage updates across
    all accounts.
    """

    if not _execution_allowed(caller, "set_leverage_for_account"):
        return None

    clients = _get_clients()
    if not clients:
        raise RuntimeError("No Binance clients available")

    try:
        idx = int(account_index)
    except Exception:
        idx = 0
    if idx < 0 or idx >= len(clients):
        idx = 0

    if idx < len(_ACCOUNT_TRADE_ENABLED) and not _ACCOUNT_TRADE_ENABLED[idx]:
        raise RuntimeError(f"Account index {idx} is trade_enabled=false")

    lev_to_use = leverage
    if idx < len(_ACCOUNT_LEVERAGE_OVERRIDES):
        override = _ACCOUNT_LEVERAGE_OVERRIDES[idx]
        if override is not None:
            lev_to_use = int(override)

    client = clients[idx]
    cached_eff = _get_cached_effective_leverage(idx, symbol, int(lev_to_use))
    if cached_eff is not None:
        eff_lev = int(cached_eff)
    else:
        eff_lev = _set_leverage_with_fallback(client, idx, symbol, int(lev_to_use))
    return {"symbol": str(symbol).upper(), "account_index": int(idx), "leverage": int(eff_lev)}


_ACCOUNT_USES_ALGO: Dict[int, bool] = {}
_EXCHANGE_ORDER_ERR_LAST_LOG_TS: Dict[str, float] = {}
_EXCHANGE_ORDER_ERR_THROTTLE_SEC = 45.0


def _log_exchange_order_error_throttled(key: str, message: str) -> None:
    now = time.time()
    last = _EXCHANGE_ORDER_ERR_LAST_LOG_TS.get(key) or 0.0
    if (now - last) < _EXCHANGE_ORDER_ERR_THROTTLE_SEC:
        return
    _EXCHANGE_ORDER_ERR_LAST_LOG_TS[key] = now
    try:
        log(message)
    except Exception:
        pass


def _tick_decimals(tick: float) -> int:
    """Return the number of decimal places implied by a tick/step size."""
    if _CPP_AVAILABLE:
        try:
            return _cpp.tick_decimals(tick)
        except Exception:
            pass
    if tick <= 0 or tick >= 1:
        return 0
    s = f"{tick:.15f}".rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0


def place_exchange_stop_order(
    account_index: int,
    symbol: str,
    side: str,
    stop_price: float,
    order_type: str = "STOP_MARKET",
    quantity: float = None,
    close_position: bool = False,
    position_side: Optional[str] = None,
    working_type: str = "MARK_PRICE",
    caller: str = "UNKNOWN",
) -> Optional[Dict[str, Any]]:
    """Place a STOP_MARKET or TAKE_PROFIT_MARKET order on Binance.

    Returns dict {"id": <int>, "algo": <bool>} on success, None on failure.
    Tries /fapi/v1/order first; on -4120 falls back to /fapi/v1/algoOrder.
    For SL use order_type='STOP_MARKET' + close_position=True.
    For TP use order_type='TAKE_PROFIT_MARKET' + quantity.
    """
    if not _execution_allowed(caller, "place_exchange_stop_order"):
        log("[BLOCKED] Unauthorized SL/TP modification attempt")
        return None

    clients = _get_clients()
    if not clients or account_index < 0 or account_index >= len(clients):
        return None

    client = clients[account_index]

    adj_price = adjust_price(symbol, stop_price)
    if adj_price <= 0:
        log(f"[EXCHANGE ORDER] Skipping {order_type} for {symbol} account {account_index}: adj_price={adj_price} (raw={stop_price})")
        return None

    ps_use = None
    if position_side is not None:
        try:
            ps = str(position_side).upper().strip()
        except Exception:
            ps = ""
        if ps and ps != "BOTH":
            ps_use = ps

    adj_qty = None
    use_reduce_only = False
    if not close_position and quantity is not None:
        adj_qty = adjust_close_quantity(symbol, abs(float(quantity)))
        if adj_qty <= 0:
            log(f"[EXCHANGE ORDER] Skipping {order_type} for {symbol} account {account_index}: adj_qty<=0 (raw={quantity})")
            return None
        if not ps_use:
            use_reduce_only = True

    # Format price and qty as strings with correct decimal precision
    # to avoid Binance -1111 "Precision is over the maximum" errors
    # caused by floating-point artifacts (e.g. 0.09500000000000001).
    filters = _get_symbol_filters(symbol)
    if filters:
        tick = filters.get("tick_size") or 0.0
        if tick > 0:
            adj_price = f"{adj_price:.{_tick_decimals(tick)}f}"
        step = filters.get("step_size") or 0.0
        if step > 0 and adj_qty is not None:
            adj_qty = f"{adj_qty:.{_tick_decimals(step)}f}"

    # If this account previously required algo endpoint, skip straight to it.
    use_algo = _ACCOUNT_USES_ALGO.get(account_index, False)

    if not use_algo:
        try:
            result = client.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                stop_price=adj_price,
                quantity=adj_qty,
                close_position="true" if close_position else None,
                reduce_only=use_reduce_only,
                position_side=ps_use,
                working_type=working_type,
            )
            if isinstance(result, dict) and result.get("orderId"):
                oid = int(result["orderId"])
                log(f"[EXCHANGE ORDER] Placed {order_type} for {symbol} account {account_index}: orderId={oid} price={adj_price} qty={adj_qty} closePos={close_position}")
                return {"id": oid, "algo": False}
            log(f"[EXCHANGE ORDER] No orderId in response for {order_type} {symbol} account {account_index}: {result}")
            return None
        except Exception as e:
            err_str = str(e)
            if "-4120" in err_str:
                log(f"[EXCHANGE ORDER] Account {account_index} requires Algo Order API, switching.")
                _ACCOUNT_USES_ALGO[account_index] = True
                use_algo = True
            else:
                log(f"[EXCHANGE ORDER] Failed {order_type} for {symbol} account {account_index}: {e}")
                return None

    # Algo Order API fallback: POST /fapi/v1/algoOrder
    try:
        result = client.place_algo_order(
            symbol=symbol,
            side=side,
            algo_type="CONDITIONAL",
            order_type=order_type,
            quantity=adj_qty,
            reduce_only=use_reduce_only,
            trigger_price=adj_price,
            close_position=close_position,
            working_type=working_type,
            position_side=ps_use,
        )
        if isinstance(result, dict) and result.get("algoId"):
            aid = int(result["algoId"])
            log(f"[EXCHANGE ORDER] Placed ALGO {order_type} for {symbol} account {account_index}: algoId={aid} price={adj_price} qty={adj_qty} closePos={close_position}")
            return {"id": aid, "algo": True}
        log(f"[EXCHANGE ORDER] No algoId in ALGO response for {order_type} {symbol} account {account_index}: {result}")
        return None
    except Exception as e:
        err_s = str(e)
        # Known, noisy exchange rejections that are usually handled by retries/rebuild.
        if ('"code":-2021' in err_s) or ("would immediately trigger" in err_s):
            _log_exchange_order_error_throttled(
                f"algo:{account_index}:{str(symbol).upper()}:{order_type}:-2021",
                f"[EXCHANGE ORDER][WARN] ALGO {order_type} rejected (immediate-trigger) for "
                f"{symbol} account {account_index}; will rely on next rebuild/retry",
            )
            return None
        if ('"code":-4130' in err_s) or ("open stop or take profit order with GTE" in err_s):
            _log_exchange_order_error_throttled(
                f"algo:{account_index}:{str(symbol).upper()}:{order_type}:-4130",
                f"[EXCHANGE ORDER][WARN] ALGO {order_type} duplicate/conflict for "
                f"{symbol} account {account_index}; existing protective order already present",
            )
            return None
        log(f"[EXCHANGE ORDER] Failed ALGO {order_type} for {symbol} account {account_index}: {e}")
        return None


def cancel_order_by_id(
    account_index: int,
    symbol: str,
    order_id: int,
    is_algo: bool = False,
    caller: str = "UNKNOWN",
) -> bool:
    """Cancel a specific order by ID. Supports both regular and algo orders."""
    if not _execution_allowed(caller, "cancel_order_by_id"):
        return False

    clients = _get_clients()
    if not clients or account_index < 0 or account_index >= len(clients):
        return False

    client = clients[account_index]
    if is_algo:
        try:
            client.cancel_algo_order(int(order_id))
            return True
        except Exception:
            return False
    else:
        try:
            client.cancel_order(str(symbol).upper(), order_id=int(order_id))
            return True
        except Exception:
            return False


def cancel_symbol_open_orders(account_index: int, symbol: str, caller: str = "UNKNOWN") -> int:
    """Cancel all open orders for a symbol in safe priority order.

    Sequence (most dangerous first):
      1. DCA / additional-buy LIMIT orders  — cancel individually so they
         cannot fill between calls and re-open a position.
      2. SL orders (STOP_MARKET / STOP)
      3. TP orders (TAKE_PROFIT_MARKET / TAKE_PROFIT)
      4. Bulk DELETE /fapi/v1/allOpenOrders for any remaining regular orders.
      5. Algo orders in parallel.

    This guarantees that even if a step fails, the most dangerous orders
    (DCA buys) are always addressed first.
    """
    if not _execution_allowed(caller, "cancel_symbol_open_orders"):
        return 0

    clients = _get_clients()
    if not clients or account_index < 0 or account_index >= len(clients):
        return 0

    client = clients[account_index]
    sym = str(symbol).upper()
    canceled = 0

    # ── Step 1-3: fetch open orders and cancel by priority ────────────────
    try:
        orders = client.open_orders(symbol=sym)
        if not isinstance(orders, list):
            orders = []
    except Exception:
        orders = []

    def _order_priority(o: dict) -> int:
        """Lower = cancel first.  DCA buys=0, SL=1, TP=2, rest=3."""
        ot = str(o.get("type") or "").upper()
        ro = bool(o.get("reduceOnly") or o.get("reduce_only") or o.get("closePosition"))
        if ot == "LIMIT" and not ro:
            return 0   # DCA / additional buy — most dangerous
        if ot in ("STOP_MARKET", "STOP"):
            return 1   # Stop loss
        if ot in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
            return 2   # Take profit
        return 3

    for o in sorted(orders, key=_order_priority):
        oid = o.get("orderId")
        if oid is None:
            continue
        try:
            client.cancel_order(sym, int(oid))
            canceled += 1
        except Exception:
            pass

    # ── Step 4: bulk cancel anything that slipped through ─────────────────
    try:
        client.cancel_all_open_orders(sym)
        canceled += 1
    except Exception:
        pass

    # ── Step 5: algo orders in background (no bulk endpoint on Binance) ───
    import threading as _thr_cso
    def _cancel_algo():
        try:
            algo_orders = client.get_algo_open_orders(symbol=sym)
            if not isinstance(algo_orders, list):
                return
            for o in algo_orders:
                if not isinstance(o, dict):
                    continue
                aid = o.get("algoId")
                if aid is None:
                    continue
                try:
                    client.cancel_algo_order(int(aid))
                except Exception:
                    continue
        except Exception:
            pass

    t = _thr_cso.Thread(target=_cancel_algo, daemon=True)
    t.start()
    t.join(timeout=8.0)

    return canceled


def place_algo_order(
    symbol,
    side,
    algo_type="CONDITIONAL",
    order_type="STOP_MARKET",
    quantity=None,
    reduce_only=False,
    trigger_price=None,
    close_position=None,
    working_type="CONTRACT_PRICE",
    time_in_force=None,
    caller: str = "UNKNOWN",
):
    """Disabled helper for Binance algo (conditional) orders.

    This project no longer creates any exchange-side conditional SL/TP orders.
    All exit logic is implemented via MARKET reduce-only orders in the bot's
    exit manager. This function is kept only for backwards compatibility but
    intentionally performs no requests.
    """

    if not _execution_allowed(caller, "place_algo_order"):
        return None
    return


def place_sl_tp_per_account(
    symbol: str,
    signal: str,
    entry_price: float,
    atr_value: float,
    qty: float,
    tp_mode: int,
    auto_sl_tp_enabled: bool = True,
):
    """Place SL/TP algo orders per-account using each account's own settings.

    NOTE: This helper is now intentionally a no-op. All SL/TP management is
    handled by the bot's internal exit manager using MARKET reduce-only
    orders, not by exchange-side algo orders.
    """

    # Explicitly disabled to avoid creating any STOP/TAKE_PROFIT algo orders.
    return


def get_open_positions_snapshot():
    clients = _get_clients()
    if not clients:
        return []

    # Pre-compute per-account metadata (no I/O).
    account_meta = []
    for idx, client in enumerate(clients):
        try:
            trade_enabled = True
            if idx < len(_ACCOUNT_TRADE_ENABLED):
                trade_enabled = bool(_ACCOUNT_TRADE_ENABLED[idx])
        except Exception:
            trade_enabled = True

        settings = {}
        if idx < len(_ACCOUNT_SETTINGS):
            s = _ACCOUNT_SETTINGS[idx]
            if isinstance(s, dict):
                settings = s

        sl_pct = settings.get("sl_pct")
        tp_pcts = settings.get("tp_pcts")
        tp_mode = settings.get("tp_mode")
        auto_sl_tp = settings.get("auto_sl_tp")
        move_sl_to_entry = settings.get("move_sl_to_entry_on_first_tp")
        multi_entry_enabled = bool(settings.get("multi_entry_enabled"))
        multi_entry_levels = settings.get("multi_entry_levels") or []

        if multi_entry_enabled and isinstance(multi_entry_levels, list) and multi_entry_levels:
            first_level = multi_entry_levels[0]
            if isinstance(first_level, dict):
                lvl_sl = first_level.get("sl_pct")
                lvl_tp_pcts = first_level.get("tp_pcts")
                lvl_tp_mode = first_level.get("tp_mode")
                if lvl_sl is not None:
                    sl_pct = lvl_sl
                if lvl_tp_pcts is not None:
                    tp_pcts = lvl_tp_pcts
                if lvl_tp_mode is not None:
                    tp_mode = lvl_tp_mode

        early_breakeven_enabled = bool(settings.get("early_breakeven_enabled"))
        early_breakeven_pct = settings.get("early_breakeven_pct")
        tp_profit_lock_enabled = bool(settings.get("tp_profit_lock_enabled"))
        tp_profit_lock_pct = settings.get("tp_profit_lock_pct")
        combined_pnl_close_enabled = bool(settings.get("combined_pnl_close_enabled"))
        combined_pnl_close_pct = settings.get("combined_pnl_close_pct")
        ladder_enabled = bool(settings.get("ladder_sl_tp_enabled"))
        ladder_sl_range_pct = settings.get("ladder_sl_range_pct")
        ladder_sl_steps = settings.get("ladder_sl_steps")
        ladder_tp_range_pct = settings.get("ladder_tp_range_pct")
        ladder_tp_steps = settings.get("ladder_tp_steps")
        adaptive_reentry_master_enabled = bool(settings.get("adaptive_reentry_ladder_enabled"))
        adaptive_reentry_small_enabled = bool(settings.get("adaptive_reentry_small_enabled"))
        adaptive_reentry_large_enabled = bool(settings.get("adaptive_reentry_large_enabled"))
        signal_source = str(settings.get("signal_source") or "small").strip().lower()
        adaptive_reentry_enabled = bool(adaptive_reentry_master_enabled)
        if not adaptive_reentry_enabled:
            if signal_source == "large":
                adaptive_reentry_enabled = bool(adaptive_reentry_large_enabled)
            elif signal_source == "small":
                adaptive_reentry_enabled = bool(adaptive_reentry_small_enabled)
            else:
                adaptive_reentry_enabled = bool(adaptive_reentry_small_enabled or adaptive_reentry_large_enabled)
        adaptive_reentry_tp_range_pct = settings.get("adaptive_reentry_tp_range_pct")
        adaptive_reentry_tp_steps = settings.get("adaptive_reentry_tp_steps")
        adaptive_reentry_add_range_pct = settings.get("adaptive_reentry_add_range_pct")
        adaptive_reentry_add_steps = settings.get("adaptive_reentry_add_steps")
        adaptive_reentry_sl_pct = settings.get("adaptive_reentry_sl_pct")
        adaptive_reentry_add_total_multiplier = settings.get("adaptive_reentry_add_total_multiplier")
        adaptive_reentry_tighten_factor = settings.get("adaptive_reentry_tighten_factor")

        account_meta.append({
            "idx": idx,
            "client": client,
            "trade_enabled": trade_enabled,
            "sl_pct": sl_pct,
            "tp_pcts": tp_pcts,
            "tp_mode": tp_mode,
            "auto_sl_tp": auto_sl_tp,
            "move_sl_to_entry": move_sl_to_entry,
            "multi_entry_enabled": multi_entry_enabled,
            "multi_entry_levels": multi_entry_levels,
            "early_breakeven_enabled": early_breakeven_enabled,
            "early_breakeven_pct": early_breakeven_pct,
            "tp_profit_lock_enabled": tp_profit_lock_enabled,
            "tp_profit_lock_pct": tp_profit_lock_pct,
            "combined_pnl_close_enabled": combined_pnl_close_enabled,
            "combined_pnl_close_pct": combined_pnl_close_pct,
            "ladder_enabled": ladder_enabled,
            "ladder_sl_range_pct": ladder_sl_range_pct,
            "ladder_sl_steps": ladder_sl_steps,
            "ladder_tp_range_pct": ladder_tp_range_pct,
            "ladder_tp_steps": ladder_tp_steps,
            "adaptive_reentry_enabled": adaptive_reentry_enabled,
            "adaptive_reentry_tp_range_pct": adaptive_reentry_tp_range_pct,
            "adaptive_reentry_tp_steps": adaptive_reentry_tp_steps,
            "adaptive_reentry_add_range_pct": adaptive_reentry_add_range_pct,
            "adaptive_reentry_add_steps": adaptive_reentry_add_steps,
            "adaptive_reentry_sl_pct": adaptive_reentry_sl_pct,
            "adaptive_reentry_add_total_multiplier": adaptive_reentry_add_total_multiplier,
            "adaptive_reentry_tighten_factor": adaptive_reentry_tighten_factor,
        })

    # Fetch positions from ALL accounts in parallel instead of sequentially.
    def _fetch_positions(meta):
        try:
            return meta["client"].position_risk()
        except Exception:
            return []

    from concurrent.futures import ThreadPoolExecutor as _TPE

    raw_results = [None] * len(account_meta)
    if len(account_meta) == 1:
        raw_results[0] = _fetch_positions(account_meta[0])
    elif account_meta:
        with _TPE(max_workers=min(len(account_meta), 10)) as pool:
            futs = {pool.submit(_fetch_positions, m): i for i, m in enumerate(account_meta)}
            for fut in futs:
                try:
                    raw_results[futs[fut]] = fut.result()
                except Exception:
                    raw_results[futs[fut]] = []

    snapshot = []
    for meta, positions in zip(account_meta, raw_results):
        if not positions or not isinstance(positions, list):
            continue
        idx = meta["idx"]
        trade_enabled = meta["trade_enabled"]
        sl_pct = meta["sl_pct"]
        tp_pcts = meta["tp_pcts"]
        tp_mode = meta["tp_mode"]
        auto_sl_tp = meta["auto_sl_tp"]
        move_sl_to_entry = meta["move_sl_to_entry"]
        multi_entry_enabled = meta["multi_entry_enabled"]
        multi_entry_levels = meta["multi_entry_levels"]
        early_breakeven_enabled = meta["early_breakeven_enabled"]
        early_breakeven_pct = meta["early_breakeven_pct"]
        tp_profit_lock_enabled = meta["tp_profit_lock_enabled"]
        tp_profit_lock_pct = meta["tp_profit_lock_pct"]
        combined_pnl_close_enabled = meta["combined_pnl_close_enabled"]
        combined_pnl_close_pct = meta["combined_pnl_close_pct"]
        ladder_enabled = bool(meta.get("ladder_enabled"))
        ladder_sl_range_pct = meta.get("ladder_sl_range_pct")
        ladder_sl_steps = meta.get("ladder_sl_steps")
        ladder_tp_range_pct = meta.get("ladder_tp_range_pct")
        ladder_tp_steps = meta.get("ladder_tp_steps")
        adaptive_reentry_enabled = bool(meta.get("adaptive_reentry_enabled"))
        adaptive_reentry_tp_range_pct = meta.get("adaptive_reentry_tp_range_pct")
        adaptive_reentry_tp_steps = meta.get("adaptive_reentry_tp_steps")
        adaptive_reentry_add_range_pct = meta.get("adaptive_reentry_add_range_pct")
        adaptive_reentry_add_steps = meta.get("adaptive_reentry_add_steps")
        adaptive_reentry_sl_pct = meta.get("adaptive_reentry_sl_pct")
        adaptive_reentry_add_total_multiplier = meta.get("adaptive_reentry_add_total_multiplier")
        adaptive_reentry_tighten_factor = meta.get("adaptive_reentry_tighten_factor")

        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue

            position_side = None
            try:
                if pos.get("positionSide") is not None:
                    position_side = str(pos.get("positionSide"))
            except Exception:
                position_side = None

            amt_raw = pos.get("positionAmt")
            try:
                amt = float(amt_raw)
            except (TypeError, ValueError):
                continue

            if not amt:
                continue

            entry_raw = pos.get("entryPrice")
            try:
                entry_price = float(entry_raw)
            except (TypeError, ValueError):
                entry_price = 0.0

            update_time_ms = None
            try:
                if pos.get("updateTime") is not None:
                    update_time_ms = int(pos.get("updateTime"))
            except Exception:
                update_time_ms = None

            snapshot.append(
                {
                    "account_index": idx,
                    "symbol": symbol,
                    "position_amt": amt,
                    "entry_price": entry_price,
                    "update_time_ms": update_time_ms,
                    "sl_pct": sl_pct,
                    "tp_pcts": tp_pcts,
                    "tp_mode": tp_mode,
                    "auto_sl_tp": auto_sl_tp,
                    "move_sl_to_entry_on_first_tp": move_sl_to_entry,
                    "multi_entry_enabled": multi_entry_enabled,
                    "multi_entry_levels": multi_entry_levels,
                    "position_side": position_side,
                    "trade_enabled": trade_enabled,
                    "early_breakeven_enabled": early_breakeven_enabled,
                    "early_breakeven_pct": early_breakeven_pct,
                    "tp_profit_lock_enabled": tp_profit_lock_enabled,
                    "tp_profit_lock_pct": tp_profit_lock_pct,
                    "combined_pnl_close_enabled": combined_pnl_close_enabled,
                    "combined_pnl_close_pct": combined_pnl_close_pct,
                    "ladder_sl_tp_enabled": ladder_enabled,
                    "ladder_sl_range_pct": ladder_sl_range_pct,
                    "ladder_sl_steps": ladder_sl_steps,
                    "ladder_tp_range_pct": ladder_tp_range_pct,
                    "ladder_tp_steps": ladder_tp_steps,
                    "adaptive_reentry_ladder_enabled": adaptive_reentry_enabled,
                    "adaptive_reentry_tp_range_pct": adaptive_reentry_tp_range_pct,
                    "adaptive_reentry_tp_steps": adaptive_reentry_tp_steps,
                    "adaptive_reentry_add_range_pct": adaptive_reentry_add_range_pct,
                    "adaptive_reentry_add_steps": adaptive_reentry_add_steps,
                    "adaptive_reentry_sl_pct": adaptive_reentry_sl_pct,
                    "adaptive_reentry_add_total_multiplier": adaptive_reentry_add_total_multiplier,
                    "adaptive_reentry_tighten_factor": adaptive_reentry_tighten_factor,
                }
            )

    return snapshot


def close_position_market(
    account_index: int,
    symbol: str,
    position_amt: float,
    force_full_close: bool = False,
    position_side: Optional[str] = None,
    caller: str = "UNKNOWN",
) -> bool:
    if not _execution_allowed(caller, "close_position_market"):
        return False

    clients = _get_clients()
    if not clients:
        raise RuntimeError("No Binance clients available")

    if position_amt is None:
        try:
            log(
                f"[EXIT MANAGER] Failed to close {symbol} position on account {account_index}: position_amt is None"
            )
        except Exception:
            pass
        return False

    try:
        amt = float(position_amt)
    except (TypeError, ValueError):
        try:
            log(
                f"[EXIT MANAGER] Failed to close {symbol} position on account {account_index}: invalid position_amt={position_amt}"
            )
        except Exception:
            pass
        return False

    if not amt:
        return False

    if account_index < 0 or account_index >= len(clients):
        raise RuntimeError("Invalid account index")

    side = "SELL" if amt > 0 else "BUY"
    qty = adjust_close_quantity(symbol, abs(amt), round_up=bool(force_full_close))
    if qty <= 0:
        try:
            log(
                f"[EXIT MANAGER] Failed to close {symbol} position on account {account_index}: adjusted close qty<=0 "
                f"(pos_amt={amt}, force_full_close={bool(force_full_close)})"
            )
        except Exception:
            pass
        return False

    client = clients[account_index]

    ps_use = None
    if position_side is not None:
        try:
            ps = str(position_side).upper().strip()
        except Exception:
            ps = ""
        if ps and ps != "BOTH":
            ps_use = ps

    try:
        client.place_order(
            symbol=symbol,
            side=side,
            quantity=qty,
            reduce_only=True,
            position_side=ps_use,
        )
    except Exception as e:
        try:
            log(
                f"[EXIT MANAGER] Failed to close {symbol} position on account {account_index}: {e}"
            )
        except Exception:
            pass
        return False

    return True