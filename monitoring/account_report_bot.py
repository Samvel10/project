import os
import sys
import csv
import json
import time
import hmac
import hashlib
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from urllib.parse import urlencode

import requests
from ruamel.yaml import YAML

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.telegram import send_telegram

from execution import binance_futures
from config.proxies import build_proxy_from_string


_STATE_PATH = ROOT_DIR / "data" / "account_report_bot_state.json"
_DAILY_SNAPSHOTS_PATH = ROOT_DIR / "data" / "account_report_bot_daily_snapshots.json"
_YAML = YAML(typ="safe")

_ACCOUNTS_LOADED = False
_ACCOUNTS: List[Dict[str, Any]] = []
_RESTART_REQUESTED = False

_FEE_CFG_LOADED = False
_TAKER_FEE = 0.0004
_MAKER_FEE = 0.0002

_AUTH_CHAT_ID_LOADED = False
_AUTH_CHAT_ID: Optional[int] = None

_AUTH_USER_IDS_LOADED = False
_AUTH_USER_IDS: List[int] = []

_BALANCE_CACHE: Dict[str, Any] = {"ts": 0.0, "text": ""}
_BALANCE_CACHE_TTL_SEC = 20.0
_BALANCE_HTTP_TIMEOUT_SEC = 5.0
_BALANCE_MAX_WORKERS = 8

_BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"


def _ensure_daily_snapshots_file() -> None:
    if not _DAILY_SNAPSHOTS_PATH.parent.exists():
        _DAILY_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _DAILY_SNAPSHOTS_PATH.exists():
        _DAILY_SNAPSHOTS_PATH.write_text(json.dumps({"days": {}}, ensure_ascii=False), encoding="utf-8")


def _load_daily_snapshots() -> Dict[str, Any]:
    _ensure_daily_snapshots_file()
    try:
        raw = _DAILY_SNAPSHOTS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            return {"days": {}}
        if "days" not in data or not isinstance(data.get("days"), dict):
            data["days"] = {}
        return data
    except Exception:
        return {"days": {}}


def _save_daily_snapshots(data: Dict[str, Any]) -> None:
    _ensure_daily_snapshots_file()
    try:
        _DAILY_SNAPSHOTS_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _today_key_local() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _local_midnight_epoch_sec() -> float:
    lt = time.localtime()
    try:
        tup = (lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, lt.tm_wday, lt.tm_yday, lt.tm_isdst)
        return float(time.mktime(tup))
    except Exception:
        # Fallback: ~24h before now
        return time.time() - 24 * 3600


def _filter_rows_since_ms(rows: List[Dict[str, Any]], start_ms: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            ts_ms = int(float(r.get("timestamp_ms") or 0))
        except (TypeError, ValueError):
            continue
        if ts_ms >= int(start_ms):
            out.append(r)
    return out


def _get_real_accounts() -> List[Dict[str, Any]]:
    _load_accounts()
    return [a for a in _ACCOUNTS if isinstance(a, dict) and a.get("type") == "real"]


def _fetch_real_futures_balances() -> List[Tuple[Dict[str, Any], Optional[Dict[str, float]]]]:
    accounts = _get_real_accounts()
    if not accounts:
        return []

    def _fetch_one(acc: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, float]], Optional[str]]:
        api_key = str(acc.get("api_key") or "")
        api_secret = str(acc.get("api_secret") or "")
        proxy_cfg = acc.get("proxy") if isinstance(acc, dict) else None
        bal, err = _binance_futures_get_balance_usdt(api_key, api_secret, proxies=proxy_cfg)
        return acc, bal, err

    # Fetch balances concurrently so one slow account/proxy does not block all.
    items_by_id: Dict[int, Tuple[Dict[str, Any], Optional[Dict[str, float]]]] = {}
    max_workers = min(_BALANCE_MAX_WORKERS, max(1, len(accounts)))

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="balance_fetch") as pool:
        fut_map = {pool.submit(_fetch_one, acc): idx for idx, acc in enumerate(accounts)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            acc = accounts[idx]
            try:
                acc_res, bal, err = fut.result()
                # Keep original account object for downstream code consistency.
                acc = acc_res if isinstance(acc_res, dict) else acc
            except Exception as e:
                bal = None
                err = f"fetch_exception:{str(e)[:180]}"

            if err:
                acc["_balance_err"] = str(err)
            else:
                acc.pop("_balance_err", None)
            items_by_id[idx] = (acc, bal)

    # Preserve original accounts order in output.
    return [items_by_id[i] for i in range(len(accounts))]


def _ensure_state_file() -> None:
    if not _STATE_PATH.parent.exists():
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        _STATE_PATH.write_text(json.dumps({"offset": 0}, ensure_ascii=False), encoding="utf-8")


def _load_state() -> Dict[str, Any]:
    _ensure_state_file()
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {"offset": 0}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"offset": 0}
        # Ensure minimal structure
        if "offset" not in data:
            data["offset"] = 0
        return data
    except Exception:
        return {"offset": 0}


def _save_state(state: Dict[str, Any]) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _ensure_fee_config() -> None:
    global _FEE_CFG_LOADED, _TAKER_FEE, _MAKER_FEE
    if _FEE_CFG_LOADED:
        return
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}
        fees = cfg.get("fees") or {}
        taker = fees.get("taker")
        maker = fees.get("maker")
        if taker is not None:
            try:
                _TAKER_FEE = float(taker)
            except (TypeError, ValueError):
                pass
        if maker is not None:
            try:
                _MAKER_FEE = float(maker)
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    _FEE_CFG_LOADED = True


def _load_report_bot_token() -> str:
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}
    except Exception:
        return ""
    reports_cfg = cfg.get("reports_bot") or {}
    token = reports_cfg.get("token") or ""
    return str(token)


def _ensure_auth_chat_id_loaded() -> None:
    global _AUTH_CHAT_ID_LOADED, _AUTH_CHAT_ID
    if _AUTH_CHAT_ID_LOADED:
        return
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    chat_id: Optional[int] = None
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}
        raw = cfg.get("telegram_chat_id")
        if raw is not None:
            chat_id = int(raw)
    except Exception:
        chat_id = None
    _AUTH_CHAT_ID = chat_id
    _AUTH_CHAT_ID_LOADED = True


def _ensure_auth_user_ids_loaded() -> None:
    global _AUTH_USER_IDS_LOADED, _AUTH_USER_IDS
    if _AUTH_USER_IDS_LOADED:
        return
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    ids: List[int] = []
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}

        reports_cfg = cfg.get("reports_bot") or {}
        raw = reports_cfg.get("admin_user_ids") or cfg.get("telegram_admin_user_ids") or cfg.get("admin_user_ids")
        if raw is not None:
            if isinstance(raw, list):
                for v in raw:
                    try:
                        ids.append(int(v))
                    except Exception:
                        continue
            else:
                try:
                    ids.append(int(raw))
                except Exception:
                    pass
    except Exception:
        ids = []

    try:
        _AUTH_USER_IDS = sorted(list({int(x) for x in ids if x is not None}))
    except Exception:
        _AUTH_USER_IDS = []
    _AUTH_USER_IDS_LOADED = True


def _is_authorized_chat(chat_id: Optional[int]) -> bool:
    _ensure_auth_chat_id_loaded()
    if chat_id is None:
        return False
    if _AUTH_CHAT_ID is None:
        return False
    try:
        return int(chat_id) == int(_AUTH_CHAT_ID)
    except Exception:
        return False


def _is_authorized_sender(msg: Dict[str, Any]) -> bool:
    _ensure_auth_user_ids_loaded()
    if not _AUTH_USER_IDS:
        return True
    sender = msg.get("from") or {}
    sender_id = sender.get("id")
    try:
        sender_id_int = int(sender_id) if sender_id is not None else None
    except Exception:
        sender_id_int = None
    if sender_id_int is None:
        return False
    return sender_id_int in _AUTH_USER_IDS


def _handle_my_id_command(msg: Dict[str, Any]) -> List[str]:
    chat = msg.get("chat") or {}
    sender = msg.get("from") or {}
    chat_id = chat.get("id")
    user_id = sender.get("id")
    username = sender.get("username")
    first = sender.get("first_name")
    last = sender.get("last_name")
    name = " ".join([x for x in [first, last] if x])
    lines = []
    lines.append("Your Telegram IDs:")
    lines.append(f"- chat_id: {chat_id}")
    lines.append(f"- user_id: {user_id}")
    if username:
        lines.append(f"- username: @{username}")
    if name:
        lines.append(f"- name: {name}")
    lines.append("")
    lines.append("To restrict access, add this in config/trading.yaml:")
    lines.append("telegram_admin_user_ids: [<your_user_id>]")
    return ["\n".join(lines)]


def _binance_futures_get_balance_usdt(
    api_key: str,
    api_secret: str,
    proxies: Optional[Dict[str, str]] = None,
) -> tuple[Optional[Dict[str, float]], Optional[str]]:
    """Fetch futures balance for USDT.

    Returns dict with keys: balance, availableBalance (when available), otherwise None.
    """

    if not api_key or not api_secret:
        return None, "missing_api_key_or_secret"

    params: Dict[str, Any] = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 50000,
    }

    query = urlencode(params)
    try:
        sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception:
        return None
    url = f"{_BINANCE_FUTURES_BASE_URL}/fapi/v2/balance?{query}&signature={sig}"

    try:
        resp = requests.get(
            url,
            headers={"X-MBX-APIKEY": api_key},
            timeout=_BALANCE_HTTP_TIMEOUT_SEC,
            proxies=proxies,
        )
        try:
            resp.raise_for_status()
        except Exception:
            try:
                return None, f"http_{int(resp.status_code)}:{str(resp.text)[:200]}"
            except Exception:
                return None, "http_error"
        try:
            data = resp.json()
        except Exception:
            return None, "bad_json"
    except Exception as e:
        return None, f"request_error:{str(e)[:200]}"

    if not isinstance(data, list):
        return None, "unexpected_payload"

    for it in data:
        if not isinstance(it, dict):
            continue
        if str(it.get("asset") or "").upper() != "USDT":
            continue
        out: Dict[str, float] = {}
        try:
            out["balance"] = float(it.get("balance") or 0.0)
        except Exception:
            out["balance"] = 0.0
        try:
            out["availableBalance"] = float(it.get("availableBalance") or 0.0)
        except Exception:
            out["availableBalance"] = out["balance"]
        return out, None
    return None, "usdt_asset_not_found"


def _load_accounts() -> None:
    global _ACCOUNTS_LOADED, _ACCOUNTS
    if _ACCOUNTS_LOADED:
        return

    accounts: List[Dict[str, Any]] = []

    # Real Binance accounts
    real_cfg_path = ROOT_DIR / "config" / "binance_accounts.yaml"
    if real_cfg_path.exists():
        try:
            yaml = YAML(typ="safe")
            with real_cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f) or {}
        except Exception:
            cfg = {}
        real_accounts = cfg.get("accounts") or []
        if isinstance(real_accounts, list):
            for idx, acc in enumerate(real_accounts):
                if not isinstance(acc, dict):
                    continue
                name = str(acc.get("name") or f"real_{idx}")
                api_key = str(acc.get("api_key") or "")
                api_secret = str(acc.get("api_secret") or "")
                proxy_str = acc.get("proxy")
                proxy_cfg = None
                try:
                    if proxy_str:
                        proxy_cfg = build_proxy_from_string(str(proxy_str))
                except Exception:
                    proxy_cfg = None
                try:
                    lev = int(acc.get("leverage")) if acc.get("leverage") is not None else None
                except Exception:
                    lev = None
                hist_path = ROOT_DIR / "data" / "trade_history" / f"account_{idx}.csv"

                settings = acc.get("settings") or {}
                if not isinstance(settings, dict):
                    settings = {}
                sl_pct = settings.get("sl_pct")
                tp_pcts = settings.get("tp_pcts") or []

                accounts.append(
                    {
                        "type": "real",
                        "index": idx,
                        "name": name,
                        "key": name.strip().lower(),
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "proxy": proxy_cfg,
                        "leverage": lev,
                        "history_path": hist_path,
                        "sl_pct": sl_pct,
                        "tp_pcts": tp_pcts,
                    }
                )

    # Paper (simulated) accounts
    paper_cfg_path = ROOT_DIR / "config" / "paper_accounts.yaml"
    if paper_cfg_path.exists():
        try:
            yaml = YAML(typ="safe")
            with paper_cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f) or {}
        except Exception:
            cfg = {}
        paper_accounts = cfg.get("accounts") or []
        if isinstance(paper_accounts, list):
            for idx, acc in enumerate(paper_accounts):
                if not isinstance(acc, dict):
                    continue
                name = str(acc.get("name") or f"paper_{idx}")
                hist_path = ROOT_DIR / "data" / "paper_trade_history" / f"account_{idx}.csv"

                settings = acc.get("settings") or {}
                if not isinstance(settings, dict):
                    settings = {}
                sl_pct = settings.get("sl_pct")
                tp_pcts = settings.get("tp_pcts") or []

                accounts.append(
                    {
                        "type": "paper",
                        "index": idx,
                        "name": name,
                        "key": name.strip().lower(),
                        "history_path": hist_path,
                        "sl_pct": sl_pct,
                        "tp_pcts": tp_pcts,
                    }
                )

    _ACCOUNTS = accounts
    _ACCOUNTS_LOADED = True


def _format_accounts_list() -> str:
    _load_accounts()
    if not _ACCOUNTS:
        return "No accounts found. Make sure binance_accounts.yaml / paper_accounts.yaml are configured."

    lines = ["Available accounts (real + paper):"]
    for acc in _ACCOUNTS:
        acc_type = acc.get("type", "?")
        idx = acc.get("index", "?")
        name = acc.get("name", "?")
        sl_pct = acc.get("sl_pct")
        tp_pcts = acc.get("tp_pcts") or []
        tp_str = ", ".join(str(x) for x in tp_pcts) if tp_pcts else "-"
        if sl_pct is None:
            sl_str = "?"
        else:
            try:
                sl_str = f"{float(sl_pct):.2f}%"
            except (TypeError, ValueError):
                sl_str = str(sl_pct)

        lines.append(
            f"- [{acc_type}] {name} (index={idx}) | SL={sl_str} | TP={tp_str}"
        )
    lines.append("")
    lines.append("Use /summary <name> [hours] [limit] to see stats. Example:")
    lines.append("/summary Samo 24 50  – last 24 hours, max 50 exits.")
    lines.append("/summary Paper_1 limit 30  – last 30 exits for Paper_1.")
    lines.append("")
    lines.append("You can also change SL/TP from Telegram:")
    lines.append("/set_sl <name> <percent>  – change SL distance in % for that account")
    lines.append("/set_tp <name> p1 p2 [p3] – change TP% levels (1–3 levels)")
    return "\n".join(lines)


def _resolve_account(key: str) -> Optional[Dict[str, Any]]:
    _load_accounts()
    k = (key or "").strip().lower()
    if not k:
        return None

    # First, match by name (case-insensitive)
    for acc in _ACCOUNTS:
        name = str(acc.get("name") or "").strip().lower()
        if name == k:
            return acc

    # Fallbacks: allow type_index (real_0, paper_0) or raw index for real
    for acc in _ACCOUNTS:
        idx = acc.get("index")
        acc_type = acc.get("type")
        if acc_type and idx is not None:
            if k == f"{acc_type}_{idx}" or k == f"{acc_type[0]}{idx}":
                return acc
            if acc_type == "real" and k == str(idx):
                return acc

    return None


def _load_history_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _filter_by_hours(rows: List[Dict[str, Any]], hours: Optional[float]) -> List[Dict[str, Any]]:
    if hours is None or hours <= 0:
        return rows
    now_ms = int(time.time() * 1000)
    cutoff = now_ms - int(hours * 3600 * 1000)
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        try:
            ts_ms = int(float(r.get("timestamp_ms") or 0))
        except (TypeError, ValueError):
            continue
        if ts_ms >= cutoff:
            filtered.append(r)
    return filtered


def _summarise_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build per-account stats from trade_history CSV rows.

    - Winrate / best / worst մնում են exit-ների տոկոսային վրա
    - Avg PnL-ը հաշվարկվում է notional-ով (qty * entry_price) քաշավորված
    - Realized PnL-ը մոտարկված է USDT-ում՝ ըստ qty, entry_price և pnl_pct
    """

    _ensure_fee_config()

    entries = [r for r in rows if (r.get("event") or "").upper() == "ENTRY"]
    exits = [r for r in rows if (r.get("event") or "").upper() == "EXIT"]

    total_exits = len(exits)
    win = 0
    loss = 0
    flat = 0

    pnl_values: List[float] = []
    weighted_pnl_numer: float = 0.0
    notional_sum: float = 0.0
    realized_pnl_usd: float = 0.0

    for r in exits:
        raw_pct = r.get("pnl_pct")
        try:
            if raw_pct in ("", None):
                continue
            v = float(raw_pct)
        except (TypeError, ValueError):
            continue

        # Փորձում ենք հաշվել notional-ը, կոմիսիաները և fee-ով ճշգրտված PnL%-ը։
        qty_raw = r.get("qty")
        entry_raw = r.get("entry_price")
        exit_raw = r.get("exit_price")

        try:
            qty_val = float(qty_raw) if qty_raw not in ("", None) else 0.0
        except (TypeError, ValueError):
            qty_val = 0.0

        try:
            entry_val = float(entry_raw) if entry_raw not in ("", None) else 0.0
        except (TypeError, ValueError):
            entry_val = 0.0

        try:
            exit_val = float(exit_raw) if exit_raw not in ("", None) else 0.0
        except (TypeError, ValueError):
            exit_val = 0.0

        effective_pct = v

        if qty_val > 0.0 and entry_val > 0.0:
            notional = qty_val * entry_val
            notional_sum += notional

            # Gross PnL-ը USDT-ում (առանց फीस):
            gross_pnl = (v / 100.0) * notional

            # Կոմիսիաներ՝ մոտարկված՝ երկու կողմն էլ taker
            fee_open = notional * _TAKER_FEE
            if exit_val > 0.0:
                exit_notional = qty_val * exit_val
            else:
                exit_notional = notional
            fee_close = exit_notional * _TAKER_FEE
            total_fees = fee_open + fee_close

            net_pnl = gross_pnl - total_fees
            realized_pnl_usd += net_pnl

            if notional > 0.0:
                effective_pct = (net_pnl / notional) * 100.0

            weighted_pnl_numer += effective_pct * notional

        pnl_values.append(effective_pct)
        if effective_pct > 0:
            win += 1
        elif effective_pct < 0:
            loss += 1
        else:
            flat += 1

    avg_pnl_simple = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
    if notional_sum > 0.0:
        avg_pnl_weighted = weighted_pnl_numer / notional_sum
    else:
        # Եթե notional չենք կարող հաշվարկել, fallback՝ պարզ միջին
        avg_pnl_weighted = avg_pnl_simple

    pnl_min = min(pnl_values) if pnl_values else 0.0
    pnl_max = max(pnl_values) if pnl_values else 0.0
    winrate = (win / total_exits * 100.0) if total_exits > 0 else 0.0

    return {
        "entries": len(entries),
        "exits": total_exits,
        "wins": win,
        "losses": loss,
        "flats": flat,
        # Հիմնական ցուցադրվող միջինը՝ size-weighted
        "avg_pnl": avg_pnl_weighted,
        # Օգտակար է, եթե պետք է պարզ միջին էլ տեսնել
        "avg_pnl_simple": avg_pnl_simple,
        "pnl_min": pnl_min,
        "pnl_max": pnl_max,
        "winrate": winrate,
        "realized_pnl_usd": realized_pnl_usd,
    }


def _parse_time_window_to_hours(token: Optional[str]) -> Optional[float]:
    if token is None:
        return 24.0
    t = str(token).strip().lower()
    if not t:
        return 24.0
    if t in ("all", "full", "history"):
        return None
    if t.endswith("h"):
        try:
            return float(t[:-1])
        except Exception:
            return 24.0
    if t.endswith("d"):
        try:
            return float(t[:-1]) * 24.0
        except Exception:
            return 24.0
    if t.endswith("w"):
        try:
            return float(t[:-1]) * 7.0 * 24.0
        except Exception:
            return 24.0
    try:
        return float(t)
    except Exception:
        return 24.0


def _compute_exit_net_metrics(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    _ensure_fee_config()
    raw_pct = row.get("pnl_pct")
    try:
        if raw_pct in ("", None):
            return None, None, None
        v = float(raw_pct)
    except (TypeError, ValueError):
        return None, None, None

    qty_raw = row.get("qty")
    entry_raw = row.get("entry_price")
    exit_raw = row.get("exit_price")

    try:
        qty_val = float(qty_raw) if qty_raw not in ("", None) else 0.0
    except (TypeError, ValueError):
        qty_val = 0.0

    try:
        entry_val = float(entry_raw) if entry_raw not in ("", None) else 0.0
    except (TypeError, ValueError):
        entry_val = 0.0

    try:
        exit_val = float(exit_raw) if exit_raw not in ("", None) else 0.0
    except (TypeError, ValueError):
        exit_val = 0.0

    if qty_val <= 0.0 or entry_val <= 0.0:
        return v, None, None

    notional = qty_val * entry_val
    gross_pnl = (v / 100.0) * notional
    fee_open = notional * _TAKER_FEE
    if exit_val > 0.0:
        exit_notional = qty_val * exit_val
    else:
        exit_notional = notional
    fee_close = exit_notional * _TAKER_FEE
    net_pnl = gross_pnl - (fee_open + fee_close)
    effective_pct = (net_pnl / notional) * 100.0 if notional > 0 else v
    return effective_pct, net_pnl, notional


def _format_report_message(acc: Dict[str, Any], rows: List[Dict[str, Any]], hours: Optional[float]) -> str:
    entries = [r for r in rows if (r.get("event") or "").upper() == "ENTRY"]
    exits = [r for r in rows if (r.get("event") or "").upper() == "EXIT"]

    by_hour = defaultdict(lambda: {"exits": 0, "wins": 0, "losses": 0, "realized": 0.0, "notional": 0.0, "pnl_num": 0.0})
    best_trade: Optional[Tuple[float, float, Dict[str, Any]]] = None
    worst_trade: Optional[Tuple[float, float, Dict[str, Any]]] = None

    exits_with_ts: List[Tuple[int, float, float, float, Dict[str, Any]]] = []
    realized_total = 0.0
    notional_sum = 0.0
    pnl_weighted_numer = 0.0

    for r in exits:
        try:
            ts_ms = int(float(r.get("timestamp_ms") or 0))
        except (TypeError, ValueError):
            continue

        eff_pct, net_pnl, notional = _compute_exit_net_metrics(r)
        if eff_pct is None:
            continue
        if net_pnl is None:
            net_pnl_val = 0.0
        else:
            net_pnl_val = float(net_pnl)
        if notional is None:
            notional_val = 0.0
        else:
            notional_val = float(notional)

        realized_total += net_pnl_val
        if notional_val > 0:
            notional_sum += notional_val
            pnl_weighted_numer += float(eff_pct) * notional_val

        ts_local = dt.datetime.fromtimestamp(ts_ms / 1000.0)
        h = int(ts_local.hour)
        b = by_hour[h]
        b["exits"] += 1
        b["realized"] += net_pnl_val
        b["notional"] += notional_val
        b["pnl_num"] += float(eff_pct) * notional_val
        if float(eff_pct) > 0:
            b["wins"] += 1
        elif float(eff_pct) < 0:
            b["losses"] += 1

        if best_trade is None or float(eff_pct) > best_trade[0]:
            best_trade = (float(eff_pct), float(net_pnl_val), r)
        if worst_trade is None or float(eff_pct) < worst_trade[0]:
            worst_trade = (float(eff_pct), float(net_pnl_val), r)

        exits_with_ts.append((ts_ms, float(eff_pct), net_pnl_val, notional_val, r))

    exits_with_ts.sort(key=lambda x: x[0])
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for _ts, _pct, net_pnl_val, _notional_val, _r in exits_with_ts:
        cum += float(net_pnl_val)
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd

    exits_count = len(exits_with_ts)
    wins = sum(by_hour[h]["wins"] for h in range(24))
    losses = sum(by_hour[h]["losses"] for h in range(24))
    flats = max(0, exits_count - wins - losses)
    winrate = (wins / exits_count * 100.0) if exits_count > 0 else 0.0
    avg_pnl = (pnl_weighted_numer / notional_sum) if notional_sum > 0 else 0.0

    best_hour: Optional[int] = None
    worst_hour: Optional[int] = None
    best_hour_val: Optional[float] = None
    worst_hour_val: Optional[float] = None
    for h in range(24):
        v = float(by_hour[h]["realized"])
        if by_hour[h]["exits"] <= 0:
            continue
        if best_hour is None or best_hour_val is None or v > float(best_hour_val):
            best_hour = h
            best_hour_val = v
        if worst_hour is None or worst_hour_val is None or v < float(worst_hour_val):
            worst_hour = h
            worst_hour_val = v

    lines: List[str] = []
    name = str(acc.get("name") or "?")
    acc_type = str(acc.get("type") or "?")

    now_local = dt.datetime.now()
    if hours is None:
        start_local = None
        try:
            if exits_with_ts:
                start_local = dt.datetime.fromtimestamp(int(exits_with_ts[0][0]) / 1000.0)
        except Exception:
            start_local = None
        period_label = "ամբողջ պատմությունը"
    else:
        start_local = now_local - dt.timedelta(hours=float(hours))
        period_label = f"վերջին {int(round(float(hours)))} ժամ"

    lines.append(f"ՀԱՇՎԵՏՎՈՒԹՅՈՒՆ — {name} [{acc_type}]")
    if start_local is not None:
        lines.append(f"Ժամանակահատված (local): {start_local.strftime('%Y-%m-%d %H:%M')} → {now_local.strftime('%Y-%m-%d %H:%M')}  ({period_label})")
    else:
        lines.append(f"Ժամանակահատված: {period_label}")
    lines.append("")

    lines.append("ԱՄՓՈՓ ՎԻՃԱԿԱԳՐՈՒԹՅՈՒՆ")
    lines.append(f"- Entries: {len(entries)}")
    lines.append(f"- Exits: {exits_count}")
    lines.append(f"- Wins/Losses/Flats: {wins}/{losses}/{flats}  (Winrate՝ {winrate:.1f}%)")
    lines.append(f"- Avg PnL (size-weighted, net≈): {avg_pnl:+.2f}%")
    lines.append(f"- Realized≈ (net, USDT): {realized_total:+.2f}")
    lines.append(f"- Max Drawdown≈ (realized curve, USDT): {max_dd:.2f}")

    if best_hour is not None and best_hour_val is not None:
        lines.append(f"- Լավագույն ժամ: {int(best_hour):02d}:00  (Realized≈ {float(best_hour_val):+.2f} USDT)")
    if worst_hour is not None and worst_hour_val is not None:
        lines.append(f"- Վատագույն ժամ: {int(worst_hour):02d}:00 (Realized≈ {float(worst_hour_val):+.2f} USDT)")
    lines.append("")

    if best_trade is not None:
        r = best_trade[2]
        ts = str(r.get("timestamp_iso") or "?")
        sym = str(r.get("symbol") or "?")
        side = str(r.get("side") or "?")
        lines.append(f"ԼԱՎԱԳՈՒՅՆ TRADE: {ts} | {sym} {side} | PnL≈ {best_trade[0]:+.2f}% | Realized≈ {best_trade[1]:+.2f} USDT")
    if worst_trade is not None:
        r = worst_trade[2]
        ts = str(r.get("timestamp_iso") or "?")
        sym = str(r.get("symbol") or "?")
        side = str(r.get("side") or "?")
        lines.append(f"ՎԱՏԱԳՈՒՅՆ TRADE: {ts} | {sym} {side} | PnL≈ {worst_trade[0]:+.2f}% | Realized≈ {worst_trade[1]:+.2f} USDT")

    lines.append("")

    # Top 5 hours by realized
    hour_items: List[Tuple[int, float, int, float]] = []
    for h in range(24):
        ex = int(by_hour[h]["exits"])
        if ex <= 0:
            continue
        realized = float(by_hour[h]["realized"])
        wr = (float(by_hour[h]["wins"]) / ex * 100.0) if ex > 0 else 0.0
        hour_items.append((h, realized, ex, wr))

    if hour_items:
        top_best = sorted(hour_items, key=lambda x: x[1], reverse=True)[:5]
        top_worst = sorted(hour_items, key=lambda x: x[1])[:5]
        lines.append("ԼԱՎԱԳՈՒՅՆ ԺԱՄԵՐ (Top 5, Realized≈ USDT)")
        for h, realized, ex, wr in top_best:
            lines.append(f"- {h:02d}:00 | exits={ex} | winrate={wr:.1f}% | realized={realized:+.2f}")
        lines.append("")
        lines.append("ՎԱՏԱԳՈՒՅՆ ԺԱՄԵՐ (Top 5, Realized≈ USDT)")
        for h, realized, ex, wr in top_worst:
            lines.append(f"- {h:02d}:00 | exits={ex} | winrate={wr:.1f}% | realized={realized:+.2f}")
        lines.append("")

    lines.append("ԺԱՄԱՅԻՆ ԱՄՓՈՓՈՒՄ (միայն ժամերը, որտեղ exits կան)")
    lines.append("```")
    lines.append("Hour  Exits  Win%   Realized≈USDT")
    for h in range(24):
        b = by_hour[h]
        ex = int(b["exits"])
        if ex <= 0:
            continue
        wr = (float(b["wins"]) / ex * 100.0) if ex > 0 else 0.0
        realized = float(b["realized"])
        lines.append(f"{h:02d}    {ex:>5}  {wr:>5.1f}  {realized:>+12.2f}")
    lines.append("```")

    lines.append("")
    lines.append("ԲԱՑԱՏՐՈՒԹՅՈՒՆ")
    lines.append("- Realized≈ (USDT)՝ մոտարկված 'մաքուր' արդյունք է՝ հաշվի առնելով qty*entry_price notional-ը և taker fee≈ (երկու կողմ)։")
    lines.append("- Max Drawdown≈՝ realized curve-ի peak-ից մինչև հաջորդ նվազագույն կետը (միայն փակված trades-ով)։")
    lines.append("")
    lines.append("Օգտագործում: /report <account> [24h|7d|30d|all]")
    return "\n".join(lines).strip()


def _handle_report_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 2:
        return ["Usage: /report <account_name|index> [24h|7d|30d|all]"]

    acc_key = parts[1].strip()
    acc = _resolve_account(acc_key)
    if not acc:
        msg = _format_accounts_list()
        return ["Unknown account name. Use /accounts to see the list.", "", msg]

    window_token = parts[2].strip() if len(parts) >= 3 else None
    hours = _parse_time_window_to_hours(window_token)

    path: Path = acc.get("history_path")
    rows = _load_history_rows(path)
    rows = _filter_by_hours(rows, hours)
    if not rows:
        if hours is None:
            return [f"No trades found for account {acc['name']}."]
        return [f"No trades found for account {acc['name']} in the last {hours:.0f}h."]

    return [_format_report_message(acc, rows, hours)]


def _update_account_settings_yaml(
    acc: Dict[str, Any],
    sl_pct: Optional[float] = None,
    tp_pcts: Optional[List[float]] = None,
    tp_mode: Optional[int] = None,
    confidence: Optional[float] = None,
) -> bool:
    """Update per-account sl_pct / tp_pcts in the corresponding YAML config.

    For real accounts: config/binance_accounts.yaml
    For paper accounts: config/paper_accounts.yaml
    """

    acc_type = acc.get("type")
    idx = acc.get("index")
    if acc_type not in ("real", "paper") or idx is None:
        return False

    if acc_type == "real":
        cfg_path = ROOT_DIR / "config" / "binance_accounts.yaml"
    else:
        cfg_path = ROOT_DIR / "config" / "paper_accounts.yaml"

    yaml = YAML()  # round-trip where possible
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.load(f) or {}
    except Exception:
        return False

    accounts = cfg.get("accounts") or []
    if not isinstance(accounts, list):
        return False
    if not (0 <= int(idx) < len(accounts)):
        return False

    node = accounts[int(idx)]
    if not isinstance(node, dict):
        return False
    settings = node.get("settings")
    if not isinstance(settings, dict):
        settings = {}
        node["settings"] = settings

    if sl_pct is not None:
        settings["sl_pct"] = float(sl_pct)
    if tp_pcts is not None:
        settings["tp_pcts"] = tp_pcts
    if tp_mode is not None:
        try:
            settings["tp_mode"] = int(tp_mode)
        except (TypeError, ValueError):
            return False
    if confidence is not None:
        try:
            settings["confidence"] = float(confidence)
        except (TypeError, ValueError):
            return False

    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
    except Exception:
        return False

    # Invalidate cached accounts so that /accounts shows updated values next time
    global _ACCOUNTS_LOADED, _ACCOUNTS
    _ACCOUNTS_LOADED = False
    _ACCOUNTS = []
    return True


def _add_real_account(name: str) -> bool:
    """Append a new real (Binance) account stub to binance_accounts.yaml.

    API key/secret will be left empty; the user should fill them manually
    in the YAML file after creation.
    """

    cfg_path = ROOT_DIR / "config" / "binance_accounts.yaml"
    yaml = YAML()

    try:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}

    accounts = cfg.get("accounts") or []
    if not isinstance(accounts, list):
        accounts = []
        cfg["accounts"] = accounts

    # Avoid duplicates by name (case-insensitive)
    lower_name = name.strip().lower()
    for acc in accounts:
        if not isinstance(acc, dict):
            continue
        existing_name = str(acc.get("name") or "").strip().lower()
        if existing_name == lower_name:
            return False

    # Default leverage: try from trading.yaml, otherwise 2
    lev_default = 2
    trading_cfg_path = ROOT_DIR / "config" / "trading.yaml"
    try:
        with trading_cfg_path.open("r", encoding="utf-8") as tf:
            tcfg = _YAML.load(tf) or {}
        lev_block = tcfg.get("leverage") or {}
        if isinstance(lev_block, dict):
            dv = lev_block.get("default")
            if dv is not None:
                lev_default = int(dv)
    except Exception:
        pass

    new_acc = {
        "name": name,
        "trade_enabled": False,
        "leverage": lev_default,
        "api_key": "",
        "api_secret": "",
        "settings": {
            "tp_mode": 1,
            "sl_pct": 2.5,
            "tp_pcts": [1.5, 2.5, 3.5],
            "fixed_notional_usd": 0,
        },
    }
    accounts.append(new_acc)

    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
    except Exception:
        return False

    global _ACCOUNTS_LOADED, _ACCOUNTS
    _ACCOUNTS_LOADED = False
    _ACCOUNTS = []
    return True


def _add_paper_account(name: str) -> bool:
    """Append a new paper account to paper_accounts.yaml."""

    cfg_path = ROOT_DIR / "config" / "paper_accounts.yaml"
    yaml = YAML()

    try:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}

    if "enabled" not in cfg:
        cfg["enabled"] = True

    accounts = cfg.get("accounts") or []
    if not isinstance(accounts, list):
        accounts = []
        cfg["accounts"] = accounts

    lower_name = name.strip().lower()
    for acc in accounts:
        if not isinstance(acc, dict):
            continue
        existing_name = str(acc.get("name") or "").strip().lower()
        if existing_name == lower_name:
            return False

    new_acc = {
        "name": name,
        "trade_enabled": True,
        "fixed_notional_usd": 10,
        "settings": {
            "sl_pct": 2.5,
            "tp_pcts": [1.5, 2.5, 3.5],
            "tp_mode": 3,
        },
    }
    accounts.append(new_acc)

    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
    except Exception:
        return False

    global _ACCOUNTS_LOADED, _ACCOUNTS
    _ACCOUNTS_LOADED = False
    _ACCOUNTS = []
    return True


def _delete_account_by_name(name: str) -> Optional[str]:
    """Delete a real or paper account by its name.

    Returns a string describing what was deleted, or None if nothing matched.
    """

    lower_name = name.strip().lower()

    # Real accounts first
    real_path = ROOT_DIR / "config" / "binance_accounts.yaml"
    yaml = YAML()
    deleted_desc: Optional[str] = None

    if real_path.exists():
        try:
            with real_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f) or {}
        except Exception:
            cfg = {}

        accounts = cfg.get("accounts") or []
        if isinstance(accounts, list):
            new_accounts = []
            for acc in accounts:
                if not isinstance(acc, dict):
                    new_accounts.append(acc)
                    continue
                existing_name = str(acc.get("name") or "").strip().lower()
                if existing_name == lower_name:
                    deleted_desc = f"real account '{acc.get('name', name)}'"
                    continue
                new_accounts.append(acc)

            if deleted_desc is not None:
                cfg["accounts"] = new_accounts
                try:
                    with real_path.open("w", encoding="utf-8") as f:
                        yaml.dump(cfg, f)
                except Exception:
                    deleted_desc = None

    # If not deleted from real, try paper
    if deleted_desc is None:
        paper_path = ROOT_DIR / "config" / "paper_accounts.yaml"
        if paper_path.exists():
            try:
                with paper_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.load(f) or {}
            except Exception:
                cfg = {}

            accounts = cfg.get("accounts") or []
            if isinstance(accounts, list):
                new_accounts = []
                for acc in accounts:
                    if not isinstance(acc, dict):
                        new_accounts.append(acc)
                        continue
                    existing_name = str(acc.get("name") or "").strip().lower()
                    if existing_name == lower_name:
                        deleted_desc = f"paper account '{acc.get('name', name)}'"
                        continue
                    new_accounts.append(acc)

                if deleted_desc is not None:
                    cfg["accounts"] = new_accounts
                    try:
                        with paper_path.open("w", encoding="utf-8") as f:
                            yaml.dump(cfg, f)
                    except Exception:
                        deleted_desc = None

    if deleted_desc is not None:
        global _ACCOUNTS_LOADED, _ACCOUNTS
        _ACCOUNTS_LOADED = False
        _ACCOUNTS = []
    return deleted_desc


def _format_summary_message(
    acc: Dict[str, Any],
    rows: List[Dict[str, Any]],
    hours: Optional[float],
    limit: int,
) -> str:
    if not rows:
        return "No trades found for this account in the selected period."

    stats = _summarise_rows(rows)

    exits = [r for r in rows if (r.get("event") or "").upper() == "EXIT"]
    # Sort by timestamp descending
    exits_sorted = []
    for r in exits:
        try:
            ts_ms = int(float(r.get("timestamp_ms") or 0))
        except (TypeError, ValueError):
            ts_ms = 0
        exits_sorted.append((ts_ms, r))
    exits_sorted.sort(key=lambda x: x[0], reverse=True)
    if limit > 0:
        exits_sorted = exits_sorted[:limit]

    lines: List[str] = []
    acc_type = acc.get("type", "?")
    name = acc.get("name", "?")
    idx = acc.get("index", "?")

    header = f"Account: {name} (type={acc_type}, index={idx})"
    lines.append(header)
    if hours is not None and hours > 0:
        lines.append(f"Period: last {hours}h")
    else:
        lines.append("Period: all available history")
    lines.append("")

    lines.append(
        f"Entries: {stats['entries']} | Exits: {stats['exits']} | Winrate: {stats['winrate']:.1f}%"
    )
    lines.append(
        f"Wins: {stats['wins']} | Losses: {stats['losses']} | Flats: {stats['flats']}"
    )
    # Avg PnL now is notional-weighted (qty * entry_price), ավելի մոտ իրական արդյունքին
    lines.append(
        f"Avg PnL (size-weighted): {stats['avg_pnl']:.2f}% | Approx realized PnL: {stats['realized_pnl_usd']:.2f} USDT"
    )
    lines.append(
        f"Best exit: {stats['pnl_max']:.2f}% | Worst exit: {stats['pnl_min']:.2f}%"
    )
    lines.append("")

    if exits_sorted:
        lines.append(f"Last {len(exits_sorted)} exits:")
        for ts_ms, r in exits_sorted:
            ts_iso = r.get("timestamp_iso") or "?"
            symbol = r.get("symbol", "?")
            side = r.get("side", "?")
            qty = r.get("qty", "?")
            entry_price = r.get("entry_price", "")
            exit_price = r.get("exit_price", "")
            pnl = r.get("pnl_pct", "")
            reason = r.get("reason", "")
            line = (
                f"{ts_iso} | {symbol} | {side} | qty={qty} | "
                f"entry={entry_price} | exit={exit_price} | pnl={pnl}% | {reason}"
            )
            lines.append(line)

    lines.append("")
    lines.append("Usage: /summary <name> [hours] [limit]")
    return "\n".join(lines)


def _handle_command_start() -> List[str]:
    lines = [
        "Account Report Bot — Help",
        "",
        "Այս բոտը կարդում է per-account trade history ֆայլերը (real + paper)",
        "և Telegram-ի մեջ ցույց է տալիս հաշվետվություններ/վիճակագրություն։",
        "",
        "ՀՐԱՄԱՆՆԵՐ",
        "- /accounts",
        "  Ցույց է տալիս բոլոր account-ների ցանկը (real + paper), index-ներով և SL/TP կարգավորումներով",
        "- /balances",
        "  Real account-ների Binance Futures USDT balance/availableBalance (Binance-ից)",
        "- /positions [account_name|index]",
        "  Բաց պոզիցիաների live վիճակ՝ qty, entry/mark, unreal%, SL/TP, և օրդերների coverage",
        "- /daily կամ /today",
        "  Օրական հաշվետվություն (start-of-day snapshot + հիմա) + այսօր կատարված trades-երի ամփոփ",
        "- /summary <account_name|index> [hours] [limit]",
        "  Արագ ամփոփ + վերջին exits ցուցակ",
        "- /report <account_name|index> [24h|7d|30d|all]",
        "  Ընդլայնված հաշվետվություն՝ ժամային breakdown, top ժամեր, max drawdown, best/worst trade",
        "- /pause_status",
        "  Ցույց է տալիս drawdown pause եղած account-ները",
        "- /resume [account_index]",
        "  Resume paused account-ը (կամ բոլոր paused account-ները) և reset անել baseline-ը",
        "- /set_sl <account_name> <percent>",
        "- /set_tp <account_name> p1 p2 [p3]",
        "- /set_tp_mode <account_name> <mode>",
        "- /set_confidence <account_name> <percent>",
        "- /add_real <name>",
        "- /add_paper <name>",
        "- /delete_account <name>",
        "- /restart",
        "  Վերագործարկում է հիմնական trading բոտը (main.py)",
        "- /my_id կամ /whoami",
        "  Ցույց է տալիս քո user_id/chat_id-ը (setup-ի համար)",
        "",
        "ՕՐԻՆԱԿՆԵՐ (ԹԵՍՏ)՝",
        "  /accounts",
        "  /balances",
        "  /positions",
        "  /positions Vardan",
        "  /positions 1",
        "  /daily",
        "  /summary Samo 24 20",
        "  /report Samo 24h",
        "  /report Samo 7d",
        "  /report 0 24h",
        "  /pause_status",
        "  /resume 4",
        "",
        "/balances մանրամասներ՝",
        "- Ցույց է տալիս real account-ների Futures USDT balance-ը և availableBalance-ը (Binance-ից)",
        "- Հնարավոր է փոքր cache (~20 վրկ), որ չծանրաբեռնենք API-ը",
        "",
        "/positions մանրամասներ՝",
        "- Ցույց է տալիս open trades-ը ըստ account-ի՝ symbol/side/qty/entry/mark/unrealized%",
        "- Ցույց է տալիս նաև SL/TP config-ը և բորսայում դրված open TP/SL order-ների coverage-ը",
        "- Կարող ես ընտրել միայն մեկ account՝ /positions <name|index>",
        "",
        "/daily (/today) մանրամասներ՝",
        "- Առաջին կանչի պահին օրվա համար պահում է start-of-day balance snapshot (ֆայլում)",
        "- Հետո հաշվարկում է Now vs Start տարբերությունը՝ ΔUSDT և Δ%",
        "- 'Today trades' և 'Realized≈' թվերը գալիս են data/trade_history/account_X.csv history-ից",
        "- 'Avg move (net)%' = price move %-ի մոտարկված միջին (fee-ով ճշգրտված), ոչ թե leverage ROI",
        "- 'ROI@lev≈' = մոտարկում՝ Avg move% * leverage (եթե leverage-ը գրված է binance_accounts.yaml-ում)",
        "",
        "Անվտանգություն՝",
        "- Այս բոտը պատասխանում է միայն այն chat_id-ին, որը նշված է config/trading.yaml → telegram_chat_id",
        "",
        "Drawdown pause (20%)՝",
        "- /pause_status – տեսնել paused account-ները",
        "- /resume – բացել բոլոր paused account-ները և reset անել baseline-ը",
        "- /resume <index> – բացել մեկ paused account-ը և reset անել baseline-ը",
        "",
        "Փոփոխված SL/TP կարգավորումները գրվում են config/binance_accounts.yaml",
        "և config/paper_accounts.yaml ֆայլերի մեջ:",
        "Հիմնական trading բոտը պետք է վերագործարկվի, որպեսզի նոր արժեքները ուժի մեջ մտնեն։",
        "Նույնը վերաբերում է նաև նոր account-ների ստեղծմանը և ջնջմանը:",
        "Դու կարող ես գրել /restart, որպեսզի սերվերի վրա main.py պրոցեսը ինքն իրեն վերագործարկի։",
    ]
    return ["\n".join(lines)]


def _handle_balances_command() -> List[str]:
    now = time.time()
    try:
        ts = float(_BALANCE_CACHE.get("ts") or 0.0)
    except Exception:
        ts = 0.0
    if now - ts < _BALANCE_CACHE_TTL_SEC:
        try:
            cached = str(_BALANCE_CACHE.get("text") or "")
        except Exception:
            cached = ""
        if cached:
            return [cached]

    lines: List[str] = []
    lines.append("Binance Futures balances (USDT):")
    lines.append("")

    any_real = False
    for acc, bal in _fetch_real_futures_balances():
        any_real = True
        name = str(acc.get("name") or "?")
        if not bal:
            try:
                err = str(acc.get("_balance_err") or "ERR")
            except Exception:
                err = "ERR"
            lines.append(f"- {name}: balance=ERR ({err})")
            continue

        b = float(bal.get("balance") or 0.0)
        a = float(bal.get("availableBalance") or 0.0)
        lines.append(f"- {name}: balance={b:.4f} | available={a:.4f}")

    if not any_real:
        return ["No real Binance accounts configured."]

    msg = "\n".join(lines)
    _BALANCE_CACHE["ts"] = now
    _BALANCE_CACHE["text"] = msg
    return [msg]


def _fetch_open_orders_for_symbol(account_index: int, symbol: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        clients = binance_futures._get_clients()  # type: ignore[attr-defined]
        if not clients:
            return out
        idx = int(account_index)
        if idx < 0 or idx >= len(clients):
            return out
        client = clients[idx]
    except Exception:
        return out

    def _to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v or "").strip().lower()
        return s in ("1", "true", "yes", "y")

    def _norm(raw: Dict[str, Any], is_algo: bool) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        try:
            order_type = str(raw.get("type") or raw.get("orderType") or "").upper()
            side = str(raw.get("side") or "").upper()
            qty = float(raw.get("origQty") if raw.get("origQty") is not None else (raw.get("quantity") or 0.0))
            if qty <= 0:
                qty = float(raw.get("qty") or 0.0)
            trig = float(
                raw.get("stopPrice")
                if raw.get("stopPrice") is not None
                else (raw.get("triggerPrice") if raw.get("triggerPrice") is not None else (raw.get("price") or 0.0))
            )
            close_pos = _to_bool(raw.get("closePosition"))
        except Exception:
            return None
        return {
            "type": order_type,
            "side": side,
            "qty": float(max(qty, 0.0)),
            "trigger_price": float(max(trig, 0.0)),
            "close_position": bool(close_pos),
            "is_algo": bool(is_algo),
        }

    try:
        regular = client.open_orders(symbol=str(symbol).upper())
        if isinstance(regular, list):
            for r in regular:
                n = _norm(r, False)
                if n is not None:
                    out.append(n)
    except Exception:
        pass

    try:
        algo = client.get_algo_open_orders(symbol=str(symbol).upper())
        algo_list = algo if isinstance(algo, list) else (algo.get("orders") or [] if isinstance(algo, dict) else [])
        if isinstance(algo_list, list):
            for r in algo_list:
                n = _norm(r, True)
                if n is not None:
                    out.append(n)
    except Exception:
        pass
    return out


def _handle_positions_command(text: str) -> List[str]:
    """Show live open positions + protection-order state.

    Usage:
      /positions
      /positions <account_name|index>
    """
    parts = text.split(maxsplit=1)
    selected_acc: Optional[Dict[str, Any]] = None
    if len(parts) > 1 and parts[1].strip():
        key = parts[1].strip()
        selected_acc = _resolve_account(key)
        if not selected_acc:
            msg = _format_accounts_list()
            return ["Unknown account name/index. Use /accounts to see the list.", "", msg]
        if str(selected_acc.get("type") or "").lower() != "real":
            return [f"Account '{selected_acc.get('name')}' is paper type. Live Binance positions are available only for real accounts."]

    real_accounts = _get_real_accounts()
    if selected_acc is not None:
        target_accounts = [a for a in real_accounts if int(a.get("index", -1)) == int(selected_acc.get("index", -9999))]
    else:
        target_accounts = list(real_accounts)

    if not target_accounts:
        return ["No real Binance accounts configured."]

    try:
        snapshot = binance_futures.get_open_positions_snapshot() or []
    except Exception as e:
        return [f"Failed to load open positions snapshot: {e}"]

    by_acc: Dict[int, List[Dict[str, Any]]] = {}
    for row in snapshot:
        if not isinstance(row, dict):
            continue
        try:
            ai = int(row.get("account_index"))
            amt = float(row.get("position_amt") or 0.0)
        except Exception:
            continue
        if amt == 0.0:
            continue
        by_acc.setdefault(ai, []).append(row)

    lines: List[str] = []
    lines.append("Live positions report (all open trades):")
    lines.append("")

    total_positions = 0
    for acc in target_accounts:
        try:
            ai = int(acc.get("index"))
        except Exception:
            continue
        name = str(acc.get("name") or f"acc_{ai}")
        rows = by_acc.get(ai, []) or []
        total_positions += len(rows)
        lines.append(f"Account: {name} (index={ai})")
        lines.append(f"Open positions: {len(rows)}")

        if not rows:
            lines.append("- No open positions")
            lines.append("")
            continue

        syms = [str(r.get("symbol") or "").upper() for r in rows if str(r.get("symbol") or "").strip()]
        marks = binance_futures.get_public_mark_prices(syms, account_index=ai) if syms else {}

        for r in rows:
            symbol = str(r.get("symbol") or "").upper()
            amt = float(r.get("position_amt") or 0.0)
            qty_abs = abs(amt)
            side = "LONG" if amt > 0 else "SHORT"
            entry = float(r.get("entry_price") or 0.0)
            mark = float(marks.get(symbol) or 0.0)
            unreal_pct = 0.0
            if entry > 0 and mark > 0:
                direction = 1.0 if amt > 0 else -1.0
                unreal_pct = ((mark - entry) / entry) * 100.0 * direction

            sl_pct = r.get("sl_pct")
            tp_mode = r.get("tp_mode")
            tp_raw = r.get("tp_pcts") if isinstance(r.get("tp_pcts"), list) else []
            tp_pcts: List[float] = []
            for v in tp_raw:
                try:
                    fv = float(v)
                    if fv > 0:
                        tp_pcts.append(fv)
                except Exception:
                    continue
            if tp_mode is None:
                tp_mode = min(2, len(tp_pcts)) if tp_pcts else 0
            try:
                tp_mode_i = int(tp_mode)
            except Exception:
                tp_mode_i = 0
            if tp_mode_i < 0:
                tp_mode_i = 0

            tp_levels = tp_pcts[:tp_mode_i] if tp_mode_i > 0 else []
            tp_prices: List[str] = []
            if entry > 0 and tp_levels:
                for p in tp_levels:
                    if side == "LONG":
                        tp_prices.append(f"{entry * (1.0 + p / 100.0):.6f}")
                    else:
                        tp_prices.append(f"{entry * (1.0 - p / 100.0):.6f}")

            orders = _fetch_open_orders_for_symbol(ai, symbol)
            close_side = "SELL" if side == "LONG" else "BUY"
            tp_types = {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
            sl_types = {"STOP", "STOP_MARKET"}
            tp_orders = [o for o in orders if str(o.get("type") or "") in tp_types and str(o.get("side") or "") == close_side]
            sl_orders = [o for o in orders if str(o.get("type") or "") in sl_types and str(o.get("side") or "") == close_side]
            tp_qty = sum(float(o.get("qty") or 0.0) for o in tp_orders)
            sl_qty = sum(float(o.get("qty") or 0.0) for o in sl_orders if not bool(o.get("close_position")))
            sl_has_close_pos = any(bool(o.get("close_position")) for o in sl_orders)

            lines.append(
                f"- {symbol} | {side} | qty={qty_abs:.6f} | entry={entry:.6f} | mark={mark:.6f} | unreal={unreal_pct:+.2f}%"
            )
            lines.append(
                f"  risk: SL%={sl_pct if sl_pct is not None else '?'} | TP mode={tp_mode_i} | TP%={tp_levels if tp_levels else '-'}"
            )
            if tp_prices:
                lines.append(f"  targets: {', '.join(tp_prices)}")
            else:
                lines.append("  targets: -")
            lines.append(
                f"  orders: TP={len(tp_orders)} (qty={tp_qty:.6f}) | SL={len(sl_orders)} "
                f"(qty={sl_qty:.6f}, closePos={'yes' if sl_has_close_pos else 'no'})"
            )
        lines.append("")

    lines.append(f"TOTAL open positions: {total_positions}")
    lines.append("")
    lines.append("Usage: /positions [account_name|index]")
    return ["\n".join(lines).strip()]


def _handle_daily_command() -> List[str]:
    day_key = _today_key_local()
    snaps = _load_daily_snapshots()
    days = snaps.get("days") if isinstance(snaps, dict) else None
    if not isinstance(days, dict):
        days = {}
        snaps = {"days": days}

    day_node = days.get(day_key)
    if not isinstance(day_node, dict):
        day_node = {"ts": time.time(), "accounts": {}}
        days[day_key] = day_node

    acc_snaps = day_node.get("accounts")
    if not isinstance(acc_snaps, dict):
        acc_snaps = {}
        day_node["accounts"] = acc_snaps

    # Ensure start-of-day snapshot exists for today.
    if not acc_snaps:
        for acc, bal in _fetch_real_futures_balances():
            key = str(acc.get("key") or acc.get("name") or "").strip().lower()
            if not key:
                continue
            if not bal:
                continue
            try:
                acc_snaps[key] = {
                    "name": str(acc.get("name") or key),
                    "balance": float(bal.get("balance") or 0.0),
                    "availableBalance": float(bal.get("availableBalance") or 0.0),
                }
            except Exception:
                continue
        _save_daily_snapshots(snaps)

    start_ms = int(_local_midnight_epoch_sec() * 1000.0)

    lines: List[str] = []
    lines.append(f"Daily report — {day_key}")
    lines.append("")

    total_start = 0.0
    total_now = 0.0
    total_realized = 0.0
    any_real = False

    for acc, bal in _fetch_real_futures_balances():
        any_real = True
        key = str(acc.get("key") or acc.get("name") or "").strip().lower()
        name = str(acc.get("name") or "?")
        lev = acc.get("leverage")

        snap = acc_snaps.get(key) if key else None
        try:
            start_bal = float(snap.get("balance") or 0.0) if isinstance(snap, dict) else 0.0
        except Exception:
            start_bal = 0.0

        if bal:
            now_bal = float(bal.get("balance") or 0.0)
            now_av = float(bal.get("availableBalance") or 0.0)
        else:
            now_bal = 0.0
            now_av = 0.0

        total_start += start_bal
        total_now += now_bal

        delta = now_bal - start_bal
        pct = (delta / start_bal * 100.0) if start_bal > 0 else 0.0

        # Trade stats for today based on trade_history CSV.
        path: Path = acc.get("history_path")
        rows = _load_history_rows(path)
        rows_today = _filter_rows_since_ms(rows, start_ms)
        stats = _summarise_rows(rows_today) if rows_today else None

        if isinstance(stats, dict):
            realized = float(stats.get("realized_pnl_usd") or 0.0)
            total_realized += realized
            exits = int(stats.get("exits") or 0)
            wins = int(stats.get("wins") or 0)
            losses = int(stats.get("losses") or 0)
            avg_move = float(stats.get("avg_pnl") or 0.0)
        else:
            realized = 0.0
            exits = 0
            wins = 0
            losses = 0
            avg_move = 0.0

        roi_lev_txt = "-"
        try:
            if lev is not None:
                roi_lev = float(avg_move) * float(lev)
                roi_lev_txt = f"{roi_lev:.2f}%"
        except Exception:
            roi_lev_txt = "-"

        lines.append(
            "\n".join(
                [
                    f"{name}",
                    f"Start: {start_bal:.4f} | Now: {now_bal:.4f} (avail {now_av:.4f})",
                    f"Δ: {delta:+.4f} USDT | Δ%: {pct:+.2f}%",
                    f"Today trades: exits={exits} (W={wins}, L={losses}) | Realized≈ {realized:+.2f} USDT",
                    f"Avg move (net)≈ {avg_move:+.2f}% | ROI@lev≈ {roi_lev_txt}",
                ]
            )
        )
        lines.append("")

    if not any_real:
        return ["No real Binance accounts configured."]

    total_delta = total_now - total_start
    total_pct = (total_delta / total_start * 100.0) if total_start > 0 else 0.0

    lines.append("TOTAL")
    lines.append(f"Start: {total_start:.4f} | Now: {total_now:.4f}")
    lines.append(f"Δ: {total_delta:+.4f} USDT | Δ%: {total_pct:+.2f}%")
    lines.append(f"Realized PnL (approx from history): {total_realized:+.2f} USDT")

    return ["\n".join(lines).strip()]


def _handle_pause_status_command() -> List[str]:
    try:
        paused = binance_futures.get_paused_accounts_snapshot()
    except Exception:
        paused = []
    if not paused:
        return ["No paused accounts."]
    lines = ["Paused accounts:"]
    for it in paused:
        if not isinstance(it, dict):
            continue
        idx = it.get("index")
        name = it.get("name")
        dd_pct = it.get("dd_pct")
        cur = it.get("current_balance")
        base = it.get("baseline_balance")
        try:
            dd_txt = f"{float(dd_pct):.2f}%" if dd_pct is not None else "?"
        except Exception:
            dd_txt = "?"
        try:
            cur_txt = f"{float(cur):.2f}" if cur is not None else "?"
        except Exception:
            cur_txt = "?"
        try:
            base_txt = f"{float(base):.2f}" if base is not None else "?"
        except Exception:
            base_txt = "?"
        lines.append(f"- index={idx} name={name} dd={dd_txt} (start {base_txt} -> now {cur_txt} USDT)")
    lines.append("")
    lines.append("Use /resume to resume ALL paused accounts.")
    lines.append("Use /resume <index> to resume a single account.")
    return ["\n".join(lines)]


def _handle_resume_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 2:
        try:
            res = binance_futures.resume_all_accounts_from_drawdown_pause()
        except Exception:
            res = {"ok": False, "resumed": 0, "failed": 0}
        resumed = int(res.get("resumed") or 0)
        failed = int(res.get("failed") or 0)
        if resumed <= 0 and failed <= 0:
            return ["No paused accounts to resume."]
        if failed > 0:
            return [f"Resumed {resumed} account(s). Failed: {failed} (see logs)."]
        return [f"Resumed {resumed} account(s). Baselines reset to current balances."]
    try:
        idx = int(parts[1])
    except Exception:
        return ["Invalid account index."]
    try:
        ok = binance_futures.resume_account_from_drawdown_pause(idx)
    except Exception:
        ok = False
    if not ok:
        return [f"Failed to resume account {idx} (or account not paused)."]
    return [f"Account {idx} resumed. Baseline reset to current balance."]


def _handle_summary_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 2:
        return ["Usage: /summary <account_name> [hours] [limit]"]

    acc_key = parts[1].strip()
    acc = _resolve_account(acc_key)
    if not acc:
        msg = _format_accounts_list()
        return ["Unknown account name. Use /accounts to see the list.", "", msg]

    hours: Optional[float] = None
    limit: int = 10

    if len(parts) >= 3:
        # Support either 'hours' value or the word 'limit'
        p = parts[2].strip()
        if p.lower() == "limit" and len(parts) >= 4:
            try:
                limit = int(parts[3])
            except (TypeError, ValueError):
                limit = 10
        else:
            try:
                hours = float(p)
            except (TypeError, ValueError):
                hours = None

    if len(parts) >= 4 and hours is not None:
        try:
            limit = int(parts[3])
        except (TypeError, ValueError):
            limit = 10

    if limit <= 0:
        limit = 10

    path: Path = acc.get("history_path")
    rows = _load_history_rows(path)
    rows = _filter_by_hours(rows, hours)

    if not rows:
        if hours is not None and hours > 0:
            return [
                f"No trades found for account {acc['name']} in the last {hours}h.",
            ]
        return [f"No trades found for account {acc['name']}."]

    msg = _format_summary_message(acc, rows, hours, limit)
    return [msg]


def _handle_set_sl_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 3:
        return ["Usage: /set_sl <account_name> <percent>"]

    acc_key = parts[1].strip()
    try:
        sl_pct = float(parts[2])
    except (TypeError, ValueError):
        return ["Invalid percent value."]

    acc = _resolve_account(acc_key)
    if not acc:
        msg = _format_accounts_list()
        return ["Unknown account name. Use /accounts to see the list.", "", msg]

    if not _update_account_settings_yaml(acc, sl_pct=sl_pct, tp_pcts=None, tp_mode=None):
        return ["Failed to update SL% in YAML. Please check server logs."]

    name = acc.get("name", acc_key)
    return [
        f"Updated SL% for account {name} to {sl_pct}%.",
        "Please restart your main trading bot so that new settings take effect.",
    ]


def _handle_set_tp_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 3:
        return ["Usage: /set_tp <account_name> p1 p2 [p3]"]

    acc_key = parts[1].strip()
    raw_levels = parts[2:]
    levels: List[float] = []
    for val in raw_levels:
        try:
            v = float(val)
            if v > 0:
                levels.append(v)
        except (TypeError, ValueError):
            continue

    if not levels:
        return ["No valid TP levels provided."]

    acc = _resolve_account(acc_key)
    if not acc:
        msg = _format_accounts_list()
        return ["Unknown account name. Use /accounts to see the list.", "", msg]

    if not _update_account_settings_yaml(acc, sl_pct=None, tp_pcts=levels, tp_mode=None):
        return ["Failed to update TP% levels in YAML. Please check server logs."]

    name = acc.get("name", acc_key)
    levels_str = ", ".join(str(x) for x in levels)
    return [
        f"Updated TP% levels for account {name} to: {levels_str}",
        "Please restart your main trading bot so that new settings take effect.",
    ]


def _handle_add_real_command(text: str) -> List[str]:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return ["Usage: /add_real <name>"]

    name = parts[1].strip()
    if not name:
        return ["Account name cannot be empty."]

    if not _add_real_account(name):
        return [f"Real account with name '{name}' already exists or could not be created."]

    return [
        f"Real account '{name}' added to config/binance_accounts.yaml.",
        "Please open that file to fill api_key/api_secret, and then restart your main trading bot.",
    ]


def _handle_add_paper_command(text: str) -> List[str]:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return ["Usage: /add_paper <name>"]

    name = parts[1].strip()
    if not name:
        return ["Account name cannot be empty."]

    if not _add_paper_account(name):
        return [f"Paper account with name '{name}' already exists or could not be created."]

    return [
        f"Paper account '{name}' added to config/paper_accounts.yaml.",
        "Please restart your main trading bot so that new settings take effect.",
    ]


def _handle_delete_account_command(text: str) -> List[str]:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return ["Usage: /delete_account <name>"]

    name = parts[1].strip()
    if not name:
        return ["Account name cannot be empty."]

    deleted = _delete_account_by_name(name)
    if not deleted:
        return [
            f"No real or paper account found with name '{name}'.",
            "Use /accounts to see the current list.",
        ]

    return [
        f"Deleted {deleted}.",
        "Please restart your main trading bot so that new settings take effect.",
    ]


def _handle_set_tp_mode_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 3:
        return ["Usage: /set_tp_mode <account_name> <mode>"]

    acc_key = parts[1].strip()
    try:
        mode_val = int(parts[2])
    except (TypeError, ValueError):
        return ["Invalid TP mode value. Use 1, 2 or 3."]

    if mode_val not in (1, 2, 3):
        return ["TP mode must be 1, 2 or 3."]

    acc = _resolve_account(acc_key)
    if not acc:
        msg = _format_accounts_list()
        return ["Unknown account name. Use /accounts to see the list.", "", msg]

    if not _update_account_settings_yaml(acc, sl_pct=None, tp_pcts=None, tp_mode=mode_val):
        return ["Failed to update TP mode in YAML. Please check server logs."]

    name = acc.get("name", acc_key)
    return [
        f"Updated TP mode for account {name} to {mode_val}.",
        "Please restart your main trading bot so that new settings take effect.",
    ]


def _handle_set_confidence_command(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 3:
        return ["Usage: /set_confidence <account_name> <percent>"]

    acc_key = parts[1].strip()
    try:
        conf_val = float(parts[2])
    except (TypeError, ValueError):
        return ["Invalid confidence value. Provide a positive number (e.g. 70 or 0.7)."]

    if conf_val <= 0:
        return ["Confidence threshold must be greater than 0."]

    acc = _resolve_account(acc_key)
    if not acc:
        msg = _format_accounts_list()
        return ["Unknown account name. Use /accounts to see the list.", "", msg]

    if not _update_account_settings_yaml(
        acc,
        sl_pct=None,
        tp_pcts=None,
        tp_mode=None,
        confidence=conf_val,
    ):
        return ["Failed to update confidence threshold in YAML. Please check server logs."]

    name = acc.get("name", acc_key)
    return [
        f"Updated confidence threshold for account {name} to {conf_val}.",
        "Please restart your main trading bot so that new settings take effect.",
    ]


def _handle_restart_command() -> List[str]:
    global _RESTART_REQUESTED
    _RESTART_REQUESTED = True
    return [
        "Restart requested.",
        "The main trading bot process (main.py) will now restart to apply new settings.",
    ]


def _handle_text_message(msg: Dict[str, Any]) -> List[str]:
    text = (msg.get("text") or "").strip()
    if not text:
        return []

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if not _is_authorized_chat(chat_id):
        return []

    if not _is_authorized_sender(msg):
        return []

    if text.startswith("/my_id") or text.startswith("/whoami"):
        return _handle_my_id_command(msg)

    if text.startswith("/start") or text.startswith("/help") or text.lower().startswith("help"):
        return _handle_command_start()

    if text.startswith("/accounts"):
        return [_format_accounts_list()]

    if text.startswith("/balances"):
        return _handle_balances_command()

    if text.startswith("/positions"):
        return _handle_positions_command(text)

    if text.startswith("/daily") or text.startswith("/today"):
        return _handle_daily_command()

    if text.startswith("/pause_status"):
        return _handle_pause_status_command()

    if text.startswith("/resume"):
        return _handle_resume_command(text)

    if text.startswith("/summary"):
        return _handle_summary_command(text)

    if text.startswith("/report"):
        return _handle_report_command(text)

    if text.startswith("/set_sl"):
        return _handle_set_sl_command(text)

    if text.startswith("/set_tp"):
        return _handle_set_tp_command(text)

    if text.startswith("/set_tp_mode"):
        return _handle_set_tp_mode_command(text)

    if text.startswith("/set_confidence"):
        return _handle_set_confidence_command(text)

    if text.startswith("/add_real"):
        return _handle_add_real_command(text)

    if text.startswith("/add_paper"):
        return _handle_add_paper_command(text)

    if text.startswith("/delete_account"):
        return _handle_delete_account_command(text)

    if text.startswith("/restart") or text.lower().strip() == "restart":
        return _handle_restart_command()

    return [
        "Unknown command. Use /start to see available commands.",
    ]


def run_account_report_bot() -> None:
    token = _load_report_bot_token()
    if not token:
        print("[ACCOUNT REPORT BOT] reports_bot.token is not configured in config/trading.yaml")
        return

    print("[ACCOUNT REPORT BOT] Starting account report bot polling loop...")

    state = _load_state()
    offset = int(state.get("offset", 0))

    # Եթե նախկինում state-ը գրվել է ուրիշ token-ով, ապա offset-ը կարող է սխալ լինել
    # և նոր բոտի համար update-ները երբեք չեն երեւալու։
    if state.get("token") != token:
        offset = 0
        state = {"offset": 0, "token": token}
        _save_state(state)

    while True:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 10, "offset": offset + 1}

        try:
            resp = requests.get(url, params=params, timeout=15)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                status = resp.status_code
                text = resp.text[:300]
                print(
                    f"[ACCOUNT REPORT BOT] getUpdates HTTP error {status}: {text}"
                )

                # Եթե webhook է ակտիվ, Telegram-ը տալիս է 409 Conflict.
                if status == 409 and "webhook" in text.lower():
                    try:
                        del_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
                        del_resp = requests.get(del_url, timeout=10)
                        print(
                            f"[ACCOUNT REPORT BOT] deleteWebhook status {del_resp.status_code}: {del_resp.text[:200]}"
                        )
                    except Exception as del_e:
                        print(
                            f"[ACCOUNT REPORT BOT] Failed to delete webhook automatically: {del_e}"
                        )

                time.sleep(5)
                continue

            try:
                payload = resp.json()
            except Exception as je:
                print(
                    f"[ACCOUNT REPORT BOT] Failed to parse getUpdates JSON response: {je}"
                )
                time.sleep(5)
                continue
        except Exception as e:
            print(f"[ACCOUNT REPORT BOT] Error while calling getUpdates: {e}")
            time.sleep(5)
            continue

        if not isinstance(payload, dict) or not payload.get("ok"):
            print(
                f"[ACCOUNT REPORT BOT] getUpdates returned non-ok payload: {payload}"
            )
            time.sleep(5)
            continue

        results = payload.get("result", []) or []
        if not results:
            # Debug log to see, որ update-ներ ընդհանրապես չկան
            # print("[ACCOUNT REPORT BOT] getUpdates returned 0 updates (waiting...)")
            time.sleep(2)
            continue

        for update in results:
            try:
                upd_id = int(update.get("update_id"))
                if upd_id > offset:
                    offset = upd_id

                message = update.get("message") or update.get("channel_post")
                if not message:
                    continue

                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                text = (message.get("text") or "").strip()
                print(
                    f"[ACCOUNT REPORT BOT] Update {upd_id} chat_id={chat_id} text={repr(text)}"
                )

                replies = _handle_text_message(message)
                if not replies:
                    continue

                if chat_id is None:
                    continue

                for part in replies:
                    try:
                        send_telegram(part, token, chat_id)
                    except Exception as e:
                        print(f"[ACCOUNT REPORT BOT] Failed to send reply: {e}")
                        continue
            except Exception as e:
                print(f"[ACCOUNT REPORT BOT] Failed to process update: {e}")
                continue

        state["offset"] = offset
        state["token"] = token
        _save_state(state)

        # If a restart was requested via Telegram, restart the main.py process.
        global _RESTART_REQUESTED
        if _RESTART_REQUESTED:
            _RESTART_REQUESTED = False
            main_path = ROOT_DIR / "main.py"
            try:
                print("[ACCOUNT REPORT BOT] /restart received, exec'ing main.py ...")
                os.execl(sys.executable, sys.executable, str(main_path))
            except Exception as e:
                print(f"[ACCOUNT REPORT BOT] Failed to restart main.py via /restart: {e}")

        time.sleep(2)


if __name__ == "__main__":
    run_account_report_bot()
