import re
import time
import threading
from typing import Set

import requests

from monitoring.logger import log
from config.proxies import get_random_proxy

_BINANCE_ANN_URL = "https://www.binance.com/en/support/announcement/delisting"
_FAPI_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_CACHE_TTL_SECONDS = 3600
_LAST_FETCH_TS: float = 0.0
_DELISTING_BASES: Set[str] = set()
_REFRESH_LOCK = threading.Lock()
_LAST_ATTEMPT_TS: float = 0.0


def _safe_get(url: str, headers=None, timeout: int = 10):
    try:
        proxies = get_random_proxy()
        resp = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        if resp.status_code >= 300:
            return None
        return resp
    except Exception as e:
        try:
            log(f"[DELIST] Request failed for {url}: {e}")
        except Exception:
            pass
        return None


def _refresh_delisting_bases_if_needed() -> None:
    global _LAST_FETCH_TS, _DELISTING_BASES, _LAST_ATTEMPT_TS
    now = time.time()
    if _LAST_FETCH_TS and now - _LAST_FETCH_TS < _CACHE_TTL_SECONDS:
        return

    # If another thread is already refreshing, do not block the trading loop.
    acquired = _REFRESH_LOCK.acquire(blocking=False)
    if not acquired:
        return

    try:
        # Double-check after acquiring the lock.
        now = time.time()
        if _LAST_FETCH_TS and now - _LAST_FETCH_TS < _CACHE_TTL_SECONDS:
            return

        # Avoid hammering endpoints if they are currently failing.
        if _LAST_ATTEMPT_TS and now - _LAST_ATTEMPT_TS < 60.0:
            return
        _LAST_ATTEMPT_TS = now
        resp_ann = _safe_get(_BINANCE_ANN_URL, headers=_HEADERS, timeout=10)
        if not resp_ann:
            return

        text = resp_ann.text.upper()
        raw = set(re.findall(r"\b[A-Z0-9]{2,10}\b", text))

        blacklist = {
            "BINANCE",
            "WILL",
            "SPOT",
            "FUTURES",
            "TRADING",
            "DELIST",
            "DELISTING",
            "PAIR",
            "MARGIN",
            "USD",
            "USDT",
            "USDC",
            "BUSD",
            "TOKEN",
        }

        delisting = raw - blacklist

        resp_info = _safe_get(_FAPI_EXCHANGE_INFO_URL, timeout=10)
        if not resp_info:
            return

        try:
            data = resp_info.json()
        except Exception:
            return

        futures_bases: Set[str] = set()
        try:
            for s in data.get("symbols", []):
                if s.get("status") == "TRADING":
                    base = s.get("baseAsset")
                    if isinstance(base, str):
                        futures_bases.add(base.upper())
        except Exception:
            futures_bases = set()

        _DELISTING_BASES = delisting.intersection(futures_bases)
        _LAST_FETCH_TS = time.time()
    finally:
        try:
            _REFRESH_LOCK.release()
        except Exception:
            pass


def is_delisting_symbol(symbol: str) -> bool:
    if not symbol:
        return False

    _refresh_delisting_bases_if_needed()

    if not _DELISTING_BASES:
        return False

    sym = str(symbol).upper()
    base = sym
    if sym.endswith("USDT"):
        base = sym[:-4]

    return base in _DELISTING_BASES
