import requests
import time
from config.proxies import get_random_proxy

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


def load_klines(
    symbol: str,
    interval: str,
    limit: int = 1500,
    start_time: int | None = None,
):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time

    attempts = 0
    resp = None
    while attempts < 3:
        attempts += 1
        proxies = get_random_proxy()
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10, proxies=proxies)
            resp.raise_for_status()
            break
        except requests.HTTPError as e:
            r = e.response
            if r is not None and r.status_code == 451:
                # try another proxy
                continue
            # other HTTP errors bubble up
            raise
        except requests.RequestException:
            # proxy/connection errors – try another proxy
            continue

    if resp is None:
        # all attempts failed with proxies; skip this symbol silently
        return []

    try:
        raw = resp.json()
    except ValueError:
        # invalid JSON – skip symbol
        return []

    if not isinstance(raw, list):
        # unexpected structure – skip symbol
        return []

    candles = []
    for k in raw:
        # protect against malformed rows
        if not isinstance(k, (list, tuple)) or len(k) < 7:
            continue
        try:
            candle = {
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            }
            if len(k) > 8:
                try:
                    candle["trades"] = int(k[8])
                except (TypeError, ValueError):
                    pass
            candles.append(candle)
        except (TypeError, ValueError):
            # skip malformed row
            continue

    return candles
