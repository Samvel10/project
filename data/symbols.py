import requests
from config.proxies import get_random_proxy

BINANCE_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def get_all_usdt_futures(min_volume=5_000_000):
    attempts = 0
    resp = None

    # Try up to 3 different proxies
    while attempts < 3:
        attempts += 1
        proxies = get_random_proxy()
        try:
            r = requests.get(BINANCE_FUTURES_EXCHANGE_INFO, timeout=10, proxies=proxies)
            r.raise_for_status()
            resp = r
            break
        except requests.HTTPError as e:
            re = e.response
            if re is not None and re.status_code == 451:
                # try another proxy on 451
                continue
            raise
        except Exception:
            # other transient proxy errors: try another proxy
            continue

    # Final fallback: try without proxy
    if resp is None:
        try:
            r = requests.get(BINANCE_FUTURES_EXCHANGE_INFO, timeout=10)
            r.raise_for_status()
            resp = r
        except Exception:
            # If even direct call fails (e.g. region blocked), return empty list
            return []

    data = resp.json()

    symbols = []
    for s in data["symbols"]:
        if (
            s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and s["contractType"] == "PERPETUAL"
        ):
            symbols.append(s["symbol"])

    return symbols
