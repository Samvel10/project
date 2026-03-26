import json
import random
import threading
import time
from typing import Dict, Iterable, Optional, List

try:
    import websocket  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    websocket = None  # type: ignore[assignment]

from monitoring.logger import log
from config.proxies import PROXIES
from execution.binance_futures import _should_use_proxy_for_trading


_LAST_PRICES: Dict[str, float] = {}
_LAST_PRICE_TS: Dict[str, float] = {}
_LAST_PRICE_TTL_SEC: float = 30.0
_KLINE_BUFFERS: Dict[str, List[dict]] = {}
_WS_THREAD: Optional[threading.Thread] = None
_WS_RUNNING = False
_WS_LOCK = threading.Lock()


def _get_ws_proxy_kwargs() -> dict:
    """Return proxy kwargs for websocket-client.

    For now we explicitly disable proxies for WebSocket connections to avoid
    protocol compatibility issues. All WS traffic goes directly from the
    current host to Binance, while REST requests may still use proxies.
    """

    return {}


def _on_message(ws, message: str) -> None:  # type: ignore[override]
    try:
        data = json.loads(message)
    except Exception:
        return

    payload = data.get("data") or {}
    kline = payload.get("k")
    if isinstance(kline, dict) and kline:
        symbol = kline.get("s")
        close_str = kline.get("c")
        if not symbol or close_str is None:
            return

        try:
            price = float(close_str)
        except (TypeError, ValueError):
            return

        if price <= 0:
            return

        sym_u = str(symbol).upper()

        with _WS_LOCK:
            _LAST_PRICES[sym_u] = price
            try:
                _LAST_PRICE_TS[sym_u] = time.time()
            except Exception:
                pass

            try:
                open_time = kline.get("t")
                close_time = kline.get("T")
                open_str = kline.get("o")
                high_str = kline.get("h")
                low_str = kline.get("l")
                vol_str = kline.get("v")

                if (
                    open_time is None
                    or close_time is None
                    or open_str is None
                    or high_str is None
                    or low_str is None
                    or vol_str is None
                ):
                    return

                o = float(open_str)
                h = float(high_str)
                l = float(low_str)
                v = float(vol_str)
            except (TypeError, ValueError):
                return

            candle = {
                "open_time": open_time,
                "open": o,
                "high": h,
                "low": l,
                "close": price,
                "volume": v,
                "close_time": close_time,
            }

            buf = _KLINE_BUFFERS.get(sym_u)
            if buf is None:
                buf = []
                _KLINE_BUFFERS[sym_u] = buf

            if buf and buf[-1].get("open_time") == open_time:
                buf[-1] = candle
            else:
                buf.append(candle)
                if len(buf) > 500:
                    del buf[:-500]
        return

    symbol = payload.get("s")
    close_str = payload.get("c")
    if not symbol or close_str is None:
        return

    try:
        price = float(close_str)
    except (TypeError, ValueError):
        return
    if price <= 0:
        return

    sym_u = str(symbol).upper()
    with _WS_LOCK:
        _LAST_PRICES[sym_u] = price
        try:
            _LAST_PRICE_TS[sym_u] = time.time()
        except Exception:
            pass


def _on_error(ws, error) -> None:  # type: ignore[override]
    try:
        ""
        # log(f"[WS PRICE] WebSocket error: {error}")
    except Exception:
        pass


def _on_close(ws, close_status_code, close_msg) -> None:  # type: ignore[override]
    try:
        ""
        # log(
        #     f"[WS PRICE] WebSocket closed: code={close_status_code}, msg={close_msg}"
        # )
    except Exception:
        pass


def _run_ws(url: str) -> None:
    global _WS_RUNNING

    if websocket is None:
        log("[WS PRICE] websocket-client not installed; price stream disabled")
        return

    proxy_kwargs = _get_ws_proxy_kwargs()

    while _WS_RUNNING:
        try:
            ws = websocket.WebSocketApp(  # type: ignore[call-arg]
                url,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10, **proxy_kwargs)  # type: ignore[arg-type]
        except Exception as e:
            try:
                log(f"[WS PRICE] run_forever error: {e}")
            except Exception:
                pass
            time.sleep(5.0)


def start_price_stream(symbols: Iterable[str], interval: str) -> bool:
    """Start a background Binance Futures kline WebSocket stream.

    - symbols: iterable of symbol strings, e.g. ["BTCUSDT", "ETHUSDT"].
    - interval: kline interval string, e.g. "1m", "5m".

    If the stream is already running, this function is a no-op.
    Returns True when a WS thread is started (or already active), False when
    streaming cannot be started (e.g. missing websocket-client or empty symbols).
    """

    global _WS_THREAD, _WS_RUNNING

    sym_list = [str(s).lower() for s in symbols if s]
    if not sym_list:
        return False

    if websocket is None:
        try:
            log("[WS PRICE] websocket-client not installed; price stream disabled")
        except Exception:
            pass
        return False

    with _WS_LOCK:
        if _WS_THREAD is not None and _WS_THREAD.is_alive():
            return True
        _WS_RUNNING = True

    parts = []
    for sym in sym_list:
        parts.append(f"{sym}@kline_{interval}")
        parts.append(f"{sym}@miniTicker")
    stream = "/".join(parts)
    url = f"wss://fstream.binance.com/stream?streams={stream}"

    t = threading.Thread(target=_run_ws, args=(url,), daemon=True)
    _WS_THREAD = t
    try:
        log(
            f"[WS PRICE] Starting Binance Futures kline stream for {len(sym_list)} symbols, interval={interval}"
        )
    except Exception:
        pass
    t.start()
    return True


def get_last_price(symbol: str) -> Optional[float]:
    """Return the latest streamed price for symbol, or None if unknown."""

    if not symbol:
        return None

    sym_u = str(symbol).upper()
    with _WS_LOCK:
        px = _LAST_PRICES.get(sym_u)
        ts = _LAST_PRICE_TS.get(sym_u)

    if px is None:
        return None
    try:
        if float(px) <= 0:
            return None
    except Exception:
        return None

    if ts is None:
        return float(px)
    try:
        age = time.time() - float(ts)
    except Exception:
        age = 0.0
    try:
        ttl = float(_LAST_PRICE_TTL_SEC)
    except Exception:
        ttl = 30.0
    if ttl > 0 and age > ttl:
        return None
    return float(px)


def get_recent_candles(symbol: str, limit: int = 200) -> List[dict]:
    if not symbol or limit <= 0:
        return []

    sym_u = str(symbol).upper()
    with _WS_LOCK:
        buf = _KLINE_BUFFERS.get(sym_u) or []
        if not buf:
            return []
        if len(buf) <= limit:
            return list(buf)
        return buf[-limit:]


def set_recent_candles(symbol: str, candles: List[dict]) -> None:
    if not symbol or not candles:
        return

    sym_u = str(symbol).upper()
    cleaned = []
    for c in candles:
        if isinstance(c, dict) and c.get("open_time") is not None:
            cleaned.append(c)
    if not cleaned:
        return

    if len(cleaned) > 500:
        cleaned = cleaned[-500:]

    last_px = None
    try:
        last_px = float(cleaned[-1].get("close"))
    except Exception:
        last_px = None

    with _WS_LOCK:
        _KLINE_BUFFERS[sym_u] = list(cleaned)
        if last_px is not None and last_px > 0:
            _LAST_PRICES[sym_u] = float(last_px)
            try:
                _LAST_PRICE_TS[sym_u] = time.time()
            except Exception:
                pass
