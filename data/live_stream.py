import time
from data.historical_loader import load_klines
from data.ws_price_stream import get_recent_candles, set_recent_candles


class LiveDataStream:
    def __init__(self, symbol, interval, window=200):
        self.symbol = symbol
        self.interval = interval
        self.window = window
        self.buffer = []

    def update(self):
        candles = load_klines(
            self.symbol,
            self.interval,
            limit=self.window,
        )
        self.buffer = candles
        return self.buffer

    def latest(self):
        return self.buffer[-1] if self.buffer else None


def fetch_candles(symbol, interval, limit=200):
    def _interval_to_seconds(iv: str) -> int:
        try:
            s = str(iv)
        except Exception:
            s = ""
        if s.endswith("m"):
            try:
                return int(s[:-1]) * 60
            except Exception:
                return 60
        if s.endswith("h"):
            try:
                return int(s[:-1]) * 3600
            except Exception:
                return 3600
        if s.endswith("d"):
            try:
                return int(s[:-1]) * 86400
            except Exception:
                return 86400
        return 60

    def _is_fresh(cands) -> bool:
        if not cands:
            return False
        last = cands[-1] if isinstance(cands, list) and cands else None
        if not isinstance(last, dict):
            return False
        ts_raw = last.get("open_time") or last.get("close_time")
        try:
            ts_ms = int(float(ts_raw))
        except Exception:
            return False
        if ts_ms <= 0:
            return False
        now = time.time()
        age = now - (float(ts_ms) / 1000.0)
        try:
            iv_sec = float(_interval_to_seconds(interval))
        except Exception:
            iv_sec = 60.0
        if iv_sec <= 0:
            iv_sec = 60.0
        if age < -2.0 * iv_sec:
            return False
        return age <= 3.0 * iv_sec

    candles = get_recent_candles(symbol, limit=limit)
    if candles and len(candles) >= limit and _is_fresh(candles):
        return candles

    candles = load_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    if candles and len(candles) >= max(20, int(limit)):
        try:
            set_recent_candles(symbol, candles)
        except Exception:
            pass

    return candles

