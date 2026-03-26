def resample(candles, factor: int):
    """
    factor=5 → 1m → 5m
    """
    out = []
    for i in range(0, len(candles), factor):
        chunk = candles[i : i + factor]
        if len(chunk) < factor:
            continue

        out.append(
            {
                "open_time": chunk[0]["open_time"],
                "open": chunk[0]["open"],
                "high": max(c["high"] for c in chunk),
                "low": min(c["low"] for c in chunk),
                "close": chunk[-1]["close"],
                "volume": sum(c["volume"] for c in chunk),
                "close_time": chunk[-1]["close_time"],
            }
        )
    return out
