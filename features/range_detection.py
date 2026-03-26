def detect_range(candles, window=50, threshold=0.35):
    try:
        window_i = int(window)
    except Exception:
        window_i = 50
    if window_i <= 0:
        window_i = 50

    recent = candles[-window_i:] if candles else []
    closes = []
    for c in recent:
        try:
            closes.append(float(c.get("close")))
        except Exception:
            continue

    if len(closes) < 2:
        return {
            "type": "TREND",
            "direction": "UP",
        }

    high = max(closes)
    low = min(closes)

    range_size = high - low
    move = abs(closes[-1] - closes[0])

    compression = move / (range_size + 1e-9)

    if compression < threshold:
        return {
            "type": "RANGE",
            "high": high,
            "low": low,
        }

    return {
        "type": "TREND",
        "direction": "UP" if closes[-1] > closes[0] else "DOWN",
    }
