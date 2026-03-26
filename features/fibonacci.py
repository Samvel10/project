def detect_swing(candles, lookback=100):
    highs = [c["high"] for c in candles[-lookback:]]
    lows = [c["low"] for c in candles[-lookback:]]

    high_price = max(highs)
    low_price = min(lows)

    high_index = highs.index(high_price)
    low_index = lows.index(low_price)

    if high_index > low_index:
        return low_price, high_price, "UP"
    else:
        return high_price, low_price, "DOWN"


def fibonacci_levels(candles):
    a, b, direction = detect_swing(candles)

    diff = abs(b - a)
    levels = {
        "0.236": b - diff * 0.236 if direction == "UP" else b + diff * 0.236,
        "0.382": b - diff * 0.382 if direction == "UP" else b + diff * 0.382,
        "0.5": b - diff * 0.5 if direction == "UP" else b + diff * 0.5,
        "0.618": b - diff * 0.618 if direction == "UP" else b + diff * 0.618,
        "0.786": b - diff * 0.786 if direction == "UP" else b + diff * 0.786,
    }

    return {
        "direction": direction,
        "anchor_a": a,
        "anchor_b": b,
        "levels": levels,
    }
