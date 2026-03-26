import math
import statistics


def rsi(closes, period=14):
    try:
        p = int(period)
    except Exception:
        p = 14
    if p <= 0:
        p = 14

    if not closes or len(closes) < 2:
        return 50.0

    vals = []
    for c in closes:
        try:
            vals.append(float(c))
        except Exception:
            pass
    if len(vals) < 2:
        return 50.0

    deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [(-d) if d < 0 else 0.0 for d in deltas]

    gains_slice = gains[-p:] if len(gains) >= p else gains
    losses_slice = losses[-p:] if len(losses) >= p else losses

    if not gains_slice or not losses_slice:
        return 50.0

    avg_gain = statistics.fmean(gains_slice)
    avg_loss = statistics.fmean(losses_slice) + 1e-9

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles, period=14):
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)

    try:
        p = int(period)
    except Exception:
        p = 14
    if p <= 0:
        p = 14

    if not trs:
        return 0.0

    sl = trs[-p:] if len(trs) >= p else trs
    return float(statistics.fmean(sl))


def volatility(closes, window=20):
    try:
        w = int(window)
    except Exception:
        w = 20
    if w <= 1:
        w = 20

    if not closes:
        return 0.0

    vals = []
    for c in closes[-w:]:
        try:
            v = float(c)
        except Exception:
            continue
        if v > 0:
            vals.append(v)

    if len(vals) < 2:
        return 0.0

    rets = []
    for i in range(1, len(vals)):
        prev_v = vals[i - 1]
        cur_v = vals[i]
        rets.append(math.log((cur_v + 1e-9) / (prev_v + 1e-9)))

    if len(rets) < 2:
        return 0.0

    return float(statistics.pstdev(rets))
