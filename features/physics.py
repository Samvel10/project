def momentum(closes, lookback=10):
    return closes[-1] - closes[-lookback]


def acceleration(closes, short=5, long=15):
    v_short = closes[-1] - closes[-short]
    v_long = closes[-short] - closes[-long]
    return v_short - v_long
