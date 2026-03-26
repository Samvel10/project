def validate_candles(candles):
    if len(candles) < 50:
        return False

    for c in candles:
        if c["high"] < c["low"]:
            return False
        if any(v is None for v in c.values()):
            return False

    return True
