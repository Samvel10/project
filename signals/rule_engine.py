def rule_signal(candles):
    """
    Basic rule-based signal generator.
    Returns:
        direction: "BUY", "SELL", "NO_TRADE"
        confidence: float (0..1)
    """
    closes = [c["close"] for c in candles]
    if len(closes) < 20:
        return "NO_TRADE", 0.0

    # Simple moving average crossover rule
    short_ma = sum(closes[-5:]) / 5
    long_ma = sum(closes[-20:]) / 20
    if long_ma <= 0:
        return "NO_TRADE", 0.0

    # Dynamic confidence (instead of hardcoded 0.70):
    # - Primary driver: MA separation in %
    # - Secondary driver: short-term momentum alignment
    # This keeps confidence variable while preserving same BUY/SELL direction rule.
    ma_gap_pct = abs(short_ma - long_ma) / long_ma * 100.0
    prev_short_ma = sum(closes[-8:-3]) / 5 if len(closes) >= 8 else short_ma
    short_momentum_pct = abs(short_ma - prev_short_ma) / max(abs(prev_short_ma), 1e-12) * 100.0

    # Normalize features into [0..1] with practical crypto intraday bounds.
    gap_score = min(1.0, ma_gap_pct / 1.20)
    mom_score = min(1.0, short_momentum_pct / 0.80)
    strength = min(1.0, max(0.0, 0.75 * gap_score + 0.25 * mom_score))

    # Keep confidence in a realistic operational band.
    # Strong trends can reach ~0.92, weaker setups remain around ~0.60-0.70.
    signal_conf = 0.58 + 0.34 * strength

    if short_ma > long_ma:
        return "BUY", float(signal_conf)
    elif short_ma < long_ma:
        return "SELL", float(signal_conf)
    # Flat crossover area should stay low confidence.
    return "NO_TRADE", 0.30
