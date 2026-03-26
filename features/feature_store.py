from features.indicators import rsi, atr, volatility
from features.physics import momentum, acceleration
from features.fibonacci import fibonacci_levels
from features.range_detection import detect_range
from features.structure import detect_structure


def build_features(candles):
    closes = [c["close"] for c in candles]

    features = {
        "rsi": rsi(closes),
        "momentum": momentum(closes),
        "acceleration": acceleration(closes),
        "volatility": volatility(closes),
        "atr": atr(candles),
        "structure": detect_structure(candles),
        "range": detect_range(candles),
        "fibonacci": fibonacci_levels(candles),
    }

    return features
