import numpy as np
from features.feature_store import build_features


def build_dataset(candles, horizon=1):
    X, y = [], []

    for i in range(60, len(candles) - horizon):
        window = candles[:i]
        features = build_features(window)

        # encode categorical → numeric
        structure_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
        range_map = {"RANGE": 0, "TREND": 1}

        x = [
            features["rsi"],
            features["momentum"],
            features["acceleration"],
            features["volatility"],
            features["atr"],
            structure_map[features["structure"]],
            range_map[features["range"]["type"]],
        ]

        future_close = candles[i + horizon]["close"]
        current_close = candles[i]["close"]

        label = 1 if future_close > current_close else 0

        X.append(x)
        y.append(label)

    return np.array(X), np.array(y)
