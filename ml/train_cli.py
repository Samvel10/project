import sys
import time
from pathlib import Path

from ruamel.yaml import YAML

# Ensure project root is on sys.path so that 'data' and 'ml' can be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.historical_loader import load_klines
from ml.trainer import train_walk_forward
from ml.model_registry import save_model


def _load_history(symbol: str, interval: str, total_bars: int) -> list[dict]:
    """Load at least total_bars candles using paginated klines requests."""
    candles: list[dict] = []
    start_time: int | None = None

    while len(candles) < total_bars:
        remaining = total_bars - len(candles)
        # Binance futures max limit is 1500 per request
        batch_limit = min(1500, remaining)

        batch = load_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_limit,
            start_time=start_time,
        )
        if not batch:
            break

        candles.extend(batch)
        # Next page starts after last close_time
        start_time = int(batch[-1]["close_time"]) + 1
        # Be polite to the API
        time.sleep(0.2)

    return candles


def main():
    args = sys.argv[1:]
    symbol = args[0] if len(args) >= 1 else "BTCUSDT"
    interval = args[1] if len(args) >= 2 else "1h"

    yaml = YAML()
    with open("config/ml.yaml") as f:
        ml_config = yaml.load(f)

    train_window = ml_config["training"]["train_window"]
    test_window = ml_config["training"]["test_window"]

    # Number of walk-forward steps to target for training
    steps = 5
    total_bars = train_window + test_window * steps

    print(f"[ML TRAIN] Loading ~{total_bars} candles for {symbol} {interval}...")
    candles = _load_history(symbol, interval, total_bars)

    if len(candles) < train_window + test_window:
        print(
            f"[ML TRAIN] Not enough data for walk-forward: got {len(candles)} candles, "
            f"need at least {train_window + test_window}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[ML TRAIN] Starting walk-forward training...")
    model, mean_score = train_walk_forward(candles, ml_config)

    print(f"[ML TRAIN] Training finished, mean score = {mean_score:.4f}")
    path = save_model(model, mean_score)
    print(f"[ML TRAIN] Saved model to: {path}")


if __name__ == "__main__":
    main()
