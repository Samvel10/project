import sys
from pathlib import Path
import csv
from typing import List

import numpy as np
from ruamel.yaml import YAML

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.signal_log import get_log_path
from ml.trainer import _create_model
from ml.model_registry import save_model
from ml.ensemble_model import EnsembleModel


def _build_dataset_from_log() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_path = get_log_path()
    if not log_path.exists():
        raise FileNotFoundError("signal_log.csv not found")

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("label") in ("0", "1")]

    if not rows:
        raise RuntimeError("No labeled signals found in log.")

    X: List[List[float]] = []
    y: List[int] = []
    sample_weights: List[float] = []

    structure_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
    range_map = {"RANGE": 0, "TREND": 1}

    for row in rows:
        try:
            rsi = float(row.get("rsi")) if row.get("rsi") not in (None, "") else 0.0
            momentum = float(row.get("momentum")) if row.get("momentum") not in (None, "") else 0.0
            acceleration = float(row.get("acceleration")) if row.get("acceleration") not in (None, "") else 0.0
            volatility = float(row.get("volatility")) if row.get("volatility") not in (None, "") else 0.0
            atr = float(row.get("atr")) if row.get("atr") not in (None, "") else 0.0
        except ValueError:
            continue

        structure = row.get("structure") or "NEUTRAL"
        range_type = row.get("range_type") or "RANGE"

        s_val = structure_map.get(structure, 0)
        r_val = range_map.get(range_type, 0)

        label = row.get("label")
        try:
            y_val = int(label)
        except (TypeError, ValueError):
            continue

        X.append([rsi, momentum, acceleration, volatility, atr, s_val, r_val])
        y.append(y_val)

        # Use pnl_pct magnitude as a sample weight so that more profitable or
        # more losing trades have a stronger impact during training. This does
        # not introduce future leakage at inference time, since weights are
        # only used while fitting the model on historical data.
        pnl_raw = row.get("pnl_pct")
        try:
            pnl_val = float(pnl_raw) if pnl_raw not in (None, "", " ") else 0.0
        except ValueError:
            pnl_val = 0.0

        # Base weight is 1.0. Scale by |pnl|, but clip to keep things stable.
        # Example: 5% move -> weight around 2.0, 20%+ move -> capped.
        weight = 1.0 + min(abs(pnl_val) / 5.0, 4.0)
        sample_weights.append(weight)

    if not X:
        raise RuntimeError("No valid feature rows built from signal log.")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    w_arr = np.asarray(sample_weights, dtype=float)
    return X_arr, y_arr, w_arr


def main() -> None:
    X, y, sample_weights = _build_dataset_from_log()

    n = len(y)
    if n < 20:
        raise RuntimeError(f"Not enough labeled signals to train: {n}")

    split = int(n * 0.8)
    if split <= 0 or split >= n:
        split = n // 2

    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    w_train = sample_weights[:split]
    w_test = sample_weights[split:]

    yaml = YAML()
    with open("config/ml.yaml") as f:
        ml_cfg = yaml.load(f)

    model_cfg = ml_cfg.get("model", {})
    model_type = (model_cfg or {}).get("type", "random_forest")

    if model_type == "ensemble":
        base_models_cfg = model_cfg.get("base_models") or [
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

        base_models = []
        for base_type in base_models_cfg:
            single_cfg = dict(model_cfg)
            single_cfg["type"] = base_type

            # Create model instance; skip if creation fails for this type
            try:
                model = _create_model(single_cfg)
            except Exception as e:
                print(f"[ML TRAIN LOG] Skipping base model '{base_type}' due to creation error: {e}")
                continue

            # Train model; prefer sample_weight when supported, otherwise fall back.
            try:
                model.fit(X_train, y_train, sample_weight=w_train)
            except TypeError:
                try:
                    model.fit(X_train, y_train)
                except Exception as e:
                    print(f"[ML TRAIN LOG] Skipping base model '{base_type}' due to training error: {e}")
                    continue
            except Exception as e:
                print(f"[ML TRAIN LOG] Skipping base model '{base_type}' due to training error: {e}")
                continue

            base_models.append(model)

        if not base_models:
            raise RuntimeError("No base models could be trained successfully for ensemble.")

        ensemble = EnsembleModel(base_models)
        preds = ensemble.predict(X_test)
        if len(y_test) > 0:
            if w_test.size == len(y_test):
                acc = float(np.average((preds == y_test), weights=w_test))
            else:
                acc = float((preds == y_test).mean())
        else:
            acc = 1.0
        path = save_model(ensemble, acc)
    else:
        try:
            model = _create_model(model_cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to create model of type '{model_type}': {e}")

        try:
            model.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            model.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"Training failed for model type '{model_type}': {e}")

        preds = model.predict(X_test)
        if len(y_test) > 0:
            if w_test.size == len(y_test):
                acc = float(np.average((preds == y_test), weights=w_test))
            else:
                acc = float((preds == y_test).mean())
        else:
            acc = 1.0
        path = save_model(model, acc)

    print(f"[ML TRAIN LOG] Trained on {n} signals, accuracy={acc:.4f}")
    print(f"[ML TRAIN LOG] Saved model to: {path}")


if __name__ == "__main__":
    main()
