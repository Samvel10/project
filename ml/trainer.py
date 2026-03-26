from sklearn.ensemble import RandomForestClassifier

from ml.dataset import build_dataset
from ml.walk_forward import walk_forward_split
from ml.metrics import accuracy
import numpy as np

try:  # optional, only needed if user selects these model types
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


def _create_model(model_cfg):
    """Factory to create an ML model based on config['model']['type'].

    Supported types:
      - random_forest (default)
      - xgboost
      - lightgbm
      - catboost
    """

    model_type = (model_cfg or {}).get("type", "random_forest")

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            n_jobs=-1,
        )

    if model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it with 'pip install xgboost' or "
                "change model.type back to 'random_forest' in config/ml.yaml."
            )
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )

    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Install it with 'pip install lightgbm' or "
                "change model.type back to 'random_forest' in config/ml.yaml."
            )
        return LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=-1,
        )

    if model_type == "catboost":
        if CatBoostClassifier is None:
            raise ImportError(
                "catboost is not installed. Install it with 'pip install catboost' or "
                "change model.type back to 'random_forest' in config/ml.yaml."
            )
        return CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
        )

    raise ValueError(f"Unknown model.type '{model_type}' in config/ml.yaml")


def train_walk_forward(candles, config):
    splits = walk_forward_split(
        candles,
        train_window=config["training"]["train_window"],
        test_window=config["training"]["test_window"],
        step=config["training"]["step"],
    )

    scores = []
    final_model = None
    model_cfg = config.get("model", {})

    for train_candles, test_candles in splits:
        X_train, y_train = build_dataset(train_candles)
        X_test, y_test = build_dataset(test_candles)

        model = _create_model(model_cfg)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = accuracy(y_test, preds)
        scores.append(score)
        final_model = model

    return final_model, float(np.mean(scores))
