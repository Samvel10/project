import numpy as np
import warnings


class EnsembleModel:
    """Simple ensemble wrapper over multiple binary classifiers.

    Each base model must implement predict_proba(X) and predict(X) in the
    usual scikit-learn style. This wrapper averages the probability of the
    positive class (label 1) across all models.
    """

    def __init__(self, models):
        self.models = list(models)
        base = self.models[0] if self.models else None
        if base is not None and hasattr(base, "classes_"):
            self.classes_ = getattr(base, "classes_")
        else:
            self.classes_ = np.array([0, 1], dtype=int)

    @staticmethod
    def _proba_up_from_row(model, row):
        """Extract probability of class 1 from a single probability row."""
        try:
            length = len(row)
        except TypeError:
            return 0.5

        if length >= 2:
            try:
                return float(row[1])
            except Exception:
                return 0.5

        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 1:
            cls = classes[0]
            try:
                val = float(row[0])
            except Exception:
                val = 1.0
            if cls == 1:
                return val
            if cls == 0:
                return 0.0

        return 0.5

    def _aggregate_up_probs(self, X):
        n = len(X)
        if n == 0:
            return np.array([], dtype=float)

        agg = np.zeros(n, dtype=float)
        used_models = 0

        for model in self.models:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "X does not have valid feature names, "
                            "but .* was fitted with feature names"
                        ),
                        category=UserWarning,
                    )
                    probs = model.predict_proba(X)
            except Exception:
                # Fallback: use hard labels as probabilities
                labels = np.asarray(model.predict(X), dtype=float)
                probs = np.vstack([1.0 - labels, labels]).T

            arr = np.asarray(probs)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != n:
                continue

            for i in range(n):
                row = arr[i]
                p_up = self._proba_up_from_row(model, row)
                agg[i] += p_up

            used_models += 1

        if used_models == 0:
            return np.full(n, 0.5, dtype=float)

        agg /= float(used_models)
        return agg

    def predict_proba(self, X):
        up = self._aggregate_up_probs(X)
        down = 1.0 - up
        return np.vstack([down, up]).T

    def predict(self, X):
        up = self._aggregate_up_probs(X)
        return (up >= 0.5).astype(int)
