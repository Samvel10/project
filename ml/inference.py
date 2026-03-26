from ml.model_registry import load_latest
from features.feature_store import build_features

# First neural network (direction NN) master kill-switch.
# User-requested hard disable: inference stays neutral (0.5) and does not load
# any model from ml/models while this is False.
FIRST_NN_ENABLED = False

class InferenceEngine:
    """
    Production-ready ML inference engine.
    Can be extended for ensemble usage, probability prediction, and more.
    """

    def __init__(self):
        if not FIRST_NN_ENABLED:
            self.model = None
            return
        try:
            self.model = load_latest()
        except FileNotFoundError:
            self.model = None

    def predict_proba(self, candles, features=None):
        """
        Predict probability of upward move using trained ML model.

        Args:
            candles: list of dicts with 'close', 'open', 'high', 'low', 'volume'

        Returns:
            float: probability of upward move (0..1)
        """
        if len(candles) < 20:
            return 0.5  # neutral if not enough data

        if self.model is None:
            return 0.5

        try:
            feats = features if isinstance(features, dict) else None
        except Exception:
            feats = None
        if feats is None:
            feats = build_features(candles)

        structure_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
        range_map = {"RANGE": 0, "TREND": 1}

        x = [[
            feats["rsi"],
            feats["momentum"],
            feats["acceleration"],
            feats["volatility"],
            feats.get("atr", 0),
            structure_map.get(feats.get("structure", "NEUTRAL"), 0),
            range_map.get(feats.get("range", {}).get("type", "RANGE"), 0),
        ]]

        # Some models (or degenerate training where only one class is seen)
        # may return a single probability column. Handle this defensively
        # instead of assuming predict_proba(x)[0][1] always exists.
        probs = self.model.predict_proba(x)

        # Try to get first row of probabilities
        try:
            row = probs[0]
        except Exception:
            row = probs

        try:
            length = len(row)
        except TypeError:
            return 0.5

        # Normal binary case: [P(class0), P(class1)]
        if length >= 2:
            return float(row[1])

        # Single-column case: use model.classes_ if available
        classes = getattr(self.model, "classes_", None)
        try:
            if classes is not None and len(classes) == 1:
                cls = classes[0]
                # If the only class is the positive class (1), row[0] is P(up)
                if cls == 1:
                    return float(row[0])
                # If the only class is the negative class (0), probability of up is ~0
                if cls == 0:
                    return 0.0
        except Exception:
            pass

        # Fallback: neutral probability
        return 0.5


# Shortcut function for legacy usage
_engine = InferenceEngine()

def reload_model():
    """Reload the latest trained model into the shared inference engine."""
    global _engine
    _engine = InferenceEngine()

def predict(candles, features=None):
    """
    Legacy shortcut: directly predict probability
    """
    return _engine.predict_proba(candles, features=features)
