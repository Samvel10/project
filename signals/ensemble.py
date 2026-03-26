from pathlib import Path

try:
    from ruamel.yaml import YAML  # type: ignore
except Exception:
    YAML = None
import yaml as _pyyaml

from signals.rule_engine import rule_signal
from ml.inference import predict, FIRST_NN_ENABLED


# Load ML thresholds from config/ml.yaml (fallback to 0.7 if missing)
_ENTER_THRESHOLD = 0.7
try:
    _cfg_path = Path(__file__).resolve().parents[1] / "config" / "ml.yaml"
    with open(_cfg_path) as _f:
        if YAML is not None:
            _yaml = YAML()
            _ml_cfg = _yaml.load(_f)
        else:
            _ml_cfg = _pyyaml.safe_load(_f)
    _ENTER_THRESHOLD = float(_ml_cfg.get("thresholds", {}).get("enter", 0.7))
except Exception:
    _ENTER_THRESHOLD = 0.7


def generate_signal(candles, features=None):
    """Combine rule-based and ML-based signals into one final signal.

    Less strict version: use rule direction as primary, blend ML confidence,
    and only apply a single threshold on the final probability.
    """

    rule_dir, rule_prob = rule_signal(candles)
    ml_prob = predict(candles, features=features)  # probability of UP move

    # If rule says NO_TRADE, respect that immediately
    if rule_dir == "NO_TRADE":
        return "NO_TRADE", 0.0

    final_dir = rule_dir
    try:
        ml_prob_val = float(ml_prob) if ml_prob is not None else 0.5
    except Exception:
        ml_prob_val = 0.5

    # Directional ML confidence: prob_up for BUY, prob_down for SELL
    ml_conf = ml_prob_val if rule_dir == "BUY" else 1.0 - ml_prob_val

    # If first NN is disabled or ML is effectively neutral, avoid overconfident
    # rule-only outputs (e.g., repeated ~0.92).
    ml_available = bool(FIRST_NN_ENABLED and abs(float(ml_conf) - 0.5) > 1e-6)
    if not ml_available:
        centered = 0.5 + (float(rule_prob) - 0.5) * 0.55
        final_prob = min(0.82, max(0.52, centered))
    else:
        # Softer ensemble: blend rule confidence with ML confirmation
        final_prob = rule_prob * 0.4 + ml_conf * 0.6

    # Single threshold on final probability only
    if final_dir != "NO_TRADE" and final_prob >= _ENTER_THRESHOLD:
        return final_dir, final_prob

    return "NO_TRADE", final_prob
