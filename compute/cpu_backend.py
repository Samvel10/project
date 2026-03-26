"""
CPU compute backend — wraps existing pure-Python code with zero behavioral changes.

This is the baseline compatibility mode.  Every function call delegates
directly to the original module so results are bit-identical to the
pre-backend codebase.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from compute.backend_base import ComputeBackend, FeatureResult, SignalOutput

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import existing production modules (unchanged)
# ---------------------------------------------------------------------------

try:
    from features.feature_store import build_features as _build_features
except ImportError:
    _build_features = None

try:
    from signals.rule_engine import rule_signal as _rule_signal
except ImportError:
    _rule_signal = None

try:
    from signals.ensemble import generate_signal as _generate_signal
except ImportError:
    _generate_signal = None

try:
    from ml.inference import InferenceEngine as _InferenceEngine
except ImportError:
    _InferenceEngine = None


# ---------------------------------------------------------------------------
# Feature dict → FeatureResult helper
# ---------------------------------------------------------------------------

def _dict_to_feature_result(symbol: str, d: Dict[str, Any]) -> FeatureResult:
    return FeatureResult(
        symbol=symbol,
        rsi=float(d.get("rsi", 50.0)),
        momentum=float(d.get("momentum", 0.0)),
        acceleration=float(d.get("acceleration", 0.0)),
        volatility=float(d.get("volatility", 0.0)),
        atr=float(d.get("atr", 0.0)),
        structure=str(d.get("structure", "NEUTRAL")),
        range_info=d.get("range", {"type": "RANGE"}),
        fibonacci=d.get("fibonacci", {}),
    )


# ---------------------------------------------------------------------------
# CPU Backend
# ---------------------------------------------------------------------------

class CpuBackend(ComputeBackend):
    """Pure-CPU backend wrapping existing production code."""

    def __init__(self, backend_mode: str = "parity") -> None:
        self._backend_mode = "parity" if backend_mode != "experimental" else "experimental"
        self._inference_engine = None
        if _InferenceEngine is not None:
            try:
                self._inference_engine = _InferenceEngine()
            except Exception:
                pass
        log.info("[CPU_BACKEND] Initialized (pure Python, no GPU)")

    # -- Properties ---------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "CPU"

    @property
    def backend_name(self) -> str:
        return "CpuBackendLegacy"

    @property
    def backend_mode(self) -> str:
        return self._backend_mode

    @property
    def supports_batching(self) -> bool:
        return False

    # -- Feature computation ------------------------------------------------

    def build_features_batch(
        self,
        items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[FeatureResult]:
        return [self.compute_features(sym, candles) for sym, candles in items]

    def compute_features(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
    ) -> FeatureResult:
        if _build_features is None:
            return FeatureResult(symbol=symbol)
        try:
            d = _build_features(candles)
            return _dict_to_feature_result(symbol, d)
        except Exception:
            return FeatureResult(symbol=symbol)

    def run_ml_inference_batch(
        self,
        features_list: List[FeatureResult],
    ) -> List[float]:
        return [self.predict_proba(f) for f in features_list]

    def predict_proba(self, features: FeatureResult) -> float:
        if self._inference_engine is None or self._inference_engine.model is None:
            return 0.5

        structure_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
        range_map = {"RANGE": 0, "TREND": 1}

        x = [[
            features.rsi,
            features.momentum,
            features.acceleration,
            features.volatility,
            features.atr,
            structure_map.get(features.structure, 0),
            range_map.get(features.range_info.get("type", "RANGE") if isinstance(features.range_info, dict) else "RANGE", 0),
        ]]

        try:
            probs = self._inference_engine.model.predict_proba(x)
            row = probs[0]
            if len(row) >= 2:
                return float(row[1])
            classes = getattr(self._inference_engine.model, "classes_", None)
            if classes is not None and len(classes) == 1:
                if classes[0] == 1:
                    return float(row[0])
                if classes[0] == 0:
                    return 0.0
        except Exception:
            pass
        return 0.5

    def rule_signal(
        self,
        candles: List[Dict[str, Any]],
    ) -> Tuple[str, float]:
        if _rule_signal is None:
            return "NO_TRADE", 0.0
        try:
            return _rule_signal(candles)
        except Exception:
            return "NO_TRADE", 0.0

    def rule_signal_batch(
        self,
        candle_buffers: List[List[Dict[str, Any]]],
    ) -> List[Tuple[str, float]]:
        return [self.rule_signal(c) for c in candle_buffers]

    # -- Full signal generation (ensemble) ----------------------------------

    def generate_signal(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        features: Optional[FeatureResult] = None,
    ) -> SignalOutput:
        if _generate_signal is None:
            # Match signal_mirror legacy fallback semantics.
            direction, confidence = self.rule_signal(candles)
            return SignalOutput(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                features=features,
            )

        feat_dict = features.to_dict() if features else None
        try:
            direction, confidence = _generate_signal(candles, features=feat_dict)
            return SignalOutput(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                features=features,
            )
        except Exception:
            # On runtime failure, preserve legacy fallback behavior.
            direction, confidence = self.rule_signal(candles)
            return SignalOutput(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                features=features,
            )

    def generate_signals_batch(
        self,
        items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[SignalOutput]:
        results = []
        for sym, candles in items:
            feat = self.compute_features(sym, candles)
            sig = self.generate_signal(sym, candles, features=feat)
            results.append(sig)
        return results

    # -- Lifecycle ----------------------------------------------------------

    def warmup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
