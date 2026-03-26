"""
Backend base abstractions for device-agnostic batched compute.

This is the canonical interface used by the backtest engine.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(slots=True)
class FeatureResult:
    """Backend-agnostic feature payload for a single symbol."""
    symbol: str = ""
    rsi: float = 50.0
    momentum: float = 0.0
    acceleration: float = 0.0
    volatility: float = 0.0
    atr: float = 0.0
    structure: str = "NEUTRAL"
    range_info: Dict[str, Any] = field(default_factory=lambda: {"type": "RANGE"})
    fibonacci: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rsi": self.rsi,
            "momentum": self.momentum,
            "acceleration": self.acceleration,
            "volatility": self.volatility,
            "atr": self.atr,
            "structure": self.structure,
            "range": self.range_info,
            "fibonacci": self.fibonacci,
        }


@dataclass(slots=True)
class SignalOutput:
    """Backend-agnostic signal payload for a single symbol."""
    symbol: str = ""
    direction: str = "NO_TRADE"
    confidence: float = 0.0
    features: Optional[FeatureResult] = None


class ComputeBackend(abc.ABC):
    """Unified backend contract for CPU/GPU compute paths."""

    @property
    @abc.abstractmethod
    def device_name(self) -> str:
        """Returns CPU or GPU."""

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Concrete backend name for reporting."""

    @property
    def backend_mode(self) -> str:
        """Execution mode: parity (default) or experimental."""
        return "parity"

    @property
    def supports_batching(self) -> bool:
        return self.backend_mode == "experimental"

    @property
    def batch_size(self) -> int:
        return 256

    @property
    def precision(self) -> str:
        return "fp32"

    @abc.abstractmethod
    def build_features_batch(
        self,
        items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[FeatureResult]:
        """Build features for all symbols in one batch."""

    @abc.abstractmethod
    def run_ml_inference_batch(
        self,
        features_list: List[FeatureResult],
    ) -> List[float]:
        """Run batched ML inference and return probabilities."""

    @abc.abstractmethod
    def generate_signals_batch(
        self,
        items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[SignalOutput]:
        """Generate full ensemble signals in batch."""

    # Compatibility aliases used by existing code paths.
    def compute_features_batch(
        self,
        items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[FeatureResult]:
        return self.build_features_batch(items)

    def predict_proba_batch(
        self,
        features_list: List[FeatureResult],
    ) -> List[float]:
        return self.run_ml_inference_batch(features_list)

    def warmup(self) -> None:
        """Optional backend warmup."""

    def shutdown(self) -> None:
        """Optional backend shutdown hook."""
