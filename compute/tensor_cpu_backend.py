"""Tensorized CPU backend (same kernels as GPU backend on CPU tensors)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from compute.backend_base import ComputeBackend, FeatureResult, SignalOutput
from compute.cpu_backend import CpuBackend
from compute.gpu_backend import (
    _rsi_tensor,
    _atr_tensor,
    _volatility_tensor,
    _momentum_tensor,
    _acceleration_tensor,
    _rule_signal_tensor,
    _candle_buffers_to_numpy,
    _detect_structure_cpu,
    _detect_range_cpu,
    _fibonacci_levels_cpu,
)

log = logging.getLogger(__name__)

_O, _H, _L, _C, _V = 0, 1, 2, 3, 4


class TensorCpuBackend(ComputeBackend):
    """Tensor-accelerated CPU backend.

    Uses vectorized PyTorch operations on CPU tensors for feature computation
    and signal generation.  Delivers batching benefits without a GPU.
    """

    def __init__(
        self,
        batch_size: int = 256,
        precision: str = "fp32",
        backend_mode: str = "parity",
    ) -> None:
        self._device = torch.device("cpu")
        self._batch_size = max(1, int(batch_size))
        self._backend_mode = "experimental" if backend_mode == "experimental" else "parity"
        self._legacy = CpuBackend(backend_mode="parity")
        self._enter_threshold = 0.7

        try:
            from pathlib import Path
            import yaml
            cfg_path = Path(__file__).resolve().parents[1] / "config" / "ml.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    ml_cfg = yaml.safe_load(f) or {}
                self._enter_threshold = float(
                    ml_cfg.get("thresholds", {}).get("enter", 0.7)
                )
        except Exception:
            pass

        self._structure_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
        self._range_map = {"RANGE": 0.0, "TREND": 1.0}
        self._ml_weights = torch.tensor(
            [0.03, 0.004, 0.003, -3.5, -0.0015, 0.06, -0.03],
            dtype=torch.float32,
            device=self._device,
        )
        self._ml_bias = torch.tensor([-0.2], dtype=torch.float32, device=self._device)
        log.info("[TENSOR_CPU_BACKEND] initialized batch=%d precision=%s", self._batch_size, precision)

    @property
    def device_name(self) -> str:
        return "CPU"

    @property
    def backend_name(self) -> str:
        if self._backend_mode == "parity":
            return "TensorCpuBackend(parity-orchestrator)"
        return "TensorCpuBackend(experimental)"

    @property
    def backend_mode(self) -> str:
        return self._backend_mode

    @property
    def supports_batching(self) -> bool:
        return self._backend_mode == "experimental"

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def precision(self) -> str:
        return "fp32"

    @torch.no_grad()
    def build_features_batch(
        self, items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[FeatureResult]:
        if self._backend_mode == "parity":
            return self._legacy.build_features_batch(items)
        if not items:
            return []
        N = len(items)
        symbols = [s for s, _ in items]
        candle_buffers = [c for _, c in items]

        np_buf, _ = _candle_buffers_to_numpy(candle_buffers)
        ohlcv = torch.from_numpy(np_buf)

        closes = ohlcv[:, :, _C]
        highs = ohlcv[:, :, _H]
        lows = ohlcv[:, :, _L]

        rsi = _rsi_tensor(closes)
        atr = _atr_tensor(highs, lows, closes)
        vol = _volatility_tensor(closes)
        mom = _momentum_tensor(closes)
        acc = _acceleration_tensor(closes)

        stacked = torch.stack([rsi, atr, vol, mom, acc], dim=1).numpy()

        results: List[FeatureResult] = []
        for i in range(N):
            results.append(FeatureResult(
                symbol=symbols[i],
                rsi=float(stacked[i, 0]),
                atr=float(stacked[i, 1]),
                volatility=float(stacked[i, 2]),
                momentum=float(stacked[i, 3]),
                acceleration=float(stacked[i, 4]),
                structure=_detect_structure_cpu(candle_buffers[i]),
                range_info=_detect_range_cpu(candle_buffers[i]),
                fibonacci=_fibonacci_levels_cpu(candle_buffers[i]),
            ))
        return results

    def run_ml_inference_batch(self, features_list: List[FeatureResult]) -> List[float]:
        if self._backend_mode == "parity":
            return self._legacy.run_ml_inference_batch(features_list)
        if not features_list:
            return []
        N = len(features_list)
        x_np = np.empty((N, 7), dtype=np.float32)
        for i, f in enumerate(features_list):
            rng = f.range_info if isinstance(f.range_info, dict) else {}
            x_np[i] = [
                f.rsi, f.momentum, f.acceleration, f.volatility, f.atr,
                self._structure_map.get(f.structure, 0.0),
                self._range_map.get(rng.get("type", "RANGE"), 0.0),
            ]
        x_t = torch.from_numpy(x_np).to(self._device)
        logits = (x_t * self._ml_weights).sum(dim=1) + self._ml_bias
        probs = torch.sigmoid(logits)
        return probs.tolist()

    @torch.no_grad()
    def generate_signals_batch(
        self, items: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[SignalOutput]:
        if self._backend_mode == "parity":
            return self._legacy.generate_signals_batch(items)
        if not items:
            return []
        N = len(items)
        symbols = [s for s, _ in items]
        candle_buffers = [c for _, c in items]

        feat_results = self.build_features_batch(items)

        np_buf, _ = _candle_buffers_to_numpy(candle_buffers)
        closes = torch.from_numpy(np_buf[:, :, _C])
        rule_dirs, rule_confs = _rule_signal_tensor(closes)

        ml_probs_list = self.run_ml_inference_batch(feat_results)
        ml_probs = torch.tensor(ml_probs_list, dtype=torch.float32)

        is_buy = (rule_dirs == 1).float()
        is_sell = (rule_dirs == 2).float()
        is_trade = is_buy + is_sell
        ml_conf = ml_probs * is_buy + (1.0 - ml_probs) * is_sell + 0.5 * (1.0 - is_trade)
        is_ml_half = (torch.abs(ml_conf - 0.5) < 1e-6).float()
        final_prob = is_ml_half * rule_confs + (1.0 - is_ml_half) * (rule_confs * 0.4 + ml_conf * 0.6)
        assert final_prob.dtype == torch.float32, "Decision logic must be fp32"
        passes = (final_prob >= self._enter_threshold).float() * is_trade

        rule_dirs_cpu = rule_dirs.tolist()
        final_prob_cpu = final_prob.tolist()
        passes_cpu = passes.tolist()
        dir_map = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}

        outputs: List[SignalOutput] = []
        for i in range(N):
            direction = dir_map[rule_dirs_cpu[i]] if passes_cpu[i] > 0.5 else "NO_TRADE"
            outputs.append(SignalOutput(
                symbol=symbols[i],
                direction=direction,
                confidence=final_prob_cpu[i],
                features=feat_results[i],
            ))
        return outputs

    def warmup(self) -> None:
        if self._backend_mode == "parity":
            return
        dummy = torch.randn(128, 128, 5, device=self._device)
        closes = dummy[:, :, _C]
        _ = _rsi_tensor(closes)
        _ = _atr_tensor(dummy[:, :, _H], dummy[:, :, _L], closes)
        _ = _volatility_tensor(closes)
        _ = _momentum_tensor(closes)
        _ = _acceleration_tensor(closes)
        _ = _rule_signal_tensor(closes)
        log.info("[TENSOR_CPU_BACKEND] warmup complete")

    def shutdown(self) -> None:
        pass
