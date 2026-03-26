"""GPU compute backend with tensorized batched feature/signal generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from compute.backend_base import ComputeBackend, FeatureResult, SignalOutput
from compute.cpu_backend import CpuBackend

log = logging.getLogger(__name__)

_O, _H, _L, _C, _V = 0, 1, 2, 3, 4


@torch.no_grad()
def _rsi_tensor(closes: torch.Tensor, period: int = 14) -> torch.Tensor:
    if closes.shape[1] < 2:
        return torch.full((closes.shape[0],), 50.0, device=closes.device)
    deltas = closes[:, 1:] - closes[:, :-1]
    gains = torch.clamp(deltas, min=0.0)
    losses = torch.clamp(-deltas, min=0.0)
    p = min(period, gains.shape[1])
    avg_gain = gains[:, -p:].mean(dim=1)
    avg_loss = losses[:, -p:].mean(dim=1) + 1e-9
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


@torch.no_grad()
def _atr_tensor(
    highs: torch.Tensor, lows: torch.Tensor, closes: torch.Tensor,
    period: int = 14,
) -> torch.Tensor:
    if closes.shape[1] < 2:
        return torch.zeros(closes.shape[0], device=closes.device)
    prev_close = closes[:, :-1]
    h = highs[:, 1:]
    l_ = lows[:, 1:]
    tr = torch.max(
        torch.max(h - l_, torch.abs(h - prev_close)),
        torch.abs(l_ - prev_close),
    )
    p = min(period, tr.shape[1])
    return tr[:, -p:].mean(dim=1)


@torch.no_grad()
def _volatility_tensor(closes: torch.Tensor, window: int = 20) -> torch.Tensor:
    w = min(window, closes.shape[1])
    if w < 2:
        return torch.zeros(closes.shape[0], device=closes.device)
    seg = closes[:, -w:]
    log_ret = torch.log(seg[:, 1:] / (seg[:, :-1] + 1e-9) + 1e-9)
    return log_ret.std(dim=1, unbiased=False)


@torch.no_grad()
def _momentum_tensor(closes: torch.Tensor, lookback: int = 10) -> torch.Tensor:
    lb = min(lookback, closes.shape[1])
    return closes[:, -1] - closes[:, -lb]


@torch.no_grad()
def _acceleration_tensor(closes: torch.Tensor, short: int = 5, long: int = 15) -> torch.Tensor:
    s = min(short, closes.shape[1])
    l_ = min(long, closes.shape[1])
    v_short = closes[:, -1] - closes[:, -s]
    v_long = closes[:, -s] - closes[:, -l_]
    return v_short - v_long


@torch.no_grad()
def _rule_signal_tensor(closes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    N = closes.shape[0]
    T = closes.shape[1]
    if T < 20:
        return (
            torch.zeros(N, dtype=torch.long, device=closes.device),
            torch.zeros(N, device=closes.device),
        )
    short_ma = closes[:, -5:].mean(dim=1)
    long_ma = closes[:, -20:].mean(dim=1)
    dirs = torch.zeros(N, dtype=torch.long, device=closes.device)
    dirs[short_ma > long_ma] = 1
    dirs[short_ma < long_ma] = 2
    conf = torch.full((N,), 0.3, device=closes.device)
    conf[dirs != 0] = 0.7
    return dirs, conf


def _candle_buffers_to_numpy(
    candle_buffers: List[List[Dict[str, Any]]],
) -> Tuple[np.ndarray, int]:
    max_len = max(len(c) for c in candle_buffers) if candle_buffers else 0
    N = len(candle_buffers)
    buf = np.zeros((N, max_len, 5), dtype=np.float32)
    for i, candles in enumerate(candle_buffers):
        T = len(candles)
        if T == 0:
            continue
        arr = np.array(
            [[c["open"], c["high"], c["low"], c["close"], c.get("volume", 0.0)]
             for c in candles],
            dtype=np.float32,
        )
        buf[i, max_len - T:] = arr  # right-align
    return buf, max_len


def _detect_structure_cpu(candles: List[Dict[str, Any]], lookback: int = 30) -> str:
    recent = candles[-lookback:]
    if len(recent) < 2:
        return "NEUTRAL"
    if recent[-1]["high"] > recent[-2]["high"] and recent[-1]["low"] > recent[-2]["low"]:
        return "BULLISH"
    if recent[-1]["high"] < recent[-2]["high"] and recent[-1]["low"] < recent[-2]["low"]:
        return "BEARISH"
    return "NEUTRAL"


def _detect_range_cpu(
    candles: List[Dict[str, Any]], window: int = 50, threshold: float = 0.35,
) -> Dict[str, Any]:
    recent = candles[-window:]
    if len(recent) < 2:
        return {"type": "TREND", "direction": "UP"}
    closes = np.array([c["close"] for c in recent], dtype=np.float64)
    high = float(closes.max())
    low = float(closes.min())
    move = abs(float(closes[-1]) - float(closes[0]))
    compression = move / (high - low + 1e-9)
    if compression < threshold:
        return {"type": "RANGE", "high": high, "low": low}
    return {"type": "TREND", "direction": "UP" if closes[-1] > closes[0] else "DOWN"}


def _fibonacci_levels_cpu(candles: List[Dict[str, Any]], lookback: int = 100) -> Dict[str, Any]:
    recent = candles[-lookback:]
    if len(recent) < 2:
        return {}
    highs = np.array([c["high"] for c in recent], dtype=np.float64)
    lows = np.array([c["low"] for c in recent], dtype=np.float64)
    hi = float(highs.max())
    lo = float(lows.min())
    hi_idx = int(highs.argmax())
    lo_idx = int(lows.argmin())
    if hi_idx > lo_idx:
        a, b, direction = lo, hi, "UP"
    else:
        a, b, direction = hi, lo, "DOWN"
    diff = abs(b - a)
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    levels = {str(r): (b - diff * r) if direction == "UP" else (b + diff * r) for r in ratios}
    return {"direction": direction, "anchor_a": a, "anchor_b": b, "levels": levels}


class GpuBackend(ComputeBackend):
    """GPU backend using tensorized PyTorch operations."""

    def __init__(
        self,
        batch_size: int = 256,
        precision: str = "fp32",
        simulate_gpu_on_cpu: bool = False,
        backend_mode: str = "parity",
    ) -> None:
        can_use_cuda = torch.cuda.is_available() and not simulate_gpu_on_cpu
        if not can_use_cuda and not simulate_gpu_on_cpu:
            raise RuntimeError("GpuBackend requires CUDA unless simulate_gpu_on_cpu=True")
        self._device = torch.device("cuda:0" if can_use_cuda else "cpu")
        self._dtype = torch.float16 if (precision == "fp16" and self._device.type == "cuda") else torch.float32
        self._batch_size = max(1, int(batch_size))
        self._simulate_gpu_on_cpu = simulate_gpu_on_cpu
        self._backend_mode = "experimental" if backend_mode == "experimental" else "parity"
        self._legacy = CpuBackend(backend_mode="parity")
        self._enter_threshold = 0.7
        self._batch_calls = 0
        self._structure_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
        self._range_map = {"RANGE": 0.0, "TREND": 1.0}
        self._ml_weights = torch.tensor(
            [0.03, 0.004, 0.003, -3.5, -0.0015, 0.06, -0.03],
            dtype=torch.float32,
            device=self._device,
        )
        self._ml_bias = torch.tensor([-0.2], dtype=torch.float32, device=self._device)
        if self._device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / 1e9
            log.info(
                "[GPU_BACKEND] device=%s gpu=%s vram=%.2fGB batch=%d precision=%s",
                self._device,
                props.name,
                mem_gb,
                self._batch_size,
                precision,
            )
        else:
            log.info("[GPU_BACKEND] simulated GPU on CPU tensors batch=%d", self._batch_size)

    @property
    def device_name(self) -> str:
        return "GPU"

    @property
    def backend_name(self) -> str:
        if self._backend_mode == "parity":
            return "GpuBackend(parity-orchestrator)"
        if self._simulate_gpu_on_cpu:
            return "GpuBackend(simulated)"
        return "GpuBackend"

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
        return "fp16" if self._dtype == torch.float16 else "fp32"

    @torch.no_grad()
    def build_features_batch(self, items: List[Tuple[str, List[Dict[str, Any]]]]) -> List[FeatureResult]:
        if self._backend_mode == "parity":
            return self._legacy.build_features_batch(items)
        if not items:
            return []
        symbols = [sym for sym, _ in items]
        candle_buffers = [candles for _, candles in items]
        np_buf, _ = _candle_buffers_to_numpy(candle_buffers)
        ohlcv = torch.from_numpy(np_buf).to(self._device, dtype=self._dtype, non_blocking=True)
        closes = ohlcv[:, :, _C]
        highs = ohlcv[:, :, _H]
        lows = ohlcv[:, :, _L]
        stacked = torch.stack(
            [_rsi_tensor(closes), _atr_tensor(highs, lows, closes), _volatility_tensor(closes), _momentum_tensor(closes), _acceleration_tensor(closes)],
            dim=1,
        ).to(torch.float32)
        cpu_arr = stacked.cpu().numpy()
        results: List[FeatureResult] = []
        for i in range(len(items)):
            results.append(FeatureResult(
                symbol=symbols[i],
                rsi=float(cpu_arr[i, 0]),
                atr=float(cpu_arr[i, 1]),
                volatility=float(cpu_arr[i, 2]),
                momentum=float(cpu_arr[i, 3]),
                acceleration=float(cpu_arr[i, 4]),
                structure=_detect_structure_cpu(candle_buffers[i]),
                range_info=_detect_range_cpu(candle_buffers[i]),
                fibonacci=_fibonacci_levels_cpu(candle_buffers[i]),
            ))
        return results

    @torch.no_grad()
    def run_ml_inference_batch(self, features_list: List[FeatureResult]) -> List[float]:
        if self._backend_mode == "parity":
            return self._legacy.run_ml_inference_batch(features_list)
        if not features_list:
            return []
        x_np = np.empty((len(features_list), 7), dtype=np.float32)
        for i, f in enumerate(features_list):
            rng = f.range_info if isinstance(f.range_info, dict) else {}
            x_np[i] = [
                f.rsi, f.momentum, f.acceleration, f.volatility, f.atr,
                self._structure_map.get(f.structure, 0.0),
                self._range_map.get(rng.get("type", "RANGE"), 0.0),
            ]
        x_t = torch.from_numpy(x_np).to(self._device, dtype=torch.float32, non_blocking=True)
        logits = (x_t * self._ml_weights).sum(dim=1) + self._ml_bias
        probs = torch.sigmoid(logits)
        return probs.cpu().tolist()

    @torch.no_grad()
    def generate_signals_batch(self, items: List[Tuple[str, List[Dict[str, Any]]]]) -> List[SignalOutput]:
        if self._backend_mode == "parity":
            return self._legacy.generate_signals_batch(items)
        if not items:
            return []
        try:
            self._batch_calls += 1
            symbols = [sym for sym, _ in items]
            candle_buffers = [candles for _, candles in items]
            feat_results = self.build_features_batch(items)
            np_buf, _ = _candle_buffers_to_numpy(candle_buffers)
            closes = torch.from_numpy(np_buf[:, :, _C]).to(self._device, dtype=self._dtype, non_blocking=True)
            rule_dirs, rule_confs = _rule_signal_tensor(closes.to(torch.float32))
            ml_probs = torch.tensor(
                self.run_ml_inference_batch(feat_results),
                dtype=torch.float32,
                device=self._device,
            )
            is_buy = (rule_dirs == 1).float()
            is_sell = (rule_dirs == 2).float()
            is_trade = is_buy + is_sell
            ml_conf = ml_probs * is_buy + (1.0 - ml_probs) * is_sell + 0.5 * (1.0 - is_trade)
            is_ml_half = (torch.abs(ml_conf - 0.5) < 1e-6).float()
            final_prob = is_ml_half * rule_confs + (1.0 - is_ml_half) * (rule_confs * 0.4 + ml_conf * 0.6)
            assert final_prob.dtype == torch.float32, "Decision logic must be fp32"
            passes_threshold = (final_prob >= self._enter_threshold).float() * is_trade
            rule_dirs_cpu = rule_dirs.cpu().tolist()
            final_prob_cpu = final_prob.cpu().tolist()
            passes_cpu = passes_threshold.cpu().tolist()
            dir_map = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}
            outputs: List[SignalOutput] = []
            for i in range(len(items)):
                direction = dir_map[rule_dirs_cpu[i]] if passes_cpu[i] > 0.5 else "NO_TRADE"
                outputs.append(SignalOutput(
                    symbol=symbols[i],
                    direction=direction,
                    confidence=final_prob_cpu[i],
                    features=feat_results[i],
                ))
            if self._device.type == "cuda" and (self._batch_calls % 200 == 0):
                log.info(
                    "[GPU_BACKEND] batch=%d symbols=%d alloc=%.2fGB reserved=%.2fGB",
                    self._batch_calls,
                    len(items),
                    torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9,
                )
            return outputs

        except torch.cuda.OutOfMemoryError:
            log.warning("[GPU_BACKEND] CUDA OOM, falling back to legacy CPU backend")
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            return self._legacy.generate_signals_batch(items)

    def warmup(self) -> None:
        if self._backend_mode == "parity":
            return
        dummy = torch.randn(256, 256, 5, device=self._device, dtype=self._dtype)
        closes = dummy[:, :, _C]
        _ = _rsi_tensor(closes)
        _ = _atr_tensor(dummy[:, :, _H], dummy[:, :, _L], closes)
        _ = _volatility_tensor(closes)
        _ = _momentum_tensor(closes)
        _ = _acceleration_tensor(closes)
        _ = _rule_signal_tensor(closes)
        x_dummy = torch.randn(256, 7, device=self._device)
        _ = torch.sigmoid((x_dummy * self._ml_weights).sum(dim=1) + self._ml_bias)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        log.info("[GPU_BACKEND] warmup complete")

    def shutdown(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        log.info("[GPU_BACKEND] shutdown complete")
