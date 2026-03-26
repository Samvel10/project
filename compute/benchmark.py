"""Benchmark CPU-legacy vs TensorCPU vs GPU backends."""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from typing import Any, Dict, List, Tuple

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_candles(n: int, base_price: float = 50000.0) -> List[Dict[str, Any]]:
    """Generate n realistic-ish candles."""
    candles = []
    price = base_price
    rng = np.random.RandomState(42)
    for i in range(n):
        change = rng.normal(0, price * 0.002)
        o = price
        c = price + change
        h = max(o, c) + abs(rng.normal(0, price * 0.001))
        l = min(o, c) - abs(rng.normal(0, price * 0.001))
        v = rng.uniform(100, 10000)
        candles.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        price = c
    return candles


def _make_batch(n_symbols: int, candle_len: int = 500) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Generate batch of (symbol, candles) tuples."""
    items = []
    for i in range(n_symbols):
        base = 1000 + i * 50
        sym = f"SYM{i}USDT"
        candles = _make_candles(candle_len, base_price=base)
        items.append((sym, candles))
    return items


def benchmark_backend(backend, n_symbols: int = 500, candle_len: int = 500):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {backend.backend_name}")
    print(f"  Symbols: {n_symbols}, Candle length: {candle_len}")
    print(f"{'='*60}\n")

    batch = _make_batch(n_symbols, candle_len)
    candle_buffers = [c for _, c in batch]
    single_candles = batch[0][1]

    results = {}

    # 1. Batch feature computation
    t0 = time.monotonic()
    feats = backend.build_features_batch(batch)
    t1 = time.monotonic()
    batch_feat_ms = (t1 - t0) * 1000
    per_sym_feat_ms = batch_feat_ms / n_symbols
    results["batch_feature_ms"] = batch_feat_ms
    results["per_sym_feature_ms"] = per_sym_feat_ms
    print(f"  Batch feature compute:   {batch_feat_ms:.1f} ms total, {per_sym_feat_ms:.3f} ms/sym")

    # 2. Batch ML inference
    t0 = time.monotonic()
    probs = backend.run_ml_inference_batch(feats)
    t1 = time.monotonic()
    ml_ms = (t1 - t0) * 1000
    results["batch_ml_ms"] = ml_ms
    print(f"  Batch ML inference:      {ml_ms:.1f} ms ({n_symbols} symbols)")

    # 3. Full ensemble (end-to-end)
    t0 = time.monotonic()
    for _ in range(3):
        signals = backend.generate_signals_batch(batch)
    t1 = time.monotonic()
    ensemble_ms = (t1 - t0) / 3 * 1000
    per_sym_ensemble = ensemble_ms / n_symbols
    results["batch_ensemble_ms"] = ensemble_ms
    results["per_sym_ensemble_ms"] = per_sym_ensemble
    print(f"  Full ensemble batch:     {ensemble_ms:.1f} ms total, {per_sym_ensemble:.3f} ms/sym")

    # 4. Throughput estimate
    kline_batches_per_year = 525_600  # one batch per minute
    batch_time_sec = ensemble_ms / 1000
    kline_processing_sec = kline_batches_per_year * batch_time_sec
    total_est_min = kline_processing_sec / 60 * 1.25  # 25% overhead for other events
    results["est_1year_minutes"] = total_est_min
    print(f"\n  === 1-Year / {n_symbols} Symbols Estimate ===")
    print(f"  Kline batches:           {kline_batches_per_year:,}")
    print(f"  Batch time:              {batch_time_sec*1000:.1f} ms")
    print(f"  Kline processing:        {kline_processing_sec/60:.1f} min")
    print(f"  Total estimate (1.25x):  {total_est_min:.1f} min")

    return results


def main():
    os.environ["BACKTEST_DEVICE"] = "CPU"
    from compute.cpu_backend import CpuBackend
    from compute.tensor_cpu_backend import TensorCpuBackend
    from compute.gpu_backend import GpuBackend

    print("\n" + "="*60)
    print("  COMPUTE BACKEND BENCHMARK")
    print("="*60)

    sizes = [100, 300, 500]
    backends = [
        CpuBackend(backend_mode="parity"),
        TensorCpuBackend(batch_size=256, backend_mode="experimental"),
    ]

    try:
        import torch
        if torch.cuda.is_available():
            backends.append(GpuBackend(batch_size=256, precision="fp16", backend_mode="experimental"))
        else:
            backends.append(GpuBackend(batch_size=256, simulate_gpu_on_cpu=True, backend_mode="experimental"))
    except Exception:
        backends.append(GpuBackend(batch_size=256, simulate_gpu_on_cpu=True, backend_mode="experimental"))

    for n_symbols in sizes:
        print(f"\n\n### BENCHMARK SIZE: {n_symbols} symbols ###")
        all_results = {}
        for backend in backends:
            backend.warmup()
            all_results[backend.backend_name] = benchmark_backend(
                backend, n_symbols=n_symbols, candle_len=500,
            )
        print(f"\n  {'Backend':<24} {'1Y Runtime(min)':>16} {'Per-sym ms':>12}")
        print(f"  {'-'*54}")
        for name, vals in all_results.items():
            print(
                f"  {name:<24} {vals['est_1year_minutes']:>16.1f} "
                f"{vals['per_sym_ensemble_ms']:>12.3f}"
            )


if __name__ == "__main__":
    main()
