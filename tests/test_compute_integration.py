from __future__ import annotations

from typing import Any, Dict, List, Tuple

from compute.gpu_backend import GpuBackend
from compute.tensor_cpu_backend import TensorCpuBackend


def _make_candles(n: int, start: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    p = start
    for i in range(n):
        nxt = p * (1.0 + (0.0005 if i % 2 == 0 else -0.0003))
        out.append(
            {
                "open": p,
                "high": max(p, nxt) * 1.001,
                "low": min(p, nxt) * 0.999,
                "close": nxt,
                "volume": 1_000.0 + i,
            }
        )
        p = nxt
    return out


def _make_items() -> List[Tuple[str, List[Dict[str, Any]]]]:
    return [
        ("BTCUSDT", _make_candles(220, 40_000.0)),
        ("ETHUSDT", _make_candles(220, 2_000.0)),
        ("SOLUSDT", _make_candles(220, 120.0)),
    ]


def test_tensor_cpu_backend_batch_generation() -> None:
    backend = TensorCpuBackend(batch_size=128)
    outs = backend.generate_signals_batch(_make_items())
    assert len(outs) == 3
    for out in outs:
        assert out.symbol
        assert out.direction in {"BUY", "SELL", "NO_TRADE"}
        assert 0.0 <= out.confidence <= 1.0
        assert out.features is not None


def test_simulated_gpu_matches_cpu_with_tolerance() -> None:
    items = _make_items()
    cpu_backend = TensorCpuBackend(batch_size=128)
    sim_gpu_backend = GpuBackend(batch_size=128, simulate_gpu_on_cpu=True)

    cpu_out = cpu_backend.generate_signals_batch(items)
    gpu_out = sim_gpu_backend.generate_signals_batch(items)

    assert len(cpu_out) == len(gpu_out)
    for a, b in zip(cpu_out, gpu_out):
        assert a.symbol == b.symbol
        assert a.direction == b.direction
        assert abs(a.confidence - b.confidence) < 1e-5
