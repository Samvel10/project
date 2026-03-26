from __future__ import annotations

import hashlib
import json
import random
import unittest
from unittest.mock import patch

import torch

from backtest.production.config import AccountConfig, BacktestConfig
from backtest.production.engine import BacktestEngine
from backtest.production.events import Event, EventType
from compute.cpu_backend import CpuBackend
from compute.gpu_backend import GpuBackend
from compute.tensor_cpu_backend import TensorCpuBackend


def _make_candles(n: int, start: float) -> list[dict]:
    out: list[dict] = []
    p = start
    for i in range(n):
        p2 = p * (1.001 if i % 2 == 0 else 0.999)
        out.append(
            {
                "open": p,
                "high": max(p, p2) * 1.001,
                "low": min(p, p2) * 0.999,
                "close": p2,
                "volume": 1000 + i,
                "open_time": i * 60_000,
                "close_time": (i + 1) * 60_000,
            }
        )
        p = p2
    return out


def _fake_dataset() -> tuple[list[Event], dict]:
    candles_a = _make_candles(40, 100.0)
    candles_b = _make_candles(40, 200.0)
    events: list[Event] = []
    seq = 0
    for i in range(40):
        ts = 1_700_000_000_000 + i * 60_000
        events.append(
            Event(
                timestamp_ms=ts,
                sequence=seq,
                event_type=EventType.KLINE_CLOSE,
                symbol="AUSDT",
                data=candles_a[i],
            )
        )
        seq += 1
        # Same-timestamp second symbol tests causality/order.
        events.append(
            Event(
                timestamp_ms=ts,
                sequence=seq,
                event_type=EventType.KLINE_CLOSE,
                symbol="BUSDT",
                data=candles_b[i],
            )
        )
        seq += 1
    return events, {"AUSDT": candles_a, "BUSDT": candles_b}


def _base_config() -> BacktestConfig:
    return BacktestConfig(
        start_date="2025-01-01",
        end_date="2025-01-02",
        symbols=["AUSDT", "BUSDT"],
        accounts=[AccountConfig(index=0, name="acc0", initial_balance=10_000.0)],
        warmup_candles=20,
        use_real_ml=False,
        verbose=False,
    )


def _trade_hash(results: dict) -> str:
    norm_signals = []
    for s in results.get("signals", []):
        if not isinstance(s, dict):
            continue
        norm_signals.append({
            "timestamp_ms": s.get("timestamp_ms"),
            "symbol": s.get("symbol"),
            "signal": s.get("signal"),
            "confidence": round(float(s.get("confidence", 0.0)), 10),
            "entry_price": round(float(s.get("entry_price", 0.0)), 10),
            "sl_price": round(float(s.get("sl_price", 0.0)), 10),
            "tp_prices": [round(float(x), 10) for x in (s.get("tp_prices") or [])],
            "atr": round(float(s.get("atr", 0.0)), 10),
        })
    norm_trades = []
    for t in results.get("trades", []):
        d = vars(t).copy()
        d.pop("hold_duration_sec", None)
        norm_trades.append(d)
    stats = results.get("statistics", {})
    payload = {
        "signals": norm_signals,
        "statistics": {
            "total_events": stats.get("total_events"),
            "total_signals": stats.get("total_signals"),
            "total_orders": stats.get("total_orders"),
            "total_trades": stats.get("total_trades"),
        },
        "accounts": results.get("accounts", {}),
        "trades": norm_trades,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ProductionParityTests(unittest.TestCase):
    def _run_engine_with_backend(self, backend) -> dict:
        random.seed(1337)
        cfg = _base_config()
        fake_events, fake_candles = _fake_dataset()
        with (
            patch("backtest.production.engine.validate_and_init", return_value={
                "device": backend.device_name,
                "requested_device": backend.device_name,
                "batch_size": backend.batch_size,
                "precision": backend.precision,
                "backend_mode": backend.backend_mode,
                "fallback_applied": False,
                "fallback_reason": "",
            }),
            patch("backtest.production.engine.get_backend", return_value=backend),
            patch("backtest.production.engine.load_backtest_data", return_value=(fake_events, fake_candles)),
        ):
            eng = BacktestEngine(cfg)
            return eng.run()

    def test_01_golden_parity_legacy_tensor_gpu(self) -> None:
        legacy = CpuBackend(backend_mode="parity")
        tensor = TensorCpuBackend(backend_mode="parity")
        gpu = GpuBackend(simulate_gpu_on_cpu=True, backend_mode="parity")
        items = [("AUSDT", _make_candles(120, 100.0)), ("BUSDT", _make_candles(120, 200.0))]

        a = legacy.generate_signals_batch(items)
        b = tensor.generate_signals_batch(items)
        c = gpu.generate_signals_batch(items)
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), len(c))
        for x, y, z in zip(a, b, c):
            self.assertEqual(x.direction, y.direction)
            self.assertEqual(x.direction, z.direction)
            self.assertAlmostEqual(x.confidence, y.confidence, places=8)
            self.assertAlmostEqual(x.confidence, z.confidence, places=8)

        # Engine-level parity (signals/trades/pnl hash equality).
        ra = self._run_engine_with_backend(legacy)
        rb = self._run_engine_with_backend(tensor)
        rc = self._run_engine_with_backend(gpu)
        self.assertEqual(_trade_hash(ra), _trade_hash(rb))
        self.assertEqual(_trade_hash(ra), _trade_hash(rc))

    def test_02_same_timestamp_causality_uses_legacy_order_in_parity(self) -> None:
        backend = TensorCpuBackend(backend_mode="parity")
        cfg = _base_config()
        fake_events, fake_candles = _fake_dataset()
        with (
            patch("backtest.production.engine.validate_and_init", return_value={
                "device": "CPU",
                "requested_device": "CPU",
                "batch_size": 256,
                "precision": "fp32",
                "backend_mode": "parity",
                "fallback_applied": False,
                "fallback_reason": "",
            }),
            patch("backtest.production.engine.get_backend", return_value=backend),
            patch("backtest.production.engine.load_backtest_data", return_value=(fake_events, fake_candles)),
        ):
            eng = BacktestEngine(cfg)
            with patch.object(eng, "_process_kline_batch", wraps=eng._process_kline_batch) as batch_call:
                eng.run()
                self.assertEqual(batch_call.call_count, 0)

    def test_03_gpu_oom_fallback_is_legacy_cpu(self) -> None:
        backend = GpuBackend(simulate_gpu_on_cpu=True, backend_mode="experimental")
        items = [("AUSDT", _make_candles(120, 100.0))]
        expected = CpuBackend(backend_mode="parity").generate_signals_batch(items)
        with patch.object(
            backend,
            "build_features_batch",
            side_effect=torch.cuda.OutOfMemoryError("forced-oom"),
        ):
            got = backend.generate_signals_batch(items)
        self.assertEqual(expected[0].direction, got[0].direction)
        self.assertAlmostEqual(expected[0].confidence, got[0].confidence, places=8)

    def test_04_precision_boundary_fp32_decision_logic(self) -> None:
        backend = GpuBackend(
            simulate_gpu_on_cpu=True,
            backend_mode="experimental",
            precision="fp16",
        )
        items = [("AUSDT", _make_candles(120, 100.0)), ("BUSDT", _make_candles(120, 99.9))]
        outs = backend.generate_signals_batch(items)
        # If internal fp32 assertion fails, generate_signals_batch will raise.
        for out in outs:
            self.assertIn(out.direction, {"BUY", "SELL", "NO_TRADE"})
            self.assertGreaterEqual(out.confidence, 0.0)
            self.assertLessEqual(out.confidence, 1.0)

    def test_05_end_to_end_snapshot_hash_stable(self) -> None:
        backend = CpuBackend(backend_mode="parity")
        r1 = self._run_engine_with_backend(backend)
        r2 = self._run_engine_with_backend(backend)
        self.assertEqual(_trade_hash(r1), _trade_hash(r2))


if __name__ == "__main__":
    unittest.main()
