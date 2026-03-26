"""
Device and runtime config resolver for backtest compute backends.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

_VALID_DEVICES = {"CPU", "GPU"}
_VALID_PRECISION = {"fp32", "fp16"}
_VALID_BACKEND_MODES = {"parity", "experimental"}


def _load_execution_config() -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {}
    root = Path(__file__).resolve().parents[1] / "config"
    # Prefer explicit execution_config.yaml, then legacy execution.yaml.
    for name in ("execution_config.yaml", "execution.yaml"):
        cfg_path = root / name
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if isinstance(data, dict):
            if isinstance(data.get("execution"), dict):
                return data["execution"]
            return data
    return {}


def _resolve_requested_device(device_override: Optional[str]) -> str:
    if device_override:
        raw = str(device_override).strip().upper()
        if raw in _VALID_DEVICES:
            return raw
        raise ValueError(f"Invalid device override: {device_override!r}")

    env_val = os.getenv("BACKTEST_DEVICE", "").strip().upper()
    if env_val:
        if env_val in _VALID_DEVICES:
            return env_val
        raise ValueError("BACKTEST_DEVICE must be CPU or GPU")

    # Backward-compatible fallback env var.
    legacy_env = os.getenv("EXECUTION_DEVICE", "").strip().upper()
    if legacy_env:
        if legacy_env in _VALID_DEVICES:
            return legacy_env
        raise ValueError("EXECUTION_DEVICE must be CPU or GPU")

    cfg = _load_execution_config()
    cfg_device = str(cfg.get("device", "CPU")).strip().upper()
    if cfg_device in _VALID_DEVICES:
        return cfg_device
    return "CPU"


def _resolve_batch_size(cfg: Dict[str, Any]) -> int:
    raw = cfg.get("batch_size", 256)
    try:
        val = int(raw)
    except Exception:
        return 256
    return max(1, min(val, 4096))


def _resolve_backend_mode(cfg: Dict[str, Any], override: Optional[str]) -> str:
    if override is not None:
        val = str(override).strip().lower()
        if val in _VALID_BACKEND_MODES:
            return val
        raise ValueError("backend_mode override must be parity or experimental")
    env_val = os.getenv("BACKTEST_BACKEND_MODE", "").strip().lower()
    if env_val:
        if env_val in _VALID_BACKEND_MODES:
            return env_val
        raise ValueError("BACKTEST_BACKEND_MODE must be parity or experimental")
    cfg_val = str(cfg.get("backend_mode", "parity")).strip().lower()
    if cfg_val in _VALID_BACKEND_MODES:
        return cfg_val
    return "parity"


def _resolve_precision(cfg: Dict[str, Any], requested_device: str) -> str:
    raw = str(cfg.get("precision", "fp32")).strip().lower()
    if raw not in _VALID_PRECISION:
        raw = "fp32"
    if requested_device != "GPU":
        return "fp32"
    return raw


def _validate_gpu() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "reason": f"torch import failed: {exc}"}

    if not torch.cuda.is_available():
        return {
            "ok": False,
            "reason": f"CUDA unavailable (torch.version.cuda={getattr(torch.version, 'cuda', None)})",
        }

    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "ok": True,
            "gpu_name": props.name,
            "gpu_memory_gb": round(props.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "gpu_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        return {"ok": False, "reason": f"CUDA device query failed: {exc}"}


def resolve_runtime_config(
    device_override: Optional[str] = None,
    backend_mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve final runtime config with automatic GPU fallback."""
    cfg = _load_execution_config()
    requested_device = _resolve_requested_device(device_override)
    backend_mode = _resolve_backend_mode(cfg, backend_mode_override)
    batch_size = _resolve_batch_size(cfg)
    precision = _resolve_precision(cfg, requested_device)

    result: Dict[str, Any] = {
        "requested_device": requested_device,
        "device": requested_device,
        "batch_size": batch_size,
        "precision": precision,
        "backend_mode": backend_mode,
        "fallback_applied": False,
        "fallback_reason": "",
    }

    if requested_device == "GPU":
        gpu_state = _validate_gpu()
        if gpu_state.get("ok"):
            result.update(gpu_state)
        else:
            result["device"] = "CPU"
            result["precision"] = "fp32"
            result["fallback_applied"] = True
            result["fallback_reason"] = str(gpu_state.get("reason", "unknown GPU validation failure"))
            result["fallback_target_backend"] = "CpuBackendLegacy"
    return result


def get_device(override: Optional[str] = None) -> str:
    """Backward-compatible accessor used by existing code."""
    return resolve_runtime_config(override)["device"]
