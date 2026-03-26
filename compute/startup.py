"""Startup validation and diagnostics for compute backend selection."""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


def _send_telegram_alert(message: str) -> None:
    """Best-effort Telegram alert via existing bot infrastructure."""
    try:
        import requests
        # Read bot tokens from environment or config
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_ALERT_CHAT_ID", "")
        if not token or not chat_id:
            # Try loading from config
            try:
                import yaml
                from pathlib import Path
                cfg_path = Path(__file__).resolve().parents[1] / "config" / "trading.yaml"
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    tg = cfg.get("telegram", {})
                    token = token or tg.get("bot_token", "")
                    chat_id = chat_id or str(tg.get("alert_chat_id", ""))
            except Exception:
                pass
        if token and chat_id:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={
                "chat_id": chat_id,
                "text": f"🚨 COMPUTE BACKEND ALERT\n\n{message}",
                "parse_mode": "HTML",
            }, timeout=10)
    except Exception:
        pass


def validate_and_init(device_override: Optional[str] = None) -> dict:
    """Resolve device, initialize backend, and install guards.

    GPU validation failure does not stop the process; it sends a Telegram alert
    and falls back to CPU automatically.
    """
    from compute.backend_factory import BackendFactory
    from compute.device_config import resolve_runtime_config

    runtime = resolve_runtime_config(device_override=device_override)
    if runtime.get("fallback_applied"):
        msg = (
            "GPU requested but unavailable. Falling back to CPU.\n"
            f"Reason: {runtime.get('fallback_reason', 'unknown')}"
        )
        log.warning("[STARTUP] %s", msg)
        _send_telegram_alert(msg)

    backend = BackendFactory.get(
        runtime.get("requested_device"),
        backend_mode=runtime.get("backend_mode"),
    )
    _print_device_banner(runtime, backend_name=backend.backend_name)
    backend.warmup()
    if runtime.get("device") == "GPU":
        _install_gpu_error_handlers()
    return runtime


def _print_device_banner(info: dict, backend_name: str) -> None:
    """Print device info to stdout."""
    device = info.get("device", "CPU")
    print(f"\n{'='*50}")
    print(f"  COMPUTE BACKEND: {device} ({backend_name})")
    print(f"{'='*50}")
    if info.get("fallback_applied"):
        print("  WARNING: GPU fallback to CPU applied")
        print(f"  REASON:  {info.get('fallback_reason', 'unknown')}")
    if device == "GPU":
        print(f"  GPU:    {info.get('gpu_name', 'Unknown')}")
        print(f"  VRAM:   {info.get('gpu_memory_gb', '?')} GB")
        print(f"  CUDA:   {info.get('cuda_version', '?')}")
        print(f"  Torch:  {info.get('torch_version', '?')}")
        print(f"  GPUs:   {info.get('gpu_count', 1)}")
    else:
        print("  Mode:   Pure CPU (no GPU initialization)")
    print(f"  Batch:  {info.get('batch_size', '?')}")
    print(f"  Prec:   {info.get('precision', '?')}")
    print(f"  Mode:   {info.get('backend_mode', 'parity')}")
    print(f"{'='*50}\n", flush=True)


def _install_gpu_error_handlers() -> None:
    """Install handlers for CUDA OOM and other GPU errors."""
    try:
        import torch

        log.info(
            "[STARTUP] GPU error handlers active (allocated=%.2fGB reserved=%.2fGB)",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )

    except ImportError:
        pass
