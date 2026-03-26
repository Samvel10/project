from __future__ import annotations

import logging
import threading
from typing import Optional

from compute.backend_base import ComputeBackend

log = logging.getLogger(__name__)


class BackendFactory:
    """Singleton factory for compute backend instances."""

    _instance: Optional[ComputeBackend] = None
    _lock = threading.Lock()
    _resolved_runtime: Optional[dict] = None

    @classmethod
    def get(
        cls,
        device: Optional[str] = None,
        backend_mode: Optional[str] = None,
    ) -> ComputeBackend:
        if cls._instance is not None and device is None:
            return cls._instance

        with cls._lock:
            if cls._instance is not None and device is None:
                return cls._instance

            from compute.device_config import resolve_runtime_config

            runtime = resolve_runtime_config(
                device_override=device,
                backend_mode_override=backend_mode,
            )
            cls._resolved_runtime = runtime
            final_device = runtime["device"]
            mode = runtime.get("backend_mode", "parity")
            force_legacy_cpu = bool(runtime.get("fallback_applied"))

            if final_device == "GPU":
                from compute.gpu_backend import GpuBackend
                cls._instance = GpuBackend(
                    batch_size=runtime["batch_size"],
                    precision=runtime["precision"],
                    backend_mode=mode,
                )
            else:
                if mode == "parity" or force_legacy_cpu:
                    from compute.cpu_backend import CpuBackend
                    cls._instance = CpuBackend()
                else:
                    from compute.tensor_cpu_backend import TensorCpuBackend
                    cls._instance = TensorCpuBackend(
                        batch_size=runtime["batch_size"],
                        precision="fp32",
                        backend_mode=mode,
                    )

            log.info(
                "[BACKEND] device=%s mode=%s backend=%s batch_size=%s precision=%s",
                cls._instance.device_name,
                cls._instance.backend_mode,
                cls._instance.backend_name,
                cls._instance.batch_size,
                cls._instance.precision,
            )
            return cls._instance

    @classmethod
    def get_runtime_info(cls) -> dict:
        return dict(cls._resolved_runtime or {})

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None
            cls._resolved_runtime = None
