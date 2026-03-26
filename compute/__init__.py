"""
Unified CPU / GPU compute backend.

Usage:
    from compute import get_backend
    backend = get_backend()          # reads BACKTEST_DEVICE env / config
    backend = get_backend("GPU")     # explicit override
"""

from compute.backend import ComputeBackend, BackendFactory

def get_backend(
    device: str | None = None,
    backend_mode: str | None = None,
) -> ComputeBackend:
    """Return the global compute backend (singleton)."""
    return BackendFactory.get(device, backend_mode=backend_mode)

__all__ = ["get_backend", "ComputeBackend", "BackendFactory"]
