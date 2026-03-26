"""
Backward-compatible exports for compute backend abstractions.
"""

from compute.backend_base import ComputeBackend, FeatureResult, SignalOutput
from compute.backend_factory import BackendFactory

__all__ = ["ComputeBackend", "FeatureResult", "SignalOutput", "BackendFactory"]
