"""
Global runtime switches shared across modules.
"""

# Single-authority trade lifecycle mode:
# When True, only AITM is allowed to modify open trades.
AITM_MASTER_ENABLED = True

# AITM rollout / safety envelope flags (default safe/off).
# When all are OFF, new experimental ML-first features are disabled.
AITM_ML_FIRST_ENABLED = True
AITM_SYMBOL_NORMALIZATION_ENABLED = True
AITM_FAST_LOOP_ENABLED = True
AITM_INTERMEDIATE_CREDIT_LOG_ENABLED = True
AITM_DYNAMIC_MODE_POLICY_ENABLED = True
AITM_OVERRIDE_TRANSPARENCY_ENABLED = True
AITM_BREAKEVEN_LOCK_ENABLED = True
AITM_TRANCHE_TP_ENABLED = True
AITM_ORPHAN_CLEANUP_ENABLED = True
AITM_ML_STRICT_MODE = True

