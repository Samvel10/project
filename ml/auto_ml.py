import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.logger import log
from monitoring.signal_log import get_log_path
from ml.label_signals import label_signals
from ml.train_from_signal_log import main as train_from_signal_log_main
from ml.inference import reload_model


_LAST_SIGNAL_COUNT = 0


def _get_signal_count() -> int:
    """Return the number of signal rows currently stored in signal_log.csv.

    We count data rows (excluding the header). Any error is treated as zero.
    """

    try:
        path = get_log_path()
        if not path.exists():
            return 0
        with path.open("r", newline="", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        # Subtract header row if present
        return max(0, total - 1)
    except Exception:
        return 0


def run_auto_ml() -> None:
    """Run self-learning step: label signals, train from log, reload model.

    This is intended to be called periodically from the main trading loop.
    """

    global _LAST_SIGNAL_COUNT

    try:
        # log("[AUTO_ML] Labeling signals...")
        label_signals()
    except SystemExit:
        # In case label_signals uses sys.exit in CLI mode
        pass
    except Exception as e:
        # log(f"[AUTO_ML] label_signals failed: {e}")
        return

    # Decide whether to train based on whether any NEW signals have been
    # appended to signal_log.csv since the last successful training run.
    current_count = _get_signal_count()
    if current_count <= _LAST_SIGNAL_COUNT:
        # log("[AUTO_ML] No new signals; skipping training.")
        return

    try:
        # log("[AUTO_ML] Training model from signal_log...")
        train_from_signal_log_main()
    except SystemExit:
        pass
    except Exception as e:
        # log(f"[AUTO_ML] train_from_signal_log failed: {e}")
        return

    try:
        reload_model()
        # log("[AUTO_ML] Reloaded latest ML model into inference engine")
        _LAST_SIGNAL_COUNT = current_count
    except Exception as e:
        ""
        # log(f"[AUTO_ML] reload_model failed: {e}")
