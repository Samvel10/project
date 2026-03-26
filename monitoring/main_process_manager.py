import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
_STATE_PATH = ROOT_DIR / "data" / "main_process_state.json"


def _ensure_state_dir() -> None:
    if not _STATE_PATH.parent.exists():
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    """Load saved state about the main.py process (PID, etc.)."""

    if not _STATE_PATH.exists():
        return {}
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    """Persist state to disk."""

    _ensure_state_dir()
    try:
        _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Failing to save state should not crash the caller
        pass


def _is_pid_running(pid: int) -> bool:
    """Return True if a process with given PID appears to be running."""

    try:
        # Sending signal 0 only checks for existence / permissions
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we may not have permission; treat as running
        return True
    else:
        return True


def _get_tracked_pid() -> Optional[int]:
    """Return tracked PID if it is still running, otherwise None."""

    state = _load_state()
    pid = state.get("pid")
    if isinstance(pid, int) and _is_pid_running(pid):
        return pid
    return None


def status_main() -> str:
    """Return human-readable status of the main trading bot process."""

    pid = _get_tracked_pid()
    if pid is not None:
        return f"Main trading bot is RUNNING (PID {pid})."
    return "Main trading bot is NOT running."


def start_main() -> str:
    """Start main.py as a background process if it is not already running."""

    existing_pid = _get_tracked_pid()
    if existing_pid is not None:
        return f"Main trading bot already running (PID {existing_pid})."

    main_path = ROOT_DIR / "main.py"
    if not main_path.exists():
        return f"main.py not found at {main_path}"

    try:
        proc = subprocess.Popen(
            [sys.executable, str(main_path)],
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as e:
        return f"Failed to start main trading bot: {e}"

    _save_state({"pid": proc.pid})
    return f"Started main trading bot (PID {proc.pid})."


def stop_main() -> str:
    """Stop the tracked main.py process if it appears to be running."""

    pid = _get_tracked_pid()
    if pid is None:
        # Clear any stale state just in case
        _save_state({})
        return "Main trading bot is NOT running (no active PID)."

    try:
        os.kill(pid, signal.SIGTERM)
        msg = f"Sent SIGTERM to main trading bot (PID {pid})."
    except ProcessLookupError:
        msg = f"No process found with PID {pid}. Clearing state."
    except Exception as e:
        msg = f"Failed to terminate main trading bot (PID {pid}): {e}"

    # Regardless of outcome, clear stored state so we do not reuse a stale PID
    _save_state({})
    return msg


def restart_main() -> str:
    """Restart the main trading bot process (stop then start)."""

    stop_msg = stop_main()
    start_msg = start_main()
    return stop_msg + "\n" + start_msg


if __name__ == "__main__":
    # CLI helper so you can manage main.py manually from shell:
    #   python -m monitoring.main_process_manager status|start|stop|restart
    import argparse

    parser = argparse.ArgumentParser(description="Manage main.py trading bot process.")
    parser.add_argument(
        "command",
        choices=["status", "start", "stop", "restart"],
        help="Action to perform on main.py process.",
    )
    args = parser.parse_args()

    if args.command == "status":
        print(status_main())
    elif args.command == "start":
        print(start_main())
    elif args.command == "stop":
        print(stop_main())
    elif args.command == "restart":
        print(restart_main())
