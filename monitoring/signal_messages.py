from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict


_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "signal_messages.json"
_LOCK = threading.Lock()


def _ensure_state_file() -> None:
    if not _STATE_PATH.parent.exists():
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        _STATE_PATH.write_text("{}", encoding="utf-8")


def _load_state() -> dict:
    _ensure_state_file()
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def record_signal_message(symbol: str, timestamp_ms: int, chat_id: int, message_id: int) -> None:
    """Store Telegram message_id for a given signal (symbol + timestamp_ms)."""

    key = f"{symbol}|{int(timestamp_ms)}"
    with _LOCK:
        state = _load_state()
        entry = state.get(key) or {}
        entry[str(int(chat_id))] = int(message_id)
        state[key] = entry
        _save_state(state)


def get_signal_messages(symbol: str, timestamp_ms: int) -> Dict[int, int]:
    """Return mapping chat_id -> message_id for a given signal, if known."""

    key = f"{symbol}|{int(timestamp_ms)}"
    with _LOCK:
        state = _load_state()
        raw_entry = state.get(key) or {}

    result: Dict[int, int] = {}
    if isinstance(raw_entry, dict):
        for k, v in raw_entry.items():
            try:
                cid = int(k)
                mid = int(v)
                result[cid] = mid
            except (TypeError, ValueError):
                continue
    return result
