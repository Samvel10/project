import json
import time
from pathlib import Path

import requests

try:
    from ruamel.yaml import YAML  # type: ignore
except Exception:
    YAML = None
import yaml as _pyyaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "trading.yaml"
STATE_PATH = ROOT_DIR / "data" / "telegram_start_subscribers.json"

POLL_INTERVAL_SEC = 2.0
HTTP_TIMEOUT_SEC = 15.0

_YAML = YAML(typ="safe") if YAML is not None else None


def _load_log_bot_token() -> str | None:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            if _YAML is not None:
                cfg = _YAML.load(f) or {}
            else:
                cfg = _pyyaml.safe_load(f) or {}
    except Exception:
        return None

    if not isinstance(cfg, dict):
        return None

    log_cfg = cfg.get("log_bot") or {}
    if not isinstance(log_cfg, dict):
        return None

    token = log_cfg.get("token")
    if token is None:
        return None
    try:
        token_str = str(token).strip()
    except Exception:
        return None
    return token_str or None


def _ensure_state_file() -> None:
    if not STATE_PATH.parent.exists():
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        STATE_PATH.write_text(
            json.dumps({"offset": 0, "chat_ids": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _load_state() -> dict:
    _ensure_state_file()
    try:
        raw = STATE_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
    except Exception:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}
    if "offset" not in parsed:
        parsed["offset"] = 0
    if "chat_ids" not in parsed or not isinstance(parsed["chat_ids"], list):
        parsed["chat_ids"] = []
    return parsed


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _collect_once(token: str) -> None:
    state = _load_state()
    try:
        offset = int(state.get("offset", 0))
    except Exception:
        offset = 0

    try:
        known_ids = {int(x) for x in (state.get("chat_ids") or [])}
    except Exception:
        known_ids = set()

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": offset + 1, "timeout": 10}

    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return

    if not isinstance(payload, dict) or not payload.get("ok"):
        return

    results = payload.get("result") or []
    if not isinstance(results, list) or not results:
        return

    new_offset = offset
    changed = False

    for upd in results:
        if not isinstance(upd, dict):
            continue
        try:
            upd_id = int(upd.get("update_id"))
            if upd_id > new_offset:
                new_offset = upd_id
        except Exception:
            pass

        msg = upd.get("message") or upd.get("channel_post") or {}
        if not isinstance(msg, dict):
            continue
        text = msg.get("text")
        if not isinstance(text, str):
            continue
        if not text.startswith("/start"):
            continue

        chat = msg.get("chat") or {}
        if not isinstance(chat, dict):
            continue
        chat_id = chat.get("id")
        try:
            chat_id_int = int(chat_id)
        except Exception:
            continue

        if chat_id_int not in known_ids:
            known_ids.add(chat_id_int)
            changed = True

    if new_offset != offset:
        state["offset"] = new_offset
        changed = True
    if changed:
        state["chat_ids"] = sorted(known_ids)
        _save_state(state)


def main() -> None:
    _ensure_state_file()
    while True:
        token = _load_log_bot_token()
        if token:
            _collect_once(token)
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
