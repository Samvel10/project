from __future__ import annotations

import json
import hashlib
import threading
from pathlib import Path
from typing import List, Optional

import requests

from monitoring.telegram import send_telegram
from monitoring.updater import download_and_apply_update


_STATE_DIR = Path(__file__).resolve().parents[1] / "data"
_LEGACY_STATE_PATH = _STATE_DIR / "telegram_state.json"
_LOCK = threading.Lock()


def _token_key(token: Optional[str]) -> str:
    raw = (token or "").strip()
    if not raw:
        return "default"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _state_path_for_token(token: Optional[str]) -> Path:
    # Keep legacy path for empty token callers (backward compatibility).
    key = _token_key(token)
    if key == "default":
        return _LEGACY_STATE_PATH
    return _STATE_DIR / f"telegram_state_{key}.json"


def _ensure_state_file(token: Optional[str]) -> Path:
    if not _STATE_DIR.exists():
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path_for_token(token)
    if not path.exists():
        path.write_text(json.dumps({"offset": 0, "chat_ids": []}), encoding="utf-8")
    return path


def _load_state(token: Optional[str]) -> dict:
    path = _ensure_state_file(token)
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {"offset": 0, "chat_ids": []}
        return json.loads(raw)
    except Exception:
        # On any parse error, reset to empty state
        return {"offset": 0, "chat_ids": []}


def _save_state(state: dict, token: Optional[str]) -> None:
    path = _ensure_state_file(token)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def update_subscribers(
    token: Optional[str],
    instance_id: Optional[str] = None,
    update_base_url: Optional[str] = None,
    include_config: bool = False,
    include_data: bool = False,
) -> None:
    """Poll Telegram getUpdates and register /start senders as subscribers.

    This keeps track of:
      - last processed update_id (offset)
      - list of unique chat_ids that sent /start
    """

    if not token:
        return

    with _LOCK:
        state = _load_state(token)
        offset = int(state.get("offset", 0))

        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 0, "offset": offset + 1}

        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Network / API errors should never break trading loop
            return

        if not isinstance(data, dict) or not data.get("ok"):
            return

        results = data.get("result", []) or []
        if not results:
            return

        new_offset = offset
        chat_ids = set(state.get("chat_ids") or [])

        for update in results:
            try:
                upd_id = int(update.get("update_id"))
                if upd_id > new_offset:
                    new_offset = upd_id

                message = update.get("message") or update.get("channel_post")
                if not message:
                    continue

                text = message.get("text") or ""
                if not isinstance(text, str):
                    continue

                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if chat_id is None:
                    continue

                if text.startswith("/start"):
                    chat_ids.add(chat_id)

                # Handle on-demand client update requests: /update
                if text.startswith("/update") and instance_id and update_base_url:
                    try:
                        res = download_and_apply_update(
                            instance_id,
                            update_base_url,
                            include_config=include_config,
                            include_data=include_data,
                        )
                        if res.get("ok"):
                            msg = (
                                "Update հաջողությամբ բեռնվեց և կիրառվեց. "
                                "նոր կոդը կաշխատի հաջորդ start-ի ժամանակ։"
                            )
                        else:
                            msg = f"Update ձախողվեց․ {res.get('error') or 'անհայտ սխալ'}"
                    except Exception as e:
                        msg = f"Update ընթացքում տեղի ունեցավ սխալ․ {e}"

                    try:
                        send_telegram(msg, token, int(chat_id))
                    except Exception:
                        pass
            except Exception:
                continue

        if new_offset != offset or chat_ids != set(state.get("chat_ids") or []):
            state["offset"] = new_offset
            state["chat_ids"] = list(chat_ids)
            _save_state(state, token)


def get_subscribers(extra_chat_id: Optional[int] = None, token: Optional[str] = None) -> List[int]:
    """Return list of subscriber chat_ids, optionally including a static extra ID.

    extra_chat_id can be used to always include the chat_id from config
    (e.g. the owner's personal chat or a fixed channel).
    """

    with _LOCK:
        state = _load_state(token)
        chat_ids = set(state.get("chat_ids") or [])
        if extra_chat_id is not None:
            try:
                chat_ids.add(int(extra_chat_id))
            except (TypeError, ValueError):
                pass
        return list(chat_ids)


def remove_subscriber(chat_id: int, token: Optional[str] = None) -> bool:
    """Remove a chat_id from subscriber state.

    Used when Telegram returns 403 (bot blocked by the user) so we do not
    keep retrying and spamming logs.
    """

    try:
        chat_id_int = int(chat_id)
    except Exception:
        return False

    with _LOCK:
        state = _load_state(token)
        ids = list(state.get("chat_ids") or [])
        before = len(ids)
        ids = [x for x in ids if str(x) != str(chat_id_int)]
        if len(ids) == before:
            return False
        state["chat_ids"] = ids
        _save_state(state, token)
        return True
