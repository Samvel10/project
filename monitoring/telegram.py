import requests
from pathlib import Path


_MAX_TELEGRAM_TEXT_LEN = 3500  # պահում ենք 4096-ից քիչ, որ multi-byte սիմվոլների հետ էլ ապահով լինի


def _send_telegram_chunk(message: str, token: str, chat_id: int):
    """Send a single Telegram message chunk and return its message_id if available."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
    }

    resp = requests.post(url, json=payload, timeout=10)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Enrich error with Telegram's response body for easier debugging
        raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp) from e

    try:
        data = resp.json()
        result = data.get("result") if isinstance(data, dict) else None
        if isinstance(result, dict):
            return result.get("message_id")
    except Exception:
        pass
    return None


def send_telegram(message, token, chat_id):
    """Send a Telegram message, splitting into multiple chunks if it's too long.

    This avoids 400 "message is too long" errors when we send big summaries.
    """

    if message is None:
        return None

    text = str(message)

    # Փոքր հաղորդագրությունների դեպքում ուղարկում ենք անմիջապես
    if len(text) <= _MAX_TELEGRAM_TEXT_LEN:
        return _send_telegram_chunk(text, token, chat_id)

    # Մեծ հաղորդագրությունների դեպքում կտրատում ենք տողերով
    last_message_id = None
    start = 0
    n = len(text)

    while start < n:
        end = min(start + _MAX_TELEGRAM_TEXT_LEN, n)
        chunk = text[start:end]

        # Փորձում ենք չկտրել տողի մեջ. նախընտրում ենք վերջին '\n'-ով բաժանումը
        if end < n:
            newline_pos = chunk.rfind("\n")
            if newline_pos > 0:
                real_end = start + newline_pos + 1
                chunk = text[start:real_end]
                start = real_end
            else:
                start = end
        else:
            start = end

        if not chunk.strip():
            continue

        last_message_id = _send_telegram_chunk(chunk, token, chat_id)

    return last_message_id


def forward_telegram(token, from_chat_id, to_chat_id, message_id):
    url = f"https://api.telegram.org/bot{token}/forwardMessage"
    payload = {
        "chat_id": to_chat_id,
        "from_chat_id": from_chat_id,
        "message_id": message_id,
    }

    resp = requests.post(url, json=payload, timeout=10)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp) from e
    return resp.json()


def send_telegram_document(file_path: str, token: str, chat_id: int, caption: str | None = None):
    """Send a file as a Telegram document.

    Returns Telegram message_id if available.
    """

    p = Path(str(file_path))
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Document not found: {p}")

    url = f"https://api.telegram.org/bot{token}/sendDocument"

    data = {
        "chat_id": int(chat_id),
    }
    if caption:
        data["caption"] = str(caption)[:1000]

    with p.open("rb") as f:
        files = {"document": (p.name, f)}
        resp = requests.post(url, data=data, files=files, timeout=60)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp) from e

    try:
        payload = resp.json()
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            return result.get("message_id")
    except Exception:
        pass
    return None
