import sys
import json
import time
from pathlib import Path
from typing import Dict

import requests
from ruamel.yaml import YAML

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.performance_report import build_performance_summary_since
from monitoring.telegram import send_telegram


_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "stats_bot_state.json"


def _ensure_state_file() -> None:
    if not _STATE_PATH.parent.exists():
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        _STATE_PATH.write_text(json.dumps({"offset": 0}), encoding="utf-8")


def _load_state() -> Dict:
    _ensure_state_file()
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {"offset": 0}
        return json.loads(raw)
    except Exception:
        return {"offset": 0}


def _save_state(state: Dict) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _split_message(text, max_len: int = 3900):
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_len = 0
    for line in lines:
        add_len = len(line) + 1
        if current_lines and current_len + add_len > max_len:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += add_len
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def _load_analytics_config() -> tuple[str, float]:
    yaml = YAML()
    with (ROOT_DIR / "config" / "trading.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.load(f)
    analytics_cfg = cfg.get("analytics_bot") or {}
    token = analytics_cfg.get("token") or ""
    window_hours_raw = analytics_cfg.get("window_hours", 24.0)
    try:
        window_hours = float(window_hours_raw)
    except (TypeError, ValueError):
        window_hours = 24.0
    return token, window_hours


def run_stats_bot() -> None:
    token, window_hours = _load_analytics_config()
    if not token:
        print("[STATS BOT] analytics_bot.token is not configured in config/trading.yaml")
        return

    print("[STATS BOT] Starting analytics bot polling loop...")

    state = _load_state()
    offset = int(state.get("offset", 0))

    while True:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 10, "offset": offset + 1}

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            time.sleep(5)
            continue

        if not isinstance(data, dict) or not data.get("ok"):
            time.sleep(5)
            continue

        results = data.get("result", []) or []
        if not results:
            time.sleep(2)
            continue

        for update in results:
            try:
                upd_id = int(update.get("update_id"))
                if upd_id > offset:
                    offset = upd_id

                message = update.get("message") or update.get("channel_post")
                if not message:
                    continue

                text = message.get("text") or ""
                if not isinstance(text, str):
                    continue

                if not (text.startswith("/start") or text.startswith("/stats")):
                    continue

                # Optional argument: number of detailed trades to show, e.g.
                # "/start 30" or "/stats 50". If omitted or invalid, show
                # all trades in the time window.
                parts = text.strip().split()
                max_trades = None
                if len(parts) >= 2:
                    try:
                        parsed = int(parts[1])
                        if parsed > 0:
                            max_trades = parsed
                    except (TypeError, ValueError):
                        max_trades = None

                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if chat_id is None:
                    continue

                try:
                    summary = build_performance_summary_since(
                        window_hours, max_detailed_trades=max_trades
                    )
                except Exception:
                    summary = "No performance data available."

                if summary:
                    try:
                        for chunk in _split_message(summary):
                            send_telegram(chunk, token, chat_id)
                    except Exception:
                        continue
            except Exception:
                continue

        state["offset"] = offset
        _save_state(state)

        time.sleep(2)


if __name__ == "__main__":
    run_stats_bot()
