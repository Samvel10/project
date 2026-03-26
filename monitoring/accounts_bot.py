import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from ruamel.yaml import YAML

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.telegram import send_telegram


_STATE_PATH = ROOT_DIR / "data" / "accounts_bot_state.json"
_DATA_PATH = ROOT_DIR / "data" / "user_accounts.json"


def _ensure_storage() -> None:
    if not _STATE_PATH.parent.exists():
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        _STATE_PATH.write_text(json.dumps({"offset": 0, "pending": {}}, ensure_ascii=False), encoding="utf-8")
    if not _DATA_PATH.exists():
        _DATA_PATH.write_text(json.dumps({"users": {}}, ensure_ascii=False), encoding="utf-8")


def _load_state() -> Dict[str, Any]:
    _ensure_storage()
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {"offset": 0, "pending": {}}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"offset": 0, "pending": {}}
        if "pending" not in data:
            data["pending"] = {}
        return data
    except Exception:
        return {"offset": 0, "pending": {}}


def _save_state(state: Dict[str, Any]) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _load_data() -> Dict[str, Any]:
    _ensure_storage()
    try:
        raw = _DATA_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {"users": {}}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"users": {}}
        if "users" not in data:
            data["users"] = {}
        return data
    except Exception:
        return {"users": {}}


def _save_data(data: Dict[str, Any]) -> None:
    _DATA_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_accounts_bot_config() -> str:
    yaml = YAML()
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f)
    accounts_cfg = cfg.get("accounts_bot") or {}
    token = accounts_cfg.get("token") or ""
    return str(token)


def _user_key_from_message(message: Dict[str, Any]) -> str:
    """Use Telegram user id if available, otherwise chat id."""
    from_user = message.get("from") or {}
    uid = from_user.get("id")
    if uid is None:
        chat = message.get("chat") or {}
        uid = chat.get("id")
    return str(uid) if uid is not None else "unknown"


def _get_user_record(data: Dict[str, Any], user_key: str) -> Dict[str, Any]:
    users = data.setdefault("users", {})
    rec = users.get(user_key)
    if not isinstance(rec, dict):
        rec = {"accounts": []}
        users[user_key] = rec
    if "accounts" not in rec or not isinstance(rec["accounts"], list):
        rec["accounts"] = []
    return rec


def _next_account_id(user_rec: Dict[str, Any]) -> str:
    accounts: List[Dict[str, Any]] = user_rec.get("accounts", [])
    existing_ids = []
    for acc in accounts:
        acc_id = acc.get("id")
        if isinstance(acc_id, str) and acc_id.startswith("acc_"):
            try:
                existing_ids.append(int(acc_id.split("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    n = max(existing_ids) + 1 if existing_ids else 1
    return f"acc_{n}"


def _format_accounts_for_user(user_rec: Dict[str, Any]) -> str:
    accounts: List[Dict[str, Any]] = user_rec.get("accounts", [])
    if not accounts:
        return "You have no API keys configured yet. Use /add_api to create one."

    lines = ["Your API connections:"]
    for acc in accounts:
        acc_id = acc.get("id", "?")
        name = acc.get("name", "(no name)")
        api_key = acc.get("api_key", "")
        tail = api_key[-4:] if isinstance(api_key, str) and len(api_key) >= 4 else "****"
        settings = acc.get("settings") or {}
        sl_pct = settings.get("sl_pct", "?")
        tp_pcts = settings.get("tp_pcts", [])
        tp_pcts_str = ", ".join(str(x) for x in tp_pcts) if tp_pcts else "-"
        fixed_notional = settings.get("fixed_notional_usd", 0)
        lines.append(
            f"{acc_id} | {name} | key ****{tail} | SL={sl_pct}% | TP={tp_pcts_str} | fixed={fixed_notional} USDT"
        )
    return "\n".join(lines)


def _handle_command_start(chat_id: int) -> List[str]:
    lines = [
        "Բարի գալուստ Binance հաշիվների կառավարման բոտ 👋",
        "",
        "Այս բոտը թույլ է տալիս․",
        "- պահել քո Binance Futures API key-երը (մի քանիսը միանգամից)",
        "- յուրաքանչյուր API-ի համար պահել առանձին կարգավորումներ (SL%, TP% մակարդակներ, ֆիքսված նոտիոնալ USDT)",
        "- հետո ցանկության դեպքում ջնջել կամ փոխել այդ API-ն և նրա կարգավորումները։",
        "",
        "⚠️ Անվտանգություն:",
        "Այս բոտը պահում է քո API key/secret-ը սերվերի վրա՝ \"user_accounts.json\" ֆայլում։",
        "Մի՛ տարածիր այս բոտի հետ քո անձնական չատը ուրիշների հետ և մի՛ ուղարկիր նույն key-ը այլ մարդկանց։",
        "Եթե այլևս չես ուզում, որ որևէ key օգտագործվի, ջնջիր այն /delete_api հրամանով։",
        "",
        "Քայլ առ քայլ՝ ինչպես սկսել:",
        "1) Գրի՛ր /add_api՝ նոր API ավելացնելու համար։",
        "   Բոտը երեք քայլով կխնդրի․",
        "   - Նախ անուն (օր․ 'Binance Main' կամ 'Վարդան Futures')",
        "   - Հետո API Key",
        "   - Հետո API Secret",
        "   Վերջում կստանաս նույնականացուցիչ (օրինակ acc_1) և default կարգավորումներ։",
        "",
        "2) Գրի՛ր /list_apis՝ տեսնելու համար բոլոր ավելացված API-ները և նրանց կարգավորումները.",
        "   Յուրաքանչյուր տողում կտեսնես․",
        "   - ID (օր. acc_1)",
        "   - անունը",
        "   - key-ի վերջի 4 նիշը",
        "   - SL%",
        "   - TP% մակարդակները",
        "   - ֆիքսված նոտիոնալը (USDT)։",
        "",
        "3) Եթե ուզում ես ջնջել կոնկրետ API՝ օգտվիր /delete_api հրամանով.",
        "   Օրինակ․ /delete_api acc_1",
        "",
        "4) Եթե ուզում ես փոխել Stop Loss տոկոսի արժեքը՝",
        "   Օրինակ․ /set_sl acc_1 3.0   → կսահմանի SL-ը 3.0% այդ API-ի համար։",
        "",
        "5) Եթե ուզում ես փոխել Take Profit տոկոսի մակարդակները՝",
        "   Օրինակ․ /set_tp acc_1 1.5 2.5 4.0",
        "   Կարող ես տալ 1, 2 կամ 3 մակարդակ (օրինակ միայն '2.0' կամ '2.0 3.0')։",
        "",
        "6) Եթե ուզում ես ֆիքսված նոտիոնալ դնել (յուրաքանչյուր trade իր համար USDT չափով)",
        "   Օրինակ․ /set_fixed acc_1 10   → յուրաքանչյուր trade-ի թիրախ տվյալ API-ի համար կլինի մոտ 10 USDT.",
        "   Եթե դնես 0, ապա կօգտագործվի Binance-ի նվազագույն հնարավոր չափը։",
        "",
        "Եթե մուտքագրես սխալ հրաման կամ չգիտես ինչ անել, միշտ կարող ես նորից գրել /start՝ տեսնելու այս օգնությունը։",
        "",
        "---",
        "English quick help:",
        "/add_api – create a new API connection (multi-step flow)",
        "/list_apis – list your saved APIs and their settings",
        "/delete_api <id> – delete API by id (e.g. /delete_api acc_1)",
        "/set_sl <id> <percent> – set stop-loss distance in % for that API",
        "/set_tp <id> p1 p2 [p3] – set 1–3 TP% levels",
        "/set_fixed <id> <usd> – set fixed notional size in USDT (0 = disabled)",
    ]
    return ["\n".join(lines)]


def _handle_add_api_step(msg: Dict[str, Any], state: Dict[str, Any], data: Dict[str, Any]) -> Optional[List[str]]:
    """Handle the multi-step /add_api flow.

    pending structure in state["pending"][chat_id]:
      {"action": "add_api", "step": "name"/"api_key"/"api_secret", "temp": {...}}
    """

    pending_all = state.setdefault("pending", {})
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    chat_key = str(chat_id)
    pending = pending_all.get(chat_key)
    if not isinstance(pending, dict) or pending.get("action") != "add_api":
        return None

    text = (msg.get("text") or "").strip()
    if not text:
        return ["Please send a non-empty text message."]

    # If user sends another command, cancel the flow
    if text.startswith("/"):
        pending_all.pop(chat_key, None)
        return ["Add-API flow cancelled."]

    step = pending.get("step")
    temp = pending.setdefault("temp", {})

    if step == "name":
        temp["name"] = text
        pending["step"] = "api_key"
        _save_state(state)
        return ["Now send the API key (as plain text)."]

    if step == "api_key":
        temp["api_key"] = text
        pending["step"] = "api_secret"
        _save_state(state)
        return ["Now send the API secret (as plain text).\n\nNote: it will be stored on this server. Do not share this bot or chat with others."]

    if step == "api_secret":
        temp["api_secret"] = text

        user_key = _user_key_from_message(msg)
        user_rec = _get_user_record(data, user_key)
        acc_id = _next_account_id(user_rec)

        settings = {
            "tp_mode": 1,
            "sl_pct": 2.5,
            "tp_pcts": [1.5, 2.5, 3.5],
            "fixed_notional_usd": 0.0,
        }

        new_acc = {
            "id": acc_id,
            "name": temp.get("name") or acc_id,
            "api_key": temp.get("api_key") or "",
            "api_secret": temp.get("api_secret") or "",
            "settings": settings,
        }
        user_rec.setdefault("accounts", []).append(new_acc)
        _save_data(data)

        # Clear pending for this chat
        pending_all.pop(chat_key, None)
        _save_state(state)

        tail = (new_acc["api_key"][-4:] if new_acc.get("api_key") else "****")
        return [
            f"API connection created: {new_acc['id']} ({new_acc['name']})",
            f"Key ending with ****{tail}",
            "You can adjust settings with /set_sl, /set_tp and /set_fixed.",
        ]

    return None


def _handle_text_message(msg: Dict[str, Any], state: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return []
    chat_key = str(chat_id)

    text = (msg.get("text") or "").strip()
    if not text:
        return []

    # If there is a pending /add_api flow, try to handle it first
    pending = state.get("pending", {}).get(chat_key)
    if isinstance(pending, dict) and pending.get("action") == "add_api":
        res = _handle_add_api_step(msg, state, data)
        if res is not None:
            return res

    # Commands
    if text.startswith("/start"):
        return _handle_command_start(chat_id)

    if text.startswith("/add_api"):
        state.setdefault("pending", {})[chat_key] = {"action": "add_api", "step": "name", "temp": {}}
        _save_state(state)
        return [
            "Let's create a new API connection.",
            "First, send a name for this account (e.g. 'Binance Main').",
        ]

    if text.startswith("/list_apis"):
        user_key = _user_key_from_message(msg)
        user_rec = _get_user_record(data, user_key)
        return [_format_accounts_for_user(user_rec)]

    if text.startswith("/delete_api"):
        parts = text.split()
        if len(parts) < 2:
            return ["Usage: /delete_api <id> (see /list_apis)"]
        acc_id = parts[1].strip()
        user_key = _user_key_from_message(msg)
        user_rec = _get_user_record(data, user_key)
        accounts = user_rec.get("accounts", [])
        new_accounts = [a for a in accounts if str(a.get("id")) != acc_id]
        if len(new_accounts) == len(accounts):
            return [f"No API found with id {acc_id}."]
        user_rec["accounts"] = new_accounts
        _save_data(data)
        return [f"API {acc_id} deleted."]

    if text.startswith("/set_sl"):
        parts = text.split()
        if len(parts) < 3:
            return ["Usage: /set_sl <id> <percent>"]
        acc_id = parts[1].strip()
        try:
            sl_pct = float(parts[2])
        except (TypeError, ValueError):
            return ["Invalid percent value."]
        user_key = _user_key_from_message(msg)
        user_rec = _get_user_record(data, user_key)
        updated = False
        for acc in user_rec.get("accounts", []):
            if str(acc.get("id")) == acc_id:
                settings = acc.setdefault("settings", {})
                settings["sl_pct"] = sl_pct
                updated = True
                break
        if not updated:
            return [f"No API found with id {acc_id}."]
        _save_data(data)
        return [f"Updated SL% for {acc_id} to {sl_pct}."]

    if text.startswith("/set_tp"):
        parts = text.split()
        if len(parts) < 3:
            return ["Usage: /set_tp <id> p1 p2 [p3]"]
        acc_id = parts[1].strip()
        raw_levels = parts[2:]
        levels: List[float] = []
        for val in raw_levels:
            try:
                v = float(val)
                if v > 0:
                    levels.append(v)
            except (TypeError, ValueError):
                continue
        if not levels:
            return ["No valid TP levels provided."]
        user_key = _user_key_from_message(msg)
        user_rec = _get_user_record(data, user_key)
        updated = False
        for acc in user_rec.get("accounts", []):
            if str(acc.get("id")) == acc_id:
                settings = acc.setdefault("settings", {})
                settings["tp_pcts"] = levels
                updated = True
                break
        if not updated:
            return [f"No API found with id {acc_id}."]
        _save_data(data)
        return [f"Updated TP% levels for {acc_id} to: {', '.join(str(x) for x in levels)}"]

    if text.startswith("/set_fixed"):
        parts = text.split()
        if len(parts) < 3:
            return ["Usage: /set_fixed <id> <usd>"]
        acc_id = parts[1].strip()
        try:
            usd = float(parts[2])
        except (TypeError, ValueError):
            return ["Invalid USD value."]
        if usd < 0:
            usd = 0.0
        user_key = _user_key_from_message(msg)
        user_rec = _get_user_record(data, user_key)
        updated = False
        for acc in user_rec.get("accounts", []):
            if str(acc.get("id")) == acc_id:
                settings = acc.setdefault("settings", {})
                settings["fixed_notional_usd"] = usd
                updated = True
                break
        if not updated:
            return [f"No API found with id {acc_id}."]
        _save_data(data)
        return [f"Updated fixed notional for {acc_id} to {usd} USDT."]

    # Unknown text, show a short hint
    return ["Unknown command. Use /start to see available commands."]


def run_accounts_bot() -> None:
    token = _load_accounts_bot_config()
    if not token:
        print("[ACCOUNTS BOT] accounts_bot.token is not configured in config/trading.yaml")
        return

    print("[ACCOUNTS BOT] Starting accounts bot polling loop...")

    state = _load_state()
    data = _load_data()
    offset = int(state.get("offset", 0))

    while True:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 10, "offset": offset + 1}

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            time.sleep(5)
            continue

        if not isinstance(payload, dict) or not payload.get("ok"):
            time.sleep(5)
            continue

        results = payload.get("result", []) or []
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

                replies = _handle_text_message(message, state, data)
                if not replies:
                    continue

                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if chat_id is None:
                    continue

                for part in replies:
                    try:
                        send_telegram(part, token, chat_id)
                    except Exception:
                        continue
            except Exception:
                continue

        state["offset"] = offset
        _save_state(state)

        time.sleep(2)


if __name__ == "__main__":
    run_accounts_bot()
