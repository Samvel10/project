import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import threading
import json
import ast
import time

try:
    from ruamel.yaml import YAML  # type: ignore
except Exception:
    YAML = None
import yaml as _pyyaml

from monitoring.telegram import send_telegram


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
_YAML = YAML(typ="safe") if YAML is not None else None

_LOG_BOT_CFG_LOADED = False
_LOG_BOT_TOKEN = None
_LOG_BOT_CHAT_ID = None
_LOG_BOT_FALLBACK_TOKEN = None
_START_SUBSCRIBERS_PATH = ROOT_DIR / "data" / "telegram_start_subscribers.json"
_CHAT_SEND_STATE_LOCK = threading.Lock()
_CHAT_LAST_SENT_TS: dict[int, float] = {}
_CHAT_MUTE_UNTIL_TS: dict[int, float] = {}
_CHAT_MIN_INTERVAL_SEC = 2.0


def setup_logger(name="quant_system", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    log_path = LOG_DIR / "system.log"

    # Always ensure the rotating file handler is attached. In some environments
    # (reloads / multiple imports) handlers may already exist and we still
    # want to guarantee that file logging is enabled.
    has_file = False
    try:
        for h in list(logger.handlers or []):
            try:
                if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == str(log_path):
                    has_file = True
                    break
            except Exception:
                continue
    except Exception:
        has_file = False

    if not has_file:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler is useful for interactive runs; keep it best-effort.
    has_console = False
    try:
        for h in list(logger.handlers or []):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
                has_console = True
                break
    except Exception:
        has_console = False
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


_LOGGER = setup_logger()
try:
    _LOGGER.info(f"[LOG] File logging enabled: {(LOG_DIR / 'system.log').resolve()}")
except Exception:
    pass


def _ensure_log_bot_config():
    """Lazy-load Telegram log bot config from trading.yaml.

    Expects:
      log_bot.token  – bot token used only for logs
      log_bot.chat_id – optional; if missing, falls back to telegram_chat_id
    """

    global _LOG_BOT_CFG_LOADED, _LOG_BOT_TOKEN, _LOG_BOT_CHAT_ID, _LOG_BOT_FALLBACK_TOKEN
    if _LOG_BOT_CFG_LOADED:
        return

    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            if _YAML is not None:
                cfg = _YAML.load(f) or {}
            else:
                cfg = _pyyaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    log_cfg = cfg.get("log_bot") or {}
    token = log_cfg.get("token") or ""
    chat_id = log_cfg.get("chat_id")
    fallback_token = ((cfg.get("control_bot") or {}).get("token") or "").strip()
    if chat_id is None:
        chat_id = cfg.get("telegram_chat_id")

    try:
        chat_id_int = int(chat_id) if chat_id is not None else None
    except (TypeError, ValueError):
        chat_id_int = None

    _LOG_BOT_TOKEN = str(token) if token else None
    _LOG_BOT_FALLBACK_TOKEN = fallback_token if fallback_token and fallback_token != _LOG_BOT_TOKEN else None
    _LOG_BOT_CHAT_ID = chat_id_int
    _LOG_BOT_CFG_LOADED = True


def _send_log_to_telegram(text: str) -> None:
    """Forward a single log line to the Telegram log bot (non-blocking).

    Network I/O is executed in a background thread so that trading loop
    is not blocked by slow Telegram responses.
    """

    _ensure_log_bot_config()
    if not _LOG_BOT_TOKEN:
        return

    try:
        def _pretty_log_message(raw_text: str) -> str:
            def _extract(body_text: str, key: str) -> str:
                try:
                    m = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", body_text)
                    if m:
                        return str(m.group(1))
                except Exception:
                    pass
                return "-"

            def _pretty_details(raw_details: str) -> str:
                d = (raw_details or "").strip()
                if not d:
                    return "-"
                try:
                    obj = ast.literal_eval(d)
                    return json.dumps(obj, ensure_ascii=False, indent=2)
                except Exception:
                    return d

            try:
                msg = str(raw_text).strip()
            except Exception:
                msg = ""
            if not msg:
                return ""
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            tag = "SYSTEM"
            body = msg
            try:
                m = re.match(r"^\[([^\]]+)\]\s*(.*)$", msg)
                if m:
                    tag = str(m.group(1) or "SYSTEM").strip().upper()
                    body = str(m.group(2) or "").strip() or msg
            except Exception:
                pass

            # AI-TM rich readable formatting
            if tag == "AI-TM":
                details_raw = ""
                main_part = body
                try:
                    if " details=" in body:
                        main_part, details_raw = body.split(" details=", 1)
                except Exception:
                    main_part = body
                    details_raw = ""

                acc_v = _extract(main_part, "account")
                sym_v = _extract(main_part, "symbol")
                model_v = _extract(main_part, "model")
                dec_v = _extract(main_part, "decision")
                conf_v = _extract(main_part, "conf")
                over_v = _extract(main_part, "override")
                state_v = _extract(main_part, "state")
                unreal_v = _extract(main_part, "unreal")
                st_v = _extract(main_part, "status")
                rs_v = _extract(main_part, "reason")

                if main_part.startswith("[ORDER-CHECK][RESULT]"):
                    return (
                        "LOG UPDATE\n"
                        f"Time: {ts}\n"
                        "Source (Աղբյուր): AI Trade Manager (AI գործարքի կառավարիչ)\n"
                        "Type (Տեսակ): Order Reconcile Result (Օրդերների համադրության արդյունք)\n\n"
                        f"- Account (Հաշիվ): {acc_v}\n"
                        f"- Symbol (Սիմվոլ): {sym_v}\n"
                        f"- Status (Վիճակ): {st_v}\n"
                        f"- Reason (Պատճառ): {rs_v}\n\n"
                        "Details (Մանրամասներ):\n"
                        f"{_pretty_details(details_raw)}"
                    )

                if (
                    " account=" in f" {main_part}"
                    and " symbol=" in f" {main_part}"
                    and acc_v != "-"
                    and sym_v != "-"
                ):
                    return (
                        "LOG UPDATE\n"
                        f"Time: {ts}\n"
                        "Source (Աղբյուր): AI Trade Manager (AI գործարքի կառավարիչ)\n"
                        "Type (Տեսակ): Position Decision (Պոզիցիայի որոշում)\n\n"
                        f"- Account (Հաշիվ): {acc_v}\n"
                        f"- Symbol (Սիմվոլ): {sym_v}\n"
                        f"- Model (Մոդել): {model_v}\n"
                        f"- Decision (Որոշում): {dec_v} (conf/վստահություն={conf_v})\n"
                        f"- Override (Գերակայող պատճառ): {over_v}\n"
                        f"- State (Շուկայի վիճակ): {state_v}\n"
                        f"- Unrealized PnL (Չփակված PnL): {unreal_v}\n"
                        f"- Result (Արդյունք): {st_v} / {rs_v}\n\n"
                        "Details (Մանրամասներ):\n"
                        f"{_pretty_details(details_raw)}"
                    )

            if tag == "EXCHANGE ORDER":
                kind = "Warning" if body.startswith("[WARN]") else "Exchange Event"
                return (
                    "LOG UPDATE\n"
                    f"Time: {ts}\n"
                    "Source (Աղբյուր): Exchange Orders (Բորսայի օրդերներ)\n"
                    f"Type (Տեսակ): {kind}\n\n"
                    f"{body}"
                )

            return (
                "LOG UPDATE\n"
                f"Time: {ts}\n"
                f"Source (Աղբյուր): {tag}\n\n"
                f"{body}"
            )

        chat_ids = []
        try:
            if _START_SUBSCRIBERS_PATH.exists():
                raw = _START_SUBSCRIBERS_PATH.read_text(encoding="utf-8")
                payload = json.loads(raw) if raw.strip() else {}
                ids = payload.get("chat_ids") if isinstance(payload, dict) else []
                if isinstance(ids, list):
                    seen = set()
                    for cid in ids:
                        try:
                            c = int(cid)
                        except Exception:
                            continue
                        if c in seen:
                            continue
                        seen.add(c)
                        chat_ids.append(c)
        except Exception:
            chat_ids = []

        # Always include explicit configured owner/admin chat id.
        if _LOG_BOT_CHAT_ID is not None:
            try:
                owner_id = int(_LOG_BOT_CHAT_ID)
                if owner_id not in chat_ids:
                    chat_ids.append(owner_id)
            except Exception:
                pass

        if not chat_ids:
            return

        def _extract_retry_after_seconds(err_text: str) -> int:
            try:
                m = re.search(r"retry after\\s+(\\d+)", err_text, flags=re.IGNORECASE)
                if m:
                    return max(0, int(m.group(1)))
            except Exception:
                pass
            try:
                m = re.search(r'"retry_after"\\s*:\\s*(\\d+)', err_text)
                if m:
                    return max(0, int(m.group(1)))
            except Exception:
                pass
            return 0

        def _send_to_many(message: str, token: str, ids: list[int]) -> None:
            for cid in ids:
                now = time.time()
                with _CHAT_SEND_STATE_LOCK:
                    mute_until = float(_CHAT_MUTE_UNTIL_TS.get(int(cid), 0.0))
                    if mute_until > now:
                        continue
                    last_ts = float(_CHAT_LAST_SENT_TS.get(int(cid), 0.0))
                    if (now - last_ts) < float(_CHAT_MIN_INTERVAL_SEC):
                        continue
                try:
                    send_telegram(message, token, cid)
                    with _CHAT_SEND_STATE_LOCK:
                        _CHAT_LAST_SENT_TS[int(cid)] = time.time()
                except Exception as e:
                    err_txt = str(e)
                    retry_after = _extract_retry_after_seconds(err_txt)
                    if retry_after > 0:
                        # Owner chat: fall back to control bot token so logs continue
                        # even while log-bot token is under Telegram flood cooldown.
                        try:
                            if (
                                _LOG_BOT_FALLBACK_TOKEN
                                and _LOG_BOT_CHAT_ID is not None
                                and int(cid) == int(_LOG_BOT_CHAT_ID)
                            ):
                                send_telegram(message, _LOG_BOT_FALLBACK_TOKEN, int(cid))
                                with _CHAT_SEND_STATE_LOCK:
                                    _CHAT_LAST_SENT_TS[int(cid)] = time.time()
                                continue
                        except Exception:
                            pass
                        with _CHAT_SEND_STATE_LOCK:
                            _CHAT_MUTE_UNTIL_TS[int(cid)] = time.time() + float(retry_after)
                    continue

        pretty = _pretty_log_message(text)
        if not pretty:
            return
        threading.Thread(
            target=_send_to_many,
            args=(pretty, _LOG_BOT_TOKEN, chat_ids),
            daemon=True,
        ).start()
    except Exception:
        # Logging failures to Telegram should never break main code
        pass


def _should_forward_to_telegram(text: str) -> bool:
    try:
        t = str(text)
    except Exception:
        return False
    return bool(t.strip())


def log(msg):
    text = str(msg)
    try:
        _LOGGER.info(text)
    except Exception:
        pass
    try:
        print(text)
    except Exception:
        pass
    if _should_forward_to_telegram(text):
        _send_log_to_telegram(text)