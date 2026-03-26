import json
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
from ruamel.yaml import YAML
from instance_security import (
    get_instances,
    set_instance_status,
    register_startup as license_register_startup,
)
from monitoring.instance_factory import create_instance_project, build_update_package

import config.proxies as proxies_cfg

from monitoring.telegram import send_telegram, send_telegram_document
from monitoring.main_process_manager import (
    start_main,
    stop_main,
    status_main,
    restart_main,
)
from monitoring.news_guard import get_active_news_events
from monitoring.signal_details_log import get_signals_since_minutes
from data.symbol_blocklist import (
    add_blocked_symbols,
    list_blocked_symbols,
    parse_symbols_argument,
    remove_blocked_symbols,
)


_STATE_PATH = ROOT_DIR / "data" / "main_control_bot_state.json"
_LOCK_PATH = ROOT_DIR / "data" / "main_control_bot.lock"
_SLTP_SYMBOL_STATS_PATH = ROOT_DIR / "data" / "sl_tp_nn_symbol_stats.json"
_YAML = YAML(typ="safe")


def _ensure_state_file() -> None:
    if not _STATE_PATH.parent.exists():
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        _STATE_PATH.write_text(json.dumps({"offset": 0}, ensure_ascii=False), encoding="utf-8")


def _load_state() -> Dict[str, Any]:
    _ensure_state_file()
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {"offset": 0}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"offset": 0}
        if "offset" not in data:
            data["offset"] = 0
        return data
    except Exception:
        return {"offset": 0}


def _save_state(state: Dict[str, Any]) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _acquire_single_instance_lock() -> bool:
    try:
        if _LOCK_PATH.exists():
            try:
                raw = _LOCK_PATH.read_text(encoding="utf-8").strip()
                if raw:
                    pid = int(raw)
                else:
                    pid = None
            except Exception:
                pid = None

            if pid is not None:
                try:
                    os.kill(pid, 0)
                except OSError:
                    try:
                        _LOCK_PATH.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    print(
                        f"[MAIN CONTROL BOT] Another instance appears to be running with pid={pid}. Exiting."
                    )
                    return False

        try:
            if not _LOCK_PATH.parent.exists():
                _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            _LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass
    except Exception:
        return True

    return True


def _load_control_bot_config() -> Dict[str, Any]:
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}
    except Exception:
        cfg = {}

    control_cfg = cfg.get("control_bot") or {}
    token = control_cfg.get("token") or ""

    # Optionally restrict commands to a single chat_id (same as main signals bot)
    chat_id = cfg.get("telegram_chat_id")
    try:
        allowed_chat_id: Optional[int] = int(chat_id) if chat_id is not None else None
    except (TypeError, ValueError):
        allowed_chat_id = None

    admin_ids: List[int] = []
    raw_admin = None
    try:
        raw_admin = (
            control_cfg.get("admin_user_ids")
            or cfg.get("telegram_admin_user_ids")
            or cfg.get("admin_user_ids")
        )
    except Exception:
        raw_admin = None
    if raw_admin is not None:
        if isinstance(raw_admin, list):
            for v in raw_admin:
                try:
                    admin_ids.append(int(v))
                except Exception:
                    continue
        else:
            try:
                admin_ids.append(int(raw_admin))
            except Exception:
                pass

    admin_ids = sorted(list({int(x) for x in admin_ids if x is not None}))

    return {
        "token": str(token),
        "allowed_chat_id": allowed_chat_id,
        "admin_user_ids": admin_ids,
    }


def _get_news_guard_enabled() -> bool:
    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}
    except Exception:
        return False

    ng = cfg.get("news_guard") or {}
    return bool(ng.get("enabled", False))


def _handle_command_signals(text: str) -> List[str]:
    parts = text.split()
    minutes = 30.0
    if len(parts) > 1:
        try:
            minutes = float(parts[1])
        except (TypeError, ValueError):
            minutes = 30.0

    if minutes <= 0:
        minutes = 30.0
    if minutes > 1440.0:
        minutes = 1440.0

    signals = get_signals_since_minutes(minutes)
    if not signals:
        return [
            f"Վերջին {int(minutes)} րոպեում սիգնալներ չեն գտնվել (signal_details_log.csv-ում).",
        ]

    replies: List[str] = []

    for rec in signals:
        symbol = rec.get("symbol") or "?"
        direction = rec.get("direction") or "?"
        ts_am = rec.get("timestamp_am") or "?"

        entry = rec.get("entry") or "?"
        sl = rec.get("sl") or "?"
        tp1 = rec.get("tp1") or "?"
        tp2 = rec.get("tp2") or "?"
        tp3 = rec.get("tp3") or "?"

        conf = rec.get("confidence") or "?"
        vol_1h = rec.get("vol_1h_pct") or "?"

        last_5m_trades = rec.get("last_5m_trades") or "?"
        avg_5m_trades = rec.get("avg_5m_trades") or "?"
        act_status = rec.get("activity_status") or "?"

        follow_ts_am = rec.get("followup_ts_am") or None
        follow_vol = rec.get("followup_vol_1h_pct") or None

        last_act_ts_am = rec.get("last_activity_check_ts_am") or None
        last_act_trades = rec.get("last_activity_trades") or None
        last_act_avg = rec.get("last_activity_avg_trades") or None
        last_act_status = rec.get("last_activity_status") or None

        lines: List[str] = []
        lines.append(f"Սիգնալ՝ {symbol} {direction}")
        lines.append(f"Ժամ (AM): {ts_am}")
        lines.append("")
        lines.append(f"Entry: {entry}")
        lines.append(f"SL: {sl}")
        lines.append(f"TP1: {tp1} | TP2: {tp2} | TP3: {tp3}")
        lines.append(f"Confidence: {conf}")
        lines.append("")
        lines.append(f"1h տատանում (vol_1h): {vol_1h} %")
        lines.append(
            f"5m activity (սկզբնական): last={last_5m_trades}, avg={avg_5m_trades}, status={act_status}"
        )

        if follow_ts_am or follow_vol:
            lines.append("")
            lines.append("Follow-up 1h (մոտ 1 ժամ հետո):")
            lines.append(f"  Ժամ (AM): {follow_ts_am or '?'}")
            lines.append(f"  vol_1h: {follow_vol or '?'} %")

        if last_act_ts_am or last_act_trades or last_act_status:
            lines.append("")
            lines.append("Վերջին activity ստուգում:")
            lines.append(f"  Ժամ (AM): {last_act_ts_am or '?'}")
            lines.append(
                f"  last={last_act_trades or '?'} | avg={last_act_avg or '?'} | status={last_act_status or '?'}"
            )

        replies.append("\n".join(lines))

    return replies


def _handle_command_sl_tp_stats(text: str) -> List[str]:
    """Show summary statistics for SL/TP NN per symbol/side.

    Reads data/sl_tp_nn_symbol_stats.json produced by ml/train_sl_tp_nn.py and
    prints:
      - total number of symbol/side keys
      - top-N strong pairs (highest win_rate, with enough trades)
      - top-N weak pairs (lowest win_rate)

    Syntax:
      /sl_tp_stats [min_trades] [top_n]
    """

    parts = text.split()
    min_trades = 20.0
    top_n = 10

    if len(parts) > 1:
        try:
            min_trades = float(parts[1])
        except (TypeError, ValueError):
            min_trades = 20.0
    if len(parts) > 2:
        try:
            top_n = int(parts[2])
        except (TypeError, ValueError):
            top_n = 10

    try:
        if not _SLTP_SYMBOL_STATS_PATH.exists():
            return [
                "SL/TP NN symbol stats ֆայլը դեռ գոյություն չունի (sl_tp_nn_symbol_stats.json). ",
            ]

        raw = _SLTP_SYMBOL_STATS_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return ["sl_tp_nn_symbol_stats.json դատարկ է (ոչ մի սիգնալ/գործարք չկա)"]

        data = json.loads(raw)
        if not isinstance(data, dict):
            return ["sl_tp_nn_symbol_stats.json schema-ն սպասված dict չէ"]
    except Exception as e:
        return [f"SL/TP NN stats կարդալ չհաջողվեց․ {e}"]

    items = []
    for key, rec in data.items():
        if not isinstance(rec, dict):
            continue
        try:
            n = float(rec.get("n", 0.0))
            win_rate = float(rec.get("win_rate", 0.0))
            mean_pnl = float(rec.get("mean_pnl", 0.0))
        except (TypeError, ValueError):
            continue
        if n < min_trades:
            continue
        items.append((key, n, win_rate, mean_pnl))

    if not items:
        return [
            f"SL/TP NN stats՝ min_trades={int(min_trades)} ֆիլտրով զույգեր չկան։ Փորձիր նվազեցնել շեմը, օրինակ՝ /sl_tp_stats 5",
        ]

    total_keys = len(data)
    items_sorted = sorted(items, key=lambda t: t[2], reverse=True)

    top_strong = items_sorted[:top_n]
    top_weak = list(reversed(items_sorted))[:top_n]

    lines: List[str] = []
    lines.append(
        f"SL/TP NN symbol stats (ընդհանուր {total_keys} SYMBOL:SIDE key, min_trades={int(min_trades)}, top_n={top_n})"
    )
    lines.append("")

    if top_strong:
        lines.append("Լավագույն զույգեր (բարձր win_rate):")
        for key, n, win_r, mean_pnl in top_strong:
            try:
                sym, side = key.split(":", 1)
            except ValueError:
                sym, side = key, "?"
            lines.append(
                f"  {sym} {side}: n={int(n)}, win_rate={win_r*100:.1f}%, mean_pnl={mean_pnl:.2f}%"
            )

    if top_weak:
        lines.append("")
        lines.append("Թույլ զույգեր (ցածր win_rate):")
        for key, n, win_r, mean_pnl in top_weak:
            try:
                sym, side = key.split(":", 1)
            except ValueError:
                sym, side = key, "?"
            lines.append(
                f"  {sym} {side}: n={int(n)}, win_rate={win_r*100:.1f}%, mean_pnl={mean_pnl:.2f}%"
            )

    return ["\n".join(lines)]


def _handle_command_news_status() -> List[str]:
    enabled = _get_news_guard_enabled()

    try:
        active_events = get_active_news_events()
    except Exception as e:
        lines = [
            "NEWS GUARD վիճակը ստուգել չհաջողվեց (economic calendar սխալ)",
            f"Մանրամասներ․ {e}",
        ]
        return ["\n".join(lines)]


def _handle_command_block_coin(text: str) -> List[str]:
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return ["Օգտագործում․ /block_coin <symbol[,symbol2,...]>"]

    symbols = parse_symbols_argument(parts[1])
    if not symbols:
        return ["Չի գտնվել ճիշտ symbol։ Օրինակ՝ /block_coin PIPPINUSDT,DYDXUSDT"]

    added, already = add_blocked_symbols(symbols)
    blocked_now = list_blocked_symbols()

    lines: List[str] = []
    lines.append("Coin blocklist-ը թարմացվեց։")
    if added:
        lines.append(f"Ավելացվեց․ {', '.join(added)}")
    if already:
        lines.append(f"Արդեն block-ի մեջ էր․ {', '.join(already)}")
    lines.append(f"Ընդհանուր block-ում․ {len(blocked_now)}")
    return ["\n".join(lines)]


def _handle_command_unblock_coin(text: str) -> List[str]:
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return ["Օգտագործում․ /unblock_coin <symbol[,symbol2,...]>"]

    symbols = parse_symbols_argument(parts[1])
    if not symbols:
        return ["Չի գտնվել ճիշտ symbol։ Օրինակ՝ /unblock_coin PIPPINUSDT,DYDXUSDT"]

    removed, missing = remove_blocked_symbols(symbols)
    blocked_now = list_blocked_symbols()

    lines: List[str] = []
    lines.append("Coin blocklist-ը թարմացվեց։")
    if removed:
        lines.append(f"Հեռացվեց․ {', '.join(removed)}")
    if missing:
        lines.append(f"Blocklist-ում չկար․ {', '.join(missing)}")
    lines.append(f"Ընդհանուր block-ում․ {len(blocked_now)}")
    return ["\n".join(lines)]


def _handle_command_blocked_coins() -> List[str]:
    symbols = list_blocked_symbols()
    if not symbols:
        return ["Արգելափակված coin-ների ցուցակը դատարկ է։"]

    lines: List[str] = []
    lines.append(f"Արգելափակված coin-ներ ({len(symbols)})․")
    lines.append(", ".join(symbols))
    return ["\n".join(lines)]


def _handle_command_instances() -> List[str]:
    instances = get_instances()
    if not instances:
        return [
            "Instance registry-ն դատարկ է։ Քո կողմից դեռ instance չի ստեղծվել կամ չի աշխատեցվել։",
        ]

    lines: List[str] = []
    lines.append("Գրանցված instance-ների ցուցակն այս պահին․")
    lines.append("")

    for inst_id, inst in instances.items():
        status = inst.get("status", "unknown")
        last_host = inst.get("last_host", "-")
        last_seen = inst.get("last_seen", "-")
        run_count = inst.get("run_count", 0)
        label = inst.get("label") or ""

        if label:
            header = f"ID: {inst_id}  ({label})"
        else:
            header = f"ID: {inst_id}"

        lines.append(header)
        lines.append(f"  Status: {status}")
        lines.append(f"  Last host: {last_host}")
        lines.append(f"  Last seen: {last_seen}")
        lines.append(f"  Run count: {run_count}")
        lines.append("")

    return ["\n".join(lines)]


def _handle_command_instance_status(text: str, status: str) -> List[str]:
    parts = text.split()
    if len(parts) < 2:
        return [
            "Խնդրում եմ օգտագործել՝ /block_instance <id> կամ /unblock_instance <id> կամ /delete_instance <id>",
        ]

    instance_id = parts[1].strip()
    if not instance_id:
        return [
            "Instance ID չի նշված։ Օրինակ՝ /block_instance 1234-ABCD",
        ]

    ok = set_instance_status(instance_id, status)
    if not ok:
        return [
            f"Instance '{instance_id}' գտնվե՛ց չի registry-ում կամ չի ստացվել թարմացնել վիճակը։",
        ]

    return [
        f"Instance '{instance_id}' վիճակը թարմացվել է՝ {status}։",
    ]


def _handle_command_instance_info(text: str) -> List[str]:
    parts = text.split()
    if len(parts) < 2:
        return [
            "Օգտագործիր՝ /instance_info <id>",
        ]

    instance_id = parts[1].strip()
    if not instance_id:
        return [
            "Instance ID չի նշված։ Օրինակ՝ /instance_info 1234-ABCD",
        ]

    instances = get_instances()
    inst = instances.get(instance_id)
    if not inst:
        return [
            f"Instance '{instance_id}' չի գտնվել registry-ում։",
        ]

    status = inst.get("status", "unknown")
    created_at = inst.get("created_at", "-")
    last_seen = inst.get("last_seen", "-")
    last_host = inst.get("last_host", "-")
    last_client = inst.get("last_client", "-")
    run_count = inst.get("run_count", 0)
    label = inst.get("label", "") or ""
    blocked_reason = inst.get("blocked_reason", "") or ""
    meta = inst.get("meta") or {}
    hosts = inst.get("hosts") or {}

    lines: List[str] = []
    lines.append(f"Instance ID: {instance_id}")
    if label:
        lines.append(f"Label: {label}")
    lines.append(f"Status: {status}")
    if blocked_reason:
        lines.append(f"Blocked reason: {blocked_reason}")
    lines.append(f"Created at: {created_at}")
    lines.append(f"Last seen: {last_seen}")
    lines.append(f"Last host: {last_host}")
    lines.append(f"Last client: {last_client}")
    lines.append(f"Run count: {run_count}")

    if meta and isinstance(meta, dict):
        lines.append("")
        lines.append("Meta տվյալներ․")
        for k, v in meta.items():
            lines.append(f"- {k}: {v}")

    if hosts and isinstance(hosts, dict):
        lines.append("")
        lines.append("Հյուրընկալներ (hosts)․")
        for h, info in hosts.items():
            if not isinstance(info, dict):
                info = {}
            first = info.get("first_seen", "-")
            h_last = info.get("last_seen", "-")
            lines.append(f"- {h}: first_seen={first}, last_seen={h_last}")

    return ["\n".join(lines)]


def _handle_command_proxies_status() -> List[str]:
    try:
        # Always reload from current config/proxies.py so /proxies reflects
        # latest edits even if control bot process wasn't restarted.
        importlib.reload(proxies_cfg)
        proxies_cfg.validate_proxies(force=True)
        working = proxies_cfg.get_working_proxies()
        blocked = proxies_cfg.get_blocked_proxies()
    except Exception as e:
        lines = [
            "PROXY վիճակը ստուգել չհաջողվեց (config.proxies սխալ)",
            f"Մանրամասներ․ {e}",
        ]
        return ["\n".join(lines)]

    total = len(working) + len(blocked)

    lines: List[str] = []
    lines.append("PROXY վիճակն այս պահին․")
    lines.append("")
    lines.append(f"Ընդհանուր PROXIES: {total}")
    lines.append(f"✅ Working: {len(working)}")
    lines.append(f"❌ Blocked: {len(blocked)}")

    if working:
        lines.append("")
        lines.append("✅ Working proxies․")
        for p in working:
            lines.append(f"- {p}")

    if blocked:
        lines.append("")
        lines.append("❌ Blocked proxies․")
        for p in blocked:
            lines.append(f"- {p}")

    return ["\n".join(lines)]


def _handle_command_commit(text: str) -> List[str]:
    """Handle /commit and /commit_all admin commands.

    Syntax:
      /commit_all                       – build full update for all active instances
      /commit_all <path>                – build update only for given file/dir for all active instances
      /commit <id>                      – build full update for single instance
      /commit <id> <path>              – build update only for given file/dir for single instance

    The resulting ZIP files are written to updates/{instance_id}.zip and can
    be pulled by clients via the /update command in their own Telegram bots.
    """

    parts = text.split()
    if not parts:
        return ["Օգտագործում․ /commit_all [path] կամ /commit <id> [path]"]

    cmd = parts[0].lower()
    rest = parts[1:]

    instances = get_instances()
    if not instances:
        return ["Instance registry-ն դատարկ է, commit անելու բան չկա։"]

    lines: List[str] = []

    def _do_build(inst_id: str, path_spec: Optional[str]) -> None:
        try:
            res = build_update_package(inst_id, path_spec=path_spec)
        except Exception as e:
            lines.append(f"[commit] {inst_id}: սխալ build_update_package-ի ընթացքում – {e}")
            return

        if not res.get("ok"):
            err = res.get("pyarmor_error") or res.get("zip_error") or "անհայտ սխալ"
            lines.append(f"[commit] {inst_id}: ձախողվեց ({err})")
        else:
            archive = res.get("archive_path") or f"updates/{inst_id}.zip"
            spec = res.get("path_spec") or "ամբողջ project"
            lines.append(f"[commit] {inst_id}: հաջողությամբ ստեղծվեց update ({spec}) → {archive}")

    if cmd == "/commit_all":
        path_spec = rest[0].strip() if rest else None
        for inst_id, inst in instances.items():
            status = str(inst.get("status") or "active").lower()
            if status != "active":
                continue
            _do_build(inst_id, path_spec)

        if not lines:
            lines.append("Չկան active instance-ներ commit անելու համար։")
        return ["\n".join(lines)]

    if cmd == "/commit":
        if not rest:
            return ["Օրինակ՝ /commit <instance_id> [path]"]

        instance_id = rest[0].strip()
        if not instance_id:
            return ["Instance ID չի նշված։ Օրինակ՝ /commit 1234-ABCD"]

        if instance_id not in instances:
            return [f"Instance {instance_id} չի գտնվել registry-ում։"]

        path_spec = rest[1].strip() if len(rest) > 1 else None
        _do_build(instance_id, path_spec)
        return ["\n".join(lines)]

    return ["Օգտագործում․ /commit_all [path] կամ /commit <id> [path]"]


def _handle_command_create_instance(text: str, chat_id: Optional[int]) -> List[str]:
    rest = ""
    parts = text.split(maxsplit=1)
    if len(parts) > 1:
        rest = parts[1].strip()

    label: str = ""
    description: str = ""

    if rest:
        if "|" in rest:
            lbl, desc = rest.split("|", 1)
            label = lbl.strip()
            description = desc.strip()
        else:
            label = rest

    if not label:
        label = f"client_{int(time.time())}"

    requested_by = int(chat_id) if chat_id is not None else None

    result = create_instance_project(
        label=label,
        description=description or None,
        requested_by_chat_id=requested_by,
    )

    instance_id = result.get("instance_id", "?")
    out_dir = result.get("output_dir") or "-"
    archive_path = result.get("archive_path") or "-"
    archive_name = "-"
    archive_ok = False
    try:
        if isinstance(archive_path, str) and archive_path.strip() and archive_path != "-":
            archive_name = Path(archive_path).name
            archive_ok = Path(archive_path).exists()
    except Exception:
        archive_name = "-"
        archive_ok = False

    pyarmor_ok = bool(result.get("pyarmor_ok"))
    pyarmor_error = result.get("pyarmor_error") or ""
    requirements_ok = bool(result.get("requirements_ok"))
    pip_ok = bool(result.get("pip_ok"))
    pip_error = result.get("pip_error") or ""

    lines: List[str] = []
    lines.append("Նոր instance ստեղծելու արդյունքը․")
    lines.append("")
    lines.append(f"Instance ID: {instance_id}")
    lines.append(f"Label: {label}")
    lines.append(f"Output dir: {out_dir}")
    if archive_ok:
        lines.append(f"Archive (zip): {archive_name} (will be sent as a file)")
    else:
        lines.append(f"Archive (zip): {archive_name}")

    lines.append("")
    lines.append(f"PyArmor ok: {'YES' if pyarmor_ok else 'NO'}")
    lines.append(f"requirements.txt ok: {'YES' if requirements_ok else 'NO'}")
    lines.append(f"pip install (libs/) ok: {'YES' if pip_ok else 'NO'}")

    if (not pyarmor_ok) and pyarmor_error:
        lines.append("")
        lines.append("PyArmor սխալ․")
        lines.append(str(pyarmor_error)[:800])

    if pip_error and not pip_ok:
        lines.append("")
        lines.append("pip սխալ․")
        lines.append(pip_error[:800])

    replies: List[str] = ["\n".join(lines)]
    if archive_ok and chat_id is not None:
        try:
            payload = {
                "path": str(archive_path),
                "caption": f"Client instance {instance_id} ({label})",
            }
            replies.append("__SEND_DOC_JSON__" + json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

        guide_lines: List[str] = []
        guide_lines.append("Client-ի օգտագործման արագ ուղեցույց․")
        guide_lines.append("")
        guide_lines.append(f"1) ZIP-ը բացիր առանձին պանակի մեջ (INSTANCE_ID={instance_id})")
        guide_lines.append("2) Պահանջներ՝ Linux VPS, Python 3.10+, unzip")
        guide_lines.append("3) Կարգավորումներ՝ edit արա config/trading.yaml և config/binance_accounts.yaml")
        guide_lines.append("   - Binance API key/secret")
        guide_lines.append("   - Telegram token/chat_id (signal-ների համար)")
        guide_lines.append("   - update.base_url (որ /update-ը աշխատի) — տալիս է admin-ը")
        guide_lines.append("4) Գործարկում՝ python3 main.py")
        guide_lines.append("   Եթե ուզում ես background՝ screen/nohup/systemd")
        guide_lines.append("5) Թարմացում՝ client Telegram bot-ում /update (կքաշի updates/<instance_id>.zip)")
        guide_lines.append("6) Եթե instance-ը paused/blocked/deleted է, trading-ը չի աշխատի — գրի admin-ին")
        replies.append("\n".join(guide_lines))
    return replies


def _handle_command_start() -> List[str]:
    lines = [
        "Բարի գալուստ main control բոտ 👋",
        "",
        "Այս բոտը կառավարում է հիմնական trading բոտի (main.py) պրոցեսը սերվերի վրա և նաև բոլոր client instance-ների լիցենզիաները:",
        "",
        "Գլխավոր հրամաններ main.py պրոցեսի համար՝",
        "  /status",
        "      Ցույց տալ, աշխատո՞ւմ է main.py-ը, թե ոչ (running / stopped)",
        "  /start",
        "      Գործարկել main.py-ը, եթե չի աշխատում։ Եթե արդեն աշխատում է, նորից չի բացի։",
        "  /stop",
        "      Կանգնեցնել main.py պրոցեսը (kill)։",
        "  /restart",
        "      Կատարել stop + start՝ main.py-ը վերագործարկելու համար։",
        "",
        "NEWS GUARD և proxy հրամաններ՝",
        "  /news_status կամ /news",
        "      Ցույց տալ NEWS GUARD-ի վիճակը և ակտիվ macro news window-ը։",
        "  /proxies_status կամ /proxies",
        "      Ցույց տալ proxy-ների վիճակը (working / blocked ցուցակ)։",
        "",
        "Coin blocklist հրամաններ՝",
        "  /block_coin <symbol[,symbol2,...]>",
        "      Նշված coin-ների համար նոր մուտքերը կփակվեն (block)։",
        "      Օրինակ՝ /block_coin PIPPINUSDT,DYDXUSDT",
        "  /unblock_coin <symbol[,symbol2,...]>",
        "      Բացում է block-ը նշված coin-ներից։",
        "      Օրինակ՝ /unblock_coin PIPPINUSDT",
        "  /blocked_coins",
        "      Ցույց է տալիս blocklist-ի ամբողջ ցուցակը։",
        "",
        "Նույն հրամանների երկար տարբերակները՝ /status_main, /start_main, /stop_main, /restart_main։",
        "",
        "Instance / լիցենզիա կառավարելու հրամաններ՝",
        "  /create_instance <label>",
        "  /create_instance <label> | <description>",
        "      Ստեղծում է նոր client project եզակի INSTANCE_ID-ով. Քայլերը․",
        "        - Գրանցում է instance-ը admin registry-ում (data/instances_registry.json)",
        "        - Պատճենում է code-ը հատուկ instance-ի պանակի մեջ",
        "        - Ներսում patch է անում INSTANCE_ID-ը Python կոդի մեջ",
        "        - Ակտիվացնում է PyArmor obfuscation-ն ամբողջ project-ի համար",
        "        - Տեղադրում է dependency-ները տեղական libs/ պանակում",
        "        - Ստեղծում է պատրաստ zip արխիվ client-ի համար։",
        "",
        "  /instances",
        "      Ցույց տալ բոլոր գրանցված instance-ների կարճ ցուցակը՝ ID, label, status, last host, run count։",
        "",
        "  /instance_info <id>",
        "      Ցույց տալ մեկ կոնկրետ instance-ի ամբողջական ինֆորմացիան՝",
        "        - Status (active / blocked / deleted)",
        "        - Blocked reason (օր․ multi_host)",
        "        - Ստեղծման ժամանակ (created_at)",
        "        - Վերջին աշխատած host-ը և client name-ը",
        "        - Քանի անգամ է run եղել (run_count)",
        "        - Meta տվյալներ (description և այլն)",
        "        - Բոլոր host-երի ցուցակը (first_seen / last_seen ըստ host-ի)։",
        "",
        "  /block_instance <id> կամ /block <id>",
        "      Փակել instance-ը (status=blocked), որպեսզի այդ INSTANCE_ID-ով client code-ը startup-ի պահին կանգնի և չաշխատի։",
        "",
        "  /unblock_instance <id> կամ /unblock <id>",
        "      Բացել instance-ը (status=active). Եթե նա նախապես ավտոմատ block էր արվել multi-host-ի պատճառով, ապա այս հրամանից հետո",
        "      multi-host օգտագործումը համարվում է թույլատրած (multi_host_allowed=True) և նույն INSTANCE_ID-ը կարող է աշխատել մի քանի host-ից (քո գիտությամբ)։",
        "",
        "  /delete_instance <id> կամ /delete <id>",
        "      Լիարժեք kill – դնում է status=deleted, client code-ը այլևս չպետք է օգտագործվի։",
        "",
        "Code update / commit հրամաններ՝",
        "  /commit_all",
        "  /commit_all <path>",
        "      Կառուցել hashed update package բոլոր ACTIVE instance-ների համար.",
        "      Առանց path-ի թարմացվում է ամբողջ project code-ը։ Եթե նշում ես path, օրինակ",
        "      'main.py' կամ 'signals/', ապա update package-ի մեջ կմտնի միայն այդ ֆայլը կամ",
        "      տվյալ պանակը (PyArmor runtime-ը ավտոմատ կմնա ներսում)",
        "",
        "  /commit <id>",
        "  /commit <id> <path>",
        "      Նույնը, բայց միայն մեկ instance-ի համար։ Ստացված update ZIP-фայլը",
        "      պահվում է updates/<id>.zip-ով։ Client-ը իր Telegram բոտից կարող է գրել",
        "      /update, որպեսզի քաշի և կիրառի այդ update-ը (config/ և data/ պանակները",
        "      client-ի մոտ չեն փոփոխվի, եթե update.include_config / include_data False են)",
        "",
        "Նշում․ այս control բոտը արձագանքում է միայն այն Telegram chat-ից, որի chat_id-ն սահմանված է config/trading.yaml ֆայլում (telegram_chat_id)",
        "հետևաբար միայն դու, որպես admin, կարող ես օգտվել այս հրամաններից։",
    ]
    return ["\n".join(lines)]


def _handle_text_message(
    msg: Dict[str, Any],
    allowed_chat_id: Optional[int],
    admin_user_ids: Optional[List[int]] = None,
) -> List[str]:
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return []

    if admin_user_ids:
        sender = msg.get("from") or {}
        sender_id = sender.get("id")
        try:
            sender_id_int = int(sender_id) if sender_id is not None else None
        except Exception:
            sender_id_int = None
        if sender_id_int is None:
            return []
        if sender_id_int not in admin_user_ids:
            return []

    # If allowed_chat_id is configured, ignore commands from other chats
    if allowed_chat_id is not None and int(chat_id) != allowed_chat_id:
        return []

    text = (msg.get("text") or "").strip()
    if not text:
        return []

    lower = text.lower()

    # Help
    if lower.startswith("/help") or lower == "help":
        return _handle_command_start()

    if lower.startswith("/news_status") or lower.strip() == "/news":
        return _handle_command_news_status()

    if lower.startswith("/sl_tp_stats"):
        return _handle_command_sl_tp_stats(text)

    if lower.startswith("/proxies_status") or lower.strip() == "/proxies":
        return _handle_command_proxies_status()

    if lower.startswith("/signals"):
        return _handle_command_signals(text)

    if lower.startswith("/create_instance"):
        return _handle_command_create_instance(text, chat_id)

    if lower.startswith("/commit_all") or lower.startswith("/commit ") or lower.strip() == "/commit":
        return _handle_command_commit(text)

    if lower.startswith("/instance_info"):
        return _handle_command_instance_info(text)

    if lower.startswith("/instances"):
        return _handle_command_instances()

    if lower.startswith("/block_coin ") or lower.strip() == "/block_coin":
        return _handle_command_block_coin(text)

    if lower.startswith("/unblock_coin ") or lower.strip() == "/unblock_coin":
        return _handle_command_unblock_coin(text)

    if lower.startswith("/blocked_coins") or lower.startswith("/coin_blocklist"):
        return _handle_command_blocked_coins()

    if lower.startswith("/block_instance") or lower.startswith("/block "):
        return _handle_command_instance_status(text, "blocked")

    if lower.startswith("/unblock_instance") or lower.startswith("/unblock "):
        return _handle_command_instance_status(text, "active")

    if lower.startswith("/delete_instance") or lower.startswith("/delete "):
        return _handle_command_instance_status(text, "deleted")

    if lower.startswith("/pause_instance") or lower.startswith("/pause "):
        return _handle_command_instance_status(text, "paused")

    if lower.startswith("/resume_instance") or lower.startswith("/resume "):
        return _handle_command_instance_status(text, "active")

    # Status first (check more specific suffix _main before short form)
    if lower.startswith("/status_main"):
        return [status_main()]
    if lower.startswith("/status") or lower == "status":
        return [status_main()]

    # Start main
    if lower.startswith("/start_main") or lower.strip() in ("/start", "start"):
        return [start_main()]

    # Stop main
    if lower.startswith("/stop_main"):
        return [stop_main()]
    if lower.startswith("/stop") or lower.strip() == "stop":
        return [stop_main()]
    if lower == "stop":
        return [stop_main()]

    # Restart main
    if lower.startswith("/restart_main"):
        return [restart_main()]
    if lower.startswith("/restart") or lower == "restart":
        return [restart_main()]

    return [
        "Unknown command. Use /start to see available commands.",
    ]


def run_main_control_bot() -> None:
    cfg = _load_control_bot_config()
    token = cfg.get("token") or ""
    allowed_chat_id = cfg.get("allowed_chat_id")
    admin_user_ids = cfg.get("admin_user_ids") or []

    if not token:
        print("[MAIN CONTROL BOT] control_bot.token is not configured in config/trading.yaml")
        return

    print("[MAIN CONTROL BOT] Starting main control bot polling loop...")
    if allowed_chat_id is not None:
        print(f"[MAIN CONTROL BOT] Restricted to chat_id={allowed_chat_id}")
    if admin_user_ids:
        try:
            print(f"[MAIN CONTROL BOT] Restricted to admin_user_ids={admin_user_ids}")
        except Exception:
            pass

    state = _load_state()
    offset = int(state.get("offset", 0))

    try:
        _ = license_register_startup("main_control_bot.py", token, allowed_chat_id)
    except Exception:
        pass

    # Ensure main.py is running when control bot starts.
    # This gives "single command" operational behavior: starting the control
    # bot also starts the trading process if it is not already running.
    try:
        start_msg = start_main()
        print(f"[MAIN CONTROL BOT] {start_msg}")
    except Exception as e:
        print(f"[MAIN CONTROL BOT] Auto-start main.py failed: {e}")

    last_healthcheck_ts = 0.0

    while True:
        # Self-heal: if main.py dies later, auto-start it again.
        now_ts = time.time()
        if now_ts - last_healthcheck_ts >= 30.0:
            last_healthcheck_ts = now_ts
            try:
                msg = start_main()
                # Print only when an action/failure happened to avoid noisy logs.
                if msg.startswith("Started") or msg.startswith("Failed") or "not found" in msg:
                    print(f"[MAIN CONTROL BOT] {msg}")
            except Exception as e:
                print(f"[MAIN CONTROL BOT] Auto-heal check failed: {e}")

        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 10, "offset": offset + 1}

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"[MAIN CONTROL BOT] Error while calling getUpdates: {e}")
            time.sleep(5)
            continue

        if not isinstance(payload, dict) or not payload.get("ok"):
            print(f"[MAIN CONTROL BOT] getUpdates returned non-ok payload: {payload}")
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

                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                text = (message.get("text") or "").strip()
                print(
                    f"[MAIN CONTROL BOT] Update {upd_id} chat_id={chat_id} text={repr(text)}"
                )

                replies = _handle_text_message(message, allowed_chat_id, admin_user_ids)
                if not replies:
                    continue

                if chat_id is None:
                    continue

                for part in replies:
                    try:
                        if isinstance(part, str) and part.startswith("__SEND_DOC_JSON__"):
                            raw = part[len("__SEND_DOC_JSON__") :]
                            try:
                                data = json.loads(raw)
                            except Exception:
                                data = {}
                            path = data.get("path") if isinstance(data, dict) else None
                            caption = data.get("caption") if isinstance(data, dict) else None
                            if isinstance(path, str) and path.strip():
                                send_telegram_document(path, token, chat_id, caption=str(caption) if caption else None)
                            continue

                        send_telegram(part, token, chat_id)
                    except Exception as e:
                        print(f"[MAIN CONTROL BOT] Failed to send reply: {e}")
                        continue
            except Exception as e:
                print(f"[MAIN CONTROL BOT] Failed to process update: {e}")
                continue

        state["offset"] = offset
        _save_state(state)
        time.sleep(2)


if __name__ == "__main__":
    if _acquire_single_instance_lock():
        run_main_control_bot()
