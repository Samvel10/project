from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from flask import Blueprint, render_template, jsonify, g, request
from webapp.auth import login_required, admin_or_above, superadmin_required
from webapp import bot_data

dashboard_bp = Blueprint("dashboard", __name__)

# ─── Constants ────────────────────────────────────────────────────────────────

INSTANCES_PATH = Path('/var/www/html/new_example_bot/data/instances_registry.json')

CONFIG_FILE_MAP = {
    "trading":           "trading.yaml",
    "accounts":          "binance_accounts.yaml",
    "market_regime":     "market_regime.yaml",
    "execution":         "execution.yaml",
    "risk":              "risk.yaml",
    "ml":                "ml.yaml",
    "secondary_signals": "secondary_signals.yaml",
    "symbols":           "symbols.yaml",
}

CFG_ROOT = Path('/var/www/html/new_example_bot/config')


# ─── Page routes ──────────────────────────────────────────────────────────────

@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
@login_required
def overview():
    regime   = bot_data.get_market_regime_state()
    quality  = bot_data.get_market_quality_cache()
    procs    = bot_data.get_process_status()
    blocklist = bot_data.get_symbol_blocklist()
    accounts  = bot_data.get_all_accounts_with_state()
    signals   = bot_data.get_signal_log(limit=20)

    # quality summary
    good = sum(1 for v in quality.values() if isinstance(v, dict) and v.get("verdict") == "GOOD")
    suspect = sum(1 for v in quality.values() if isinstance(v, dict) and v.get("verdict") == "SUSPECT")
    manip = sum(1 for v in quality.values() if isinstance(v, dict) and v.get("verdict") == "MANIPULATED")

    return render_template(
        "dashboard/overview.html",
        user=g.current_user,
        regime=regime,
        quality_summary={"good": good, "suspect": suspect, "manipulated": manip, "total": good+suspect+manip},
        processes=procs,
        blocklist=blocklist,
        accounts=accounts,
        recent_signals=signals,
    )


@dashboard_bp.route("/dashboard/market-quality")
@login_required
def market_quality():
    cache = bot_data.get_market_quality_cache()
    items = []
    for symbol, data in cache.items():
        if isinstance(data, dict):
            items.append({**data, "symbol": symbol})
    items.sort(key=lambda x: x.get("score", 100))
    return render_template("dashboard/market_quality.html", user=g.current_user, items=items)


@dashboard_bp.route("/dashboard/market-regime")
@login_required
def market_regime():
    state  = bot_data.get_market_regime_state()
    config = bot_data.get_market_regime_config()
    return render_template("dashboard/market_regime.html", user=g.current_user, state=state, config=config)


@dashboard_bp.route("/dashboard/models")
@login_required
def ml_models():
    from pathlib import Path
    import os
    models_dir = Path(__file__).resolve().parent.parent.parent / "ml" / "models"
    models = []
    try:
        for f in sorted(models_dir.glob("model_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            parts = f.stem.split("_")
            score = float(parts[-1]) if len(parts) >= 3 else 0.0
            ts    = int(parts[1])    if len(parts) >= 3 else 0
            models.append({"name": f.name, "score": score, "timestamp": ts, "size_kb": f.stat().st_size // 1024})
    except Exception:
        pass
    total = len(list(models_dir.glob("model_*.pkl"))) if models_dir.exists() else 0
    return render_template("dashboard/ml_models.html", user=g.current_user, models=models, total=total)


@dashboard_bp.route("/dashboard/signals")
@login_required
def signals_log():
    signals = bot_data.get_signal_log(limit=1000)
    signals.reverse()
    large = [s for s in signals if s.get("category") == "large"]
    small = [s for s in signals if s.get("category") != "large"]
    return render_template("dashboard/signals.html", user=g.current_user,
                           signals_large=large, signals_small=small)


@dashboard_bp.route("/dashboard/blocklist")
@admin_or_above
def blocklist():
    bl = bot_data.get_symbol_blocklist()
    return render_template("dashboard/blocklist.html", user=g.current_user, blocklist=bl)


@dashboard_bp.route("/dashboard/config")
@admin_or_above
def config_view():
    configs = bot_data.get_all_configs()
    # Add extra configs not in the original list
    extra_names = ["secondary_signals", "symbols"]
    for name in extra_names:
        fname = CONFIG_FILE_MAP.get(name)
        if fname:
            configs[name] = bot_data.read_yaml(CFG_ROOT / fname)

    from webapp.database import SessionLocal
    from webapp.models import ConfigHistory
    db = SessionLocal()
    try:
        history = db.query(ConfigHistory).order_by(ConfigHistory.applied_at.desc()).limit(50).all()
    finally:
        db.close()
    return render_template("dashboard/config.html", user=g.current_user, configs=configs, history=history)


@dashboard_bp.route("/dashboard/news")
@login_required
def news():
    return render_template("dashboard/news.html", user=g.current_user)


@dashboard_bp.route("/api/news")
@login_required
def api_news():
    """Fetch latest news from multiple free RSS feeds."""
    import urllib.request
    import xml.etree.ElementTree as ET
    import html as html_mod

    FEEDS = [
        {"name": "CoinDesk",       "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "category": "Crypto"},
        {"name": "Cointelegraph",   "url": "https://cointelegraph.com/rss",                  "category": "Crypto"},
        {"name": "CryptoNews",      "url": "https://cryptonews.com/news/feed/",               "category": "Crypto"},
        {"name": "Reuters Business","url": "https://feeds.reuters.com/reuters/businessNews",  "category": "Markets"},
        {"name": "Decrypt",         "url": "https://decrypt.co/feed",                         "category": "Crypto"},
    ]

    articles = []
    for feed in FEEDS:
        try:
            req = urllib.request.Request(feed["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            for item in items[:12]:
                def txt(tag):
                    el = item.find(tag)
                    return html_mod.unescape(el.text or "") if el is not None and el.text else ""
                title = txt("title")
                link  = txt("link") or txt("guid")
                pub   = txt("pubDate") or txt("published") or txt("updated")
                desc  = txt("description") or txt("summary")
                # strip html tags from description
                import re
                desc = re.sub(r"<[^>]+>", "", desc)[:280]
                if title and link:
                    articles.append({
                        "title":    title,
                        "link":     link,
                        "pub":      pub,
                        "desc":     desc,
                        "source":   feed["name"],
                        "category": feed["category"],
                    })
        except Exception:
            pass

    # Sort by publication date (best effort)
    from email.utils import parsedate_to_datetime
    def _sort_key(a):
        try:
            return parsedate_to_datetime(a["pub"]).timestamp()
        except Exception:
            return 0
    articles.sort(key=_sort_key, reverse=True)
    return jsonify({"articles": articles[:80]})


@dashboard_bp.route("/dashboard/audit")
@admin_or_above
def audit_log():
    from webapp.database import SessionLocal
    from webapp.models import AuditLog
    db = SessionLocal()
    try:
        logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(200).all()
    finally:
        db.close()
    return render_template("dashboard/audit.html", user=g.current_user, logs=logs)


@dashboard_bp.route("/admin/control")
@superadmin_required
def admin_control():
    return render_template("admin/control.html", user=g.current_user)


# ─── API endpoints for JS ─────────────────────────────────────────────────────

@dashboard_bp.route("/api/processes")
@login_required
def api_processes():
    return jsonify(bot_data.get_process_status())


@dashboard_bp.route("/api/regime")
@login_required
def api_regime():
    return jsonify(bot_data.get_market_regime_state())


@dashboard_bp.route("/api/quality/<symbol>")
@login_required
def api_quality_symbol(symbol):
    cache = bot_data.get_market_quality_cache()
    return jsonify(cache.get(symbol.upper(), {}))


@dashboard_bp.route("/api/klines/<symbol>")
@login_required
def api_klines(symbol):
    interval = request.args.get("interval", "1m")
    limit    = min(int(request.args.get("limit", 200)), 1000)
    return jsonify(bot_data.get_klines(symbol.upper(), interval, limit))


@dashboard_bp.route("/api/config/save", methods=["POST"])
@admin_or_above
def api_save_config():
    from pathlib import Path
    from webapp.auth import audit
    from webapp.database import SessionLocal
    from webapp.models import ConfigHistory
    import yaml

    data = request.get_json() or {}
    config_name = data.get("config")
    field_path  = data.get("field")
    new_value   = data.get("value")

    allowed = ["trading", "market_regime", "execution", "risk", "ml"]
    if config_name not in allowed:
        return jsonify({"error": "Config not allowed"}), 400

    cfg_file = Path(__file__).resolve().parent.parent.parent / "config" / f"{config_name}.yaml"
    try:
        cfg = bot_data.read_yaml(cfg_file)
        # Navigate and set nested key
        keys = field_path.split(".")
        obj = cfg
        for k in keys[:-1]:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        old_value = str(obj.get(keys[-1], ""))
        obj[keys[-1]] = new_value
        bot_data.write_yaml(cfg_file, cfg)

        db = SessionLocal()
        try:
            ch = ConfigHistory(
                user_id=g.current_user.id,
                username=g.current_user.username,
                config_file=config_name,
                field_path=field_path,
                old_value=old_value,
                new_value=str(new_value),
            )
            db.add(ch)
            db.commit()
        finally:
            db.close()

        audit("CONFIG_CHANGE", f"{config_name}.{field_path}", f"{old_value} → {new_value}")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Config raw edit API ──────────────────────────────────────────────────────

@dashboard_bp.route("/api/config/<name>/raw", methods=["GET"])
@admin_or_above
def api_config_raw_get(name):
    if name not in CONFIG_FILE_MAP:
        return jsonify({"error": "Unknown config"}), 400
    cfg_file = CFG_ROOT / CONFIG_FILE_MAP[name]
    try:
        content = cfg_file.read_text(encoding="utf-8")
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/config/<name>/raw", methods=["POST"])
@admin_or_above
def api_config_raw_save(name):
    import yaml
    from webapp.auth import audit
    if name not in CONFIG_FILE_MAP:
        return jsonify({"error": "Unknown config"}), 400
    data = request.get_json() or {}
    content = data.get("content", "")
    try:
        # Validate YAML
        yaml.safe_load(content)
    except Exception as e:
        return jsonify({"error": f"Invalid YAML: {e}"}), 400
    cfg_file = CFG_ROOT / CONFIG_FILE_MAP[name]
    try:
        cfg_file.write_text(content, encoding="utf-8")
        audit("CONFIG_RAW_SAVE", name, f"Raw yaml save by {g.current_user.username}")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Admin Bot Control API ────────────────────────────────────────────────────

@dashboard_bp.route("/api/admin/bot/status")
@superadmin_required
def api_admin_bot_status():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from monitoring.main_process_manager import status_main, _get_tracked_pid
        info = status_main()
        pid = None
        try:
            pid = _get_tracked_pid()
        except Exception:
            pass
        running = bool(info.get("running", False)) if isinstance(info, dict) else False
        return jsonify({"running": running, "pid": pid, "info": info if isinstance(info, dict) else {}})
    except Exception as e:
        # Fallback: check process list
        procs = bot_data.get_process_status()
        engine = procs.get("Trading Engine", {})
        return jsonify({"running": engine.get("running", False), "pid": engine.get("pid"), "info": {}, "error": str(e)})


@dashboard_bp.route("/api/admin/bot/start", methods=["POST"])
@superadmin_required
def api_admin_bot_start():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from monitoring.main_process_manager import start_main
        result = start_main()
        return jsonify({"ok": True, "message": str(result) if result else "Start requested"})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@dashboard_bp.route("/api/admin/bot/stop", methods=["POST"])
@superadmin_required
def api_admin_bot_stop():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from monitoring.main_process_manager import stop_main
        result = stop_main()
        return jsonify({"ok": True, "message": str(result) if result else "Stop requested"})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@dashboard_bp.route("/api/admin/bot/restart", methods=["POST"])
@superadmin_required
def api_admin_bot_restart():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from monitoring.main_process_manager import restart_main
        result = restart_main()
        return jsonify({"ok": True, "message": str(result) if result else "Restart requested"})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


# ─── Admin Blocklist API ──────────────────────────────────────────────────────

@dashboard_bp.route("/api/admin/blocklist")
@admin_or_above
def api_admin_blocklist_get():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from data.symbol_blocklist import list_blocked_symbols
        symbols = list_blocked_symbols()
        return jsonify({"symbols": symbols})
    except Exception as e:
        # Fallback to bot_data
        return jsonify({"symbols": bot_data.get_symbol_blocklist(), "error": str(e)})


@dashboard_bp.route("/api/admin/blocklist/add", methods=["POST"])
@admin_or_above
def api_admin_blocklist_add():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from data.symbol_blocklist import add_blocked_symbols, list_blocked_symbols
        data = request.get_json() or {}
        raw = data.get("symbols", "")
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not syms:
            return jsonify({"error": "No symbols provided"}), 400
        add_blocked_symbols(syms)
        total = list_blocked_symbols()
        return jsonify({"ok": True, "added": syms, "total": total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@dashboard_bp.route("/api/admin/blocklist/remove", methods=["POST"])
@admin_or_above
def api_admin_blocklist_remove():
    try:
        sys.path.insert(0, '/var/www/html/new_example_bot')
        from data.symbol_blocklist import remove_blocked_symbols, list_blocked_symbols
        data = request.get_json() or {}
        symbol = data.get("symbol", "").strip().upper()
        if not symbol:
            return jsonify({"error": "No symbol provided"}), 400
        remove_blocked_symbols([symbol])
        total = list_blocked_symbols()
        return jsonify({"ok": True, "removed": symbol, "total": total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ─── Admin Instances API ──────────────────────────────────────────────────────

def _get_instances():
    d = json.loads(INSTANCES_PATH.read_text(encoding="utf-8"))
    return d, d.get('instances', d)


def _set_instance_status(inst_id, status):
    d = json.loads(INSTANCES_PATH.read_text(encoding="utf-8"))
    insts = d.get('instances', d)
    if inst_id in insts:
        insts[inst_id]['status'] = status
        INSTANCES_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return True
    return False


@dashboard_bp.route("/api/admin/instances")
@superadmin_required
def api_admin_instances():
    try:
        _, insts = _get_instances()
        result = []
        for inst_id, inst in insts.items():
            result.append({
                "id": inst_id,
                "id_short": inst_id[:8],
                "label": inst.get("label") or "",
                "status": inst.get("status", "active"),
                "run_count": inst.get("run_count", 0),
                "last_client": inst.get("last_client") or "",
                "last_host": inst.get("last_host") or "",
                "last_seen": inst.get("last_seen") or "",
            })
        return jsonify({"instances": result})
    except Exception as e:
        return jsonify({"instances": [], "error": str(e)}), 500


@dashboard_bp.route("/api/admin/instances/<inst_id>/action", methods=["POST"])
@superadmin_required
def api_admin_instance_action(inst_id):
    try:
        data = request.get_json() or {}
        action = data.get("action", "")
        status_map = {"block": "blocked", "unblock": "active", "delete": "deleted"}
        if action not in status_map:
            return jsonify({"error": "Invalid action"}), 400
        ok = _set_instance_status(inst_id, status_map[action])
        if not ok:
            return jsonify({"error": "Instance not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ─── Admin Proxies API ────────────────────────────────────────────────────────

@dashboard_bp.route("/api/admin/proxies")
@superadmin_required
def api_admin_proxies():
    try:
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys; sys.path.insert(0,"/var/www/html/new_example_bot")
import config.proxies as p
import importlib; importlib.reload(p)
try:
    working = p.get_working_proxies()
except Exception:
    working = []
try:
    blocked = p.get_blocked_proxies()
except Exception:
    blocked = []
import json; print(json.dumps({"working": working, "blocked": blocked}))
'''],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            return jsonify(data)
        return jsonify({"working": [], "blocked": [], "error": result.stderr[:200]})
    except Exception as e:
        return jsonify({"working": [], "blocked": [], "error": str(e)}), 500


# ─── Admin News Guard API ─────────────────────────────────────────────────────

@dashboard_bp.route("/api/admin/news")
@superadmin_required
def api_admin_news():
    try:
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys; sys.path.insert(0,"/var/www/html/new_example_bot")
from monitoring.news_guard import get_active_news_events
import json; print(json.dumps(get_active_news_events()))
'''],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            events = json.loads(result.stdout.strip())
            enabled = isinstance(events, list)
            return jsonify({"enabled": enabled, "events": events if isinstance(events, list) else []})
        return jsonify({"enabled": False, "events": [], "error": result.stderr[:200]})
    except Exception as e:
        return jsonify({"enabled": False, "events": [], "error": str(e)}), 500
