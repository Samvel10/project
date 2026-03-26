import json
import os
import socket
import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from monitoring.telegram import send_telegram
import requests


ROOT_DIR = Path(__file__).resolve().parent
_REGISTRY_PATH = ROOT_DIR / "data" / "instances_registry.json"
_PUBLIC_LICENSE_DIR = ROOT_DIR / "updates" / "licenses"

# NOTE:
#   This placeholder value will be replaced per-distribution before obfuscation
#   when you create a new packaged bot. In the development tree it may stay as
#   a fixed value.
INSTANCE_ID = "UNSET_INSTANCE_ID"


def _get_local_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            try:
                s.close()
            except Exception:
                pass
        if ip:
            return str(ip)
    except Exception:
        pass

    try:
        host = socket.gethostname() or ""
        if host:
            ip2 = socket.gethostbyname(host)
            if ip2:
                return str(ip2)
    except Exception:
        pass
    return None


def _get_public_ip() -> Optional[str]:
    try:
        resp = requests.get("https://api.ipify.org", timeout=5)
        resp.raise_for_status()
        ip = (resp.text or "").strip()
        if ip:
            return ip
    except Exception:
        pass
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_registry() -> Dict[str, Any]:
    """Load the local instances registry from data/instances_registry.json.

    The structure is:

    {
      "instances": {
        "<instance_id>": {
          "status": "active" | "blocked" | "deleted",
          "created_at": "...",
          "last_seen": "...",
          "last_host": "...",
          "last_client": "...",
          "last_cwd": "...",
          "last_pid": 1234,
          "run_count": 1,
          "hosts": {
            "hostname": {"first_seen": "...", "last_seen": "..."}
          }
        },
        ...
      }
    }
    """

    try:
        if not _REGISTRY_PATH.exists():
            return {"instances": {}}
        raw = _REGISTRY_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return {"instances": {}}
        data = json.loads(raw)
        if isinstance(data, dict):
            if "instances" not in data or not isinstance(data.get("instances"), dict):
                data["instances"] = {}
            return data
    except Exception:
        # On any error we fall back to an empty registry; we never want
        # licensing to crash the trading bot.
        pass

    return {"instances": {}}


def _save_registry(reg: Dict[str, Any]) -> None:
    """Persist the registry back to disk, swallowing any IO errors."""

    try:
        if not _REGISTRY_PATH.parent.exists():
            _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REGISTRY_PATH.write_text(
            json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        # Never raise from here – trading logic must not depend on this.
        pass


def _publish_instance_status(instance_id: str, inst: Dict[str, Any]) -> None:
    try:
        if not _PUBLIC_LICENSE_DIR.exists():
            _PUBLIC_LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        payload: Dict[str, Any] = {
            "id": instance_id,
            "status": inst.get("status") or "active",
            "status_updated_at": inst.get("status_updated_at") or inst.get("last_seen"),
            "blocked_reason": inst.get("blocked_reason") or "",
            "last_seen": inst.get("last_seen"),
            "last_host": inst.get("last_host"),
            "last_client": inst.get("last_client"),
        }
        out_path = _PUBLIC_LICENSE_DIR / f"{instance_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _ensure_instance_entry(reg: Dict[str, Any]) -> Dict[str, Any]:
    instances = reg.setdefault("instances", {})
    inst = instances.get(INSTANCE_ID)
    if not isinstance(inst, dict):
        now = _now_iso()
        inst = {
            "id": INSTANCE_ID,
            "status": "active",
            "created_at": now,
            "last_seen": None,
            "last_host": None,
            "last_client": None,
            "last_cwd": None,
            "last_pid": None,
            "run_count": 0,
            "hosts": {},
        }
        instances[INSTANCE_ID] = inst
    return inst


def register_new_instance(
    instance_id: str,
    label: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or update a registry record for a newly provisioned instance.

    This is used by the admin-side factory when building a new client
    distribution, before that client ever starts up.
    """

    reg = _load_registry()
    instances = reg.setdefault("instances", {})

    inst = instances.get(instance_id)
    now = _now_iso()
    if not isinstance(inst, dict):
        inst = {
            "id": instance_id,
            "status": "active",
            "created_at": now,
            "last_seen": None,
            "last_host": None,
            "last_client": "admin_factory",
            "last_cwd": None,
            "last_pid": None,
            "run_count": 0,
            "hosts": {},
        }

    if label:
        inst["label"] = label

    if meta and isinstance(meta, dict):
        existing_meta = inst.get("meta") or {}
        if not isinstance(existing_meta, dict):
            existing_meta = {}
        for k, v in meta.items():
            existing_meta[k] = v
        inst["meta"] = existing_meta

    instances[instance_id] = inst
    reg["instances"] = instances
    _save_registry(reg)
    try:
        _publish_instance_status(instance_id, inst)
    except Exception:
        pass
    return inst


def register_startup(
    client_name: str,
    token: Optional[str],
    admin_chat_id: Optional[int],
    license_base_url: Optional[str] = None,
) -> bool:
    """Register a process startup for this INSTANCE_ID.

    This function:
      - Updates data/instances_registry.json with host and run metadata
      - Optionally auto-blocks the instance on multi-host usage
      - Sends a Telegram notification (if token/chat_id are provided)
      - Returns True if the instance is allowed to run, False if it should
        terminate (status is "blocked" or "deleted").
    """

    reg = _load_registry()
    inst = _ensure_instance_entry(reg)

    hostname = socket.gethostname() or "unknown-host"
    cwd = os.getcwd()
    pid = os.getpid()
    now = _now_iso()

    hosts = inst.get("hosts") or {}
    if not isinstance(hosts, dict):
        hosts = {}

    host_entry = hosts.get(hostname) or {}
    if not isinstance(host_entry, dict):
        host_entry = {}
    if "first_seen" not in host_entry:
        host_entry["first_seen"] = now
    host_entry["last_seen"] = now
    hosts[hostname] = host_entry

    inst["hosts"] = hosts
    inst["last_seen"] = now
    inst["last_host"] = hostname
    inst["last_client"] = client_name
    inst["last_cwd"] = cwd
    inst["last_pid"] = pid
    try:
        inst["run_count"] = int(inst.get("run_count", 0)) + 1
    except Exception:
        inst["run_count"] = 1

    host_list = list(hosts.keys())
    multi_host = len(host_list) > 1
    multi_host_allowed = bool(inst.get("multi_host_allowed"))

    # If we detect the same INSTANCE_ID on multiple hosts and admin has not
    # explicitly allowed this, ask the user interactively.
    if multi_host and not multi_host_allowed:
        current_status = str(inst.get("status") or "active")
        # If previously auto-blocked for multi_host OR first time seeing
        # multi-host — prompt the user for permission.
        if current_status != "deleted":
            print()
            print("=" * 60)
            print("  WARNING: Multi-host detected!")
            print(f"  Instance '{INSTANCE_ID}' has been seen on:")
            for h in host_list:
                h_info = hosts.get(h) or {}
                print(f"    - {h} (last seen: {h_info.get('last_seen', '?')})")
            print()
            print("  This instance is being started on a new/different host.")
            print("=" * 60)
            try:
                answer = input("  Allow this instance to run on multiple hosts? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("y", "yes"):
                inst["multi_host_allowed"] = True
                inst["status"] = "active"
                inst["status_updated_at"] = now
                inst["blocked_reason"] = "multi_host_allowed"
                print("  -> Multi-host ALLOWED. Instance will start normally.")
            else:
                inst["status"] = "blocked"
                inst["status_updated_at"] = now
                inst["blocked_reason"] = "multi_host"
                print("  -> Multi-host DENIED. Instance blocked.")
            print()

    status = str(inst.get("status") or "active")

    os.environ.pop("LICENSE_STATUS", None)
    remote_status: Optional[str] = None
    base_url = (license_base_url or os.environ.get("LICENSE_BASE_URL") or "").strip()
    if base_url:
        try:
            url = base_url.rstrip("/") + f"/licenses/{INSTANCE_ID}.json"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    rs = str(payload.get("status") or "").lower().strip()
                    if rs:
                        remote_status = rs
        except Exception:
            remote_status = None

    if remote_status in ("blocked", "deleted"):
        status = remote_status
        inst["status"] = status
        inst["status_updated_at"] = now
    elif remote_status == "paused":
        status = "paused"
        inst["status"] = "paused"
        inst["status_updated_at"] = now
        os.environ["LICENSE_STATUS"] = "paused"

    allowed = status not in ("blocked", "deleted")

    reg.setdefault("instances", {})[INSTANCE_ID] = inst
    _save_registry(reg)
    try:
        _publish_instance_status(INSTANCE_ID, inst)
    except Exception:
        pass

    # Send Telegram notification if we have credentials.
    if token and admin_chat_id is not None:
        try:
            try:
                user = getpass.getuser() or None
            except Exception:
                user = None
            local_ip = _get_local_ip()
            public_ip = _get_public_ip()

            lines = [
                "[LICENSE] Instance startup notification",
                f"Instance ID: {INSTANCE_ID}",
                f"Status: {status}",
                f"Client: {client_name}",
                f"Host: {hostname}",
                f"User: {user or '-'}",
                f"Local IP: {local_ip or '-'}",
                f"Public IP: {public_ip or '-'}",
                f"CWD: {cwd}",
                f"PID: {pid}",
            ]

            if multi_host:
                lines.append("")
                lines.append(
                    "⚠ WARNING: This instance_id has been seen on multiple hosts:"  # noqa: E501
                )
                for h in host_list:
                    h_info = hosts.get(h) or {}
                    first = h_info.get("first_seen", "?")
                    last = h_info.get("last_seen", "?")
                    lines.append(f"- {h}: first_seen={first}, last_seen={last}")

            lines.append("")
            if not allowed:
                reason = str(inst.get("blocked_reason") or "")
                if reason == "multi_host":
                    lines.append(
                        "Result: BLOCKED (multi-host violation; instance auto-blocked)"
                    )
                else:
                    lines.append("Result: BLOCKED (status is blocked/deleted)")
            else:
                if status == "paused":
                    lines.append("Result: PAUSED (trading disabled)")
                else:
                    lines.append("Result: ALLOWED to run")

            msg = "\n".join(lines)
            send_telegram(msg, token, int(admin_chat_id))
        except Exception:
            # Notification errors must never break the bot.
            pass

    return allowed


def get_instances() -> Dict[str, Dict[str, Any]]:
    """Return a snapshot of all instances from the local registry."""

    reg = _load_registry()
    instances = reg.get("instances") or {}
    if not isinstance(instances, dict):
        return {}

    # Shallow copy to avoid accidental external mutation.
    return {k: dict(v) for k, v in instances.items() if isinstance(v, dict)}


def set_instance_status(instance_id: str, status: str) -> bool:
    """Update status for a given instance_id.

    Status must be one of: "active", "blocked", "deleted".
    Returns True if the instance existed and was updated.
    """

    status = str(status).lower()
    if status not in ("active", "paused", "blocked", "deleted"):
        return False

    reg = _load_registry()
    instances = reg.get("instances") or {}
    if not isinstance(instances, dict) or instance_id not in instances:
        return False

    inst = instances.get(instance_id)
    if not isinstance(inst, dict):
        return False

    inst["status"] = status
    inst["status_updated_at"] = _now_iso()

    # If admin explicitly re-activates an instance which was previously
    # blocked due to multi-host, treat this as an override and allow future
    # multi-host startups.
    if status == "active":
        blocked_reason = inst.get("blocked_reason")
        if blocked_reason == "multi_host":
            inst["blocked_reason"] = "multi_host_allowed"
            inst["multi_host_allowed"] = True
    instances[instance_id] = inst
    reg["instances"] = instances
    _save_registry(reg)
    try:
        _publish_instance_status(instance_id, inst)
    except Exception:
        pass
    return True
