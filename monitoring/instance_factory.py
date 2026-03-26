import os
import re
import sys
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ruamel.yaml import YAML

from instance_security import register_new_instance


ROOT_DIR = Path(__file__).resolve().parents[1]
_CLIENTS_ROOT = ROOT_DIR / "clients"
_BUILD_ROOT = ROOT_DIR / "build_instances"
_UPDATES_ROOT = ROOT_DIR / "updates"
_YAML = YAML(typ="safe")


def _safe_mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _generate_instance_id() -> str:
    return uuid.uuid4().hex


def _copy_source_tree(src_root: Path, dst_root: Path) -> None:
    includes_dirs = [
        "backtest",
        "config",
        "data",
        "execution",
        "features",
        "ml",
        "monitoring",
        "risk",
        "signals",
    ]
    include_files = [
        "main.py",
        "instance_security.py",
    ]

    _safe_mkdir(dst_root)

    for rel in includes_dirs:
        src = src_root / rel
        if not src.exists():
            continue
        dst = dst_root / rel
        try:
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    "instances_registry.json",
                ),
            )
        except Exception:
            continue

    for rel_file in include_files:
        src_file = src_root / rel_file
        if not src_file.exists():
            continue
        dst_file = dst_root / rel_file
        try:
            _safe_mkdir(dst_file.parent)
            shutil.copy2(src_file, dst_file)
        except Exception:
            continue


def _patch_instance_id(work_root: Path, instance_id: str) -> None:
    target = work_root / "instance_security.py"
    if not target.exists():
        return
    try:
        text = target.read_text(encoding="utf-8")
    except Exception:
        return

    marker = "INSTANCE_ID"
    if marker not in text:
        return

    new_line = f'INSTANCE_ID = "{instance_id}"'

    replaced = False
    lines = []
    for line in text.splitlines():
        if marker in line and "INSTANCE_ID" in line and "=" in line:
            lines.append(new_line)
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        return

    new_text = "\n".join(lines) + "\n"
    try:
        target.write_text(new_text, encoding="utf-8")
    except Exception:
        return


def _patch_license_overrides(work_root: Path) -> None:
    """Embed admin Telegram token/chat_id into main.py for client builds.

    We read admin-side config/trading.yaml from ROOT_DIR and patch the
    _LICENSE_TOKEN_OVERRIDE / _LICENSE_CHAT_ID_OVERRIDE placeholders in the
    copied main.py under work_root. This way, client distributions keep the
    admin notification credentials only inside obfuscated Python bytecode,
    not in plain-text YAML.
    """

    main_path = work_root / "main.py"
    if not main_path.exists():
        return

    cfg_path = ROOT_DIR / "config" / "trading.yaml"
    admin_token: Optional[str] = None
    admin_chat_id: Optional[int] = None

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = _YAML.load(f) or {}
    except Exception:
        cfg = {}

    if isinstance(cfg, dict):
        ctrl_cfg = cfg.get("control_bot") or {}
        token = ctrl_cfg.get("token") or None
        if token:
            admin_token = str(token)
        if not admin_token:
            token = cfg.get("telegram_token")
            if token:
                admin_token = str(token)

        chat_id_val = cfg.get("telegram_chat_id")
        if chat_id_val is not None:
            try:
                admin_chat_id = int(chat_id_val)
            except (TypeError, ValueError):
                admin_chat_id = None

    # If we couldn't load any credentials, do not modify main.py.
    if not admin_token and admin_chat_id is None:
        return

    try:
        text = main_path.read_text(encoding="utf-8")
    except Exception:
        return

    lines = []
    for line in text.splitlines():
        if "PATCH_LICENSE_TOKEN" in line and admin_token:
            lines.append(f'_LICENSE_TOKEN_OVERRIDE = "{admin_token}"  # PATCH_LICENSE_TOKEN')
        elif "PATCH_LICENSE_CHAT_ID" in line and admin_chat_id is not None:
            lines.append(f"_LICENSE_CHAT_ID_OVERRIDE = {admin_chat_id}  # PATCH_LICENSE_CHAT_ID")
        else:
            lines.append(line)

    new_text = "\n".join(lines) + "\n"
    try:
        main_path.write_text(new_text, encoding="utf-8")
    except Exception:
        return


def _copy_non_py(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    for root, dirs, files in os.walk(src_dir):
        rel_root = Path(root).relative_to(src_dir)
        target_root = dst_dir / rel_root
        _safe_mkdir(target_root)
        for name in files:
            if name.endswith(".py") or name.endswith(".pyc") or name.endswith(".pyo"):
                continue
            src_path = Path(root) / name
            dst_path = target_root / name
            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                continue


def _filter_client_configs(work_root: Path) -> None:
    """Adjust config files for a client build inside work_root.

    - Remove admin-only configs from the client package:
        backtest.yaml, execution.yaml, risk.yaml, symbols.yaml, paper_accounts.yaml
    - Replace ml.yaml with a minimal file that only keeps the existing
      thresholds.enter / thresholds.exit values so that the client can tune
      thresholds but not see the rest of the ML configuration.
    """

    cfg_dir = work_root / "config"
    if not cfg_dir.exists():
        return

    ml_path = cfg_dir / "ml.yaml"
    thresholds: Dict[str, Any] = {}

    # Extract existing ML thresholds so we can keep only those for clients.
    try:
        if ml_path.exists():
            with ml_path.open("r", encoding="utf-8") as f:
                ml_cfg = _YAML.load(f) or {}
            if isinstance(ml_cfg, dict):
                th = ml_cfg.get("thresholds") or {}
                if isinstance(th, dict):
                    thresholds = dict(th)
    except Exception:
        thresholds = {}

    remove_files = [
        "backtest.yaml",
        "execution.yaml",
        "risk.yaml",
        "symbols.yaml",
        "paper_accounts.yaml",
    ]

    for name in remove_files:
        try:
            p = cfg_dir / name
            if p.exists():
                p.unlink()
        except Exception:
            continue

    # Rewrite ml.yaml so that it only exposes thresholds.{enter,exit}.
    if thresholds and ml_path.exists():
        try:
            minimal_ml = {"thresholds": thresholds}
            with ml_path.open("w", encoding="utf-8") as f:
                _YAML.dump(minimal_ml, f)
        except Exception:
            pass

    # Sanitize binance_accounts.yaml: drop all real accounts so that client
    # instances never see existing users' Binance API keys or names.
    bin_path = cfg_dir / "binance_accounts.yaml"
    try:
        if bin_path.exists():
            src_path = ROOT_DIR / "config" / "binance_accounts.yaml"
            header: list[str] = []
            try:
                raw = src_path.read_text(encoding="utf-8") if src_path.exists() else ""
            except Exception:
                raw = ""

            for line in (raw or "").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    header.append(line)
                    continue
                break

            if not header:
                header = ["# Binance accounts template (client build)"]

            template_lines: list[str] = []
            template_lines.extend(header)
            if template_lines and template_lines[-1].strip() != "":
                template_lines.append("")

            template_lines.extend(
                [
                    'mode: "single"',
                    "",
                    "accounts:",
                    '  - name: "YOUR_ACCOUNT_NAME"',
                    "    trade_enabled: true",
                    "    leverage: 10",
                    '    proxy: ""',
                    '    api_key: ""',
                    '    api_secret: ""',
                    "    settings:",
                    "      margin_mode: isolated",
                    "      tp_mode: 2",
                    "      sl_pct: 3.0",
                    "      tp_pcts: [2.0, 4.0, 6.0]",
                    "      auto_sl_tp: true",
                    "      move_sl_to_entry_on_first_tp: true",
                    '      fixed_notional_type: "BALANCE_PCT"',
                    "      fixed_notional_value: 1.0",
                    "      confidence: 65",
                    "",
                ]
            )

            bin_path.write_text("\n".join(template_lines), encoding="utf-8")
    except Exception:
        pass

    # Sanitize trading.yaml: remove or blank all Telegram/API credentials so
    # that client builds do not contain admin tokens or chat IDs.
    trading_path = cfg_dir / "trading.yaml"
    try:
        if trading_path.exists():
            src_path = ROOT_DIR / "config" / "trading.yaml"
            try:
                text = src_path.read_text(encoding="utf-8") if src_path.exists() else ""
            except Exception:
                text = ""

            if not text.strip():
                # Fallback: keep existing client file if we can't load template.
                text = trading_path.read_text(encoding="utf-8")

            out_lines: list[str] = []
            for line in text.splitlines():
                s = line.strip()

                # Root-level secrets / identifiers
                if re.match(r"^telegram_token\s*:", s):
                    out_lines.append(re.sub(r"^telegram_token\s*:\s*.*$", 'telegram_token: ""', line))
                    continue
                if re.match(r"^telegram_chat_id\s*:", s):
                    out_lines.append(re.sub(r"^telegram_chat_id\s*:\s*.*$", "telegram_chat_id: 0", line))
                    continue

                if re.match(r"^api_id\s*:", s) or re.match(r"^\"api_id\"\s*:", s):
                    out_lines.append(re.sub(r"^.*api_id.*$", '"api_id": 0', line))
                    continue
                if re.match(r"^api_hash\s*:", s) or re.match(r"^\"api_hash\"\s*:", s):
                    out_lines.append(re.sub(r"^.*api_hash.*$", '"api_hash": ""', line))
                    continue

                # Blank any nested bot token lines (control/log/analytics/etc.)
                if re.match(r"^\s*token\s*:", line):
                    indent = re.match(r"^(\s*)", line).group(1) if re.match(r"^(\s*)", line) else ""
                    out_lines.append(f"{indent}token: \"\"")
                    continue

                out_lines.append(line)

            trading_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _disable_paper_trading(work_root: Path) -> None:
    """Override paper trading helpers for client builds.

    The client package should not expose functional paper accounts. We keep the
    module present for imports, but redefine the public helpers so that they
    are no-ops.
    """

    pt_path = work_root / "execution" / "paper_trading.py"
    if not pt_path.exists():
        return

    try:
        original = pt_path.read_text(encoding="utf-8")
    except Exception:
        return

    stub = """

def paper_trading_enabled() -> bool:
    return False


def open_paper_trades(
    symbol: str,
    signal: str,
    entry_price: float,
    confidence: float,
    atr_value: float | None = None,
) -> None:
    return


def check_paper_exits() -> None:
    return
"""

    try:
        pt_path.write_text(original + stub, encoding="utf-8")
    except Exception:
        return


def _clean_client_data_dir(work_root: Path) -> None:
    """Remove admin-side state and history from data/ for client builds.

    We want client instances to start from a clean slate without any of the
    admin environment's trade history, Telegram offsets, or instance registry
    information.
    """

    data_dir = work_root / "data"
    if not data_dir.exists():
        return

    # Remove trade history CSVs (real + paper) if present.
    for name in ("trade_history", "paper_trade_history"):
        try:
            path = data_dir / name
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
        except Exception:
            continue

    # Remove admin/state JSON and CSV files that contain Telegram offsets,
    # chat IDs, registry info, or historical signals.
    sensitive_files = [
        "account_report_bot_state.json",  # contains reports_bot token + offset
        "accounts_bot_state.json",
        "main_control_bot_state.json",
        "main_control_bot.lock",
        "main_process_state.json",
        "stats_bot_state.json",
        "telegram_state.json",
        "signal_log.csv",
        "signal_messages.json",
        "instances_registry.json",
    ]

    for name in sensitive_files:
        try:
            p = data_dir / name
            if p.exists():
                p.unlink()
        except Exception:
            continue


def _prune_update_tree(build_root: Path, path_spec: Optional[str]) -> None:
    """Optionally remove files outside path_spec from the build tree.

    When path_spec is None, we keep the whole tree. Otherwise we keep only:
      - pyarmor_runtime_* files (always required)
      - the exact file or directory subtree specified by path_spec.
    """

    if path_spec is None:
        return

    spec = path_spec.strip().replace("\\", "/")
    if not spec or spec.lower() in ("all", "*"):
        return

    spec_prefix = spec.rstrip("/")

    def _keep(rel_path: str) -> bool:
        norm = rel_path.replace("\\", "/")
        # Always keep PyArmor runtime files
        if norm.startswith("pyarmor_runtime_"):
            return True
        if norm == spec_prefix:
            return True
        if norm.startswith(spec_prefix + "/"):
            return True
        return False

    # Walk bottom-up so we can safely remove empty directories
    for root, dirs, files in os.walk(build_root, topdown=False):
        rel_dir = Path(root).relative_to(build_root)

        # Remove files that are outside the filtered subtree
        for name in files:
            if str(rel_dir) == ".":
                rel_path = name
            else:
                rel_path = f"{rel_dir.as_posix()}/{name}"
            if not _keep(rel_path):
                try:
                    (Path(root) / name).unlink()
                except Exception:
                    continue

        # Remove empty directories (except the root)
        try:
            if root != str(build_root) and not os.listdir(root):
                Path(root).rmdir()
        except Exception:
            continue


def _run_pyarmor(work_root: Path, output_root: Path) -> Dict[str, Any]:
    _safe_mkdir(output_root)

    cmd = [
        sys.executable,
        "-m",
        "pyarmor.cli",
        "gen",
        "-O",
        str(output_root),
        "-r",
        "main.py",
        "monitoring/main_control_bot.py",
        "instance_security.py",
        "data",
        "execution",
        "features",
        "ml",
        "monitoring",
        "signals",
        "backtest",
        "risk",
        "config",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(e),
        }

    ok = proc.returncode == 0

    # Heuristic: even if return code is non-zero, consider PyArmor successful if
    # it produced an obfuscated main.py with the standard PyArmor header. Some
    # environments may return a non-zero status while still writing all output
    # files correctly.
    if not ok:
        try:
            main_out = output_root / "main.py"
            if main_out.exists():
                header = (main_out.read_text(encoding="utf-8") or "").lstrip()
                if header.startswith("# Pyarmor"):
                    ok = True
        except Exception:
            pass

    error: Optional[str]
    if ok:
        error = None
    else:
        # Prefer stderr, fall back to stdout, then a generic message.
        error = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        if not error:
            error = f"pyarmor exited with code {proc.returncode}"
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "error": error,
    }


def _write_requirements(dest_root: Path) -> Dict[str, Any]:
    req_path = dest_root / "requirements.txt"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return {"ok": False, "path": str(req_path), "error": str(e)}

    if proc.returncode != 0:
        return {
            "ok": False,
            "path": str(req_path),
            "error": proc.stderr.strip(),
        }

    try:
        req_path.write_text(proc.stdout, encoding="utf-8")
    except Exception as e:
        return {"ok": False, "path": str(req_path), "error": str(e)}

    return {"ok": True, "path": str(req_path)}


def _install_libs(req_path: Path, libs_dir: Path) -> Dict[str, Any]:
    _safe_mkdir(libs_dir)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(req_path),
                "--target",
                str(libs_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}

    ok = proc.returncode == 0
    return {
        "ok": ok,
        "error": None if ok else proc.stderr.strip(),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def create_instance_project(
    label: Optional[str] = None,
    description: Optional[str] = None,
    requested_by_chat_id: Optional[int] = None,
) -> Dict[str, Any]:
    instance_id = _generate_instance_id()

    meta: Dict[str, Any] = {}
    if description:
        meta["description"] = description
    if requested_by_chat_id is not None:
        meta["requested_by_chat_id"] = requested_by_chat_id

    reg_entry = register_new_instance(instance_id, label=label, meta=meta)

    _safe_mkdir(_BUILD_ROOT)
    _safe_mkdir(_CLIENTS_ROOT)

    work_root = _BUILD_ROOT / f"src_{instance_id}"
    output_root = _CLIENTS_ROOT / instance_id

    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)

    _copy_source_tree(ROOT_DIR, work_root)
    _patch_instance_id(work_root, instance_id)
    _patch_license_overrides(work_root)
    _filter_client_configs(work_root)
    _disable_paper_trading(work_root)
    _clean_client_data_dir(work_root)

    pyarmor_result = _run_pyarmor(work_root, output_root)

    # Copy non-Python configs and data (YAML, CSV, etc.)
    _copy_non_py(work_root / "config", output_root / "config")
    _copy_non_py(work_root / "data", output_root / "data")

    req_info = _write_requirements(output_root)
    pip_info: Dict[str, Any] = {"ok": False, "error": "requirements not written"}
    if req_info.get("ok"):
        req_path = Path(req_info["path"])
        pip_info = _install_libs(req_path, output_root / "libs")

    archive_path: Optional[str] = None
    try:
        archive_base = str(output_root)
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=str(output_root))
    except Exception:
        archive_path = None

    try:
        shutil.rmtree(work_root, ignore_errors=True)
    except Exception:
        pass

    return {
        "instance_id": instance_id,
        "label": label or "",
        "registry_entry": reg_entry,
        "output_dir": str(output_root),
        "archive_path": archive_path,
        "pyarmor_ok": bool(pyarmor_result.get("ok")),
        "pyarmor_returncode": pyarmor_result.get("returncode"),
        "pyarmor_error": pyarmor_result.get("error"),
        "requirements_ok": bool(req_info.get("ok")),
        "requirements_path": req_info.get("path"),
        "pip_ok": bool(pip_info.get("ok")),
        "pip_error": pip_info.get("error"),
    }


def build_update_package(instance_id: str, path_spec: Optional[str] = None) -> Dict[str, Any]:
    """Build an obfuscated update package for a given instance.

    This re-obfuscates the current source tree for the specified instance_id
    (including license placeholders) and writes an update ZIP archive to
    updates/{instance_id}.zip. If path_spec is provided, only that file or
    directory subtree (plus the PyArmor runtime) is kept in the archive.
    """

    _safe_mkdir(_BUILD_ROOT)
    _safe_mkdir(_UPDATES_ROOT)

    work_root = _BUILD_ROOT / f"src_update_{instance_id}"
    build_root = _BUILD_ROOT / f"build_update_{instance_id}"

    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    if build_root.exists():
        shutil.rmtree(build_root, ignore_errors=True)

    _copy_source_tree(ROOT_DIR, work_root)
    _patch_instance_id(work_root, instance_id)
    _patch_license_overrides(work_root)
    _filter_client_configs(work_root)
    _disable_paper_trading(work_root)
    _clean_client_data_dir(work_root)

    pyarmor_result = _run_pyarmor(work_root, build_root)

    # Copy non-Python configs and data (YAML, CSV, etc.) – these will only be
    # applied on the client if update.include_config / include_data is enabled.
    _copy_non_py(work_root / "config", build_root / "config")
    _copy_non_py(work_root / "data", build_root / "data")

    _prune_update_tree(build_root, path_spec)

    archive_path: Optional[str] = None
    zip_error: Optional[str] = None
    try:
        archive_base = str(_UPDATES_ROOT / instance_id)
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=str(build_root))
    except Exception as e:
        zip_error = str(e)

    try:
        shutil.rmtree(work_root, ignore_errors=True)
    except Exception:
        pass
    try:
        shutil.rmtree(build_root, ignore_errors=True)
    except Exception:
        pass

    ok = bool(pyarmor_result.get("ok")) and bool(archive_path)

    return {
        "instance_id": instance_id,
        "ok": ok,
        "archive_path": archive_path,
        "pyarmor_ok": bool(pyarmor_result.get("ok")),
        "pyarmor_error": pyarmor_result.get("error") or "",
        "zip_error": zip_error or "",
        "path_spec": path_spec or "",
    }
