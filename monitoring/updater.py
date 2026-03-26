from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from monitoring.logger import log


ROOT_DIR = Path(__file__).resolve().parents[1]


def download_and_apply_update(
    instance_id: str,
    base_url: str,
    include_config: bool = False,
    include_data: bool = False,
) -> Dict[str, Any]:
    """Download and apply a code update for this instance.

    The update is expected to be a ZIP archive accessible at:

        {base_url.rstrip('/')}/{instance_id}.zip

    The archive may contain the full project tree or just a subset of files.
    This function will extract all entries relative to ROOT_DIR, skipping
    config/ and data/ unless explicitly allowed via flags.
    """

    result: Dict[str, Any] = {
        "ok": False,
        "downloaded": False,
        "applied": False,
        "url": None,
        "error": None,
    }

    try:
        base_url = (base_url or "").strip()
        if not base_url:
            result["error"] = "empty_base_url"
            return result

        url = base_url.rstrip("/") + f"/{instance_id}.zip"
        result["url"] = url

        try:
            resp = requests.get(url, stream=True, timeout=30)
        except Exception as e:
            result["error"] = f"request_failed: {e}"
            return result

        if resp.status_code != 200:
            result["error"] = f"http_status_{resp.status_code}"
            return result

        fd, tmp_path = tempfile.mkstemp(prefix=f"update_{instance_id}_", suffix=".zip")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)

            result["downloaded"] = True

            try:
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    for member in zf.infolist():
                        name = member.filename
                        if not name or name.endswith("/"):
                            continue

                        norm = name.replace("\\", "/")

                        if norm.startswith("config/") and not include_config:
                            continue
                        if norm.startswith("data/") and not include_data:
                            continue
                        if "__pycache__" in norm:
                            continue

                        target_path = ROOT_DIR / norm
                        target_dir = target_path.parent
                        try:
                            target_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            continue

                        try:
                            with zf.open(member) as src, open(target_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        except Exception:
                            continue

            except Exception as e:
                result["error"] = f"extract_failed: {e}"
                return result

            result["applied"] = True
            result["ok"] = True

            try:
                log(f"[UPDATE] Applied update from {url}")
            except Exception:
                pass

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        result["error"] = str(e)

    return result
