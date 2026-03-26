from __future__ import annotations
import os
import jwt
import json
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import request, redirect, url_for, session, g, jsonify, current_app
from webapp.database import SessionLocal
from webapp.models import User, AuditLog

TOKEN_TTL_HOURS = 8

# ─── secret key (consistent with app.py file-based key) ──────────────────────

def _load_secret_key() -> str:
    """Load secret key: env var > file > hardcoded fallback (same logic as app.py)."""
    env_key = os.environ.get("WEBAPP_SECRET")
    if env_key:
        return env_key
    from pathlib import Path
    key_file = Path(__file__).resolve().parent.parent / "data" / "webapp_secret.key"
    if key_file.exists():
        return key_file.read_text().strip()
    return "change-me-in-production-please-use-long-random-key"

SECRET_KEY = _load_secret_key()

# ─── token helpers ────────────────────────────────────────────────────────────

def create_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"],
                          options={"verify_sub": False})
    except Exception:
        return None


# ─── current user ─────────────────────────────────────────────────────────────

def get_current_user() -> User | None:
    token = request.cookies.get("token") or session.get("token")
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(id=payload["sub"], is_active=True).first()
        return user
    except Exception:
        return None
    finally:
        db.close()


# ─── decorators ───────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            if request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("auth.login_page"))
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def roles_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = get_current_user()
            if not user:
                if request.is_json:
                    return jsonify({"error": "Unauthorized"}), 401
                return redirect(url_for("auth.login_page"))
            if user.role not in roles:
                if request.is_json:
                    return jsonify({"error": "Forbidden"}), 403
                return redirect(url_for("dashboard.overview"))
            g.current_user = user
            return f(*args, **kwargs)
        return decorated
    return decorator


def superadmin_required(f):
    return roles_required("superadmin")(f)


def admin_or_above(f):
    return roles_required("superadmin", "admin")(f)


# ─── audit log ────────────────────────────────────────────────────────────────

def audit(action: str, target: str = None, details: str = None):
    db = SessionLocal()
    try:
        user = get_current_user()
        log = AuditLog(
            user_id=user.id if user else None,
            username=user.username if user else "anonymous",
            action=action,
            target=target,
            details=details,
            ip_address=request.remote_addr,
        )
        db.add(log)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
