from __future__ import annotations
import os
import secrets
from pathlib import Path
from flask import Flask, redirect, url_for

ROOT_DIR = Path(__file__).resolve().parent.parent


def _get_or_create_secret_key() -> str:
    """Load persistent secret key from disk, or generate once and save."""
    env_key = os.environ.get("WEBAPP_SECRET")
    if env_key:
        return env_key
    key_file = ROOT_DIR / "data" / "webapp_secret.key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    key_file.chmod(0o600)
    return key


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "webapp" / "templates"),
        static_folder=str(ROOT_DIR / "webapp" / "static"),
    )
    app.secret_key = _get_or_create_secret_key()
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB max upload

    # Init database
    from webapp.database import init_db
    init_db()

    # Seed superadmin if no users exist
    _seed_superadmin()

    # Register blueprints
    from webapp.routes.auth_routes      import auth_bp
    from webapp.routes.dashboard_routes import dashboard_bp
    from webapp.routes.financial_routes import financial_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(financial_bp)

    # Jinja2 filter: format millisecond timestamp to readable datetime
    @app.template_filter("fmt_ts")
    def fmt_ts(ms):
        try:
            import datetime
            return datetime.datetime.utcfromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ms)[:16] if ms else "—"

    @app.route("/")
    def root():
        return redirect(url_for("dashboard.overview"))

    @app.context_processor
    def inject_globals():
        from webapp.auth import get_current_user
        from datetime import datetime
        return {"current_user": get_current_user(), "now": datetime.utcnow()}

    return app


def _seed_superadmin():
    """Create default superadmin if no users exist."""
    from webapp.database import SessionLocal
    from webapp.models import User
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            u = User(username="superadmin", role="superadmin", email=None)
            u.set_password("Admin@123456")
            db.add(u)
            db.commit()
            print("[WEBAPP] Created default superadmin — username: superadmin, password: Admin@123456")
            print("[WEBAPP] CHANGE THIS PASSWORD IMMEDIATELY after first login!")
    except Exception as e:
        db.rollback()
        print(f"[WEBAPP] Seed error: {e}")
    finally:
        db.close()
