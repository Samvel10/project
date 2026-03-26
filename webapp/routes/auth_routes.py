from __future__ import annotations
import os
import secrets
from datetime import datetime, timezone, timedelta
from flask import (Blueprint, render_template, request, redirect, url_for,
                   make_response, flash, session, g, current_app, jsonify)
from webapp.database import SessionLocal
from webapp.models import User, Invitation
from webapp.auth import create_token, get_current_user, audit, login_required, admin_or_above

auth_bp = Blueprint("auth", __name__)

MAX_FAILED      = 5
LOCKOUT_MINUTES = 15
INVITE_HOURS    = 72   # invitation link valid for 72 hours

# ── Google OAuth setup (lazy) ─────────────────────────────────────────────────
def _get_google():
    from authlib.integrations.flask_client import OAuth
    oauth = OAuth(current_app)
    client_id     = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None
    google = oauth.register(
        name="google",
        client_id=client_id,
        client_secret=client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return google


# ── Login ─────────────────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET", "POST"])
def login_page():
    if get_current_user():
        return redirect(url_for("dashboard.overview"))

    google_enabled = bool(os.environ.get("GOOGLE_CLIENT_ID"))
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(username=username).first()
            if not user or not user.is_active:
                error = "Invalid credentials"
            elif user.is_locked():
                error = "Account locked — too many failed attempts. Try again later."
            elif not user.check_password(password):
                user.failed_attempts = (user.failed_attempts or 0) + 1
                if user.failed_attempts >= MAX_FAILED:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
                    error = f"Account locked for {LOCKOUT_MINUTES} minutes."
                else:
                    left = MAX_FAILED - user.failed_attempts
                    error = f"Invalid credentials ({left} attempt{'s' if left != 1 else ''} left)"
                db.commit()
            else:
                user.failed_attempts = 0
                user.locked_until    = None
                user.last_login      = datetime.now(timezone.utc)
                db.commit()
                token = create_token(user)
                session["token"] = token
                audit("LOGIN", username)
                # 303 See Other forces browser to GET the redirect target
                resp = make_response(redirect(url_for("dashboard.overview"), 303))
                resp.set_cookie("token", token, httponly=True, samesite="Lax", max_age=8*3600)
                return resp
        finally:
            db.close()

    return render_template("login.html", error=error, google_enabled=google_enabled)


@auth_bp.route("/logout")
def logout():
    audit("LOGOUT")
    session.clear()
    resp = make_response(redirect(url_for("auth.login_page"), 303))
    resp.delete_cookie("token")
    return resp


# ── Google OAuth ──────────────────────────────────────────────────────────────

@auth_bp.route("/auth/google")
def google_login():
    google = _get_google()
    if not google:
        flash("Google login is not configured on this server.", "error")
        return redirect(url_for("auth.login_page"))
    callback_url = url_for("auth.google_callback", _external=True)
    return google.authorize_redirect(callback_url)


@auth_bp.route("/auth/google/callback")
def google_callback():
    google = _get_google()
    if not google:
        flash("Google login is not configured.", "error")
        return redirect(url_for("auth.login_page"))

    try:
        token   = google.authorize_access_token()
        userinfo = token.get("userinfo") or google.userinfo()
        email    = userinfo.get("email", "").lower().strip()
        name     = userinfo.get("name") or userinfo.get("given_name") or email.split("@")[0]
        google_id = userinfo.get("sub")
    except Exception as e:
        flash(f"Google login failed: {e}", "error")
        return redirect(url_for("auth.login_page"))

    if not email:
        flash("Could not get email from Google account.", "error")
        return redirect(url_for("auth.login_page"))

    db = SessionLocal()
    try:
        # Find by email
        user = db.query(User).filter_by(email=email).first()
        if not user:
            # Auto-create with 'user' role (lowest privilege)
            username = email.split("@")[0]
            # Make username unique
            base = username
            suffix = 1
            while db.query(User).filter_by(username=username).first():
                username = f"{base}{suffix}"
                suffix += 1
            user = User(username=username, email=email, role="user")
            user.set_password(secrets.token_hex(32))  # random unusable password
            db.add(user)
            db.flush()
            audit("GOOGLE_REGISTER", email)
        if not user.is_active:
            flash("Your account is disabled.", "error")
            return redirect(url_for("auth.login_page"))

        user.last_login = datetime.now(timezone.utc)
        db.commit()

        t = create_token(user)
        session["token"] = t
        audit("GOOGLE_LOGIN", email)
        resp = make_response(redirect(url_for("dashboard.overview"), 303))
        resp.set_cookie("token", t, httponly=True, samesite="Lax", max_age=8*3600)
        return resp
    finally:
        db.close()


# ── Invitation system ─────────────────────────────────────────────────────────

@auth_bp.route("/admin/invite", methods=["GET", "POST"])
@admin_or_above
def create_invitation():
    if request.method == "POST":
        role  = request.form.get("role", "user")
        email = request.form.get("email", "").strip() or None

        # Admins cannot invite superadmins
        if role == "superadmin" and g.current_user.role != "superadmin":
            flash("Only superadmin can invite superadmins.", "error")
            return redirect(url_for("auth.manage_users"))

        token = secrets.token_urlsafe(32)
        db = SessionLocal()
        try:
            inv = Invitation(
                token=token,
                role=role,
                email=email,
                created_by=g.current_user.id,
                created_by_name=g.current_user.username,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=INVITE_HOURS),
            )
            db.add(inv)
            db.commit()
            audit("CREATE_INVITE", f"role={role}", f"email={email}")
        finally:
            db.close()

        link = url_for("auth.register_with_token", token=token, _external=True)
        flash(f"Invitation link created (valid {INVITE_HOURS}h): {link}", "success")
        return redirect(url_for("auth.manage_users"))

    return redirect(url_for("auth.manage_users"))


@auth_bp.route("/register/<token>", methods=["GET", "POST"])
def register_with_token(token: str):
    db = SessionLocal()
    try:
        inv = db.query(Invitation).filter_by(token=token).first()
        if not inv or not inv.is_valid():
            return render_template("register.html", error="This invitation link is invalid or has expired.", inv=None, token=token)

        error = None
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirm  = request.form.get("confirm_password", "")
            email    = request.form.get("email", "").strip() or None

            if len(username) < 3:
                error = "Username must be at least 3 characters"
            elif len(password) < 10:
                error = "Password must be at least 10 characters"
            elif password != confirm:
                error = "Passwords do not match"
            elif db.query(User).filter_by(username=username).first():
                error = f"Username '{username}' is already taken"
            else:
                user = User(username=username, email=email, role=inv.role, created_by=inv.created_by)
                user.set_password(password)
                db.add(user)
                db.flush()
                inv.used_at = datetime.now(timezone.utc)
                inv.used_by = user.id
                db.commit()
                audit("REGISTER_VIA_INVITE", username, f"role={inv.role} invited_by={inv.created_by_name}")
                t = create_token(user)
                session["token"] = t
                resp = make_response(redirect(url_for("dashboard.overview"), 303))
                resp.set_cookie("token", t, httponly=True, samesite="Lax", max_age=8 * 3600)
                return resp

        return render_template("register.html", inv=inv, token=token, error=error)
    finally:
        db.close()


# ── Change password ───────────────────────────────────────────────────────────

@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    error = success = None
    if request.method == "POST":
        current_pw = request.form.get("current_password", "")
        new_pw     = request.form.get("new_password", "")
        confirm    = request.form.get("confirm_password", "")
        if len(new_pw) < 10:
            error = "Password must be at least 10 characters"
        elif new_pw != confirm:
            error = "Passwords do not match"
        elif not g.current_user.check_password(current_pw):
            error = "Current password is incorrect"
        else:
            db = SessionLocal()
            try:
                u = db.query(User).get(g.current_user.id)
                u.set_password(new_pw)
                db.commit()
                audit("CHANGE_PASSWORD", g.current_user.username)
                success = "Password changed successfully"
            finally:
                db.close()
    return render_template("change_password.html", error=error, success=success, user=g.current_user)


# ── User management ───────────────────────────────────────────────────────────

@auth_bp.route("/admin/users")
@admin_or_above
def manage_users():
    db = SessionLocal()
    try:
        users       = db.query(User).order_by(User.created_at.desc()).all()
        invitations = db.query(Invitation).order_by(Invitation.expires_at.desc()).limit(20).all()
        google_enabled = bool(os.environ.get("GOOGLE_CLIENT_ID"))
        return render_template("admin/users.html",
                               users=users, invitations=invitations,
                               user=g.current_user, google_enabled=google_enabled,
                               invite_hours=INVITE_HOURS)
    finally:
        db.close()


@auth_bp.route("/admin/users/create", methods=["POST"])
@admin_or_above
def create_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role     = request.form.get("role", "user")
    email    = request.form.get("email", "").strip() or None

    if role == "superadmin" and g.current_user.role != "superadmin":
        flash("Only superadmin can create superadmin accounts", "error")
        return redirect(url_for("auth.manage_users"))
    if not username or len(password) < 10:
        flash("Username required and password must be ≥10 characters", "error")
        return redirect(url_for("auth.manage_users"))

    db = SessionLocal()
    try:
        if db.query(User).filter_by(username=username).first():
            flash(f"Username '{username}' already exists", "error")
            return redirect(url_for("auth.manage_users"))
        u = User(username=username, email=email, role=role, created_by=g.current_user.id)
        u.set_password(password)
        db.add(u)
        db.commit()
        audit("CREATE_USER", username, f"role={role}")
        flash(f"User '{username}' created", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error: {e}", "error")
    finally:
        db.close()
    return redirect(url_for("auth.manage_users"))


@auth_bp.route("/admin/users/<int:user_id>/toggle", methods=["POST"])
@admin_or_above
def toggle_user(user_id):
    db = SessionLocal()
    try:
        u = db.query(User).get(user_id)
        if not u:
            flash("User not found", "error")
        elif u.id == g.current_user.id:
            flash("Cannot deactivate yourself", "error")
        elif u.role == "superadmin" and g.current_user.role != "superadmin":
            flash("Cannot modify superadmin", "error")
        else:
            u.is_active = not u.is_active
            db.commit()
            audit("TOGGLE_USER", u.username, f"active={u.is_active}")
            flash(f"User '{u.username}' {'activated' if u.is_active else 'deactivated'}", "success")
    finally:
        db.close()
    return redirect(url_for("auth.manage_users"))


@auth_bp.route("/admin/users/<int:user_id>/reset-password", methods=["POST"])
@admin_or_above
def reset_user_password(user_id):
    new_pw = request.form.get("new_password", "")
    if len(new_pw) < 10:
        flash("Password must be ≥10 characters", "error")
        return redirect(url_for("auth.manage_users"))
    db = SessionLocal()
    try:
        u = db.query(User).get(user_id)
        if not u:
            flash("User not found", "error")
        elif u.role == "superadmin" and g.current_user.role != "superadmin":
            flash("Cannot modify superadmin", "error")
        else:
            u.set_password(new_pw)
            db.commit()
            audit("RESET_PASSWORD", u.username)
            flash(f"Password reset for '{u.username}'", "success")
    finally:
        db.close()
    return redirect(url_for("auth.manage_users"))


@auth_bp.route("/admin/users/<int:user_id>/rename", methods=["POST"])
@admin_or_above
def rename_user(user_id):
    new_name = request.form.get("new_username", "").strip()
    me = g.current_user
    if len(new_name) < 3:
        flash("Username must be at least 3 characters", "error")
        return redirect(url_for("auth.manage_users"))
    db = SessionLocal()
    try:
        u = db.query(User).get(user_id)
        if not u:
            flash("User not found", "error")
        elif u.role == "superadmin" and me.role != "superadmin":
            flash("Cannot rename a superadmin account", "error")
        elif me.role == "admin" and u.id != me.id and u.role != "user":
            flash("Admins can only rename themselves or regular users", "error")
        elif db.query(User).filter(User.username == new_name, User.id != u.id).first():
            flash(f"Username '{new_name}' is already taken", "error")
        else:
            old_name = u.username
            u.username = new_name
            db.commit()
            audit("RENAME_USER", old_name, f"new_username={new_name}")
            flash(f"Username changed from '{old_name}' to '{new_name}'", "success")
    finally:
        db.close()
    return redirect(url_for("auth.manage_users"))
