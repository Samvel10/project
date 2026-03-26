from __future__ import annotations
import bcrypt
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from webapp.database import Base

ROLES = ("superadmin", "admin", "user")


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    email         = Column(String(128), unique=True, nullable=True)
    password_hash = Column(String(256), nullable=False)
    role          = Column(Enum(*ROLES, name="role_enum"), nullable=False, default="user")
    is_active     = Column(Boolean, default=True, nullable=False)
    created_by    = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login    = Column(DateTime, nullable=True)
    failed_attempts = Column(Integer, default=0)
    locked_until  = Column(DateTime, nullable=True)

    audit_logs    = relationship("AuditLog", back_populates="user", foreign_keys="AuditLog.user_id")

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt(rounds=12)
        ).decode("utf-8")

    def check_password(self, password: str) -> bool:
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"),
                self.password_hash.encode("utf-8"),
            )
        except Exception:
            return False

    def is_locked(self) -> bool:
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until.replace(tzinfo=timezone.utc)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLog(Base):
    __tablename__ = "audit_log"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    username   = Column(String(64), nullable=True)
    action     = Column(String(128), nullable=False)
    target     = Column(String(256), nullable=True)
    details    = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="audit_logs", foreign_keys=[user_id])


class Invitation(Base):
    __tablename__ = "invitations"

    id         = Column(Integer, primary_key=True, index=True)
    token      = Column(String(128), unique=True, nullable=False, index=True)
    role       = Column(Enum(*ROLES, name="inv_role_enum"), nullable=False, default="user")
    email      = Column(String(128), nullable=True)           # optional pre-fill
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_by_name = Column(String(64), nullable=True)
    expires_at = Column(DateTime, nullable=False)
    used_at    = Column(DateTime, nullable=True)
    used_by    = Column(Integer, ForeignKey("users.id"), nullable=True)

    def is_valid(self) -> bool:
        if self.used_at:
            return False
        return datetime.now(timezone.utc) < self.expires_at.replace(tzinfo=timezone.utc)


class ConfigHistory(Base):
    __tablename__ = "config_history"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    username    = Column(String(64), nullable=True)
    config_file = Column(String(128), nullable=False)
    field_path  = Column(String(256), nullable=False)
    old_value   = Column(Text, nullable=True)
    new_value   = Column(Text, nullable=True)
    applied_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    rolled_back = Column(Boolean, default=False)
