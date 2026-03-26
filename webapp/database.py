import os
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── PostgreSQL (primary) ───────────────────────────────────────────────────────
PG_HOST     = os.environ.get("DB_HOST",     "127.0.0.1")
PG_PORT     = os.environ.get("DB_PORT",     "5432")
PG_NAME     = os.environ.get("DB_NAME",     "tradebotdb")
PG_USER     = os.environ.get("DB_USER",     "tradebot")
PG_PASSWORD = os.environ.get("DB_PASSWORD", "TradeBotDB2026")

DATABASE_URL = (
    f"postgresql+psycopg2://{quote_plus(PG_USER)}:{quote_plus(PG_PASSWORD)}"
    f"@{PG_HOST}:{PG_PORT}/{PG_NAME}"
)

# ── SQLite fallback (if Postgres is unavailable) ───────────────────────────────
_SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "webapp.db"
_SQLITE_URL  = f"sqlite:///{_SQLITE_PATH}"

def _make_engine():
    try:
        eng = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5,
                            max_overflow=10, echo=False)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return eng, "postgresql"
    except Exception as e:
        print(f"[DB] PostgreSQL unavailable ({e}), falling back to SQLite")
        _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(_SQLITE_URL,
                            connect_args={"check_same_thread": False}, echo=False)
        return eng, "sqlite"

engine, _DB_BACKEND = _make_engine()
print(f"[DB] Using backend: {_DB_BACKEND}")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from webapp.models import User, AuditLog, ConfigHistory, Invitation  # noqa: F401
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
