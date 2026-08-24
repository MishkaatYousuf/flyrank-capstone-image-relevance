"""
DB engine + session factory. SQLite by default (zero config, $0, no Docker —
plenty for ~50 images). If you outgrow it, swap DATABASE_URL to a Postgres
URL (e.g. free Docker Postgres, or a free-tier hosted Postgres) and nothing
else in the app has to change — that's the point of going through the ORM.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call repeatedly."""
    from app import models  # noqa: F401  (ensures models are registered on Base)

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
