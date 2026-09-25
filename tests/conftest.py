from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base


@pytest.fixture()
def db_session():
    """A fresh in-memory SQLite DB per test — fast, isolated, no leftover state."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    from app import models  # noqa: F401  ensure models are registered on Base

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def fast_test_settings():
    """Test-tuned settings: fewer retries, no real backoff delay."""
    from app.config import settings as real_settings

    return dataclasses.replace(
        real_settings,
        max_vision_retries=2,
        low_confidence_threshold=0.6,
        vision_retry_backoff_seconds=0.0,
        vision_provider="gemini",
        gemini_model="gemini-2.5-flash",
        max_embedding_retries=2,
        embedding_retry_backoff_seconds=0.0,
        similarity_threshold=0.55,
        match_top_k=5,
    )


@pytest.fixture()
def patched_settings(monkeypatch, fast_test_settings):
    """Point app.batch, app.matching, app.vision, app.cost_tracker at the fast test settings."""
    import app.batch as batch_mod
    import app.cost_tracker as cost_mod
    import app.matching as matching_mod

    monkeypatch.setattr(batch_mod, "settings", fast_test_settings)
    monkeypatch.setattr(cost_mod, "settings", fast_test_settings)
    monkeypatch.setattr(matching_mod, "settings", fast_test_settings)
    monkeypatch.setattr(batch_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(matching_mod.time, "sleep", lambda *_: None)
    return fast_test_settings


@pytest.fixture()
def fake_image_file(tmp_path: Path) -> Path:
    """A stand-in image file — content doesn't matter, classify_image is mocked."""
    d = tmp_path / "animal"
    d.mkdir()
    f = d / "red_fox_01.jpg"
    f.write_bytes(b"not a real jpeg, but classify_image is mocked in these tests")
    return f


@pytest.fixture()
def api_client(monkeypatch):
    """
    A FastAPI TestClient wired to its own isolated in-memory DB — no shared
    state with other tests, no network, and no touching the real
    capstone.db file on disk.
    """
    from fastapi.testclient import TestClient

    from app import models  # noqa: F401  ensure models are registered on Base
    import app.main as main_mod

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared connection — required so the
        # in-memory DB survives across the multiple sessions/threads the
        # TestClient's requests each open (otherwise each new connection
        # to ":memory:" gets its OWN empty database and "no such table").
    )
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Tables are already created above; skip the real init_db() so tests
    # never touch the real DATABASE_URL / capstone.db file.
    monkeypatch.setattr(main_mod, "init_db", lambda: None)
    main_mod.app.dependency_overrides[main_mod.get_db] = override_get_db

    client = TestClient(main_mod.app)
    client.test_sessionmaker = TestSessionLocal  # tests use this to seed data directly
    yield client

    main_mod.app.dependency_overrides.clear()