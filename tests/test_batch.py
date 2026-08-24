"""
Definition of Done:
  - "Images are processed through a batch background job with retries."
  - "Low-confidence classifications are flagged instead of accepted."
  - "Vision and embedding costs are tracked per call."

All Gemini/Ollama calls are mocked — these tests never touch the network,
never cost anything, and are fully deterministic.
"""
from __future__ import annotations

import app.batch as batch_mod
from app.models import ApiCallLog, Image, ImageStatus
from app.schemas import ImageTags
from app.vision import InvalidVisionOutput, VisionCallResult


def _make_result(confidence: float, attempt: int = 1) -> VisionCallResult:
    tags = ImageTags(
        subject="red fox",
        category="animal",
        attributes=["orange fur", "forest"],
        caption="A red fox standing in a forest",
        confidence=confidence,
    )
    return VisionCallResult(tags=tags, input_tokens=1000, output_tokens=80, raw_response="{}", attempt=attempt)


def test_high_confidence_image_is_tagged(db_session, patched_settings, fake_image_file, monkeypatch):
    monkeypatch.setattr(batch_mod, "classify_image", lambda path, attempt=1: _make_result(0.94, attempt))

    image = batch_mod._upsert_pending_image(db_session, fake_image_file, fake_image_file.parent.parent)
    status = batch_mod.process_image(db_session, image, fake_image_file)
    db_session.commit()

    assert status == ImageStatus.TAGGED
    assert image.confidence == 0.94
    assert image.subject == "red fox"

    calls = db_session.query(ApiCallLog).all()
    assert len(calls) == 1
    assert calls[0].succeeded is True
    assert calls[0].call_type == "vision"


def test_low_confidence_image_is_flagged_not_accepted(db_session, patched_settings, fake_image_file, monkeypatch):
    monkeypatch.setattr(batch_mod, "classify_image", lambda path, attempt=1: _make_result(0.42, attempt))

    image = batch_mod._upsert_pending_image(db_session, fake_image_file, fake_image_file.parent.parent)
    status = batch_mod.process_image(db_session, image, fake_image_file)

    assert status == ImageStatus.FLAGGED
    assert status != ImageStatus.TAGGED  # never silently accepted


def test_invalid_output_is_retried_then_succeeds(db_session, patched_settings, fake_image_file, monkeypatch):
    calls = {"n": 0}

    def flaky_classify(path, attempt=1):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InvalidVisionOutput("model returned malformed JSON")
        return _make_result(0.9, attempt)

    monkeypatch.setattr(batch_mod, "classify_image", flaky_classify)

    image = batch_mod._upsert_pending_image(db_session, fake_image_file, fake_image_file.parent.parent)
    status = batch_mod.process_image(db_session, image, fake_image_file)

    assert status == ImageStatus.TAGGED
    assert calls["n"] == 2  # failed once, retried, succeeded

    log_rows = db_session.query(ApiCallLog).order_by(ApiCallLog.attempt).all()
    assert [r.succeeded for r in log_rows] == [False, True]


def test_persistent_invalid_output_becomes_error_after_max_retries(
    db_session, patched_settings, fake_image_file, monkeypatch
):
    def always_invalid(path, attempt=1):
        raise InvalidVisionOutput("model never produces valid JSON")

    monkeypatch.setattr(batch_mod, "classify_image", always_invalid)

    image = batch_mod._upsert_pending_image(db_session, fake_image_file, fake_image_file.parent.parent)
    status = batch_mod.process_image(db_session, image, fake_image_file)

    assert status == ImageStatus.ERROR
    assert image.error_message is not None

    # patched_settings.max_vision_retries == 2 -> exactly 2 attempts logged, all failed
    log_rows = db_session.query(ApiCallLog).all()
    assert len(log_rows) == 2
    assert all(r.succeeded is False for r in log_rows)


def test_every_attempt_gets_a_cost_log_entry_even_on_failure(
    db_session, patched_settings, fake_image_file, monkeypatch
):
    monkeypatch.setattr(
        batch_mod, "classify_image",
        lambda path, attempt=1: (_ for _ in ()).throw(InvalidVisionOutput("bad")),
    )

    image = batch_mod._upsert_pending_image(db_session, fake_image_file, fake_image_file.parent.parent)
    batch_mod.process_image(db_session, image, fake_image_file)

    log_rows = db_session.query(ApiCallLog).all()
    assert len(log_rows) == patched_settings.max_vision_retries
    for row in log_rows:
        assert row.estimated_cost_usd is not None  # every attempt attributed, even failures


def test_upsert_is_idempotent_on_filepath(db_session, fake_image_file):
    images_dir = fake_image_file.parent.parent
    first = batch_mod._upsert_pending_image(db_session, fake_image_file, images_dir)
    db_session.commit()
    second = batch_mod._upsert_pending_image(db_session, fake_image_file, images_dir)

    assert first.id == second.id
    assert db_session.query(Image).count() == 1


def test_category_hint_comes_from_folder_name(db_session, fake_image_file):
    images_dir = fake_image_file.parent.parent  # fake_image_file is .../animal/red_fox_01.jpg
    image = batch_mod._upsert_pending_image(db_session, fake_image_file, images_dir)
    assert image.source_category_hint == "animal"
