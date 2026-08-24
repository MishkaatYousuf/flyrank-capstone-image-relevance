"""
Definition of Done: "Vision model produces structured output validated
against a schema; invalid responses are never trusted." These tests prove
the schema actually rejects the shapes it should.
"""
import pytest
from pydantic import ValidationError

from app.schemas import ImageCategory, ImageTags


def _valid_payload(**overrides):
    payload = {
        "subject": "red fox",
        "category": "animal",
        "attributes": ["orange fur", "wild", "forest"],
        "caption": "A red fox standing in a forest",
        "confidence": 0.94,
    }
    payload.update(overrides)
    return payload


def test_valid_payload_parses():
    tags = ImageTags.model_validate(_valid_payload())
    assert tags.subject == "red fox"
    assert tags.category == ImageCategory.ANIMAL
    assert tags.confidence == 0.94


def test_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        ImageTags.model_validate(_valid_payload(confidence=1.4))


def test_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        ImageTags.model_validate(_valid_payload(confidence=-0.1))


def test_rejects_missing_subject():
    payload = _valid_payload()
    del payload["subject"]
    with pytest.raises(ValidationError):
        ImageTags.model_validate(payload)


def test_rejects_blank_subject():
    with pytest.raises(ValidationError):
        ImageTags.model_validate(_valid_payload(subject="   "))


def test_rejects_category_outside_enum():
    with pytest.raises(ValidationError):
        ImageTags.model_validate(_valid_payload(category="spaceship"))


def test_rejects_empty_attributes():
    with pytest.raises(ValidationError):
        ImageTags.model_validate(_valid_payload(attributes=[]))


def test_rejects_attributes_all_blank():
    with pytest.raises(ValidationError):
        ImageTags.model_validate(_valid_payload(attributes=["   ", ""]))


def test_low_confidence_helper():
    tags = ImageTags.model_validate(_valid_payload(confidence=0.4))
    assert tags.is_low_confidence(threshold=0.6) is True
    assert tags.is_low_confidence(threshold=0.3) is False
