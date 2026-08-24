"""
Pydantic models = the contract between the vision model and the rest of the
system. Nothing downstream is ever allowed to see a raw, unvalidated model
response — it either parses into one of these shapes, or it's treated as a
failure and retried/flagged (see app/vision.py and app/batch.py).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ImageCategory(str, Enum):
    """
    Kept as an open-ish enum: broad enough to cover the ~50-image demo corpus
    (§7 realistic scope: a handful of animal categories) plus a catch-all so
    the model never has to force a bad fit. Extend as your corpus grows.
    """

    ANIMAL = "animal"
    LANDSCAPE = "landscape"
    PEOPLE = "people"
    OBJECT = "object"
    OTHER = "other"


class ImageTags(BaseModel):
    """
    The structured output every vision call must conform to. Mirrors the
    schema given in the capstone brief §4:

        {
          "subject": "red fox",
          "category": "animal",
          "attributes": ["orange fur", "wild", "forest"],
          "caption": "A red fox standing in a forest",
          "confidence": 0.94
        }
    """

    subject: str = Field(
        ..., min_length=1, max_length=80,
        description="The single main subject of the image, e.g. 'red fox'.",
    )
    category: ImageCategory = Field(
        ..., description="Broad bucket the subject belongs to."
    )
    attributes: list[str] = Field(
        ..., min_length=1, max_length=8,
        description="Short descriptive attributes, e.g. ['orange fur', 'forest'].",
    )
    caption: str = Field(
        ..., min_length=1, max_length=300,
        description="One-sentence natural-language caption of the image.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="The model's own confidence in this classification, 0-1.",
    )

    @field_validator("subject", "caption")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank/whitespace")
        return v

    @field_validator("attributes")
    @classmethod
    def _attrs_not_blank(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a.strip()]
        if not cleaned:
            raise ValueError("attributes must contain at least one non-blank entry")
        return cleaned

    def is_low_confidence(self, threshold: float) -> bool:
        return self.confidence < threshold


class VisionCallResult(BaseModel):
    """
    Wraps a validated ImageTags result together with the bookkeeping the
    batch job / cost tracker need. This is what app.vision returns.
    """

    tags: ImageTags
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: str = ""
    attempt: int = 1
