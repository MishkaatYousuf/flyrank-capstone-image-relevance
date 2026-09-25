"""
Phase 4: the HTTP-facing shapes for the Review API (§4.5). Kept separate
from app/schemas.py (the vision-output contract) — these describe what the
API returns and accepts, not what the vision model returns. Validation at
the boundary (shared requirement #2): a malformed request body never
reaches app/review.py, FastAPI + Pydantic reject it with a 422 first.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PostOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    title: str


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    subject: str | None
    category: str | None
    attributes: list[str] | None
    caption: str | None
    confidence: float | None


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    decision: str
    reviewer: str
    notes: str | None
    created_at: datetime


class SuggestionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    rank: int
    similarity_score: float
    guard_status: Literal["approved", "rejected"]
    guard_reason: str
    image: ImageOut
    review: ReviewOut | None = None

    @classmethod
    def from_orm_obj(cls, suggestion) -> "SuggestionOut":
        """
        Explicit constructor (called while the DB session is still open)
        instead of relying on FastAPI's automatic response_model
        conversion, which can run after the request-scoped session has
        already been closed and lazy relationships (image, review) can no
        longer be loaded.
        """
        return cls(
            id=suggestion.id,
            rank=suggestion.rank,
            similarity_score=suggestion.similarity_score,
            guard_status=suggestion.guard_status,
            guard_reason=suggestion.guard_reason,
            image=ImageOut.model_validate(suggestion.image),
            review=ReviewOut.model_validate(suggestion.review) if suggestion.review else None,
        )


class PostImagesResponse(BaseModel):
    post_id: str
    post_slug: str
    post_title: str
    status: Literal["matched", "no_confident_match"]
    explanation: str
    best: SuggestionOut | None
    candidates: list[SuggestionOut] = Field(default_factory=list)


class ReviewCreateRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=2000)


class ErrorResponse(BaseModel):
    detail: str


class CostSummaryResponse(BaseModel):
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_estimated_cost_usd: float
    by_call_type: dict[str, int]