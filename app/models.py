"""
Database models — see DESIGN.md for the full ER sketch and rationale.

Populated in Phase 2:  Image, ApiCallLog
Populated in Phase 3:  ImageEmbedding, PostEmbedding, Suggestion
Populated in Phase 4:  Post (seeded), Review

Keeping the whole schema here from Phase 1 means later phases are additive
(no migrations rewriting Phase 2 tables) and a reviewer can see the intended
shape of the whole system in one file.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ImageStatus(str, enum.Enum):
    PENDING = "pending"        # queued, not yet processed
    TAGGED = "tagged"          # schema-valid tags, confidence OK
    FLAGGED = "flagged"        # schema-valid tags, but LOW confidence — needs human review
    ERROR = "error"            # exhausted retries without a valid response


class ReviewDecision(str, enum.Enum):
    APPROVE = "approve"
    REJECT = "reject"


# --------------------------------------------------------------------------
# Phase 2 tables
# --------------------------------------------------------------------------


class Image(Base):
    """One row per source image in the corpus."""

    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)

    # Expected category from the download manifest (if seeded from Pexels
    # search terms) — used later as a sanity check / eval label, never as a
    # substitute for actually running the vision model.
    source_category_hint: Mapped[str | None] = mapped_column(String(80), nullable=True)

    status: Mapped[ImageStatus] = mapped_column(
        Enum(ImageStatus), nullable=False, default=ImageStatus.PENDING
    )

    # Validated ImageTags fields, flattened for easy querying.
    subject: Mapped[str | None] = mapped_column(String(80), nullable=True)
    category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    attributes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Full raw model response kept for debugging/audit — never trusted
    # directly, only the validated fields above are used downstream.
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    embeddings: Mapped[list["ImageEmbedding"]] = relationship(back_populates="image")
    suggestions: Mapped[list["Suggestion"]] = relationship(back_populates="image")

    __table_args__ = (
        Index("ix_images_status", "status"),
        Index("ix_images_category", "category"),
    )


class ApiCallLog(Base):
    """
    One row per AI API call (vision or embedding). This is the cost-tracking
    table required by §4.4 / Probe 6 — every call attributed with a cost
    entry, even though the free tier bills $0.
    """

    __tablename__ = "api_call_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    call_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "vision" | "embedding"
    provider: Mapped[str] = mapped_column(String(20), nullable=False)   # "gemini" | "ollama"
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)

    # What the call was for (nullable — embeddings may reference a post instead)
    image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("images.id"), nullable=True
    )
    post_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("posts.id"), nullable=True
    )

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    actual_billed_usd: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 on free tier

    succeeded: Mapped[bool] = mapped_column(default=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_api_call_log_created_at", "created_at"),
        Index("ix_api_call_log_call_type", "call_type"),
    )


# --------------------------------------------------------------------------
# Phase 3 tables (schema defined now, populated later)
# --------------------------------------------------------------------------


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(String(36), ForeignKey("images.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    # Stored as JSON list of floats. At ~50 images this is simpler and just
    # as fast as pgvector; swap to pgvector only if the corpus grows a lot.
    vector: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    image: Mapped["Image"] = relationship(back_populates="embeddings")

    __table_args__ = (Index("ix_image_embeddings_image_id", "image_id"),)


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    embeddings: Mapped[list["PostEmbedding"]] = relationship(back_populates="post")
    suggestions: Mapped[list["Suggestion"]] = relationship(back_populates="post")


class PostEmbedding(Base):
    __tablename__ = "post_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    post_id: Mapped[str] = mapped_column(String(36), ForeignKey("posts.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    vector: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    post: Mapped["Post"] = relationship(back_populates="embeddings")

    __table_args__ = (Index("ix_post_embeddings_post_id", "post_id"),)


class Suggestion(Base):
    """One ranked image candidate for one post, after the mismatch guard."""

    __tablename__ = "suggestions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    post_id: Mapped[str] = mapped_column(String(36), ForeignKey("posts.id"), nullable=False)
    image_id: Mapped[str] = mapped_column(String(36), ForeignKey("images.id"), nullable=False)

    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    guard_status: Mapped[str] = mapped_column(String(20), nullable=False)  # approved|rejected|no_match
    guard_reason: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    post: Mapped["Post"] = relationship(back_populates="suggestions")
    image: Mapped["Image"] = relationship(back_populates="suggestions")
    review: Mapped["Review | None"] = relationship(back_populates="suggestion", uselist=False)

    __table_args__ = (
        Index("ix_suggestions_post_id", "post_id"),
        Index("ix_suggestions_image_id", "image_id"),
    )


# --------------------------------------------------------------------------
# Phase 4 table
# --------------------------------------------------------------------------


class Review(Base):
    """Human approve/reject decision on a suggestion (the Review API, §4.5)."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    suggestion_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("suggestions.id"), nullable=False, unique=True
    )
    decision: Mapped[ReviewDecision] = mapped_column(Enum(ReviewDecision), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(100), default="local-reviewer")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    suggestion: Mapped["Suggestion"] = relationship(back_populates="review")
