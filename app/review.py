"""
Phase 4, part of §4.5 "Review API": the business logic behind approve /
reject / inspect-why, kept separate from app/main.py's HTTP layer (shared
requirement #1, layered architecture — swap FastAPI for Flask/Express and
nothing here changes).

Two behaviors worth calling out because they're deliberate, not accidental:

  - `get_or_compute_suggestions` is idempotent by default: calling
    GET /posts/:id/images twice does NOT insert duplicate Suggestion rows.
    A post only gets re-ranked when explicitly asked (`recompute=True`),
    and even then, already-reviewed suggestions are preserved so the audit
    trail in `reviews` never loses history out from under a human decision.
    (Shared requirement #5, idempotency where it matters.)

  - `create_review` enforces one review per suggestion at the data layer
    (Review.suggestion_id is unique) and raises a specific exception the
    API layer turns into 409 Conflict — a retried "approve" click can't
    silently double-write or silently overwrite a prior "reject".
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.matching import match_and_guard_post
from app.models import Image, Post, Review, ReviewDecision, Suggestion


class PostNotFound(Exception):
    pass


class SuggestionNotFound(Exception):
    pass


class AlreadyReviewed(Exception):
    def __init__(self, existing_review: Review):
        self.existing_review = existing_review
        super().__init__(f"Suggestion already reviewed as '{existing_review.decision.value}'.")


def get_post_or_404(session: Session, post_identifier: str) -> Post:
    """Looks up by DB id OR slug, so both UUIDs and human-readable slugs work in the URL."""
    post = (
        session.query(Post)
        .filter(or_(Post.id == post_identifier, Post.slug == post_identifier))
        .one_or_none()
    )
    if post is None:
        raise PostNotFound(f"No post with id or slug '{post_identifier}'.")
    return post


def get_or_compute_suggestions(session: Session, post: Post, *, recompute: bool = False) -> list[Suggestion]:
    existing = (
        session.query(Suggestion)
        .filter_by(post_id=post.id)
        .order_by(Suggestion.rank)
        .all()
    )
    if existing and not recompute:
        return existing

    if existing and recompute:
        reviewed_suggestion_ids = {
            row.suggestion_id
            for row in session.query(Review.suggestion_id)
            .filter(Review.suggestion_id.in_([s.id for s in existing]))
            .all()
        }
        for suggestion in existing:
            if suggestion.id not in reviewed_suggestion_ids:
                session.delete(suggestion)
        session.flush()

    match_and_guard_post(session, post)  # persists fresh Suggestion rows
    session.commit()

    return (
        session.query(Suggestion)
        .filter_by(post_id=post.id)
        .order_by(Suggestion.rank)
        .all()
    )


@dataclass
class MatchSummary:
    status: str  # "matched" | "no_confident_match"
    best: Suggestion | None
    explanation: str


def summarize(suggestions: list[Suggestion]) -> MatchSummary:
    approved = [s for s in suggestions if s.guard_status == "approved"]
    if approved:
        best = min(approved, key=lambda s: s.rank)
        return MatchSummary(status="matched", best=best, explanation=best.guard_reason)

    if not suggestions:
        return MatchSummary(
            status="no_confident_match", best=None,
            explanation="No candidate images are available (corpus empty, or nothing embedded yet).",
        )

    top = min(suggestions, key=lambda s: s.rank)
    return MatchSummary(status="no_confident_match", best=None, explanation=f"No confident match found. {top.guard_reason}")


def get_suggestion_or_404(session: Session, suggestion_id: str) -> Suggestion:
    suggestion = session.query(Suggestion).filter_by(id=suggestion_id).one_or_none()
    if suggestion is None:
        raise SuggestionNotFound(f"No suggestion with id '{suggestion_id}'.")
    return suggestion


def create_review(
    session: Session, *, suggestion_id: str, decision: str, reviewer: str, notes: str | None
) -> Review:
    suggestion = get_suggestion_or_404(session, suggestion_id)  # 404 before 409

    existing = session.query(Review).filter_by(suggestion_id=suggestion.id).one_or_none()
    if existing is not None:
        raise AlreadyReviewed(existing)

    review = Review(
        suggestion_id=suggestion.id,
        decision=ReviewDecision(decision),
        reviewer=reviewer or "local-reviewer",
        notes=notes,
    )
    session.add(review)
    session.commit()
    session.refresh(review)
    return review