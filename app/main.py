"""
Phase 4: the Review API (§4.5) — "a simple workflow to approve or reject a
suggested pairing and inspect why an image was selected or refused."

This file is intentionally thin: every route parses/validates the request,
calls into app/review.py or app/matching.py for the actual logic, and maps
domain exceptions to HTTP status codes. No business logic lives here
(shared requirement #1, layered architecture) — swapping FastAPI for Flask
or Express would only touch this file.

Run it:
    uvicorn app.main:app --reload
Docs (auto-generated, free):
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from app import review as review_logic
from app.api_schemas import (
    CostSummaryResponse,
    ImageOut,
    PostImagesResponse,
    PostOut,
    ReviewCreateRequest,
    ReviewOut,
    SuggestionOut,
)
from app.cost_tracker import total_estimated_cost
from app.database import SessionLocal, init_db
from app.models import ApiCallLog, Image, Post


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # safe to call repeatedly; creates tables if missing
    yield


app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    description="FlyRank capstone — Review API (Phase 4).",
    version="0.4.0",
    lifespan=lifespan,
)


def get_db():
    """Request-scoped DB session — the one place app/main.py touches persistence directly."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Health + listing
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/posts", response_model=list[PostOut])
def list_posts(db: Session = Depends(get_db)):
    return db.query(Post).order_by(Post.title).all()


@app.get("/images", response_model=list[ImageOut])
def list_images(status: str | None = Query(default=None), db: Session = Depends(get_db)):
    """Mainly useful for demo/debugging — e.g. GET /images?status=flagged to see review queue."""
    q = db.query(Image)
    if status:
        q = q.filter(Image.status == status)
    return q.order_by(Image.filename).all()


# --------------------------------------------------------------------------
# The core matching endpoint — §5 architecture diagram: GET /posts/:id/images
# --------------------------------------------------------------------------


@app.get("/posts/{post_id}/images", response_model=PostImagesResponse)
def get_post_images(
    post_id: str,
    recompute: bool = Query(
        default=False,
        description="Force re-ranking instead of reusing stored suggestions. "
        "Already-reviewed suggestions are preserved either way.",
    ),
    include_candidates: bool = Query(
        default=True,
        description="Include every ranked candidate (with rejection reasons), not just the best match.",
    ),
    db: Session = Depends(get_db),
):
    """
    Probe 2 / Probe 3 / Probe 4 all land here: the fox post's top candidate
    should be a fox image; a forced wolf candidate is visible (and rejected)
    in `candidates`; a post with nothing good enough returns
    status="no_confident_match" with `explanation` set.
    """
    try:
        post = review_logic.get_post_or_404(db, post_id)
    except review_logic.PostNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        suggestions = review_logic.get_or_compute_suggestions(db, post, recompute=recompute)
    except ValueError as e:
        # e.g. post or images not embedded yet — a client error, not a 500
        raise HTTPException(status_code=422, detail=str(e)) from e

    summary = review_logic.summarize(suggestions)

    return PostImagesResponse(
        post_id=post.id,
        post_slug=post.slug,
        post_title=post.title,
        status=summary.status,
        explanation=summary.explanation,
        best=SuggestionOut.from_orm_obj(summary.best) if summary.best else None,
        candidates=[SuggestionOut.from_orm_obj(s) for s in suggestions] if include_candidates else [],
    )


# --------------------------------------------------------------------------
# Inspecting + reviewing individual suggestions
# --------------------------------------------------------------------------


@app.get("/suggestions/{suggestion_id}", response_model=SuggestionOut)
def get_suggestion(suggestion_id: str, db: Session = Depends(get_db)):
    """Inspect why one specific candidate was selected or refused."""
    try:
        suggestion = review_logic.get_suggestion_or_404(db, suggestion_id)
    except review_logic.SuggestionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return SuggestionOut.from_orm_obj(suggestion)


@app.post("/suggestions/{suggestion_id}/review", response_model=ReviewOut, status_code=201)
def review_suggestion(suggestion_id: str, payload: ReviewCreateRequest, db: Session = Depends(get_db)):
    """
    Approve or reject a suggested pairing. Idempotency (shared requirement
    #5): a suggestion can only ever be reviewed once — a retried click (or
    a genuine double-submit) gets 409 Conflict, never a silent double-write
    or a silently overwritten decision.
    """
    try:
        review = review_logic.create_review(
            db,
            suggestion_id=suggestion_id,
            decision=payload.decision,
            reviewer=payload.reviewer or "local-reviewer",
            notes=payload.notes,
        )
    except review_logic.SuggestionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except review_logic.AlreadyReviewed as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return ReviewOut.model_validate(review)


# --------------------------------------------------------------------------
# Cost tracking — Probe 6: "check the cost log"
# --------------------------------------------------------------------------


@app.get("/costs/summary", response_model=CostSummaryResponse)
def cost_summary(db: Session = Depends(get_db)):
    rows = db.query(ApiCallLog).all()
    by_type = Counter(r.call_type for r in rows)
    return CostSummaryResponse(
        total_calls=len(rows),
        successful_calls=sum(1 for r in rows if r.succeeded),
        failed_calls=sum(1 for r in rows if not r.succeeded),
        total_estimated_cost_usd=total_estimated_cost(db),
        by_call_type=dict(by_type),
    )