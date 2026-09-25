"""
Definition of Done:
  - "API endpoints validated; the review workflow (approve / reject /
    inspect why) exists."
  - Shared requirement #5, idempotency: "the retried action happens once."

These test app/review.py directly (no HTTP layer involved) — the logic
should be correct independent of which web framework sits on top of it.
"""
from __future__ import annotations

import pytest

from app.models import Image, ImageEmbedding, ImageStatus, Post, PostEmbedding, Review, Suggestion
from app.review import (
    AlreadyReviewed,
    PostNotFound,
    SuggestionNotFound,
    create_review,
    get_or_compute_suggestions,
    get_post_or_404,
    get_suggestion_or_404,
    summarize,
)

MODEL = "test-embedding-model"


def _seed_fox_post_with_candidates(session):
    post = Post(slug="fox-post", title="Red Foxes", body_text="About the red fox.")
    session.add(post)
    session.flush()
    session.add(PostEmbedding(post_id=post.id, model_name=MODEL, vector=[1.0, 0.0]))

    fox = Image(
        filename="fox.jpg", filepath="/tmp/fox.jpg", subject="red fox", category="animal",
        confidence=0.9, status=ImageStatus.TAGGED, attributes=["orange"], caption="A red fox",
    )
    session.add(fox)
    session.flush()
    session.add(ImageEmbedding(image_id=fox.id, model_name=MODEL, vector=[0.9, 0.1]))

    wolf = Image(
        filename="wolf.jpg", filepath="/tmp/wolf.jpg", subject="gray wolf", category="animal",
        confidence=0.9, status=ImageStatus.TAGGED, attributes=["gray"], caption="A gray wolf",
    )
    session.add(wolf)
    session.flush()
    session.add(ImageEmbedding(image_id=wolf.id, model_name=MODEL, vector=[0.5, 0.5]))

    session.commit()
    return post, fox, wolf


# --------------------------------------------------------------------------
# get_post_or_404
# --------------------------------------------------------------------------


def test_get_post_or_404_finds_by_slug_or_id(db_session):
    post, _, _ = _seed_fox_post_with_candidates(db_session)

    assert get_post_or_404(db_session, post.slug).id == post.id
    assert get_post_or_404(db_session, post.id).id == post.id


def test_get_post_or_404_raises_for_unknown(db_session):
    with pytest.raises(PostNotFound):
        get_post_or_404(db_session, "does-not-exist")


# --------------------------------------------------------------------------
# get_or_compute_suggestions: idempotency
# --------------------------------------------------------------------------


def test_get_or_compute_suggestions_does_not_duplicate_on_repeat_calls(db_session):
    post, _, _ = _seed_fox_post_with_candidates(db_session)

    first = get_or_compute_suggestions(db_session, post)
    second = get_or_compute_suggestions(db_session, post)  # same call again, no recompute

    assert len(first) == 2
    assert {s.id for s in first} == {s.id for s in second}
    assert db_session.query(Suggestion).filter_by(post_id=post.id).count() == 2


def test_recompute_preserves_already_reviewed_suggestions(db_session):
    post, fox, wolf = _seed_fox_post_with_candidates(db_session)

    suggestions = get_or_compute_suggestions(db_session, post)
    fox_suggestion = next(s for s in suggestions if s.image_id == fox.id)

    review = create_review(db_session, suggestion_id=fox_suggestion.id, decision="approve", reviewer="tester", notes=None)

    # Recompute — the reviewed suggestion must survive with its review intact.
    new_suggestions = get_or_compute_suggestions(db_session, post, recompute=True)
    still_there = next(s for s in new_suggestions if s.id == fox_suggestion.id)

    assert still_there is not None
    assert db_session.query(Review).filter_by(suggestion_id=fox_suggestion.id).one().id == review.id


# --------------------------------------------------------------------------
# create_review: 404 / 409 domain errors
# --------------------------------------------------------------------------


def test_create_review_approve_then_double_review_conflicts(db_session):
    post, fox, _ = _seed_fox_post_with_candidates(db_session)
    suggestions = get_or_compute_suggestions(db_session, post)
    fox_suggestion = next(s for s in suggestions if s.image_id == fox.id)

    create_review(db_session, suggestion_id=fox_suggestion.id, decision="approve", reviewer="tester", notes=None)

    with pytest.raises(AlreadyReviewed):
        create_review(db_session, suggestion_id=fox_suggestion.id, decision="reject", reviewer="tester2", notes=None)

    # The original decision must be untouched — no silent overwrite.
    stored = db_session.query(Review).filter_by(suggestion_id=fox_suggestion.id).one()
    assert stored.decision.value == "approve"


def test_review_unknown_suggestion_raises_not_found(db_session):
    with pytest.raises(SuggestionNotFound):
        create_review(db_session, suggestion_id="does-not-exist", decision="approve", reviewer="tester", notes=None)


def test_get_suggestion_or_404(db_session):
    post, fox, _ = _seed_fox_post_with_candidates(db_session)
    suggestions = get_or_compute_suggestions(db_session, post)
    found = get_suggestion_or_404(db_session, suggestions[0].id)
    assert found.id == suggestions[0].id

    with pytest.raises(SuggestionNotFound):
        get_suggestion_or_404(db_session, "nope")


# --------------------------------------------------------------------------
# summarize()
# --------------------------------------------------------------------------


def test_summarize_returns_best_approved_ranked_first(db_session):
    post, fox, wolf = _seed_fox_post_with_candidates(db_session)
    suggestions = get_or_compute_suggestions(db_session, post)

    result = summarize(suggestions)

    assert result.status == "matched"
    assert result.best.image_id == fox.id


def test_summarize_no_confident_match_when_nothing_approved(db_session):
    post = Post(slug="lakes-post", title="Mountain Lakes", body_text="About still alpine lakes.")
    db_session.add(post)
    db_session.flush()
    db_session.add(PostEmbedding(post_id=post.id, model_name=MODEL, vector=[0.0, 1.0]))

    fox = Image(
        filename="fox2.jpg", filepath="/tmp/fox2.jpg", subject="red fox", category="animal",
        confidence=0.9, status=ImageStatus.TAGGED, attributes=["orange"], caption="A red fox",
    )
    db_session.add(fox)
    db_session.flush()
    db_session.add(ImageEmbedding(image_id=fox.id, model_name=MODEL, vector=[1.0, 0.0]))  # orthogonal -> low sim
    db_session.commit()

    suggestions = get_or_compute_suggestions(db_session, post)
    result = summarize(suggestions)

    assert result.status == "no_confident_match"
    assert result.best is None
    assert "threshold" in result.explanation.lower() or "no confident match" in result.explanation.lower()