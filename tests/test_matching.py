"""
Definition of Done:
  - "Image and post embeddings are stored; posts return ranked image
    suggestions."
  - "Semantic matching works for equivalent concepts."
  - "The mismatch guard rejects incorrect recommendations — the
    wolf-on-a-fox-post scenario provably fails" — tested here at the FULL
    pipeline level (ranking + guard together), complementing the guard-only
    unit tests in test_guard.py.

No network calls: embeddings are inserted directly as hand-crafted vectors,
and app.matching.embed_text is monkeypatched where the embed-and-store path
itself is under test.
"""
from __future__ import annotations

import app.matching as matching_mod
from app.embeddings import EmbeddingError, EmbeddingResult
from app.matching import embed_missing_images, embed_missing_posts, match_and_guard_post, rank_images_for_post
from app.models import ApiCallLog, Image, ImageEmbedding, ImageStatus, Post, PostEmbedding

MODEL = "test-embedding-model"


def _add_post(session, slug: str, title: str, body: str) -> Post:
    post = Post(slug=slug, title=title, body_text=body)
    session.add(post)
    session.flush()
    return post


def _add_image(session, subject: str, category: str = "animal", confidence: float = 0.9, status=ImageStatus.TAGGED, suffix: str = "") -> Image:
    unique = f"{subject.replace(' ', '_')}{suffix}"
    image = Image(
        filename=f"{unique}.jpg",
        filepath=f"/tmp/{unique}.jpg",
        subject=subject,
        category=category,
        confidence=confidence,
        status=status,
        attributes=["placeholder"],
        caption=f"A {subject}",
    )
    session.add(image)
    session.flush()
    return image


def _embed(session, *, image=None, post=None, vector):
    if image is not None:
        session.add(ImageEmbedding(image_id=image.id, model_name=MODEL, vector=vector))
    if post is not None:
        session.add(PostEmbedding(post_id=post.id, model_name=MODEL, vector=vector))
    session.flush()


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_rank_images_for_post_orders_by_similarity_descending(db_session):
    post = _add_post(db_session, "fox-post", "Red Foxes", "About the red fox.")
    _embed(db_session, post=post, vector=[1.0, 0.0, 0.0])

    fox = _add_image(db_session, "red fox")
    _embed(db_session, image=fox, vector=[0.95, 0.05, 0.0])  # very close

    wolf = _add_image(db_session, "gray wolf")
    _embed(db_session, image=wolf, vector=[0.3, 0.9, 0.0])  # far

    dog = _add_image(db_session, "dog")
    _embed(db_session, image=dog, vector=[0.1, 0.1, 0.95])  # farthest

    ranked = rank_images_for_post(db_session, post)

    assert [img.subject for img, _score in ranked] == ["red fox", "gray wolf", "dog"]
    assert ranked[0][1] > ranked[1][1] > ranked[2][1]


def test_ranking_excludes_errored_images(db_session):
    post = _add_post(db_session, "fox-post", "Red Foxes", "About the red fox.")
    _embed(db_session, post=post, vector=[1.0, 0.0])

    good = _add_image(db_session, "red fox")
    _embed(db_session, image=good, vector=[0.9, 0.1])

    errored = _add_image(db_session, "red fox", status=ImageStatus.ERROR, suffix="_2")
    _embed(db_session, image=errored, vector=[1.0, 0.0])  # would rank #1 if included

    ranked = rank_images_for_post(db_session, post)
    assert len(ranked) == 1
    assert ranked[0][0].id == good.id


# --------------------------------------------------------------------------
# Full pipeline: ranking + guard together
# --------------------------------------------------------------------------


def test_fox_post_matches_fox_even_when_wolf_scores_higher_on_similarity_alone(db_session):
    """
    The headline demo scenario (§13 of the brief), tested adversarially: the
    wolf image is given a DELIBERATELY higher raw similarity score than the
    fox image (e.g. a noisy embedding, or a wolf caption that happens to
    share more words with the post). The guard must still refuse it on the
    category mismatch and let the correct fox image through instead.
    """
    post = _add_post(
        db_session, "fox-post", "The Secret Life of Red Foxes", "The red fox (Vulpes vulpes) is a cunning hunter."
    )
    _embed(db_session, post=post, vector=[1.0, 0.0])

    wolf = _add_image(db_session, "gray wolf")
    _embed(db_session, image=wolf, vector=[0.99, 0.01])  # ranks #1 on raw similarity

    fox = _add_image(db_session, "red fox")
    _embed(db_session, image=fox, vector=[0.9, 0.1])  # ranks #2 on raw similarity

    # Sanity check the adversarial setup actually ranks wolf first:
    ranked = rank_images_for_post(db_session, post)
    assert ranked[0][0].id == wolf.id

    result = match_and_guard_post(db_session, post)

    assert result.status == "matched"
    assert result.suggestion.image_id == fox.id
    assert "fox" in result.explanation.lower()

    # The wolf must appear among the considered candidates, explicitly rejected.
    wolf_candidate = next(c for c in result.candidates if c[0].image_id == wolf.id)
    assert wolf_candidate[1].approved is False
    assert "mismatch" in wolf_candidate[1].reason.lower()


def test_no_confident_match_when_nothing_clears_threshold(db_session):
    post = _add_post(db_session, "lakes-post", "Mountain Lakes", "A piece about still alpine lakes at sunrise.")
    _embed(db_session, post=post, vector=[0.0, 0.0, 1.0])

    fox = _add_image(db_session, "red fox")
    _embed(db_session, image=fox, vector=[1.0, 0.0, 0.0])  # orthogonal -> ~0 similarity

    result = match_and_guard_post(db_session, post)

    assert result.status == "no_confident_match"
    assert result.suggestion is None
    assert "no confident match" in result.explanation.lower()
    assert "threshold" in result.explanation.lower()


def test_match_result_persists_a_suggestion_row_per_candidate(db_session):
    from app.models import Suggestion

    post = _add_post(db_session, "fox-post", "Red Foxes", "About the red fox.")
    _embed(db_session, post=post, vector=[1.0, 0.0])
    fox = _add_image(db_session, "red fox")
    _embed(db_session, image=fox, vector=[0.9, 0.1])
    wolf = _add_image(db_session, "gray wolf")
    _embed(db_session, image=wolf, vector=[0.5, 0.5])

    match_and_guard_post(db_session, post)
    db_session.commit()

    rows = db_session.query(Suggestion).filter_by(post_id=post.id).all()
    assert len(rows) == 2  # both candidates recorded, so rejections stay inspectable


# --------------------------------------------------------------------------
# Embed-and-store: retries + cost tracking (mirrors test_batch.py's pattern)
# --------------------------------------------------------------------------


def test_embed_missing_images_stores_vector_and_logs_cost(db_session, monkeypatch):
    fox = _add_image(db_session, "red fox")
    db_session.commit()

    monkeypatch.setattr(
        matching_mod, "embed_text",
        lambda text, **kw: EmbeddingResult(vector=[0.1, 0.2, 0.3], model_name=MODEL, input_tokens=12),
    )

    count = embed_missing_images(db_session)

    assert count == 1
    stored = db_session.query(ImageEmbedding).filter_by(image_id=fox.id).one()
    assert stored.vector == [0.1, 0.2, 0.3]

    calls = db_session.query(ApiCallLog).filter_by(call_type="embedding").all()
    assert len(calls) == 1
    assert calls[0].succeeded is True


def test_embed_missing_images_skips_already_embedded(db_session, monkeypatch):
    fox = _add_image(db_session, "red fox")
    _embed(db_session, image=fox, vector=[1.0, 0.0])
    db_session.commit()

    call_count = {"n": 0}

    def tracked_embed(text, **kw):
        call_count["n"] += 1
        return EmbeddingResult(vector=[0.5, 0.5], model_name=MODEL)

    monkeypatch.setattr(matching_mod, "embed_text", tracked_embed)

    count = embed_missing_images(db_session)

    assert count == 0
    assert call_count["n"] == 0  # never called — already embedded


def test_embed_missing_posts_retries_then_succeeds(db_session, patched_settings, monkeypatch):
    post = _add_post(db_session, "fox-post", "Red Foxes", "About the red fox.")
    db_session.commit()

    calls = {"n": 0}

    def flaky_embed(text, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmbeddingError("simulated transient failure")
        return EmbeddingResult(vector=[0.2, 0.4], model_name=MODEL, input_tokens=5)

    monkeypatch.setattr(matching_mod, "embed_text", flaky_embed)
    monkeypatch.setattr(matching_mod.time, "sleep", lambda *_: None)

    count = embed_missing_posts(db_session)

    assert count == 1
    assert calls["n"] == 2
    log_rows = db_session.query(ApiCallLog).order_by(ApiCallLog.attempt).all()
    assert [r.succeeded for r in log_rows] == [False, True]