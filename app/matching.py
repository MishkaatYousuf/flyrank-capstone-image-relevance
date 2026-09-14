"""
Phase 3, part 1+2: "Create embeddings for image descriptions and blog post
content, store them in a vector index, and for each post rank the most
relevant images" (§4.2).

At ~50 images and a handful of posts, a plain Python cosine-similarity scan
over rows pulled from SQLite IS the vector index — no pgvector, no Docker,
no extra infra needed. If the corpus ever grows past a few thousand items,
swap the linear scan in `rank_images_for_post` for a real vector index;
nothing else here would need to change.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import settings
from app.cost_tracker import log_api_call
from app.embeddings import EmbeddingError, embed_text
from app.models import Image, ImageEmbedding, ImageStatus, Post, PostEmbedding, Suggestion

logger = logging.getLogger("matching")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)} (embedded with different models?)")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------
# Embed-and-store, with the same retry + cost-tracking discipline as vision
# --------------------------------------------------------------------------


def _embed_with_retries(session: Session, *, text: str, ref_image_id: str | None, ref_post_id: str | None):
    last_error: Exception | None = None
    for attempt in range(1, settings.max_embedding_retries + 1):
        try:
            result = embed_text(text)
            log_api_call(
                session,
                call_type="embedding",
                provider=settings.vision_provider,
                model_name=result.model_name,
                input_tokens=result.input_tokens,
                image_id=ref_image_id,
                post_id=ref_post_id,
                succeeded=True,
                attempt=attempt,
            )
            return result
        except (EmbeddingError, Exception) as e:  # noqa: BLE001
            last_error = e
            log_api_call(
                session,
                call_type="embedding",
                provider=settings.vision_provider,
                model_name=settings.gemini_embedding_model
                if settings.vision_provider == "gemini"
                else settings.ollama_embedding_model,
                image_id=ref_image_id,
                post_id=ref_post_id,
                succeeded=False,
                attempt=attempt,
                error_message=str(e)[:500],
            )
            logger.warning("embedding call failed (attempt %d/%d): %s", attempt, settings.max_embedding_retries, e)
            if attempt < settings.max_embedding_retries:
                time.sleep(settings.embedding_retry_backoff_seconds * attempt)

    raise EmbeddingError(f"Embedding failed after {settings.max_embedding_retries} attempts: {last_error}")


def embed_missing_images(session: Session) -> int:
    """Embed every TAGGED/FLAGGED image's caption that doesn't have an embedding yet. Returns count embedded."""
    already_embedded_ids = {row.image_id for row in session.query(ImageEmbedding.image_id).all()}
    images = (
        session.query(Image)
        .filter(Image.status.in_([ImageStatus.TAGGED, ImageStatus.FLAGGED]))
        .filter(Image.caption.isnot(None))
        .all()
    )

    count = 0
    for image in images:
        if image.id in already_embedded_ids:
            continue
        logger.info("embedding image caption: %s -> %r", image.filename, image.caption)
        result = _embed_with_retries(session, text=image.caption, ref_image_id=image.id, ref_post_id=None)
        session.add(ImageEmbedding(image_id=image.id, model_name=result.model_name, vector=result.vector))
        session.commit()
        count += 1

    return count


def embed_missing_posts(session: Session) -> int:
    """Embed every post's body text that doesn't have an embedding yet. Returns count embedded."""
    already_embedded_ids = {row.post_id for row in session.query(PostEmbedding.post_id).all()}
    posts = session.query(Post).all()

    count = 0
    for post in posts:
        if post.id in already_embedded_ids:
            continue
        logger.info("embedding post: %s", post.slug)
        result = _embed_with_retries(session, text=post.body_text, ref_image_id=None, ref_post_id=post.id)
        session.add(PostEmbedding(post_id=post.id, model_name=result.model_name, vector=result.vector))
        session.commit()
        count += 1

    return count


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def rank_images_for_post(session: Session, post: Post, top_k: int | None = None) -> list[tuple[Image, float]]:
    """
    Returns up to top_k (image, similarity_score) pairs, ranked descending.
    Only considers images that were successfully tagged (TAGGED or FLAGGED —
    excluding FLAGGED entirely would hide them from review; the mismatch
    guard is what actually blocks a flagged image from being auto-suggested).
    """
    top_k = top_k or settings.match_top_k

    post_embedding = (
        session.query(PostEmbedding)
        .filter_by(post_id=post.id)
        .order_by(PostEmbedding.created_at.desc())
        .first()
    )
    if post_embedding is None:
        raise ValueError(f"Post '{post.slug}' has no embedding yet — run embed_missing_posts() first.")

    image_embeddings = (
        session.query(ImageEmbedding)
        .join(Image, ImageEmbedding.image_id == Image.id)
        .filter(Image.status.in_([ImageStatus.TAGGED, ImageStatus.FLAGGED]))
        .filter(ImageEmbedding.model_name == post_embedding.model_name)
        .all()
    )

    scored = [
        (emb.image, cosine_similarity(post_embedding.vector, emb.vector))
        for emb in image_embeddings
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def score_pair(session: Session, post: Post, image: Image) -> float:
    """
    Similarity between one specific post and one specific image, regardless
    of where that image would naturally rank. This is what the demo's
    "force the wolf as a candidate" moment (Probe 3) and `--force` CLI flag
    use — the guard has to work even on a candidate you hand it directly.
    """
    post_embedding = (
        session.query(PostEmbedding).filter_by(post_id=post.id).order_by(PostEmbedding.created_at.desc()).first()
    )
    image_embedding = (
        session.query(ImageEmbedding).filter_by(image_id=image.id).order_by(ImageEmbedding.created_at.desc()).first()
    )
    if post_embedding is None:
        raise ValueError(f"Post '{post.slug}' has no embedding yet.")
    if image_embedding is None:
        raise ValueError(f"Image '{image.filename}' has no embedding yet.")
    if post_embedding.model_name != image_embedding.model_name:
        raise ValueError(
            f"Model mismatch: post embedded with {post_embedding.model_name}, "
            f"image embedded with {image_embedding.model_name}."
        )
    return cosine_similarity(post_embedding.vector, image_embedding.vector)


# --------------------------------------------------------------------------
# Ranking + guard together: the full "suggest an image for this post" flow
# --------------------------------------------------------------------------


@dataclass
class MatchResult:
    status: str  # "matched" | "no_confident_match"
    post: Post
    suggestion: "Suggestion | None"
    explanation: str
    candidates: list  # list[tuple[Suggestion, GuardResult]], best-ranked first


def match_and_guard_post(session: Session, post: Post, top_k: int | None = None) -> MatchResult:
    """
    The full Phase 3 flow for one post (§5 architecture diagram, right-hand
    side): rank candidates, run each through the guard in ranked order,
    persist a Suggestion row per candidate either way (so rejections are
    inspectable later, per §4.5's review API), and return the first
    approved one — or a "no confident match" result carrying the reasons.
    """
    from app.guard import evaluate_candidate  # local import: avoids a cycle with guard.py at module load

    ranked = rank_images_for_post(session, post, top_k=top_k)

    if not ranked:
        return MatchResult(
            status="no_confident_match",
            post=post,
            suggestion=None,
            explanation="No candidate images are available (corpus empty, or no images embedded yet).",
            candidates=[],
        )

    candidates = []
    for rank, (image, score) in enumerate(ranked, start=1):
        guard_result = evaluate_candidate(post, image, score)
        suggestion = Suggestion(
            post_id=post.id,
            image_id=image.id,
            similarity_score=score,
            rank=rank,
            guard_status="approved" if guard_result.approved else "rejected",
            guard_reason=guard_result.reason,
        )
        session.add(suggestion)
        candidates.append((suggestion, guard_result))

    session.flush()

    approved = [c for c in candidates if c[1].approved]
    if approved:
        best_suggestion, best_guard = approved[0]
        return MatchResult(
            status="matched",
            post=post,
            suggestion=best_suggestion,
            explanation=best_guard.reason,
            candidates=candidates,
        )

    top_reason = candidates[0][1].reason
    return MatchResult(
        status="no_confident_match",
        post=post,
        suggestion=None,
        explanation=f"No confident match found. {top_reason}",
        candidates=candidates,
    )