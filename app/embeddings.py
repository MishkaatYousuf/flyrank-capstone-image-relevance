"""
Embedding layer: turns text (an image caption, or a post body) into a
fixed-length vector so semantically related text lands close together in
vector space — "red fox", "Vulpes vulpes", and "wild fox species" all
embed near each other even though they share almost no words. This is what
makes matching concept-based instead of keyword-based (§4.2 of the brief).

Same two-provider split as app/vision.py, same $0 promise:
  - "gemini": gemini-embedding-001, free tier, via the same API key as vision
  - "ollama": all-minilm, fully local, no key, works offline

Both return an EmbeddingResult with a plain list[float] vector — nothing
provider-specific leaks past this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass
class EmbeddingResult:
    vector: list[float]
    model_name: str
    input_tokens: int = 0


class EmbeddingError(Exception):
    """Raised when an embedding call fails outright (network, auth, empty text, etc.)."""


def embed_text(text: str, *, task_type: str = "SEMANTIC_SIMILARITY") -> EmbeddingResult:
    text = (text or "").strip()
    if not text:
        raise EmbeddingError("Cannot embed empty text.")

    if settings.vision_provider == "ollama":
        return _embed_with_ollama(text)
    return _embed_with_gemini(text, task_type=task_type)


# --------------------------------------------------------------------------
# Gemini (default, free tier)
# --------------------------------------------------------------------------


def _embed_with_gemini(text: str, *, task_type: str) -> EmbeddingResult:
    settings.validate_for_vision()  # same GEMINI_API_KEY covers embeddings
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    try:
        result = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    except Exception as e:  # noqa: BLE001 — network/auth/quota errors all land here
        raise EmbeddingError(f"Gemini embedding call failed: {e}") from e

    if not result.embeddings:
        raise EmbeddingError("Gemini returned no embedding for this text.")

    vector = list(result.embeddings[0].values)
    # The embeddings API doesn't return token usage the same way generate
    # calls do on every SDK version; approximate for cost-tracking purposes
    # using a simple word-count heuristic (~1.3 tokens/word) when unavailable.
    usage = getattr(result, "metadata", None)
    input_tokens = getattr(usage, "billable_character_count", None)
    if input_tokens is None:
        input_tokens = max(1, int(len(text.split()) * 1.3))

    return EmbeddingResult(vector=vector, model_name=settings.gemini_embedding_model, input_tokens=input_tokens)


# --------------------------------------------------------------------------
# Ollama (fully local, $0, no key)
# --------------------------------------------------------------------------


def _embed_with_ollama(text: str) -> EmbeddingResult:
    import requests

    try:
        resp = requests.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.ollama_embedding_model, "input": text},
            timeout=60,
        )
    except requests.RequestException as e:
        raise EmbeddingError(f"Could not reach Ollama at {settings.ollama_host}: {e}") from e

    if resp.status_code != 200:
        raise EmbeddingError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    embeddings = body.get("embeddings") or []
    if not embeddings:
        raise EmbeddingError("Ollama returned no embedding for this text.")

    return EmbeddingResult(
        vector=list(embeddings[0]),
        model_name=settings.ollama_embedding_model,
        input_tokens=body.get("prompt_eval_count", 0) or 0,
    )