"""
Vision layer: turns an image file into a validated ImageTags object.

Two providers, both $0:
  - "gemini": Gemini Flash free tier via Google AI Studio (needs a free key,
    no credit card). Uses the Interactions API with a JSON schema so the
    model is constrained to our shape at generation time.
  - "ollama":  Fully local & free, no key, works offline. Vision models like
    `llava` or `moondream` don't support strict schema-constrained output,
    so we ask nicely in the prompt and rely on Pydantic validation +
    retries to enforce the contract.

Either way, the CONTRACT is the same: this module never returns anything
that hasn't passed ImageTags validation. If it can't produce a valid
response, it raises — the caller (app/batch.py) decides whether to retry,
flag, or fail the image.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.schemas import ImageTags, VisionCallResult

PROMPT = """You are an image-cataloging assistant for a blog's image library.
Look carefully at the image and describe it.

Respond with ONLY a JSON object with exactly these fields:
- "subject": the single main subject, e.g. "red fox" (be specific — prefer
  the common species/object name over a generic category word)
- "category": one of "animal", "landscape", "people", "object", "other"
- "attributes": a list of 2-5 short descriptive attributes (e.g. ["orange fur", "forest", "wild"])
- "caption": one plain sentence describing the image
- "confidence": your own confidence in this classification, a number from 0 to 1

Be honest about confidence. If the subject is ambiguous, blurry, or you are
guessing, use a LOW confidence number (below 0.6) rather than pretending
certainty. Do not include any text outside the JSON object.
"""


class VisionError(Exception):
    """Raised when a single vision call fails outright (network, auth, etc.)."""


class InvalidVisionOutput(Exception):
    """Raised when the model's response cannot be parsed/validated as ImageTags."""


def classify_image(image_path: Path, *, attempt: int = 1) -> VisionCallResult:
    if settings.vision_provider == "ollama":
        return _classify_with_ollama(image_path, attempt=attempt)
    return _classify_with_gemini(image_path, attempt=attempt)


# --------------------------------------------------------------------------
# Gemini (default, free tier)
# --------------------------------------------------------------------------


def _classify_with_gemini(image_path: Path, *, attempt: int) -> VisionCallResult:
    settings.validate_for_vision()
    from google import genai  # imported lazily so `ollama`-only setups don't need it

    client = genai.Client(api_key=settings.gemini_api_key)

    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    image_bytes = image_path.read_bytes()

    interaction = client.interactions.create(
        model=settings.gemini_model,
        input=[
            {"type": "text", "text": PROMPT},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
                "mime_type": mime_type,
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ImageTags.model_json_schema(),
        },
    )

    raw_text = interaction.output_text
    usage = getattr(interaction, "usage", None)
    input_tokens = getattr(usage, "total_input_tokens", 0) or 0
    output_tokens = getattr(usage, "total_output_tokens", 0) or 0

    tags = _parse_and_validate(raw_text)

    return VisionCallResult(
        tags=tags,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_response=raw_text,
        attempt=attempt,
    )


# --------------------------------------------------------------------------
# Ollama (fully local, $0, no key — the offline fallback from the brief)
# --------------------------------------------------------------------------


def _classify_with_ollama(image_path: Path, *, attempt: int) -> VisionCallResult:
    import requests

    image_bytes = image_path.read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    resp = requests.post(
        f"{settings.ollama_host}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": PROMPT,
            "images": [b64_image],
            "format": "json",
            "stream": False,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise VisionError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    raw_text = body.get("response", "")
    tags = _parse_and_validate(raw_text)

    # Ollama doesn't report token usage the same way Gemini does; approximate
    # from eval_count fields when present so cost tracking still has a number.
    input_tokens = body.get("prompt_eval_count", 0) or 0
    output_tokens = body.get("eval_count", 0) or 0

    return VisionCallResult(
        tags=tags,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_response=raw_text,
        attempt=attempt,
    )


# --------------------------------------------------------------------------
# Shared: never trust a raw response — parse + validate or raise
# --------------------------------------------------------------------------


def _parse_and_validate(raw_text: str) -> ImageTags:
    raw_text = (raw_text or "").strip()
    # Some local models wrap JSON in ```json fences despite instructions.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Treat "not even JSON" the same as "invalid schema" — both are
        # untrusted output the batch job must retry or flag, never accept.
        raise InvalidVisionOutput(f"Model response was not valid JSON: {e}") from e

    try:
        # Raises pydantic.ValidationError on anything that doesn't fit the
        # schema — this propagates up to the batch job, which is the only
        # place allowed to decide "retry" vs "flag" vs "error".
        return ImageTags.model_validate(data)
    except ValidationError as e:
        raise InvalidVisionOutput(f"Model response failed schema validation: {e}") from e
