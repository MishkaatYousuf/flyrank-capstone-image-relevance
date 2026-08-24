"""
The Phase 2 batch background job (§4.4 / §8 Phase 2):

  - Runs every pending image through the vision model
  - NEVER trusts an invalid response: on a JSON/schema failure, retries with
    backoff up to `MAX_VISION_RETRIES`; after that, the image is marked
    ERROR (not silently skipped, not silently accepted)
  - Low-confidence results are stored but marked FLAGGED, not accepted as
    "tagged" — a human (or Phase 4's review step) has to look at them
  - Every attempt, success or failure, gets a cost-log entry

This is deliberately a plain synchronous loop, not a task queue — at ~50
images that's the right amount of infrastructure. `run_ingestion.py` is the
CLI entrypoint that calls `run_batch()`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.cost_tracker import log_api_call
from app.models import Image, ImageStatus
from app.vision import InvalidVisionOutput, VisionCallResult, VisionError, classify_image

logger = logging.getLogger("batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class BatchSummary:
    total: int = 0
    tagged: int = 0
    flagged: int = 0
    errored: int = 0
    skipped_already_done: int = 0
    total_estimated_cost_usd: float = 0.0
    per_image: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "tagged": self.tagged,
            "flagged": self.flagged,
            "errored": self.errored,
            "skipped_already_done": self.skipped_already_done,
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 6),
        }


def discover_images(images_dir: Path) -> list[Path]:
    """Recursively find image files under data/images/<category>/*.jpg."""
    if not images_dir.exists():
        return []
    paths = [
        p for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return paths


def _upsert_pending_image(session: Session, path: Path, images_dir: Path) -> Image:
    """Ensure a DB row exists for this file; return it (creating if needed)."""
    filepath = str(path.resolve())
    existing = session.query(Image).filter_by(filepath=filepath).one_or_none()
    if existing:
        return existing

    # data/images/<category>/<file> -> category hint from the folder name
    try:
        category_hint = path.relative_to(images_dir).parts[0]
    except ValueError:
        category_hint = None

    image = Image(
        filename=path.name,
        filepath=filepath,
        source_category_hint=category_hint,
        status=ImageStatus.PENDING,
    )
    session.add(image)
    session.flush()
    return image


def _classify_with_retries(session: Session, image: Image, image_path: Path) -> VisionCallResult | None:
    """
    Calls the vision model up to MAX_VISION_RETRIES times. Every attempt is
    logged to the cost tracker regardless of outcome. Returns None (and
    leaves image.status == ERROR) if every attempt fails.
    """
    last_error: Exception | None = None

    for attempt in range(1, settings.max_vision_retries + 1):
        try:
            result = classify_image(image_path, attempt=attempt)
            log_api_call(
                session,
                call_type="vision",
                provider=settings.vision_provider,
                model_name=settings.gemini_model if settings.vision_provider == "gemini" else settings.ollama_model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                image_id=image.id,
                succeeded=True,
                attempt=attempt,
            )
            return result

        except (InvalidVisionOutput, VisionError, Exception) as e:  # noqa: BLE001
            last_error = e
            log_api_call(
                session,
                call_type="vision",
                provider=settings.vision_provider,
                model_name=settings.gemini_model if settings.vision_provider == "gemini" else settings.ollama_model,
                image_id=image.id,
                succeeded=False,
                attempt=attempt,
                error_message=str(e)[:500],
            )
            logger.warning(
                "vision call failed for %s (attempt %d/%d): %s",
                image.filename, attempt, settings.max_vision_retries, e,
            )
            if attempt < settings.max_vision_retries:
                time.sleep(settings.vision_retry_backoff_seconds * attempt)  # simple backoff

    image.status = ImageStatus.ERROR
    image.error_message = str(last_error)[:2000] if last_error else "unknown error"
    image.attempts = settings.max_vision_retries
    return None


def process_image(session: Session, image: Image, image_path: Path) -> ImageStatus:
    """Process a single image end-to-end. Commits the resulting status onto `image`."""
    result = _classify_with_retries(session, image, image_path)

    if result is None:
        return ImageStatus.ERROR  # already set on image by _classify_with_retries

    tags = result.tags
    image.subject = tags.subject
    image.category = tags.category.value
    image.attributes = tags.attributes
    image.caption = tags.caption
    image.confidence = tags.confidence
    image.raw_response = result.raw_response
    image.attempts = result.attempt
    image.error_message = None

    if tags.is_low_confidence(settings.low_confidence_threshold):
        image.status = ImageStatus.FLAGGED
    else:
        image.status = ImageStatus.TAGGED

    return image.status


def run_batch(session: Session, *, images_dir: Path | None = None, rerun: bool = False) -> BatchSummary:
    """
    Runs the full batch job. Set rerun=True to reprocess images that are
    already TAGGED/FLAGGED/ERROR (useful after changing the prompt/threshold).
    """
    images_dir = images_dir or settings.images_dir
    paths = discover_images(images_dir)
    summary = BatchSummary(total=len(paths))

    if not paths:
        logger.warning("No images found under %s — run scripts/download_images.py first.", images_dir)
        return summary

    for i, path in enumerate(paths, start=1):
        image = _upsert_pending_image(session, path, images_dir)

        if not rerun and image.status != ImageStatus.PENDING:
            summary.skipped_already_done += 1
            logger.info("[%d/%d] %s already %s — skipping", i, len(paths), path.name, image.status.value)
            continue

        logger.info("[%d/%d] classifying %s ...", i, len(paths), path.name)
        status = process_image(session, image, path)
        session.commit()

        if status == ImageStatus.TAGGED:
            summary.tagged += 1
            logger.info(
                "    -> TAGGED  subject=%r category=%r confidence=%.2f",
                image.subject, image.category, image.confidence,
            )
        elif status == ImageStatus.FLAGGED:
            summary.flagged += 1
            logger.info(
                "    -> FLAGGED (low confidence) subject=%r confidence=%.2f",
                image.subject, image.confidence,
            )
        else:
            summary.errored += 1
            logger.error("    -> ERROR: %s", image.error_message)

        summary.per_image.append(
            {"filename": image.filename, "status": status.value, "confidence": image.confidence}
        )

    from app.cost_tracker import total_estimated_cost
    summary.total_estimated_cost_usd = total_estimated_cost(session)
    return summary
