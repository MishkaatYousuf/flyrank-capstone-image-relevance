"""
Phase 3, part 3: the mismatch guard (§4.3) — the production-critical part
of this whole capstone. Combines three independent signals before a
suggestion is ever allowed to reach a human:

  1. Confidence gate  — a FLAGGED (low vision-confidence) image can never
     be auto-approved, no matter how well it scores.
  2. Tag/category check — does the candidate's detected subject plausibly
     match what the post is actually about? This is what makes the
     wolf-on-a-fox-post rejection *provable*, not probabilistic: it's a
     direct comparison of the subject the vision model detected in the
     image against subject words the post text actually mentions, not a
     vibe from the embedding score alone.
  3. Similarity threshold — cosine similarity must clear
     `settings.similarity_threshold` (tuned from eval data in Phase 4,
     not guessed).

Any one of these failing is enough to reject, with a human-readable reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.models import Image, ImageStatus, Post

# --------------------------------------------------------------------------
# Subject vocabulary for the tag/category check.
#
# Kept as an explicit, small mapping rather than something cleverer on
# purpose — §7 "Realistic scope" calls for a FEW animal categories, and an
# explainable guard beats a black-box one. Extend this table if you add
# categories to your corpus. The canonical names on the right must match
# (or be substrings of) the `subject` your vision model actually produces
# for images in that category — check a few rows in the `images` table
# after Phase 2 if you rename or add species.
# --------------------------------------------------------------------------
SUBJECT_SYNONYMS: dict[str, str] = {
    # scientific / alternate names -> canonical subject word
    "vulpes vulpes": "fox",
    "wild fox species": "fox",
    "red fox": "fox",
    "canis lupus": "wolf",
    "gray wolf": "wolf",
    "grey wolf": "wolf",
    "canis lupus familiaris": "dog",
    "domestic dog": "dog",
    "ursus arctos": "bear",
    "brown bear": "bear",
    "grizzly bear": "bear",
    "grizzly": "bear",
    "white-tailed deer": "deer",
    "whitetail deer": "deer",
    "odocoileus": "deer",
}
KNOWN_SUBJECTS = ["fox", "wolf", "dog", "bear", "deer"]


def _canonical_subject(text: str) -> str | None:
    """First known animal subject (via synonym table or direct word match) found in text."""
    text = (text or "").lower()
    for phrase, canonical in SUBJECT_SYNONYMS.items():
        if phrase in text:
            return canonical
    for subj in KNOWN_SUBJECTS:
        if re.search(rf"\b{re.escape(subj)}(e?s)?\b", text):
            return subj
    return None


def _all_mentioned_subjects(post_text: str) -> set[str]:
    """Every known animal subject mentioned anywhere in the post (usually just one)."""
    text = (post_text or "").lower()
    found = set()
    for phrase, canonical in SUBJECT_SYNONYMS.items():
        if phrase in text:
            found.add(canonical)
    for subj in KNOWN_SUBJECTS:
        if re.search(rf"\b{re.escape(subj)}(e?s)?\b", text):
            found.add(subj)
    return found


@dataclass
class GuardResult:
    approved: bool
    reason: str


def evaluate_candidate(post: Post, image: Image, similarity_score: float) -> GuardResult:
    """
    The single entry point every candidate (naturally ranked or force-tested)
    passes through. Order matters: cheapest/most decisive checks first.
    """
    # --- 1. Confidence gate -------------------------------------------------
    if image.status == ImageStatus.FLAGGED:
        return GuardResult(
            approved=False,
            reason=(
                f"Image confidence ({image.confidence:.2f}) is below the review threshold "
                f"({settings.low_confidence_threshold:.2f}) — this image was flagged for human "
                f"review in Phase 2 and cannot be auto-suggested until approved."
            ),
        )

    # --- 2. Tag / category check --------------------------------------------
    expected_subjects = _all_mentioned_subjects(post.body_text + " " + post.title)
    candidate_subject = _canonical_subject((image.subject or "") + " " + (image.category or ""))

    if expected_subjects:
        if candidate_subject is None:
            return GuardResult(
                approved=False,
                reason=(
                    f"Subject mismatch: post is about {'/'.join(sorted(expected_subjects))}, "
                    f"but the image's detected subject is '{image.subject}' ({image.category}), "
                    "which doesn't match any known animal category."
                ),
            )
        if candidate_subject not in expected_subjects:
            expected_str = "/".join(sorted(expected_subjects))
            return GuardResult(
                approved=False,
                reason=f"Animal category mismatch: expected {expected_str}, detected {candidate_subject}.",
            )

    # --- 3. Similarity threshold --------------------------------------------
    if similarity_score < settings.similarity_threshold:
        return GuardResult(
            approved=False,
            reason=(
                f"Similarity {similarity_score:.2f} is below the threshold "
                f"{settings.similarity_threshold:.2f} — not confident this image matches the post."
            ),
        )

    # --- All clear -----------------------------------------------------------
    return GuardResult(
        approved=True,
        reason=(
            f"Subject '{image.subject}' matches the post topic; similarity "
            f"{similarity_score:.2f} clears the {settings.similarity_threshold:.2f} threshold; "
            f"vision confidence {image.confidence:.2f} is high enough to trust."
        ),
    )