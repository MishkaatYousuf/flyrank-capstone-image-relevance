"""
Definition of Done:
  - "The mismatch guard rejects incorrect recommendations — the
    wolf-on-a-fox-post scenario provably fails."
  - "Rejections include a human-readable explanation."
  - "When no image clears the bar, the system answers 'no confident match'
    with reasons."

No network calls here — these test the guard's decision logic directly
against in-memory Image/Post objects.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.guard import evaluate_candidate
from app.models import Image, ImageStatus, Post


def _post(title: str, body: str) -> Post:
    return Post(id="post-1", slug="test-post", title=title, body_text=body)


def _image(subject: str, category: str = "animal", confidence: float = 0.9, status=ImageStatus.TAGGED) -> Image:
    return Image(
        id="img-1", filename=f"{subject.replace(' ', '_')}.jpg", filepath="/tmp/x.jpg",
        subject=subject, category=category, confidence=confidence, status=status,
        attributes=["placeholder"], caption=f"A {subject}",
    )


# --- The headline demo scenario: wolf-on-a-fox-post -----------------------

def test_wolf_rejected_for_fox_post():
    post = _post("The Secret Life of Red Foxes", "The red fox (Vulpes vulpes) is a cunning hunter.")
    wolf = _image("gray wolf")

    result = evaluate_candidate(post, wolf, similarity_score=0.85)  # even with HIGH similarity

    assert result.approved is False
    assert "mismatch" in result.reason.lower()
    assert "fox" in result.reason.lower()
    assert "wolf" in result.reason.lower()


def test_matching_fox_image_is_approved_for_fox_post():
    post = _post("The Secret Life of Red Foxes", "The red fox (Vulpes vulpes) is a cunning hunter.")
    fox = _image("red fox")

    result = evaluate_candidate(post, fox, similarity_score=0.82)

    assert result.approved is True
    assert "fox" in result.reason.lower()


def test_scientific_name_synonym_still_matches_correct_subject():
    """'red fox' post text should still recognize a fox image even when the
    post uses the scientific name only — the DoD's synonym requirement."""
    post = _post("Vulpes vulpes: A Profile", "Vulpes vulpes is found across the northern hemisphere.")
    fox = _image("red fox")

    result = evaluate_candidate(post, fox, similarity_score=0.8)
    assert result.approved is True


# --- Confidence gate --------------------------------------------------------

def test_flagged_low_confidence_image_never_auto_approved():
    post = _post("The Secret Life of Red Foxes", "A piece about the red fox.")
    flagged_fox = _image("red fox", confidence=0.4, status=ImageStatus.FLAGGED)

    result = evaluate_candidate(post, flagged_fox, similarity_score=0.9)  # great similarity, still flagged

    assert result.approved is False
    assert "confidence" in result.reason.lower()


# --- Similarity threshold ---------------------------------------------------

def test_below_threshold_similarity_rejected_even_with_matching_subject():
    post = _post("The Secret Life of Red Foxes", "A piece about the red fox.")
    fox = _image("red fox")

    result = evaluate_candidate(post, fox, similarity_score=0.1)

    assert result.approved is False
    assert "threshold" in result.reason.lower()


# --- Every rejection carries a human-readable explanation -------------------

@pytest.mark.parametrize(
    "image,score",
    [
        (_image("gray wolf"), 0.9),
        (_image("red fox", confidence=0.3, status=ImageStatus.FLAGGED), 0.9),
        (_image("red fox"), 0.05),
    ],
)
def test_every_rejection_has_a_nonempty_reason(image, score):
    post = _post("The Secret Life of Red Foxes", "A piece about the red fox.")
    result = evaluate_candidate(post, image, similarity_score=score)
    assert result.approved is False
    assert isinstance(result.reason, str) and len(result.reason) > 10


# --- Unrelated subject (dog) also correctly rejected for a fox post --------

def test_generic_dog_rejected_for_fox_post():
    post = _post("The Secret Life of Red Foxes", "A piece about the red fox in the wild.")
    dog = _image("dog")

    result = evaluate_candidate(post, dog, similarity_score=0.6)

    assert result.approved is False
    assert "mismatch" in result.reason.lower()


# --- No expected subject mentioned at all -> guard falls through to threshold only

def test_post_with_no_recognized_subject_relies_on_similarity_only():
    post = _post("Mountain Lakes at Sunrise", "A piece about still alpine lakes and mountain peaks.")
    fox = _image("red fox")

    high = evaluate_candidate(post, fox, similarity_score=0.9)
    low = evaluate_candidate(post, fox, similarity_score=0.1)

    assert high.approved is True   # no subject mismatch possible; similarity carries it
    assert low.approved is False
    assert "threshold" in low.reason.lower()