#!/usr/bin/env python3
"""
Phase 4 entrypoint: "A small labeled evaluation dataset measures top-1
precision — the number is in your README" (§6), evaluated by Probe 5:
"Run the eval script → top-1 precision reported on the labeled set,
matching the number in the README."

Top-1 precision here = of all labeled posts, the share where the system's
final decision was correct:
  - if expected_subject is set: the top APPROVED suggestion's image subject
    matches it (a wrong image OR a wrongly-withheld "no confident match"
    both count as incorrect)
  - if expected_subject is null: the system correctly returned
    "no_confident_match" (suggesting ANY image here is incorrect — this is
    what proves refusal isn't just a fallback, it's graded as a real
    outcome)

Usage:
    python scripts/eval.py                 # uses data/eval_set.json, prints precision
    python scripts/eval.py --recompute       # force fresh ranking instead of reusing stored suggestions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import review as review_logic  # noqa: E402
from app.database import get_session, init_db  # noqa: E402
from app.guard import canonical_subject  # noqa: E402
from app.models import Post  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"


def evaluate(session, recompute: bool = False) -> dict:
    entries = json.loads(EVAL_SET_PATH.read_text())
    results = []

    for entry in entries:
        slug = entry["post_slug"]
        expected = entry["expected_subject"]

        post = session.query(Post).filter_by(slug=slug).one_or_none()
        if post is None:
            results.append({"post_slug": slug, "correct": False, "detail": "post not found in DB — run scripts/run_matching.py --seed-posts-only first"})
            continue

        try:
            suggestions = review_logic.get_or_compute_suggestions(session, post, recompute=recompute)
        except ValueError as e:
            results.append({"post_slug": slug, "correct": False, "detail": f"could not rank: {e}"})
            continue

        summary = review_logic.summarize(suggestions)

        if expected is None:
            correct = summary.status == "no_confident_match"
            detail = (
                "correctly returned no confident match"
                if correct
                else f"expected no confident match, but got {summary.best.image.subject!r}"
            )
        else:
            if summary.status != "matched":
                correct = False
                detail = f"expected subject '{expected}', but got no confident match"
            else:
                got_subject = canonical_subject(summary.best.image.subject or "")
                correct = got_subject == expected
                detail = (
                    f"expected '{expected}', got '{got_subject}' (image: {summary.best.image.filename})"
                    if not correct
                    else f"correctly matched '{expected}' (image: {summary.best.image.filename})"
                )

        results.append({"post_slug": slug, "correct": correct, "detail": detail})

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    precision = (correct_count / total) if total else 0.0

    return {"results": results, "correct": correct_count, "total": total, "top1_precision": precision}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure top-1 precision against the labeled eval set.")
    parser.add_argument("--recompute", action="store_true", help="Force fresh ranking instead of reusing stored suggestions.")
    args = parser.parse_args()

    init_db()

    with get_session() as session:
        report = evaluate(session, recompute=args.recompute)

    print("=== Eval results ===")
    for r in report["results"]:
        mark = "✅" if r["correct"] else "❌"
        print(f"  {mark} {r['post_slug']}: {r['detail']}")

    precision_pct = report["top1_precision"] * 100
    print(f"\nTop-1 precision: {precision_pct:.0f}% ({report['correct']}/{report['total']})")
    print("Paste this line into README.md's eval section and EVIDENCE.md's Probe 5 proof.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())