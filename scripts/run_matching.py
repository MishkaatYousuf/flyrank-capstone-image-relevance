#!/usr/bin/env python3
"""
Phase 3 entrypoint.

Usage:
    python scripts/run_matching.py                     # seed posts + embed + match everything
    python scripts/run_matching.py --seed-posts-only     # just load data/posts.json into the DB
    python scripts/run_matching.py --embed-only          # just embed missing images/posts
    python scripts/run_matching.py --match-only           # just run ranking+guard (needs embeddings already done)
    python scripts/run_matching.py --post <slug>           # match one post, verbose (see every candidate + reason)
    python scripts/run_matching.py --force <post_slug> <image_filename>
                                                           # the Probe 3 demo: force a specific image as a
                                                           # candidate for a specific post and show the guard's verdict,
                                                           # regardless of where it would naturally rank.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import get_session, init_db  # noqa: E402
from app.matching import embed_missing_images, embed_missing_posts, match_and_guard_post, score_pair  # noqa: E402
from app.models import Image, Post  # noqa: E402


def seed_posts(session) -> int:
    """Idempotent: loads data/posts.json, skipping posts whose slug already exists."""
    posts_path = Path(__file__).resolve().parent.parent / "data" / "posts.json"
    payload = json.loads(posts_path.read_text())

    existing_slugs = {p.slug for p in session.query(Post).all()}
    added = 0
    for entry in payload:
        if entry["slug"] in existing_slugs:
            continue
        session.add(Post(slug=entry["slug"], title=entry["title"], body_text=entry["body_text"]))
        added += 1
    session.commit()
    return added


def print_match_result(result) -> None:
    print(f"\n=== {result.post.title} ({result.post.slug}) ===")
    if result.status == "matched":
        img = result.suggestion.image
        print(f"  ✅ MATCHED -> {img.filename} (subject={img.subject!r}, similarity={result.suggestion.similarity_score:.3f})")
        print(f"     reason: {result.explanation}")
    else:
        print(f"  ❌ NO CONFIDENT MATCH")
        print(f"     reason: {result.explanation}")

    if result.candidates:
        print("  -- all candidates considered, ranked --")
        for suggestion, guard in result.candidates:
            mark = "✅" if guard.approved else "✗ "
            print(
                f"     {mark} rank {suggestion.rank}: {suggestion.image.filename} "
                f"(similarity={suggestion.similarity_score:.3f}) — {guard.reason}"
            )


def run_force(session, post_slug: str, image_filename: str) -> int:
    post = session.query(Post).filter_by(slug=post_slug).one_or_none()
    if post is None:
        print(f"No post with slug '{post_slug}'. Known slugs:")
        for p in session.query(Post).all():
            print(f"  - {p.slug}")
        return 1

    image = session.query(Image).filter(Image.filename == image_filename).one_or_none()
    if image is None:
        print(f"No image with filename '{image_filename}' (match on Image.filename, not full path).")
        return 1

    from app.guard import evaluate_candidate

    try:
        score = score_pair(session, post, image)
    except ValueError as e:
        print(f"⚠️  {e}")
        return 1
    guard_result = evaluate_candidate(post, image, score)

    print(f"\n=== FORCED CANDIDATE: {image.filename} for post '{post.slug}' ===")
    print(f"  similarity: {score:.3f}")
    print(f"  guard verdict: {'✅ APPROVED' if guard_result.approved else '❌ REJECTED'}")
    print(f"  reason: {guard_result.reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 3 matching engine.")
    parser.add_argument("--seed-posts-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--match-only", action="store_true")
    parser.add_argument("--post", type=str, default=None, help="Match a single post by slug, with full candidate detail.")
    parser.add_argument(
        "--force", nargs=2, metavar=("POST_SLUG", "IMAGE_FILENAME"),
        help="Force-test one specific image against one specific post (Probe 3 demo).",
    )
    args = parser.parse_args()

    init_db()

    with get_session() as session:
        if args.force:
            return run_force(session, args.force[0], args.force[1])

        if args.seed_posts_only:
            added = seed_posts(session)
            print(f"Seeded {added} new post(s).")
            return 0

        if args.embed_only:
            try:
                n_posts = embed_missing_posts(session)
                n_images = embed_missing_images(session)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Embedding failed: {e}")
                return 1
            print(f"Embedded {n_posts} post(s) and {n_images} image(s).")
            return 0

        if args.post:
            post = session.query(Post).filter_by(slug=args.post).one_or_none()
            if post is None:
                print(f"No post with slug '{args.post}'.")
                return 1
            try:
                result = match_and_guard_post(session, post)
            except ValueError as e:
                print(f"⚠️  {e}")
                return 1
            print_match_result(result)
            return 0

        if args.match_only:
            posts = session.query(Post).all()
            for post in posts:
                try:
                    result = match_and_guard_post(session, post)
                except ValueError as e:
                    print(f"\n=== {post.title} ({post.slug}) ===\n  ⚠️  Skipped: {e}")
                    continue
                print_match_result(result)
            return 0

        # Default: full pipeline — seed, embed, match, report.
        added = seed_posts(session)
        print(f"Seeded {added} new post(s).")

        try:
            n_posts = embed_missing_posts(session)
            n_images = embed_missing_images(session)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Embedding failed: {e}")
            return 1
        print(f"Embedded {n_posts} post(s) and {n_images} image(s).")

        posts = session.query(Post).all()
        if not posts:
            print("No posts found — nothing to match.")
            return 1

        matched, no_match = 0, 0
        for post in posts:
            try:
                result = match_and_guard_post(session, post)
            except ValueError as e:
                print(f"\n=== {post.title} ({post.slug}) ===\n  ⚠️  Skipped: {e}")
                continue
            print_match_result(result)
            if result.status == "matched":
                matched += 1
            else:
                no_match += 1

        print(f"\n=== Summary: {matched} matched, {no_match} no-confident-match, out of {len(posts)} posts ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())