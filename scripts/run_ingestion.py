#!/usr/bin/env python3
"""
Phase 2 entrypoint.

Usage:
    python scripts/run_ingestion.py            # process new/pending images
    python scripts/run_ingestion.py --rerun     # reprocess everything
    python scripts/run_ingestion.py --summary   # just print current DB stats

This is the `run:`-adjacent command referenced in capstone.yaml for the
ingestion step (the API server comes in Phase 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.batch import run_batch  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import get_session, init_db  # noqa: E402
from app.models import Image, ImageStatus  # noqa: E402


def print_summary(session) -> None:
    counts = {}
    for status in ImageStatus:
        counts[status.value] = session.query(Image).filter_by(status=status).count()
    print(json.dumps({"image_status_counts": counts}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 vision ingestion batch job.")
    parser.add_argument("--rerun", action="store_true", help="Reprocess EVERY image, including already-tagged ones.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Only reprocess PENDING + ERROR images (skip TAGGED/FLAGGED). "
        "Use this to resume after hitting a rate limit / quota mid-run without "
        "re-spending quota on images that already succeeded.",
    )
    parser.add_argument("--summary", action="store_true", help="Just print DB status counts and exit.")
    args = parser.parse_args()

    init_db()

    with get_session() as session:
        if args.summary:
            print_summary(session)
            return 0

        summary = run_batch(session, rerun=args.rerun, retry_errors_only=args.retry_errors)

    print("\n=== Batch run complete ===")
    print(json.dumps(summary.as_dict(), indent=2))

    if summary.total == 0:
        print(
            "\nNo images found in data/images/. Run "
            "`python scripts/download_images.py` first (see README)."
        )
        return 1

    if summary.errored > 0:
        print(
            f"\n{summary.errored} image(s) errored after {settings.max_vision_retries} retries each. "
            "See cost_log.jsonl / the `images` table for details."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
