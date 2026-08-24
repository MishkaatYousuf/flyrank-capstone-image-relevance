#!/usr/bin/env python3
"""
Downloads the ~50-image demo corpus from Pexels (free, no credit card,
license confirmed for this use at https://www.pexels.com/license/).

Get a free key at https://www.pexels.com/api/ and put it in .env as
PEXELS_API_KEY. Images are saved under data/images/<category>/, which
run_ingestion.py uses as the source_category_hint.

Usage:
    python scripts/download_images.py                # default categories/counts
    python scripts/download_images.py --per-category 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# §7 "Realistic scope": a few animal categories, large enough to show real
# retrieval behavior, small enough to check by eye. The fox/wolf pair is the
# one the mismatch guard demo hinges on — keep both.
DEFAULT_CATEGORIES = {
    "red-fox": "red fox",
    "wolf": "gray wolf",
    "dog": "dog",
    "bear": "brown bear",
    "deer": "deer",
}


def download_category(query: str, out_dir: Path, count: int, api_key: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": count, "orientation": "landscape"}

    resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    photos = resp.json().get("photos", [])

    saved = 0
    for i, photo in enumerate(photos[:count], start=1):
        url = photo["src"]["large"]
        img_resp = requests.get(url, timeout=30)
        img_resp.raise_for_status()

        filename = out_dir / f"{query.replace(' ', '_')}_{i:02d}.jpg"
        filename.write_bytes(img_resp.content)
        saved += 1
        print(f"  saved {filename}")
        time.sleep(0.3)  # be polite to the free API

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the free demo image corpus from Pexels.")
    parser.add_argument("--per-category", type=int, default=10, help="Images per category (default 10 -> 50 total).")
    parser.add_argument("--out", type=str, default=None, help="Output dir (default: data/images).")
    args = parser.parse_args()

    if not settings.pexels_api_key:
        print(
            "PEXELS_API_KEY is not set in .env.\n"
            "Get a free key (no credit card) at https://www.pexels.com/api/ and add it to .env.\n\n"
            "Alternative: skip this script and manually drop ~10 images per category into\n"
            "data/images/<category-name>/ yourself, using Unsplash or Pexels' website directly\n"
            "(check the license page for each image, both sites listed in the capstone brief §14)."
        )
        return 1

    out_root = Path(args.out) if args.out else settings.images_dir
    total = 0
    for folder_name, query in DEFAULT_CATEGORIES.items():
        print(f"\nDownloading '{query}' -> data/images/{folder_name}/")
        total += download_category(query, out_root / folder_name, args.per_category, settings.pexels_api_key)

    print(f"\nDone. Saved {total} images across {len(DEFAULT_CATEGORIES)} categories under {out_root}")
    print("Next: python scripts/run_ingestion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
