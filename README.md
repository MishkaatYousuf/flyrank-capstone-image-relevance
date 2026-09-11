# AI Image Understanding & Content Matching Engine

FlyRank Internship · Backend Track Capstone. Understands an image library,
tags it automatically with a vision model, and (in later phases) matches
each image to the right blog post — refusing to guess when nothing fits
well enough.

**Status: Phase 1 (design) + Phase 2 (vision ingestion pipeline) complete.**
Phase 3 (embeddings + matching + mismatch guard) and Phase 4 (review API +
eval + tests-beyond-schema) are the next steps — see `DESIGN.md` §5 for
where they plug in.

## What's implemented right now

- ✅ Image metadata schema, Pydantic-validated (`app/schemas.py`)
- ✅ Matching strategy + guard rules **designed** (`DESIGN.md` §3) — not yet
  implemented (Phase 3)
- ✅ Full database schema for all phases (`app/models.py`), SQLite, $0
- ✅ ~50-image dataset via a free Pexels download script
- ✅ Vision processing with structured-output validation against the schema
  — invalid responses are retried, never trusted
- ✅ Batch background job with retries + exponential backoff
- ✅ Per-call cost tracking (even though the free tier bills $0)
- ✅ Low-confidence results flagged, not silently accepted

## Architecture

```
Images ─(batch job)─► Vision Model ─► {tags, caption, confidence} ─► image_metadata (SQLite)
                                                                    ─► api_call_log (cost)

           [Phase 3, not yet built]
           └─► embed(caption) ─► image_vectors
Posts  ──────────────────────► embed(post text) ─► post_vectors
GET /posts/:id/images
  └─► Similarity Ranking (image_vectors × post_vector)
  └─► Mismatch Guard (tags + threshold + confidence)
      ├─► Suggested image (ranked, explained)
      └─► "No good match" + explanation
  └─► Review API: approve / reject      [Phase 4, not yet built]
```

## Setup (clean machine)

Requires Python 3.11+.

```bash
git clone <your-repo-url>
cd flyrank-capstone-image-relevance
python -m venv .venv && source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env:
#   GEMINI_API_KEY=<free key from https://aistudio.google.com/apikey>
#   PEXELS_API_KEY=<free key from https://www.pexels.com/api/>
```

No credit card is required for either key.

### Local-only alternative (no API keys at all)

Set `VISION_PROVIDER=ollama` in `.env`, install [Ollama](https://ollama.com),
then `ollama pull llava`. The pipeline runs 100% offline with no key. For
the image corpus, skip `download_images.py` and manually drop licensed-free
images into `data/images/<category-name>/`.

## Run (seed → ingest)

```bash
# 1. Build the ~50-image demo corpus (free, no card)
python scripts/download_images.py

# 2. Run the Phase 2 vision batch job — tags every image, validates against
#    the schema, retries on failure, flags low-confidence results, logs cost
python scripts/run_ingestion.py

# Check status any time:
python scripts/run_ingestion.py --summary
```

### Resuming after a rate limit / quota hit

The Gemini free tier has a daily request cap (RPD) that resets at
**midnight Pacific Time**. If a run stops mid-way with a 429 "quota
exceeded" error, don't just rerun — plain `python scripts/run_ingestion.py`
already skips anything not still `pending`, but images that failed and
landed in `error` status will keep being skipped forever unless you tell it
to retry them. Use:

```bash
python scripts/run_ingestion.py --retry-errors
```

This reprocesses only `pending` + `error` images and leaves anything
already `tagged`/`flagged` untouched — so you don't re-spend quota on
images that already succeeded. (`--rerun` reprocesses *everything*,
including already-successful images — only use that after a prompt or
threshold change you want reflected across the whole corpus.)

If you switch models mid-project (e.g. a free-tier model gets deprecated),
update `GEMINI_MODEL` in `.env` first, then use `--retry-errors` to pick up
only the images that failed under the old model.

Example output:

```
[1/50] classifying red_fox_01.jpg ...
    -> TAGGED  subject='red fox' category='animal' confidence=0.94
[2/50] classifying wolf_03.jpg ...
    -> TAGGED  subject='gray wolf' category='animal' confidence=0.89
...
=== Batch run complete ===
{
  "total": 50,
  "tagged": 46,
  "flagged": 3,
  "errored": 1,
  "skipped_already_done": 0,
  "total_estimated_cost_usd": 0.000812
}
```

`total_estimated_cost_usd` is a nominal estimate at public per-token
pricing for visibility/discipline — the actual bill on the Gemini free tier
is $0. See `cost_log.jsonl` for a line-by-line log of every call.

## Tests

```bash
pytest
```

Covers: schema validation (rejects out-of-range confidence, missing
fields, bad category), and the batch job's retry/flag/error state machine
against a mocked vision provider (no API calls, no cost, deterministic).

## Project structure

See `DESIGN.md` §5 for the full layer sketch and rationale.

## Limitations (honest, as of Phase 1+2)

- No matching, ranking, or mismatch guard yet — that's Phase 3. Right now
  this repo only tags images; it does not yet suggest images for posts.
- No review API or evaluation harness yet — that's Phase 4.
- The Ollama local path is implemented but only lightly exercised — Gemini
  free tier is the primary path this was developed against.
- `data/posts.json` is a seed set for Phase 3/4 development; it is not
  consumed by anything in Phase 1/2.

## Free-tools promise

Everything above runs at $0 with no credit card: Gemini Flash free tier
(or fully local Ollama), Pexels free image API, SQLite, and open-source
Python libraries only. See `.env.example` for every key involved.
