# AI Image Understanding & Content Matching Engine

FlyRank Internship · Backend Track Capstone. Understands an image library,
tags it automatically with a vision model, matches each image to the right
blog post by meaning (not keywords), and refuses to guess when nothing fits
well enough — with a Review API to approve, reject, and inspect why.

**Status: all four phases complete.** Design → vision ingestion → semantic
matching + mismatch guard → Review API + eval. See `DESIGN.md` for the
original design doc and `BUILDLOG.md` for an honest account of where AI
helped at each phase.

## What's implemented

- ✅ Image metadata schema, Pydantic-validated (`app/schemas.py`) — invalid
  vision output is retried, never trusted
- ✅ Full database schema for all phases (`app/models.py`), SQLite, $0, no Docker
- ✅ ~50-image dataset via a free Pexels download script
- ✅ Vision processing batch job with retries, exponential backoff, and
  low-confidence flagging (`app/vision.py`, `app/batch.py`)
- ✅ Per-call cost tracking for every vision AND embedding call, even
  though the free tier bills $0 (`app/cost_tracker.py`)
- ✅ Embeddings for image captions + post text, cosine-similarity ranking
  (`app/embeddings.py`, `app/matching.py`)
- ✅ The mismatch guard: confidence gate → tag/category check → similarity
  threshold, each rejection with a human-readable reason (`app/guard.py`)
- ✅ A Review API (FastAPI): ranked suggestions per post, approve/reject,
  inspect why, idempotent by design (`app/main.py`, `app/review.py`)
- ✅ A labeled eval set measuring top-1 precision (`data/eval_set.json`,
  `scripts/eval.py`)
- ✅ 57 automated tests: schema validation, batch retry/flag/error states,
  the mismatch guard (including an adversarial case), the matching
  pipeline, the review workflow, and the full HTTP API — all mocked, no
  network, no cost, fully deterministic

## Architecture

```
Images ─(batch job)─► Vision Model ─► {tags, caption, confidence} ─► image_metadata (SQLite)
         (app/batch.py, app/vision.py)                             ─► api_call_log (cost)
                            │
                            └─► embed(caption) ──────────────────────► image_vectors
Posts ──────────────────────────► embed(post text) ───────────────────► post_vectors
         (app/embeddings.py, app/matching.py)                       ─► api_call_log (cost)

GET /posts/:id/images                              (app/main.py, app/review.py)
  └─► Similarity Ranking — cosine(image_vectors, post_vector)        (app/matching.py)
  └─► Mismatch Guard — tags + threshold + confidence, ranked order   (app/guard.py)
      ├─► Suggested image (ranked, explained)     → status="matched"
      └─► "No confident match" + explanation       → status="no_confident_match"
  └─► POST /suggestions/:id/review — approve / reject                (app/review.py)
  └─► GET  /suggestions/:id — inspect why a candidate was picked/refused
  └─► GET  /costs/summary — Probe 6: every AI call attributed with a cost
```

Layering (why each piece exists where it does — see `DESIGN.md` §5 for the
full rationale): `app/main.py` is the only file that knows about HTTP;
`app/review.py` and `app/matching.py` hold the actual logic and know
nothing about FastAPI; `app/vision.py` and `app/embeddings.py` are the only
files that know about Gemini vs. Ollama. Swap any one layer without
touching the others.

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
then `ollama pull llava` (vision) and `ollama pull all-minilm` (embeddings).
The whole pipeline runs 100% offline with no key. For the image corpus,
skip `download_images.py` and manually drop licensed-free images into
`data/images/<category-name>/`.

## Run — full pipeline

```bash
# 1. Build the ~50-image demo corpus (free, no card)
python scripts/download_images.py

# 2. Tag every image: vision model, schema validation, retries, cost log
python scripts/run_ingestion.py
python scripts/run_ingestion.py --summary       # check status any time

# 3. Seed posts, embed images + posts, rank + guard every post
python scripts/run_matching.py

# 4. Measure quality against the labeled eval set
python scripts/eval.py

# 5. Start the Review API
uvicorn app.main:app --reload
# -> interactive docs at http://127.0.0.1:8000/docs
```

### Resuming after a rate limit / quota hit

The Gemini free tier has a daily request cap (RPD) that resets at
**midnight Pacific Time**. `run_ingestion.py` and `run_matching.py`'s embed
step both skip anything already processed, but images/posts that failed
stay in an error state until retried explicitly:

```bash
python scripts/run_ingestion.py --retry-errors   # only re-tags images that errored
python scripts/run_matching.py --embed-only       # only embeds what's still missing
```

`--rerun` on `run_ingestion.py` reprocesses _everything_, including
already-successful images — only use that after a prompt/threshold change
you want reflected across the whole corpus.

## The Review API

| Method | Path                      | What it does                                                                         |
| ------ | ------------------------- | ------------------------------------------------------------------------------------ |
| GET    | `/health`                 | liveness check                                                                       |
| GET    | `/posts`                  | list all posts                                                                       |
| GET    | `/images?status=flagged`  | list images, optionally filtered by status                                           |
| GET    | `/posts/:id/images`       | ranked suggestions + guard verdicts for a post (`:id` accepts a slug or a UUID)      |
| GET    | `/suggestions/:id`        | inspect one candidate: image, score, guard reason, review status                     |
| POST   | `/suggestions/:id/review` | `{"decision": "approve"\|"reject", "reviewer"?, "notes"?}` — 409 if already reviewed |
| GET    | `/costs/summary`          | aggregated AI call cost log                                                          |

Example: force-checking the wolf-on-a-fox-post scenario from the terminal
without touching a browser:

```bash
curl "http://127.0.0.1:8000/posts/the-secret-life-of-red-foxes/images" | python -m json.tool
```

The response's `candidates` array includes every ranked image — including
a rejected wolf, if one is in your corpus — each with its own
`guard_status` and `guard_reason`.

## Evaluation — top-1 precision

`data/eval_set.json` labels each seed post with its correct animal subject
(or `null` for the one post — "Mountain Lakes at Sunrise" — that has no
matching image in the corpus on purpose, to test correct refusal).
`scripts/eval.py` runs the full matching + guard pipeline against each
labeled post and reports:

```
Top-1 precision: 83% (5/6)
```

## Tests

```bash
pytest -q
```

57 tests, all mocked (no network, no cost, fully deterministic):

- `test_schema_validation.py` — the vision-output contract rejects invalid shapes
- `test_batch.py` — retry/flag/error state machine, cost logging, quota-safe resume
- `test_guard.py` — the mismatch guard's decision logic, including the wolf-on-a-fox-post case
- `test_matching.py` — embedding storage + retries, similarity ranking, and an
  **adversarial** version of the fox/wolf test where the wolf is given a
  deliberately higher raw similarity score and the guard still rejects it
- `test_review.py` — idempotent suggestion computation, 404/409 domain errors
- `test_api.py` — the full HTTP API via FastAPI's TestClient, covering Probes 2-6

## Project structure

See `DESIGN.md` §5 for the full layer sketch and rationale.

## Limitations (honest)

- The similarity threshold (`SIMILARITY_THRESHOLD=0.55` in `.env.example`)
  is a reasonable starting default, not one tuned against a large labeled
  set — the 6-post eval set here is illustrative at the scope this
  capstone calls for (§7), not a statistically rigorous validation.
- The tag/category mismatch check in `app/guard.py` uses a small explicit
  subject vocabulary (fox/wolf/dog/bear/deer + a few scientific-name
  synonyms) rather than a general NLP subject extractor. That's
  appropriate for the "few categories" scope in §7 and keeps every
  rejection reason fully explainable and traceable to a specific rule
  rather than an opaque model judgment — but it won't generalize to an
  arbitrary corpus without extending `SUBJECT_SYNONYMS`/`KNOWN_SUBJECTS`.
- The Ollama local path (vision + embeddings) is implemented and unit-testable but only lightly exercised end-to-end — Gemini free tier is the primary path this was developed against.
- No frontend — per §7, the Review API + interactive `/docs` is the
  reviewing interface. No admin table beyond what `/docs` renders.
- `/images` and `/posts` have no pagination — fine at the ~50-image /
  handful-of-posts scope this capstone targets; would need pagination
  before growing much further.

## Free-tools promise

Everything above runs at $0 with no credit card: Gemini Flash + Gemini
embeddings free tier (or fully local Ollama for both), Pexels free image
API, SQLite, FastAPI, and open-source Python libraries only. See
`.env.example` for every key involved.
