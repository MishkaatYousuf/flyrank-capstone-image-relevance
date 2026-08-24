# Design Doc — AI Image Understanding & Content Matching Engine

This is the Phase 1 deliverable required by the capstone's Ground Rule
"Pick one, early": a one-page design doc covering the problem, data model,
API surface, layer sketch, and one explicit non-goal.

## 1. Problem

Given a library of images and a set of blog posts, automatically:
1. Understand what each image actually depicts (structured tags + caption).
2. Rank candidate images against each post by semantic similarity.
3. Refuse a suggestion when it isn't actually a good match, and explain why.

The hard requirement is not "find a plausible image" — it's "know when the
best available image is still wrong, and say so."

## 2. Image metadata schema

Every image, after vision processing, must validate against this shape
(`app/schemas.py::ImageTags`):

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

- `subject` — free-text but constrained (1-80 chars, non-blank): the single
  main subject, kept specific ("red fox", not just "animal").
- `category` — closed enum (`animal | landscape | people | object | other`)
  so downstream filtering/matching has a stable bucket to group on.
- `attributes` — 1-8 short descriptive tags.
- `caption` — one sentence, used as the text that gets embedded for
  semantic matching in Phase 3.
- `confidence` — the model's own 0-1 self-estimate. This is the field the
  mismatch guard and the low-confidence flag both key off of.

Never trusted directly: raw model output is parsed and run through
`ImageTags.model_validate()`. Anything that fails — bad JSON, missing
field, out-of-range confidence, a category outside the enum — is treated
as a failed call, not a partial success (see §5).

## 3. Matching strategy + mismatch guard (sketch — implemented in Phase 3)

Two embedding streams meet at ranking time:

```
image.caption --embed--> image_vector   (one per image, stored once)
post.body_text --embed--> post_vector   (one per post, stored once)

for each post:
    candidates = top_k(image_vectors, by=cosine_similarity(post_vector))
    for each candidate, in ranked order:
        guard_result = mismatch_guard(post, candidate)
        if guard_result.approved: return guard_result   # first approved wins
    return NoConfidentMatch(reasons=[...])
```

The **mismatch guard** combines three signals before anything reaches a
human:

1. **Tag/category check** — does the candidate's `subject`/`category`
   plausibly relate to the post's inferred subject? (e.g. post about foxes,
   candidate tagged "wolf" → hard category mismatch, reject regardless of
   embedding similarity — this is what makes the wolf-on-a-fox-post demo
   provable rather than probabilistic.)
2. **Similarity threshold** — cosine similarity must clear a
   tuned-from-eval-data cutoff (not guessed — see §7 Realistic scope /
   Phase 4). Below the cutoff → "no confident match."
3. **Confidence gate** — if the candidate image itself was FLAGGED
   (low vision confidence), it cannot be auto-approved even if similarity
   and tags look fine; it needs human review first.

Every rejection carries a machine-readable reason string, e.g.
`"Animal category mismatch: expected fox, detected wolf"` or
`"Similarity 0.31 below threshold 0.55"`.

## 4. Database design

SQLite by default (via SQLAlchemy) — $0, zero config, more than enough at
~50 images and a handful of posts. `DATABASE_URL` is the only thing that'd
change to point this at Postgres later; nothing else in the app knows or
cares which database is behind the ORM.

| Table | Purpose | Key columns | Populated |
|---|---|---|---|
| `images` | one row per source image | `status`, `subject`, `category`, `attributes` (JSON), `confidence`, `raw_response`, `attempts` | Phase 2 |
| `api_call_log` | cost tracking, one row per AI call attempt | `call_type`, `model_name`, `input_tokens`, `output_tokens`, `estimated_cost_usd`, `succeeded`, `attempt` | Phase 2 |
| `posts` | blog posts to match images against | `slug`, `title`, `body_text` | Phase 4 (seeded from `data/posts.json`) |
| `image_embeddings` | one embedding vector per image | `image_id` (FK), `model_name`, `vector` (JSON floats) | Phase 3 |
| `post_embeddings` | one embedding vector per post | `post_id` (FK), `model_name`, `vector` | Phase 3 |
| `suggestions` | ranked candidate + guard verdict per post | `post_id` (FK), `image_id` (FK), `similarity_score`, `rank`, `guard_status`, `guard_reason` | Phase 3 |
| `reviews` | human approve/reject on a suggestion | `suggestion_id` (FK, unique), `decision`, `reviewer`, `notes` | Phase 4 |

Indexes: `images.status`, `images.category` (filtering by processing state
and by category), `api_call_log.created_at` + `.call_type` (cost queries),
`suggestions.post_id` + `.image_id` (the two directions the review API
needs to query).

Full SQLAlchemy models: `app/models.py`.

## 5. Layer sketch

```
scripts/            <- CLI entrypoints (thin; no business logic)
  download_images.py    Phase 1: build the free image corpus
  run_ingestion.py       Phase 2: run the vision batch job

app/
  config.py          <- all env/config reads, nothing else touches os.getenv
  schemas.py          <- Pydantic contracts (ImageTags, VisionCallResult)
  database.py / models.py   <- SQLAlchemy engine + full schema
  vision.py           <- provider-specific API calls; ALWAYS returns a
                          validated ImageTags or raises — never a raw dict
  cost_tracker.py     <- one function every AI call goes through
  batch.py            <- orchestration: retry policy, status transitions,
                          nothing here talks to Gemini/Ollama directly

  (Phase 3+) matching.py, guard.py
  (Phase 4+) main.py (FastAPI), review routes, eval.py
```

The rule that keeps this swappable: `vision.py` is the only file that knows
about Gemini or Ollama specifically; `batch.py` only knows about
`classify_image() -> VisionCallResult | raises`. Swapping providers, or
adding a third one, never touches `batch.py`, `models.py`, or the schema.

## 6. Non-goal (explicit)

**Not building:** a general-purpose image search engine, a model-comparison
harness, or a public-facing frontend. One vision model + one embedding
model is enough (§7 of the brief) — comparing models is a stretch goal,
not core, and is out of scope for this project entirely.
