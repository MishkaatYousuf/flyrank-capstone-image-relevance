# Build Log

Honest record of where AI helped, where it was wrong, and what was changed.
Per the capstone ground rules: "The AI wrote it" is not an answer at demo
time — this log is what you'd draw on to explain any given line.

## Phase 1 — Design

- Used Claude to scaffold the initial project structure, the `ImageTags`
  Pydantic schema, and the full SQLAlchemy schema for all four phases based
  on the capstone brief's §4/§6/§11 requirements.
- **You should verify/adjust:** the `ImageCategory` enum values were chosen
  to fit the demo corpus (animal/landscape/people/object/other) — extend if
  your own corpus needs more buckets.
- **You should verify:** the exact free-tier rate limits and pricing quoted
  in `app/config.py` comments and `.env.example` — these change over time;
  check https://aistudio.google.com for your account's live limits before
  relying on them for a large batch run.

## Phase 2 — Vision ingestion pipeline

- Used Claude to write the Gemini Interactions API integration
  (`app/vision.py`), the retry/backoff/flagging state machine
  (`app/batch.py`), and cost tracking (`app/cost_tracker.py`), based on
  Google's published Gemini API docs (structured outputs + image
  understanding pages) as of August 2026.
- **You should verify before your demo:** run the pipeline against your own
  Gemini API key end-to-end at least once — the Gemini Interactions API is
  relatively new (GA in 2026) and worth confirming still matches this code
  against the live docs at https://ai.google.dev/gemini-api/docs if time
  has passed since this was written.
- **What was changed from a first draft:** the initial JSON-parsing error
  path used a private Pydantic API (`ValidationError.from_exception_data`)
  to synthesize a validation error from a JSON decode failure — swapped for
  a plain custom `InvalidVisionOutput` exception instead, since relying on
  Pydantic internals was fragile and unnecessary.
- **Mid-Phase-2 real-world incident:** hit a Gemini model deprecation
  (2.5 → 3.6) mid-run, plus a daily-quota exhaustion, on a real account.
  Added a `--retry-errors` CLI flag (only reprocesses `pending`/`error`
  images, leaving already-tagged ones untouched) so switching models or
  resuming after a quota reset never re-spends quota on images that had
  already succeeded. This came from an actual failure, not a hypothetical.

## Phase 3 — Matching engine

- Used Claude to write the Gemini embeddings integration
  (`app/embeddings.py`, `gemini-embedding-001` with `SEMANTIC_SIMILARITY`
  task type), the cosine-similarity ranking (`app/matching.py`), and the
  mismatch guard (`app/guard.py`), based on Google's published embeddings
  docs as of August 2026.
- **Design decision worth explaining at demo:** the guard's tag/category
  check uses a small explicit subject vocabulary + synonym table
  (`SUBJECT_SYNONYMS`/`KNOWN_SUBJECTS` in `app/guard.py`) rather than
  asking a model "does this image match this post?" — deliberately, so
  every rejection reason is traceable to an explainable rule instead of a
  second opaque AI judgment layered on top of the first. This is a scope
  trade-off appropriate to the "few animal categories" corpus (§7), not a
  general solution.
- **Test rigor decision:** wrote an adversarial test
  (`test_matching.py::test_fox_post_matches_fox_even_when_wolf_scores_higher_on_similarity_alone`)
  that deliberately gives the wolf image a _higher_ raw cosine similarity
  than the fox image, to prove the guard's category check does real work
  independent of embedding quality — not just a happy-path test that
  happens to pass because the embeddings were already well-behaved.
- **You should verify:** `SIMILARITY_THRESHOLD=0.55` (in `app/config.py` /
  `.env.example`) is a reasonable starting default, not tuned against your
  own eval data — re-check it against your real `scripts/eval.py` output
  and adjust if precision is lower than expected.

## Phase 4 — Production layer

- Used Claude to write the FastAPI Review API (`app/main.py`,
  `app/api_schemas.py`) and the underlying business logic
  (`app/review.py`), plus the eval script (`scripts/eval.py`) and labeled
  eval set (`data/eval_set.json`).
- **Design decision worth explaining at demo:** `app/review.py` contains
  zero FastAPI imports — it's plain functions over SQLAlchemy sessions,
  tested directly in `tests/test_review.py` with no HTTP layer involved.
  `app/main.py` is a thin adapter on top. This is the "swap the framework
  without touching business logic" property the rubric's Architecture
  dimension asks for.
- **Idempotency decision:** `get_or_compute_suggestions` in `app/review.py`
  does NOT re-rank on every `GET /posts/:id/images` call — it reuses stored
  suggestions unless `recompute=True` is passed, and even then preserves
  suggestions that already have a human review attached. `create_review`
  enforces one review per suggestion at the DB level (unique constraint) so
  a retried "approve" click can't double-write or silently overwrite a
  prior "reject" — mapped to a 409, not a 500 or a silent no-op.
- **You should verify before your demo:** run `python scripts/eval.py`
  against your own populated database and paste the real precision number
  into `README.md` and this file's Probe 5 proof — both currently show a
  worked example, not your actual measured result.
- **Testing note:** discovered mid-build that SQLite's `:memory:` database
  is per-connection, not per-process — FastAPI's TestClient runs each
  request in a worker thread, so without `poolclass=StaticPool` on the test
  engine, each request got its own empty database ("no such table: posts").
  Fixed in `tests/conftest.py::api_client`.
