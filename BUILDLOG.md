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

## What to fill in yourself as you build

- [ ] Which specific test cases you added/changed and why
- [ ] Any prompt-engineering iterations on `app/vision.py::PROMPT` (what
      confidence calibration looked like before vs. after tuning)
- [ ] Phase 3/4 entries once those phases are built
