# Evidence

One pasted proof per Definition-of-Done checkbox (capstone brief §6).
Phase 1 + 2 boxes below; Phase 3/4 boxes to be filled in as those phases land.

## Vision model produces structured output validated against a schema; invalid responses are never trusted.

`app/schemas.py::ImageTags` is the schema. `app/vision.py::_parse_and_validate`
runs every raw response through `ImageTags.model_validate()`; a JSON or
schema failure raises `InvalidVisionOutput`, which `app/batch.py` catches
and retries — it never reaches the database as a "tagged" result.

```
# TODO after your first real run: paste one PASSING test output here, e.g.
$ pytest tests/test_schema_validation.py -v
```

## Low-confidence classifications are flagged instead of accepted.

`app/batch.py::process_image` — `if tags.is_low_confidence(threshold): status = FLAGGED else TAGGED`.

```
# TODO: paste one real FLAGGED log line from your own run, e.g.:
[7/50] classifying deer_04.jpg ...
    -> FLAGGED (low confidence) subject='deer' confidence=0.52
```

## Images are processed through a batch background job with retries.

`app/batch.py::_classify_with_retries` — up to `MAX_VISION_RETRIES` attempts
with exponential backoff, driven by `scripts/run_ingestion.py`.

```
# TODO: paste one real retry warning line, e.g.:
2026-08-24 10:02:11 WARNING vision call failed for wolf_09.jpg (attempt 1/3): ...
```

## Vision and embedding costs are tracked per call.

`app/cost_tracker.py::log_api_call`, called from every branch (success and
failure) of `_classify_with_retries`. Writes to the `api_call_log` table
and `cost_log.jsonl`.

```
# TODO: paste one real line from cost_log.jsonl, e.g.:
{"id": "...", "call_type": "vision", "provider": "gemini", "model": "gemini-2.5-flash", "input_tokens": 1284, "output_tokens": 96, "estimated_cost_usd": 0.00062, ...}
```

---

*Remaining Definition-of-Done boxes (semantic matching, mismatch guard,
review API, eval precision, tests beyond schema validation) belong to
Phase 3 and Phase 4 and are not claimed as done here.*
