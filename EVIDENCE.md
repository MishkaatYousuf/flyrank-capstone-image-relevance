# Evidence

One pasted proof per Definition-of-Done checkbox (capstone brief §6).
Phase 1 + 2 boxes below; Phase 3/4 boxes to be filled in as those phases land.

## Vision model produces structured output validated against a schema; invalid responses are never trusted.

`app/schemas.py::ImageTags` is the schema. `app/vision.py::_parse_and_validate`
runs every raw response through `ImageTags.model_validate()`; a JSON or
schema failure raises `InvalidVisionOutput`, which `app/batch.py` catches
and retries — it never reaches the database as a "tagged" result.

```
# TODO
After running: $ pytest tests/test_schema_validation.py -v, we see:
![alt text](image.png)

```

## Low-confidence classifications are flagged instead of accepted.

`app/batch.py::process_image` — `if tags.is_low_confidence(threshold): status = FLAGGED else TAGGED`.

```
# TODO: paste one FLAGGED log line from my own run, e.g.:
![alt text](image-1.png)
[7/50] classifying deer_04.jpg ...
    -> FLAGGED (low confidence) subject='deer' confidence=0.52
```

## Images are processed through a batch background job with retries.

`app/batch.py::_classify_with_retries` — up to `MAX_VISION_RETRIES` attempts
with exponential backoff, driven by `scripts/run_ingestion.py`.

```
# TODO: pasted one retry warning line:
2026-08-24 10:02:11 WARNING vision call failed for wolf_09.jpg (attempt 1/3): ...
```

## Vision and embedding costs are tracked per call.

`app/cost_tracker.py::log_api_call`, called from every branch (success and
failure) of `_classify_with_retries`. Writes to the `api_call_log` table
and `cost_log.jsonl`.

```
# TODO: pasting one real line from cost_log.jsonl, e.g.:
![alt text](image-2.png)
Contains: {"id": "9144c436-9826-4eaa-ad9a-e3276c4a0cc8", "call_type": "vision", "provider": "gemini", "model": "gemini-3.6-flash", "input_tokens": 1289, "output_tokens": 88, "estimated_cost_usd": 0.00123075, "actual_billed_usd": 0.0, "succeeded": true, "attempt": 1, "image_id": "85a1d003-69d0-4b65-85b1-32b5975c9885", "post_id": null, "timestamp": "2026-08-26T09:24:44.030148+00:00"}
```

---

_Remaining Definition-of-Done boxes (semantic matching, mismatch guard,
review API, eval precision, tests beyond schema validation) belong to
Phase 3 and Phase 4 and are not claimed as done here._
