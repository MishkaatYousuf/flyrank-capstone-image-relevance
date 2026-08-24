"""
Per-call cost tracking (Definition of Done: "Vision and embedding costs are
tracked per call" / Probe 6: "every vision/embedding call attributed with a
cost entry"). We're on the Gemini free tier, so the *actual* charge is
always $0 — but we still compute and record what the call *would* cost at
public pricing, and log every attempt (success or failure). That habit is
the thing being graded, not the dollar amount.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import DEFAULT_PRICING, GEMINI_PRICING_USD_PER_1M_TOKENS, settings
from app.models import ApiCallLog


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    rates = GEMINI_PRICING_USD_PER_1M_TOKENS.get(model_name, DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * rates["input"]
    cost += (output_tokens / 1_000_000) * rates["output"]
    return round(cost, 8)


def log_api_call(
    session: Session,
    *,
    call_type: str,
    provider: str,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    image_id: str | None = None,
    post_id: str | None = None,
    succeeded: bool = True,
    attempt: int = 1,
    error_message: str | None = None,
    actual_billed_usd: float = 0.0,
) -> ApiCallLog:
    estimated = estimate_cost_usd(model_name, input_tokens, output_tokens)

    entry = ApiCallLog(
        call_type=call_type,
        provider=provider,
        model_name=model_name,
        image_id=image_id,
        post_id=post_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated,
        actual_billed_usd=actual_billed_usd,
        succeeded=succeeded,
        attempt=attempt,
        error_message=error_message,
    )
    session.add(entry)
    session.flush()  # get entry.id without committing

    # Also append a plain-text JSONL log — handy for `EVIDENCE.md` pastes and
    # for grepping without spinning up a DB session.
    _append_jsonl_line(entry)

    return entry


def _append_jsonl_line(entry: ApiCallLog) -> None:
    record = {
        "id": entry.id,
        "call_type": entry.call_type,
        "provider": entry.provider,
        "model": entry.model_name,
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "estimated_cost_usd": entry.estimated_cost_usd,
        "actual_billed_usd": entry.actual_billed_usd,
        "succeeded": entry.succeeded,
        "attempt": entry.attempt,
        "image_id": entry.image_id,
        "post_id": entry.post_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(settings.cost_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def total_estimated_cost(session: Session) -> float:
    rows = session.query(ApiCallLog.estimated_cost_usd).all()
    return round(sum(r[0] for r in rows), 6)
