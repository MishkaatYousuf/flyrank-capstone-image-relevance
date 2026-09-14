"""
Central place every module reads settings from. Nothing else in the app
should call os.getenv() directly — that keeps all the "what env var controls
this" knowledge in one file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of current working directory.
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass(frozen=True)
class Settings:
    # Vision provider
    vision_provider: str = os.getenv("VISION_PROVIDER", "gemini")

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # Ollama (free local fallback — no API key required)
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llava")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Pexels (free image corpus)
    pexels_api_key: str = os.getenv("PEXELS_API_KEY", "")

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./capstone.db")

    # Pipeline tuning
    low_confidence_threshold: float = field(
        default_factory=lambda: _get_float("LOW_CONFIDENCE_THRESHOLD", 0.6)
    )
    max_vision_retries: int = field(
        default_factory=lambda: _get_int("MAX_VISION_RETRIES", 3)
    )
    vision_retry_backoff_seconds: float = field(
        default_factory=lambda: _get_float("VISION_RETRY_BACKOFF_SECONDS", 2.0)
    )
    max_embedding_retries: int = field(
        default_factory=lambda: _get_int("MAX_EMBEDDING_RETRIES", 3)
    )
    embedding_retry_backoff_seconds: float = field(
        default_factory=lambda: _get_float("EMBEDDING_RETRY_BACKOFF_SECONDS", 2.0)
    )

    # Phase 3: matching + mismatch guard tuning
    similarity_threshold: float = field(
        default_factory=lambda: _get_float("SIMILARITY_THRESHOLD", 0.55)
    )
    match_top_k: int = field(default_factory=lambda: _get_int("MATCH_TOP_K", 5))

    # Paths
    root_dir: Path = _ROOT
    images_dir: Path = _ROOT / "data" / "images"
    cost_log_path: Path = _ROOT / "cost_log.jsonl"

    def validate_for_vision(self) -> None:
        """Fail loudly and early rather than 20 images into a batch job."""
        if self.vision_provider == "gemini" and not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "a free key from https://aistudio.google.com/apikey"
            )


settings = Settings()

# Nominal per-token pricing used ONLY for cost-tracking discipline (Definition
# of Done requires "vision and embedding costs tracked per call" even when
# the actual bill is $0 on the free tier). Source: public Gemini API pricing
# page at time of writing. Update if pricing changes; this never causes a
# real charge on the free tier, it's a shadow/estimated cost for visibility.
GEMINI_PRICING_USD_PER_1M_TOKENS = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Gemini 3.x family — 2.5 was still listed as free-tier by Google's docs
    # when this was written, but availability changes fast; always check
    # https://ai.google.dev/gemini-api/docs/models for the current live list
    # before assuming a model name still works.
    "gemini-3.6-flash": {"input": 0.75, "output": 3.00},
    "gemini-3.5-flash": {"input": 0.75, "output": 3.00},
    "gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    # Embeddings: input-only cost (no output tokens generated).
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
    "text-embedding-004": {"input": 0.00, "output": 0.0},  # legacy, was free
}
DEFAULT_PRICING = {"input": 0.30, "output": 2.50}