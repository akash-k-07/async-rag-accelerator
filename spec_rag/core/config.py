import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


LLM_TTFT_MS = _get_int("LLM_TTFT_MS", 1500)
DEFAULT_PREDICTED_LATENCY_MS = _get_float("DEFAULT_PREDICTED_LATENCY_MS", 3000.0)
QDRANT_SCORE_THRESHOLD = _get_float("QDRANT_SCORE_THRESHOLD", 0.99)
SPECULATIVE_SLEEP_SEC = _get_float("SPECULATIVE_SLEEP_SEC", 1.5)
GROUND_TRUTH_SLEEP_SEC = _get_float("GROUND_TRUTH_SLEEP_SEC", 3.0)
RECONCILE_SLEEP_SEC = _get_float("RECONCILE_SLEEP_SEC", 1.05)
L1_TTL_SECONDS = _get_int("L1_TTL_SECONDS", 300)
CACHE_HIT_RATE = _get_float("CACHE_HIT_RATE", 0.8)
BENCHMARK_QUERY_COUNT = _get_int("BENCHMARK_QUERY_COUNT", 100)
