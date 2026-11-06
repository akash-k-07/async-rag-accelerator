"""LangGraph node execution handlers for Spec-RAG."""

import time
import asyncio
import logging
import json
import xxhash
from spec_rag.core.state import AgentState
from spec_rag.core.config import (
    LLM_TTFT_MS,
    DEFAULT_PREDICTED_LATENCY_MS,
    QDRANT_SCORE_THRESHOLD,
    SPECULATIVE_SLEEP_SEC,
    GROUND_TRUTH_SLEEP_SEC,
    RECONCILE_SLEEP_SEC
)
from spec_rag.core.cache import (
    qdrant, get_dummy_embedding, get_l1_cache, set_l1_cache, set_l2_cache,
    _decrypt_payload
)
from spec_rag.tools.mock_api_server import get_shipment_status

# Configure academic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Spec-RAG] %(message)s"
)

VOLATILE_SCHEMA_PATHS = {"timestamp", "trace_id", "uuid"}


def _canonicalize_and_mask(payload_str: str, volatile_paths: set) -> str:
    """
    Parses a JSON string into a strictly ordered tree, masking any volatile
    paths (like timestamps or UUIDs) to ensure deterministic canonicalization.
    """
    try:
        payload = json.loads(payload_str)
        if not isinstance(payload, dict):
            return payload_str
    except json.JSONDecodeError:
        return payload_str

    def mask_recursive(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in volatile_paths:
                    obj[k] = "<MASKED>"
                else:
                    mask_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                mask_recursive(item)

    mask_recursive(payload)
    return json.dumps(payload, sort_keys=True)


def _compute_hash(canonical_masked_str: str) -> str:
    """Returns a high-speed xxHash3 64-bit digest of the payload."""
    return xxhash.xxh3_64_hexdigest(canonical_masked_str)


async def node_a_predictor(state: AgentState):
    """
    Node A (Fast Predictor): Checks ultra-fast L1 Memory Cache. If miss,
    embeds the query and queries the Qdrant (L2) vector database.
    """
    start_time = time.time()

    # Security/Privacy: Strictly Ephemeral, do not cache PII
    if state.get("contains_pii"):
        logging.debug("Node A: Bypassing predictor due to PII constraint.")
        return {
            "predicted_tool_response": None,
            "predicted_tool_latency": None,
            "latencies": {"predictor_ms": (time.time() - start_time) * 1000}
        }

    tool_payload = state.get('tool_payload') or {}
    payload_str = json.dumps(tool_payload, sort_keys=True)
    tool_signature = f"{state.get('tool_name', 'unknown')}:{payload_str}"

    # 1. Try L1 Cache (In-Memory, Ultra-Fast)
    predicted, predicted_latency = get_l1_cache(tool_signature)
    cache_level = "L1"
    if predicted and predicted_latency is None:
        predicted_latency = DEFAULT_PREDICTED_LATENCY_MS

    # 2. Try L2 Cache (Qdrant Vector DB) if L1 miss
    if not predicted:
        query_vector = get_dummy_embedding(tool_signature)
        search_response = qdrant.query_points(
            "speculative_cache",
            query_vector,
            limit=1,
            score_threshold=QDRANT_SCORE_THRESHOLD
        ).points

        if search_response:
            result_point = search_response[0]
            encrypted_predicted = result_point.payload.get("tool_response")
            predicted = (
                _decrypt_payload(encrypted_predicted)
                if encrypted_predicted else None
            )
            predicted_latency = result_point.payload.get(
                "tool_latency", DEFAULT_PREDICTED_LATENCY_MS
            )
            if predicted_latency is None:
                predicted_latency = DEFAULT_PREDICTED_LATENCY_MS
            cache_level = "L2 (Qdrant)"
            set_l1_cache(tool_signature, predicted, predicted_latency)

    elapsed = time.time() - start_time

    predicted_hash = None
    if predicted:
        canonical = _canonicalize_and_mask(predicted, VOLATILE_SCHEMA_PATHS)
        predicted_hash = _compute_hash(canonical)

    logging.debug(
        "Node A completed in %.4fs. Cache Hit: %s (Level: %s)",
        elapsed, predicted is not None, cache_level if predicted else "Miss"
    )
    return {
        "predicted_tool_response": predicted,
        "predicted_tool_latency": predicted_latency,
        "predicted_hash": predicted_hash,
        "latencies": {"predictor_ms": elapsed * 1000}
    }


async def node_b_speculative(state: AgentState):
    """
    Node B (Speculative Branch): Autoregressively generates a response draft
    using a local LLM based strictly on the predicted context.
    """
    start_time = time.time()
    predicted_response = state.get("predicted_tool_response")
    predicted_latency = state.get("predicted_tool_latency")
    is_safe = state.get("is_safe_to_speculate", True)
    draft = None
    tokens_generated = 0

    # Adaptive Speculation: Only speculate if tool is pure (safe) AND
    # historically slower than the LLM generation time (e.g., latency > 1000ms)
    llm_ttft_ms = LLM_TTFT_MS

    if predicted_response and is_safe:
        effective_latency = (
            predicted_latency if predicted_latency is not None
            else DEFAULT_PREDICTED_LATENCY_MS
        )
        if effective_latency >= llm_ttft_ms:
            # Simulating TTFT and Generation Time asynchronously
            await asyncio.sleep(SPECULATIVE_SLEEP_SEC)
            draft = (
                f"[Speculative Draft] The tool returned: {predicted_response}"
            )
            tokens_generated = 125  # Simulated token count
        else:
            logging.debug(
                "Node B: Bypassed speculation due to Fast Tool "
                "Latency (< %d ms).", llm_ttft_ms
            )
    elif not is_safe:
        logging.debug(
            "Node B: Bypassed speculation due to state mutation risk "
            "(not safe)."
        )

    elapsed = time.time() - start_time
    return {
        "draft_response": draft,
        "tokens_generated": tokens_generated,
        "latencies": {"speculative_llm_ms": elapsed * 1000}
    }


async def node_c_ground_truth(state: AgentState):
    """
    Node C (Ground Truth Branch): Interfaces with the high-latency
    generic API backend and updates L1 Cache with true latency.
    """
    start_time = time.time()

    # Simulate network I/O and server-side DB latency asynchronously
    if state.get("tool_name") == "get_shipment_status":
        payload = state.get("tool_payload", {})
        tracking_id = payload.get("tracking_id", "unknown")
        try:
            live_response = await asyncio.to_thread(
                get_shipment_status, tracking_id
            )
        except Exception as exc:  # pylint: disable=broad-except
            live_response = f"[API Error] {exc}"
            logging.error(
                "Node C: Live API error for %s: %s",
                tracking_id,
                exc
            )
    else:
        await asyncio.sleep(GROUND_TRUTH_SLEEP_SEC)
        tool_name = state.get('tool_name', 'unknown')
        live_response = f"Simulated response for {tool_name}"

    elapsed = time.time() - start_time
    latency_ms = elapsed * 1000

    # Update L1/L2 Caches with the real ground truth response and latency
    if not state.get("contains_pii"):
        payload_str = json.dumps(state['tool_payload'], sort_keys=True)
        tool_signature = f"{state['tool_name']}:{payload_str}"
        set_l1_cache(tool_signature, live_response, latency_ms)
        set_l2_cache(tool_signature, live_response, latency_ms)

    return {
        "live_tool_response": live_response,
        "latencies": {"api_call_ms": latency_ms}
    }


async def node_d_reconcile(state: AgentState):
    """
    Node D (Reconcile): A synchronization barrier that verifies the
    speculative draft against the ground truth using Semantic Fuzzy Matching.
    Commits on match, rolls back on violation.
    """
    start_time = time.time()

    predicted_hash = state.get("predicted_hash")

    live_resp_str = str(state.get("live_tool_response") or "")
    live_canonical = _canonicalize_and_mask(
        live_resp_str, VOLATILE_SCHEMA_PATHS
    )
    live_hash = _compute_hash(live_canonical)

    is_semantic_match = (
        predicted_hash is not None and predicted_hash == live_hash
    )
    draft_response = state.get("draft_response")

    if draft_response and is_semantic_match:
        # Cache Hit: Speculative execution was strictly equivalent (masked)!
        final_response = draft_response
        rollback = False
        tokens_wasted = 0
        logging.debug(
            "Reconciliation: Successful Deterministic Hash Match. "
            "Draft committed."
        )
    elif draft_response:
        # Cache Miss: Trigger rollback penalty and compute true response
        await asyncio.sleep(RECONCILE_SLEEP_SEC)
        live_resp = state.get("live_tool_response")
        final_response = f"[Final Truth] The tool returned: {live_resp}"
        rollback = True
        tokens_wasted = state.get("tokens_generated", 0)
        logging.debug(
            "Reconciliation: Hash Mismatch. Rollback triggered."
        )
    else:
        # No speculative draft was generated; return ground truth directly.
        live_resp = state.get("live_tool_response")
        final_response = f"[Final Truth] The tool returned: {live_resp}"
        rollback = False
        tokens_wasted = 0
        logging.debug(
            "Reconciliation: No speculative draft. Using live response."
        )

    elapsed = time.time() - start_time
    return {
        "final_response": final_response,
        "tokens_wasted": tokens_wasted,
        "latencies": {
            "reconcile_ms": elapsed * 1000,
            "rollback_flag": 1.0 if rollback else 0.0
        }
    }
