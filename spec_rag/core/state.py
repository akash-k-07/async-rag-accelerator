"""State definitions for the Spec-RAG DAG."""

import operator
from typing import TypedDict, Optional, Dict, Annotated, Any


class AgentState(TypedDict):
    """Distributed state dictionary for the Spec-RAGDAG."""
    query: str
    tool_name: str
    tool_payload: Dict[str, Any]
    predicted_tool_response: Optional[str]
    predicted_tool_latency: Optional[float]
    predicted_hash: Optional[str]
    live_tool_response: Optional[str]
    is_safe_to_speculate: bool
    contains_pii: bool
    draft_response: Optional[str]
    final_response: Optional[str]
    tokens_generated: Optional[int]
    tokens_wasted: Optional[int]
    latencies: Annotated[Dict[str, float], operator.ior]
