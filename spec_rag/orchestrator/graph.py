"""LangGraph DAG definitions and compiler."""

from langgraph.graph import StateGraph, START, END
from spec_rag.core.state import AgentState
from spec_rag.nodes.handlers import (
    node_a_predictor,
    node_b_speculative,
    node_c_ground_truth,
    node_d_reconcile
)


def build_graph():
    """Builds and compiles the Spec-RAG."""
    workflow = StateGraph(AgentState)

    workflow.add_node("Predictor", node_a_predictor)
    workflow.add_node("Speculative", node_b_speculative)
    workflow.add_node("GroundTruth", node_c_ground_truth)
    workflow.add_node("Reconcile", node_d_reconcile)

    workflow.add_edge(START, "Predictor")
    workflow.add_edge("Predictor", "Speculative")
    workflow.add_edge("Predictor", "GroundTruth")
    workflow.add_edge("Speculative", "Reconcile")
    workflow.add_edge("GroundTruth", "Reconcile")
    workflow.add_edge("Reconcile", END)

    return workflow.compile()


# Expose a compiled instance
app = build_graph()
