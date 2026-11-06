"""
Spec-RAG Benchmarking Suite
===========================

Developed by: Principal Systems Architect & Doctoral Research Fellow
Date: July 2026

Description:
A comprehensive empirical evaluation script for Spec-RAG.
Simulates high-throughput analytical workloads, capturing
End-to-End Latency, P95/P99 distributions, and cache-hit
rollback penalties. Outputs quality plots and JSON metrics.
"""

import time
import os
import json
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from spec_rag.orchestrator.graph import app
from spec_rag.core.cache import set_l2_cache
from spec_rag.core.config import BENCHMARK_QUERY_COUNT, CACHE_HIT_RATE

sns.set_theme(style="whitegrid")


def seed_qdrant_cache(n_queries: int, target_hit_rate: float):
    """
    Deterministically seeds the Qdrant vector database to achieve
    the exact target empirical cache hit rate.
    """
    num_hits = int(n_queries * target_hit_rate)

    for i in range(num_hits):
        tracking_id = f"TRK{1000 + i}"
        payload = {"tracking_id": tracking_id}
        tool_signature = (
            f"get_shipment_status:{json.dumps(payload, sort_keys=True)}"
        )
        context = json.dumps({
            "tracking_id": tracking_id,
            "status": "out_for_delivery",
            "estimated_arrival": "17:00",
            "timestamp": "2026-01-01T00:00:00",
            "trace_id": "seeded-trace-id"
        })

        set_l2_cache(tool_signature, context, 3000.0)

    print(f"[Pre-Warming] Loaded {num_hits} predictive contexts into Qdrant.")


def run_standard_rag(_query, _tracking_id):
    """
    Standard sequential RAG:
    Tool Call (3.0s) -> LLM Generation (1.5s)
    """
    start_time = time.time()
    time.sleep(3.0)  # Simulated MCP tool latency
    time.sleep(1.5)  # Simulated LLM Token Generation
    return time.time() - start_time


def run_spec_rag(query, tracking_id):
    """
    Executes the Spec-RAG parallel DAG and extracts granular latencies.
    """
    start_time = time.time()

    initial_state = {
        "query": query,
        "tool_name": "get_shipment_status",
        "tool_payload": {"tracking_id": tracking_id},
        "predicted_tool_response": None,
        "predicted_tool_latency": None,
        "live_tool_response": None,
        "is_safe_to_speculate": True,
        "contains_pii": False,
        "draft_response": None,
        "final_response": None,
        "tokens_generated": 0,
        "tokens_wasted": 0,
        "latencies": {}
    }

    result = asyncio.run(app.ainvoke(initial_state))
    e2e_time = time.time() - start_time

    rollback = result["latencies"].get("rollback_flag", 0.0) == 1.0
    return e2e_time, rollback, result["latencies"]


def main():
    """Main execution loop for advanced benchmarking metrics."""
    print("==================================================")
    print(" Spec-RAG Empirical Benchmarking Suite")
    print(" Architecture: Parallel DAG w/ Speculative Decoding")
    print("==================================================\n")

    n_queries = BENCHMARK_QUERY_COUNT
    cache_hit_rate = CACHE_HIT_RATE

    # Pre-populate Qdrant Vector DB to achieve the expected hit rate
    seed_qdrant_cache(n_queries, cache_hit_rate)

    results = []

    print("\nExecuting workload...")
    for i in tqdm(range(n_queries), desc="Simulating Queries", unit="req"):
        tracking_id = f"TRK{1000 + i}"
        query = f"Where is my package {tracking_id}?"

        # 1. Standard Execution
        std_time = run_standard_rag(query, tracking_id)

        # 2. Spec-RAG Execution
        spec_time, rollback, lat_metrics = run_spec_rag(query, tracking_id)

        results.append({
            "query_id": i,
            "tracking_id": tracking_id,
            "standard_latency_s": std_time,
            "spec_rag_latency_s": spec_time,
            "rollback_triggered": rollback,
            **lat_metrics
        })

    df = pd.DataFrame(results)

    # Generate Advanced Metrics
    metrics = {
        "workload": {
            "total_queries": n_queries,
            "target_cache_hit_rate": cache_hit_rate,
            "actual_rollback_rate": float(df["rollback_triggered"].mean()),
        },
        "standard_rag": {
            "mean_latency_s": float(df["standard_latency_s"].mean()),
            "p50_latency_s": float(df["standard_latency_s"].quantile(0.50)),
            "p95_latency_s": float(df["standard_latency_s"].quantile(0.95)),
            "p99_latency_s": float(df["standard_latency_s"].quantile(0.99)),
        },
        "spec_rag": {
            "mean_latency_s": float(df["spec_rag_latency_s"].mean()),
            "p50_latency_s": float(df["spec_rag_latency_s"].quantile(0.50)),
            "p95_latency_s": float(df["spec_rag_latency_s"].quantile(0.95)),
            "p99_latency_s": float(df["spec_rag_latency_s"].quantile(0.99)),
        },
        "performance_gain": {
            "mean_latency_reduction_s": float(
                df["standard_latency_s"].mean() -
                df["spec_rag_latency_s"].mean()
            ),
            "mean_speedup_factor": float(
                df["standard_latency_s"].mean() /
                df["spec_rag_latency_s"].mean()
            )
        }
    }

    # Export artifacts
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "benchmark_raw_data.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(run_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f_out:
        json.dump(metrics, f_out, indent=4)

    # Latency-Quality Plotting
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting
    arch_labels = ['Standard RAG'] * n_queries + ['Spec-RAG'] * n_queries
    latency_vals = pd.concat(
        [df['standard_latency_s'], df['spec_rag_latency_s']]
    )
    plot_data = pd.DataFrame({
        'Architecture': arch_labels,
        'Latency (s)': latency_vals
    })

    sns.boxplot(
        x='Architecture',
        y='Latency (s)',
        data=plot_data,
        hue='Architecture',
        palette="Set2"
    )
    sns.stripplot(
        x='Architecture',
        y='Latency (s)',
        data=plot_data,
        color=".25",
        alpha=0.5,
        jitter=True
    )

    plt.title(
        f"End-to-End Latency Distribution\n"
        f"(N={n_queries}, Target Hit Rate={cache_hit_rate*100:.0f}%, "
        f"Speedup={metrics['performance_gain']['mean_speedup_factor']:.2f}x)",
        fontsize=14,
        fontweight='bold'
    )
    plt.ylabel("Latency (Seconds)", fontsize=12)
    plt.xlabel("Architecture Pattern", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "latency_plot.png")
    plt.savefig(plot_path, dpi=300)

    print("\n==================================================")
    print(" Benchmarking Complete")
    print("==================================================")
    std_mean = metrics['standard_rag']['mean_latency_s']
    std_p95 = metrics['standard_rag']['p95_latency_s']
    spec_mean = metrics['spec_rag']['mean_latency_s']
    spec_p95 = metrics['spec_rag']['p95_latency_s']

    print(f"Standard RAG Mean:  {std_mean:.3f}s (P95: {std_p95:.3f}s)")
    print(f"Spec-RAG Mean:      {spec_mean:.3f}s (P95: {spec_p95:.3f}s)")
    speedup = metrics['performance_gain']['mean_speedup_factor']
    print(f"Average Speedup:    {speedup:.2f}x")
    rollback_rt = metrics['workload']['actual_rollback_rate'] * 100
    print(f"Rollback Penalty:   {rollback_rt:.1f}% Rollbacks")
    print(
        f"Artifacts saved in {run_dir}/: benchmark_raw_data.csv, "
        "benchmark_report.json, latency_plot.png"
    )


if __name__ == "__main__":
    main()
