"""
Comprehensive benchmark for Chapter 9.
Generates performance data for tables and figures.
"""

import json
import statistics
import time
from typing import Dict

from semantic_cache_core import SemanticCacheRAG


def run_comprehensive_benchmark(num_queries: int = 1000) -> Dict:
    """Run complete benchmark with realistic query distribution."""

    print(f"\n{'='*70}")
    print(f"RUNNING COMPREHENSIVE BENCHMARK: {num_queries} QUERIES")
    print(f"{'='*70}\n")

    # Initialize RAG system
    rag = SemanticCacheRAG(cache_threshold=0.80, ttl_seconds=86400)

    rag.clear_cache()
    
    # Base queries (from FAQ dataset)
    base_queries = [
        "How do I reset my password?",
        "What's the process for password recovery?",
        "How do I update billing information?",
        "What are your business hours?",
        "How do I cancel my subscription?",
        "Can I get a refund?",
        "How do I upgrade my account?",
        "What plans do you offer?",
        "Is there a free trial?",
        "How do I contact sales?",
        "Is my data secure?",
        "How do I delete my account?",
        "Do you offer student discounts?",
        "How do I export my data?",
        "What happens if I exceed my query limit?",
    ]

    # Generate realistic query distribution
    # 70% from top 5 FAQs (frequently asked)
    # 30% from remaining FAQs (less frequent)
    query_list = []
    for i in range(num_queries):
        if i % 10 < 7:
            query_list.append(base_queries[i % 5])
        else:
            query_list.append(base_queries[(i % 10) + 5])

    # Execute benchmark
    print("Executing queries...")
    start_time = time.time()

    for i, query in enumerate(query_list):
        rag.query(query)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_queries - i - 1) / rate if rate else 0
            print(
                f"  Progress: {i+1}/{num_queries} "
                f"({(i+1)/num_queries*100:.1f}%) - ETA: {eta:.0f}s"
            )

    total_time = time.time() - start_time

    # Calculate comprehensive statistics
    stats = rag.get_stats()

    cache_hits = [q for q in rag.query_history if q.cached]
    cache_misses = [q for q in rag.query_history if not q.cached]

    all_latencies = [q.latency_ms for q in rag.query_history]
    hit_latencies = [q.latency_ms for q in cache_hits]
    miss_latencies = [q.latency_ms for q in cache_misses]

    all_latencies_sorted = sorted(all_latencies)

    def percentile(data, p):
        if not data:
            return 0
        idx = min(int(len(data) * p), len(data) - 1)
        return data[idx]

    detailed_stats = {
        **stats,
        "latency_p50": percentile(all_latencies_sorted, 0.50),
        "latency_p75": percentile(all_latencies_sorted, 0.75),
        "latency_p90": percentile(all_latencies_sorted, 0.90),
        "latency_p95": percentile(all_latencies_sorted, 0.95),
        "latency_p99": percentile(all_latencies_sorted, 0.99),
        "latency_min": min(all_latencies) if all_latencies else 0,
        "latency_max": max(all_latencies) if all_latencies else 0,
        "hit_latency_stddev": statistics.stdev(hit_latencies)
        if len(hit_latencies) > 1
        else 0,
        "miss_latency_stddev": statistics.stdev(miss_latencies)
        if len(miss_latencies) > 1
        else 0,
        "overall_latency_stddev": statistics.stdev(all_latencies)
        if len(all_latencies) > 1
        else 0,
        "total_time_seconds": total_time,
        "throughput_qps": num_queries / total_time if total_time else 0,
        "daily_cost_10k": stats["total_cost_dollars"] * 10,
        "monthly_cost_300k": stats["total_cost_dollars"] * 10 * 30,
        "baseline_daily_cost": 10000 * 0.0015,
        "baseline_monthly_cost": 10000 * 0.0015 * 30,
        "daily_savings": (10000 * 0.0015) - (stats["total_cost_dollars"] * 10),
        "monthly_savings":
        ((10000 * 0.0015 * 30) - (stats["total_cost_dollars"] * 10 * 30)),
    }

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE - DETAILED RESULTS")
    print(f"{'='*70}")

    print("\n📊 GENERAL METRICS:")
    print(f"  Total Queries:        {detailed_stats['total_queries']}")
    print(
        f"  Cache Hits:           {detailed_stats['cache_hits']} "
        f"({detailed_stats['hit_rate_percent']:.2f}%)"
    )
    print(
        f"  Cache Misses:         {detailed_stats['cache_misses']} "
        f"({100-detailed_stats['hit_rate_percent']:.2f}%)"
    )
    print(f"  Total Time:           {detailed_stats['total_time_seconds']:.2f}s")
    print(f"  Throughput:           {detailed_stats['throughput_qps']:.2f} queries/sec")

    print("\n⚡ LATENCY ANALYSIS:")
    print(
        f"  Overall Average:      {detailed_stats['avg_latency_ms']:.2f}ms "
        f"(±{detailed_stats['overall_latency_stddev']:.2f})"
    )
    print(f"  P50 (Median):         {detailed_stats['latency_p50']:.2f}ms")
    print(f"  P75:                  {detailed_stats['latency_p75']:.2f}ms")
    print(f"  P90:                  {detailed_stats['latency_p90']:.2f}ms")
    print(f"  P95:                  {detailed_stats['latency_p95']:.2f}ms")
    print(f"  P99:                  {detailed_stats['latency_p99']:.2f}ms")
    print(f"  Min:                  {detailed_stats['latency_min']:.2f}ms")
    print(f"  Max:                  {detailed_stats['latency_max']:.2f}ms")

    print("\n💨 CACHE PERFORMANCE:")
    print(
        f"  Hit Latency:          {detailed_stats['avg_hit_latency_ms']:.2f}ms "
        f"(±{detailed_stats['hit_latency_stddev']:.2f})"
    )
    print(
        f"  Miss Latency:         {detailed_stats['avg_miss_latency_ms']:.2f}ms "
        f"(±{detailed_stats['miss_latency_stddev']:.2f})"
    )
    improvement = (
        (detailed_stats["avg_miss_latency_ms"] - detailed_stats["avg_hit_latency_ms"])
        / detailed_stats["avg_miss_latency_ms"]
        * 100
        if detailed_stats["avg_miss_latency_ms"]
        else 0
    )
    print(f"  Improvement:          {improvement:.1f}% faster on cache hits")

    print("\n💰 COST ANALYSIS:")
    print(f"  Total Cost:           ${detailed_stats['total_cost_dollars']:.4f}")
    print(f"  Cost Savings:         {detailed_stats['cost_savings_percent']:.2f}%")
    print("\n  Projections (10K queries/day):")
    print(f"    Daily Cost:         ${detailed_stats['daily_cost_10k']:.2f}")
    print(f"    Monthly Cost:       ${detailed_stats['monthly_cost_300k']:.2f}")
    print(f"    Baseline Daily:     ${detailed_stats['baseline_daily_cost']:.2f}")
    print(f"    Baseline Monthly:   ${detailed_stats['baseline_monthly_cost']:.2f}")
    print(f"    Daily Savings:      ${detailed_stats['daily_savings']:.2f}")
    print(f"    Monthly Savings:    ${detailed_stats['monthly_savings']:.2f}")

    print(f"\n{'='*70}\n")

    with open("data/benchmark_results.json", "w") as f:
        json.dump(detailed_stats, f, indent=2)

    print("✓ Results saved to: data/benchmark_results.json")
    print("✓ Ready for table and figure generation\n")

    return detailed_stats


if __name__ == "__main__":
    results = run_comprehensive_benchmark(1000)
    print(results)
