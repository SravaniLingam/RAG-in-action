Table 2.1. System Configuration and Experimental Setup

| Parameter | Value | Description |
| --- | --- | --- |
| Runtime Environment | Docker Compose | Multi-container local deployment |
| API Framework | FastAPI | REST API for RAG queries |
| Cache Backend | Redis Stack | Semantic cache storage |
| Cache Similarity Threshold | 0.80 | Minimum similarity score for cache hit |
| LLM Invocation | External LLM API | Used only on cache misses |
| Observability Stack | Prometheus + Grafana | Metrics collection and visualization |
| Metrics Collected | latency, hit/miss, throughput | Exposed via /metrics endpoint |
| Load Test Requests | 100 per run | Total requests per concurrency test |
| Concurrency Levels | 1, 5, 10, 20 | Parallel request counts evaluated |
Source: This study.
