"""
Core semantic caching implementation
Chapter 9: Reducing Costs and Latency in Production RAG Systems
Author: [Your Name]

This version implements semantic caching using:
- langchain-openai for embeddings + LLM
- redis-py for storage
No langchain-redis dependency required.
"""

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from redis import Redis

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    response: str
    latency_ms: float
    cached: bool
    cost_dollars: float
    timestamp: float = time.time()


class SemanticCacheRAG:
    """
    Production-style RAG system with semantic caching.

    This implementation:
    - Stores query embeddings and responses in Redis.
    - On each query:
      1. Embeds the incoming query.
      2. Scans cached items and computes cosine similarity.
      3. If similarity >= threshold: returns cached response (HIT).
      4. Otherwise: calls LLM, stores in cache (MISS).
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        cache_threshold: float = 0.80,
        ttl_seconds: int = 86400,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        cost_per_query: float = 0.0015,
        cache_prefix: str = "scache",
    ) -> None:
        # Configuration
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.cache_threshold = cache_threshold
        self.ttl_seconds = ttl_seconds
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.cost_per_query = cost_per_query
        self.cache_prefix = cache_prefix  # key namespace in Redis

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        # Initialize components
        self._init_redis()
        self._init_embeddings()
        self._init_llm()

        # In‑memory metrics
        self.query_history: List[QueryMetrics] = []

        print(
            f"✓ SemanticCacheRAG initialized "
            f"(threshold={self.cache_threshold}, ttl={self.ttl_seconds}s, model={self.model_name})"
        )

    # ---------------------------
    # Initialization helpers
    # ---------------------------

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        self.redis_client = Redis.from_url(self.redis_url, decode_responses=False)
        try:
            if not self.redis_client.ping():
                raise ConnectionError("Failed to connect to Redis (PING returned False).")
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to Redis at {self.redis_url}: {exc}") from exc

    def _init_embeddings(self) -> None:
        """Initialize embeddings model."""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.openai_api_key,
        )

    def _init_llm(self) -> None:
        """Initialize LLM."""
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.0,
            api_key=self.openai_api_key,
        )

    # ---------------------------
    # Redis key helpers
    # ---------------------------

    def _cache_key(self, query_id: str) -> str:
        return f"{self.cache_prefix}:item:{query_id}"

    def _index_key(self) -> str:
        """Key that stores a Redis Set of all query_ids."""
        return f"{self.cache_prefix}:index"

    # ---------------------------
    # Vector helpers
    # ---------------------------

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))

    def _embed_query(self, text: str) -> List[float]:
        """Get embedding vector for a query string."""
        # langchain-openai returns List[List[float]] for embed_documents
        return self.embeddings.embed_query(text)

    # ---------------------------
    # Cache operations
    # ---------------------------

    def _find_best_match(
        self, query_embedding: List[float]
    ) -> Optional[Tuple[str, Dict]]:
        """
        Scan cache and find the best match by cosine similarity.

        Returns:
            (query_id, payload_dict) or None if no match above threshold.
        """
        index_key = self._index_key()
        # Get all cached IDs
        query_ids = self.redis_client.smembers(index_key)
        if not query_ids:
            return None

        best_sim = -1.0
        best_item: Optional[Tuple[str, Dict]] = None

        for qid_bytes in query_ids:
            qid = qid_bytes.decode("utf-8")
            key = self._cache_key(qid)
            raw = self.redis_client.get(key)
            if not raw:
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
                emb = payload.get("embedding", [])
            except Exception:
                continue

            sim = self._cosine_similarity(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_item = (qid, payload)

        if best_item is None:
            return None

        if best_sim >= self.cache_threshold:
            return best_item
        return None

    def _store_in_cache(
        self,
        query: str,
        query_embedding: List[float],
        response: str,
    ) -> None:
        """Store query + embedding + response in Redis."""
        # Use timestamp as a simple unique ID
        query_id = str(time.time_ns())
        key = self._cache_key(query_id)
        index_key = self._index_key()

        payload = {
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "created_at": time.time(),
        }
        encoded = json.dumps(payload).encode("utf-8")

        pipe = self.redis_client.pipeline()
        pipe.set(key, encoded, ex=self.ttl_seconds)
        pipe.sadd(index_key, query_id.encode("utf-8"))
        pipe.execute()

    # ---------------------------
    # Public API
    # ---------------------------

    def query(self, question: str) -> QueryMetrics:
        """
        Process query with semantic caching.

        - Computes query embedding.
        - Searches cache for best semantic match.
        - If hit: returns cached response (cost=0).
        - If miss: calls LLM, stores in cache, charges cost.
        """
        start_time = time.time()

        # 1) Embed incoming query
        query_vec = self._embed_query(question)

        # 2) Try semantic cache
        cache_item = self._find_best_match(query_vec)

        if cache_item is not None:
            # Cache hit
            cached = True
            cached_payload = cache_item[1]
            response = cached_payload.get("response", "")
            cost = 0.0
        else:
            # Cache miss: call LLM
            cached = False
            llm_result = self.llm.invoke(question)
            response = llm_result.content if hasattr(llm_result, "content") else str(llm_result)
            cost = self.cost_per_query

            # Store in cache
            self._store_in_cache(question, query_vec, response)

        latency_ms = (time.time() - start_time) * 1000.0

        metrics = QueryMetrics(
            query=question,
            response=response,
            latency_ms=latency_ms,
            cached=cached,
            cost_dollars=cost,
        )
        self.query_history.append(metrics)
        return metrics

    def get_stats(self) -> Dict:
        """Get aggregate statistics over all queries."""
        total = len(self.query_history)
        if total == 0:
            return {
                "total_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate_percent": 0.0,
                "avg_latency_ms": 0.0,
                "avg_hit_latency_ms": 0.0,
                "avg_miss_latency_ms": 0.0,
                "total_cost_dollars": 0.0,
                "cost_savings_percent": 0.0,
            }

        hits = [q for q in self.query_history if q.cached]
        misses = [q for q in self.query_history if not q.cached]

        avg_latency = sum(q.latency_ms for q in self.query_history) / total
        avg_hit_latency = (
            sum(q.latency_ms for q in hits) / len(hits) if hits else 0.0
        )
        avg_miss_latency = (
            sum(q.latency_ms for q in misses) / len(misses) if misses else 0.0
        )

        total_cost = sum(q.cost_dollars for q in self.query_history)
        baseline_cost = total * self.cost_per_query
        cost_savings_percent = (
            (1.0 - (total_cost / baseline_cost)) * 100.0 if baseline_cost else 0.0
        )

        return {
            "total_queries": total,
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "hit_rate_percent": round(len(hits) / total * 100.0, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_hit_latency_ms": round(avg_hit_latency, 2),
            "avg_miss_latency_ms": round(avg_miss_latency, 2),
            "total_cost_dollars": round(total_cost, 4),
            "cost_savings_percent": round(cost_savings_percent, 2),
        }

    def clear_cache(self) -> None:
        """Clear all cache entries in Redis for this prefix."""
        self.clear_cache_with_count()

    def clear_cache_with_count(self) -> int:
        """Clear all cache entries and return deleted entry count."""
        pattern = f"{self.cache_prefix}:*"
        keys = list(self.redis_client.scan_iter(match=pattern))
        if not keys:
            self.query_history.clear()
            return 0

        pipe = self.redis_client.pipeline()
        for key in keys:
            pipe.delete(key)
        results = pipe.execute()
        deleted = sum(int(r) for r in results)
        self.query_history.clear()
        return deleted


# ---------------------------
# Test function
# ---------------------------

def test_semantic_cache() -> None:
    """Quick end-to-end test of semantic caching."""
    print("\n" + "=" * 70)
    print("SEMANTIC CACHE TEST")
    print("=" * 70 + "\n")

    rag = SemanticCacheRAG(cache_threshold=0.80)

    test_queries = [
        "How do I reset my password?",
        "How do I reset my password?",  # Exact duplicate
        "What's the process for resetting my password?",  # Semantic similar
        "How do I update billing information?",  # Different query
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"[Query {i}] {query}")
        result = rag.query(query)
        status = "✓ CACHE HIT" if result.cached else "✗ CACHE MISS"
        print(
            f"  {status} | Latency: {result.latency_ms:.2f}ms | "
            f"Cost: ${result.cost_dollars:.4f}"
        )
        print(f"  Response: {result.response[:80]}...\n")

    print("=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")

    rag.clear_cache()


if __name__ == "__main__":
    test_semantic_cache()
