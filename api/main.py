import json
import logging
import os
import time
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from redis import Redis

from code.semantic_cache_core import SemanticCacheRAG

load_dotenv()

app = FastAPI()

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

_cache_instance: SemanticCacheRAG | None = None

rag_requests_total = Counter("rag_requests_total", "Total number of RAG query requests")
rag_cache_hits_total = Counter("rag_cache_hits_total", "Total number of cache hits")
rag_cache_misses_total = Counter("rag_cache_misses_total", "Total number of cache misses")
rag_query_latency_ms = Histogram(
    "rag_query_latency_ms", "Latency of /query calls in milliseconds"
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    logger.info(json.dumps({
        "event": "request_start",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
    }))
    response: Response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    logger.info(json.dumps({
        "event": "request_end",
        "request_id": request_id,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
    }))
    return response


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool
    latency_ms: float


class ClearCacheResponse(BaseModel):
    status: str
    deleted: int


def get_cache() -> SemanticCacheRAG:
    global _cache_instance
    if _cache_instance is None:
        cache_threshold = float(os.getenv("CACHE_THRESHOLD", "0.80"))
        ttl_seconds = int(os.getenv("TTL_SECONDS", "86400"))
        _cache_instance = SemanticCacheRAG(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            cache_threshold=cache_threshold,
            ttl_seconds=ttl_seconds,
        )
    return _cache_instance


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        client = Redis.from_url(redis_url)
        pong = client.ping()
        if pong is True or pong == "PONG":
            return {"status": "ok", "redis": "PONG"}
    except Exception:
        pass
    return Response(
        content=json.dumps({"status": "error", "redis": "DOWN"}),
        media_type="application/json",
        status_code=503,
    )


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
def query(request: Request, payload: QueryRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    if not payload.query or not payload.query.strip():
        return Response(
            content=json.dumps({"error": "query must be non-empty"}),
            media_type="application/json",
            status_code=400,
        )
    start_time = time.time()
    rag_requests_total.inc()
    try:
        cache = get_cache()
        result = cache.query(payload.query.strip())
        latency_ms = (time.time() - start_time) * 1000.0
        rag_query_latency_ms.observe(latency_ms)
        if result.cached:
            rag_cache_hits_total.inc()
        else:
            rag_cache_misses_total.inc()
        logger.info(json.dumps({
            "event": "query_result",
            "request_id": request_id,
            "cached": result.cached,
            "latency_ms": round(latency_ms, 2),
        }))
        return QueryResponse(
            query=payload.query.strip(),
            answer=result.response,
            cached=result.cached,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000.0
        rag_query_latency_ms.observe(latency_ms)
        logger.info(json.dumps({
            "event": "query_error",
            "request_id": request_id,
            "error": str(exc),
        }))
        return Response(
            content=json.dumps({"error": "query failed"}),
            media_type="application/json",
            status_code=500,
        )


@app.post("/cache/clear", response_model=ClearCacheResponse)
def clear_cache(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    try:
        cache = get_cache()
        deleted = cache.clear_cache_with_count()
        logger.info(json.dumps({
            "event": "cache_clear",
            "request_id": request_id,
            "deleted": deleted,
        }))
        return ClearCacheResponse(status="ok", deleted=deleted)
    except Exception as exc:
        logger.info(json.dumps({
            "event": "cache_clear_error",
            "request_id": request_id,
            "error": str(exc),
        }))
        return Response(
            content=json.dumps({"error": "cache clear failed"}),
            media_type="application/json",
            status_code=500,
        )
