"""
FastAPI middleware:
- Request/response logging with latency
- Prometheus metrics export
- Structured error responses
"""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import Counter, Histogram

# Prometheus metrics
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of RAG API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval stage latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "Generation stage latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


async def latency_middleware(request: Request, call_next: Callable) -> Response:
    """Log request latency and update Prometheus metrics."""
    start = time.time()
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        duration = time.time() - start
        status = response.status_code

        REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=status
        ).inc()
        REQUEST_LATENCY.labels(endpoint=path).observe(duration)

        if duration > 2.0:
            logger.warning(f"Slow request: {method} {path} → {status} in {duration:.2f}s")
        else:
            logger.info(f"{method} {path} → {status} in {duration*1000:.0f}ms")

        return response

    except Exception as exc:
        duration = time.time() - start
        logger.error(f"{method} {path} failed after {duration:.2f}s: {exc}")
        REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=500
        ).inc()
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )
