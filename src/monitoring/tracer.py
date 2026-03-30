"""
Per-stage latency tracer.

Tracks wall-clock time for each pipeline stage so you can identify
where time is being spent and set SLA alerts per stage.

Used as a context manager:

    with Tracer() as t:
        with t.stage("query_transform"):
            result = transformer.transform(query)
        with t.stage("retrieval"):
            chunks = retriever.retrieve(...)
    print(t.summary())
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StageTrace:
    name: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class Tracer:
    """Collects timing data for named pipeline stages."""

    def __init__(self):
        self._stages: List[StageTrace] = []
        self._start_ms: float = time.time() * 1000

    @contextmanager
    def stage(self, name: str):
        trace = StageTrace(name=name, start_ms=time.time() * 1000)
        self._stages.append(trace)
        try:
            yield trace
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            trace.end_ms = time.time() * 1000

    @property
    def total_ms(self) -> float:
        return time.time() * 1000 - self._start_ms

    def summary(self) -> Dict:
        return {
            "total_ms": round(self.total_ms, 1),
            "stages": {
                s.name: {
                    "duration_ms": round(s.duration_ms, 1),
                    "error": s.error,
                }
                for s in self._stages
            },
        }

    def as_flat_dict(self) -> Dict[str, float]:
        """Flat dict suitable for W&B logging."""
        result = {"latency/total_ms": round(self.total_ms, 1)}
        for s in self._stages:
            result[f"latency/{s.name}_ms"] = round(s.duration_ms, 1)
        return result
