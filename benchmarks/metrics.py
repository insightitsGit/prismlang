"""MetricsCollector — captures wall-clock time, token counts, payload size, and memory usage.

Usage
-----
    with MetricsCollector() as m:
        m.start_llm()
        response = call_gemini(prompt)
        m.end_llm(prompt_tokens=..., output_tokens=...)
        m.record_state_bytes(len(json.dumps(state).encode()))

    result = m.snapshot()
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TurnMetrics:
    turn_id: int
    agent_id: str
    llm_ms: float
    encode_ms: float
    prompt_tokens: int
    output_tokens: int
    state_bytes_after: int     # cumulative state size after this turn
    vector_bytes_after: int    # cumulative vector-only size


@dataclass
class BenchmarkResult:
    domain: str
    mode: str                   # standard | prismlang
    model: str
    tenant_id: str
    turns: List[TurnMetrics] = field(default_factory=list)
    total_ms: float = 0.0
    peak_rss_mb: float = 0.0
    category_flow: List[str] = field(default_factory=list)
    tenant_matrix_fp: Optional[str] = None

    # Aggregated
    @property
    def llm_ms(self) -> float:
        return sum(t.llm_ms for t in self.turns)

    @property
    def encode_ms(self) -> float:
        return sum(t.encode_ms for t in self.turns)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def state_bytes(self) -> int:
        return self.turns[-1].state_bytes_after if self.turns else 0

    @property
    def vector_bytes(self) -> int:
        return self.turns[-1].vector_bytes_after if self.turns else 0


class MetricsCollector:
    """Context manager that measures a single benchmark run."""

    def __init__(self) -> None:
        self._wall_start: float = 0.0
        self._llm_start: float = 0.0
        self._encode_start: float = 0.0
        self.turns: List[TurnMetrics] = []
        self._peak_rss_mb: float = 0.0

        # Running totals
        self._total_llm_ms: float = 0.0
        self._total_encode_ms: float = 0.0
        self._total_prompt_tokens: int = 0
        self._total_output_tokens: int = 0
        self._cumulative_state_bytes: int = 0
        self._cumulative_vector_bytes: int = 0

    def __enter__(self) -> "MetricsCollector":
        tracemalloc.start()
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_rss_mb = peak / (1024 * 1024)

    # ------------------------------------------------------------------ #
    # LLM timing                                                           #
    # ------------------------------------------------------------------ #

    def start_llm(self) -> None:
        self._llm_start = time.perf_counter()

    def end_llm(self, prompt_tokens: int = 0, output_tokens: int = 0) -> float:
        elapsed = (time.perf_counter() - self._llm_start) * 1000
        self._total_llm_ms += elapsed
        self._total_prompt_tokens += prompt_tokens
        self._total_output_tokens += output_tokens
        return elapsed

    # ------------------------------------------------------------------ #
    # PrismLang encode timing                                              #
    # ------------------------------------------------------------------ #

    def start_encode(self) -> None:
        self._encode_start = time.perf_counter()

    def end_encode(self) -> float:
        elapsed = (time.perf_counter() - self._encode_start) * 1000
        self._total_encode_ms += elapsed
        return elapsed

    # ------------------------------------------------------------------ #
    # State size tracking                                                  #
    # ------------------------------------------------------------------ #

    def record_turn(
        self,
        turn_id: int,
        agent_id: str,
        state_bytes: int,
        vector_bytes: int = 0,
        llm_ms: float = 0.0,
        encode_ms: float = 0.0,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._cumulative_state_bytes = state_bytes
        self._cumulative_vector_bytes = vector_bytes
        self.turns.append(
            TurnMetrics(
                turn_id=turn_id,
                agent_id=agent_id,
                llm_ms=llm_ms,
                encode_ms=encode_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                state_bytes_after=state_bytes,
                vector_bytes_after=vector_bytes,
            )
        )

    # ------------------------------------------------------------------ #
    # Final snapshot                                                       #
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        domain: str,
        mode: str,
        model: str,
        tenant_id: str,
        category_flow: Optional[List[str]] = None,
        tenant_matrix_fp: Optional[str] = None,
    ) -> BenchmarkResult:
        total_ms = (time.perf_counter() - self._wall_start) * 1000
        result = BenchmarkResult(
            domain=domain,
            mode=mode,
            model=model,
            tenant_id=tenant_id,
            turns=self.turns,
            total_ms=total_ms,
            peak_rss_mb=self._peak_rss_mb,
            category_flow=category_flow or [],
            tenant_matrix_fp=tenant_matrix_fp,
        )
        return result
