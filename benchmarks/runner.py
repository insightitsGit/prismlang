"""BenchmarkRunner — orchestrates standard vs PrismLang graph runs and stores results.

Each domain calls runner.run_pair(standard_graph_fn, prismlang_graph_fn, ...) which:
  1. Runs the standard LangGraph graph, collecting metrics
  2. Runs the PrismLang graph with the same inputs, collecting metrics
  3. Stores both results to bench.run_results in prismLangDB
  4. Returns (standard_result, prismlang_result) for reporting
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

import psycopg2

from .metrics import BenchmarkResult

_INSERT_SQL = """
INSERT INTO bench.run_results (
    run_id, domain, mode, model, tenant_id, turns,
    total_ms, llm_ms, encode_ms,
    prompt_tokens, output_tokens,
    state_bytes, vector_bytes,
    peak_rss_mb, category_flow, tenant_matrix_fp
) VALUES (
    %(run_id)s, %(domain)s, %(mode)s, %(model)s, %(tenant_id)s, %(turns)s,
    %(total_ms)s, %(llm_ms)s, %(encode_ms)s,
    %(prompt_tokens)s, %(output_tokens)s,
    %(state_bytes)s, %(vector_bytes)s,
    %(peak_rss_mb)s, %(category_flow)s, %(tenant_matrix_fp)s
)
"""


def _store(conn, run_id: str, result: BenchmarkResult) -> None:
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_SQL,
            {
                "run_id": run_id,
                "domain": result.domain,
                "mode": result.mode,
                "model": result.model,
                "tenant_id": result.tenant_id,
                "turns": len(result.turns),
                "total_ms": round(result.total_ms, 2),
                "llm_ms": round(result.llm_ms, 2),
                "encode_ms": round(result.encode_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "state_bytes": result.state_bytes,
                "vector_bytes": result.vector_bytes,
                "peak_rss_mb": round(result.peak_rss_mb, 3),
                "category_flow": result.category_flow or None,
                "tenant_matrix_fp": result.tenant_matrix_fp,
            },
        )
    conn.commit()


class BenchmarkRunner:
    """Runs standard and PrismLang graphs for a domain and persists results."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn = psycopg2.connect(dsn)

    def close(self) -> None:
        self._conn.close()

    def run_pair(
        self,
        standard_fn: Callable[[], BenchmarkResult],
        prismlang_fn: Callable[[], BenchmarkResult],
        run_id: Optional[str] = None,
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Run both modes and store results. Returns (standard, prismlang)."""
        if run_id is None:
            run_id = str(uuid.uuid4())

        print(f"\n  [standard]   running ...")
        std = standard_fn()
        _store(self._conn, run_id, std)
        print(f"  [standard]   done  — {std.total_ms:.0f}ms | {std.prompt_tokens}+{std.output_tokens} tokens | {std.state_bytes} bytes state")

        print(f"  [prismlang]  running ...")
        prism = prismlang_fn()
        _store(self._conn, run_id, prism)
        print(f"  [prismlang]  done  — {prism.total_ms:.0f}ms | {prism.prompt_tokens}+{prism.output_tokens} tokens | {prism.vector_bytes} bytes vector | flow: {' -> '.join(prism.category_flow)}")

        return std, prism

    def fetch_latest_pairs(self, domain: Optional[str] = None, limit: int = 10) -> list:
        """Fetch the most recent benchmark pairs from the DB."""
        where = "WHERE domain = %s" if domain else ""
        params = (domain,) if domain else ()
        sql = f"""
            SELECT run_id, domain, mode, model, turns,
                   total_ms, llm_ms, encode_ms,
                   prompt_tokens, output_tokens,
                   state_bytes, vector_bytes, compression_ratio,
                   peak_rss_mb, category_flow, tenant_matrix_fp, created_at
            FROM bench.run_results
            {where}
            ORDER BY created_at DESC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, params + (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
