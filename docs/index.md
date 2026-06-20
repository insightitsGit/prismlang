# PrismLang

**Deterministic Vector Language Protocol for Multi-Agent AI Orchestration**

[![PyPI](https://img.shields.io/pypi/v/prismlang)](https://pypi.org/project/prismlang/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](../LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/prismlang)](https://pypi.org/project/prismlang/)

PrismLang replaces verbose text payloads between LangGraph agents with compact, tenant-isolated 64-dimensional vectors — reducing token consumption by up to 62% while keeping every routing decision fully auditable.

---

## Why PrismLang?

In large multi-agent systems, agents exchange text messages that are:
- **Expensive** — each hop pays full prompt-token cost
- **Opaque** — no structured trace of routing decisions
- **Insecure** — a single shared context exposes all tenants

PrismLang intercepts at the node boundary via a zero-change decorator, converts agent output to a `PrismEnvelope`, and injects an auditable `rule_chain` — all without modifying a single line of agent code.

---

## Quick Install

```bash
pip install prismlang
```

## Quickstart

```python
from prismlang import (
    TaxonomyConfig, Category, PrismProjector, PrismState, prism_node
)
from langgraph.graph import StateGraph

taxonomy = TaxonomyConfig([
    Category("risk",       "Market Risk",    ["risk", "exposure", "volatility"]),
    Category("compliance", "Compliance",     ["regulation", "audit", "KYC"]),
    Category("market",     "Market Data",    ["price", "index", "equity"]),
])
projector = PrismProjector(taxonomy, tenant_id="acme-finance")

@prism_node(agent_id="analyst", projector=projector)
def analyst(state: PrismState) -> dict:
    return {"raw_output": "Current volatility suggests elevated market risk exposure."}

graph = StateGraph(PrismState)
graph.add_node("analyst", analyst)
graph.set_entry_point("analyst")
graph.set_finish_point("analyst")

app = graph.compile()
result = app.invoke({"tenant_id": "acme-finance", "prism_sequence": [], "raw_output": ""})

envelope = result["prism_sequence"][0]
print(envelope["category_slug"])  # "risk"
print(len(envelope["vector"]))    # 64
print(envelope["rule_chain"])     # full audit trace
```

---

## Key Properties

| Property | Value |
|----------|-------|
| Vector dimension | 64 (configurable) |
| Encoder | all-MiniLM-L6-v2 (ONNX, no GPU) |
| Token reduction | 57–62% across healthcare, finance, trade |
| State size reduction | 46–50% |
| Cross-tenant cosine similarity | < 0.20 (near-orthogonal) |
| Determinism | Identical output for identical input + tenant |
| LLM dependency | None (structural reconstruction default) |

---

## Next Steps

- [Installation](getting-started/installation.md) — full setup including optional extras
- [Quickstart](getting-started/quickstart.md) — 5-minute working example
- [Architecture](ARCHITECTURE.md) — protocol internals
- [Benchmarks](BENCHMARK.md) — performance across 3 domains
- [Security](SECURITY.md) — threat model and deployment guidance
- [API Reference](api/projector.md) — full API docs
