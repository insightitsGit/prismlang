# PrismLang

**Deterministic Vector Language Protocol for Multi-Agent AI Orchestration**

*by Insight IT Solutions LLC — Amin Parva*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://github.com/langchain-ai/langgraph)

---

## What is PrismLang?

PrismLang is a **middleware protocol** for LangGraph multi-agent systems that replaces growing text-state arrays with deterministic, compressed vector envelopes — without requiring any changes to agent node code.

Every agent in your graph continues to write and read plain text. PrismLang intercepts at the state boundary, mathematically projects each output into a compact `k`-dimensional vector, and transmits that instead. A single boundary translator node at the graph exit reconstructs human-readable output.

The result: **inter-agent state that is 40–60% smaller, fully auditable back to its taxonomy rule, and cryptographically isolated per tenant** — with no retraining, no external API, and no changes to your existing agents.

---

## The Problem PrismLang Solves

In production multi-agent LangGraph pipelines, the shared state grows with every turn:

```
Turn 1 state:  800 bytes   (agent A output)
Turn 2 state: 1,600 bytes  (agent A + agent B, full text)
Turn 3 state: 2,400 bytes  (all three agents, full text)
```

This creates three compounding problems:

| Problem | Impact |
|---|---|
| **State payload bloat** | Every downstream agent re-reads the full history → prompt tokens grow linearly |
| **No tenant isolation** | Shared state in multi-tenant systems can leak across organisational boundaries |
| **No auditability** | You cannot trace why an agent said what it said back to a deterministic rule |

PrismLang solves all three at the protocol layer — no LLM retraining, no architecture changes.

---

## How It Works

```
[Agent A] → text output
               ↓
         [ PrismLang Layer ]
         1. Encode: all-MiniLM-L6-v2 ONNX → 384-d unit vector
         2. Category: keyword taxonomy → domain slug (e.g. "risk")
         3. Spherical blend: v' = normalize((1-α)·v + α·‖v‖·eᵢ)
         4. JL reduction: p = normalize(P·v')  ← P seeded by SHA-256(tenant_id)
               ↓
         PrismEnvelope { turn_id, agent_id, category_slug, vector[64], rule_chain }
               ↓
[Agent B] ← reads category_slug + prism_sequence (not raw text)
```

At the graph boundary, a `BoundaryTranslator` node reconstructs the human-readable report from the envelope sequence and audit chain.

### The Two Core Equations (from the PrismLang paper)

**Spherical Blend (domain-taxonomy enforcement):**
```
v' = normalize((1 − α) · v + α · ‖v‖ · eᵢ)
```

**Johnson-Lindenstrauss Reduction (tenant isolation):**
```
p = normalize(P · v')
```

Where `P` is a `(k × 384)` Gaussian matrix seeded deterministically from `SHA-256(tenant_id)`. A vector payload from Tenant A is geometrically meaningless to an agent operating under Tenant B's matrix.

---

## Benchmark Results

Tested across three enterprise domains against standard LangGraph text state. All results verified with Gemini 2.0 Flash and stored in PostgreSQL.

### Healthcare Domain (ICU multi-agent triage pipeline)

| Metric | Standard LangGraph | PrismLang | Change |
|---|---|---|---|
| Prompt tokens (3 turns) | 391 | 148 | **−62.1%** |
| State at turn 3 | 1,928 B | 960 B | **−50.2%** |
| LLM latency | 151 ms | 151 ms | 0% |
| Category flow | — | `clinical → lab → compliance` | ✓ VERIFIED |
| Audit trail | none | full rule chain | ✓ |

### Finance Domain (Hedge fund risk / portfolio / compliance pipeline)

| Metric | Standard LangGraph | PrismLang | Change |
|---|---|---|---|
| Prompt tokens (3 turns) | 407 | 175 | **−57.0%** |
| State at turn 3 | 1,760 B | 960 B | **−45.5%** |
| LLM latency | 151 ms | 151 ms | 0% |
| Category flow | — | `risk → portfolio → compliance` | ✓ VERIFIED |

### Trade Market Domain (Signal / execution / position-risk pipeline)

| Metric | Standard LangGraph | PrismLang | Change |
|---|---|---|---|
| Prompt tokens (3 turns) | 435 | 180 | **−58.6%** |
| State at turn 3 | 1,867 B | 960 B | **−48.6%** |
| LLM latency | 151 ms | 151 ms | 0% |
| Category flow | — | `signal → execution → risk` | ✓ VERIFIED |

> **Key finding:** LLM inference cost is unchanged. PrismLang reduces the *state transport layer* — the growing history passed between agents — by 40–62% in prompt tokens, while adding near-zero encoding overhead.

---

## Installation

```bash
pip install prismlang
```

Or from source:
```bash
git clone https://github.com/insightits/prismlang
cd prismlang
pip install -e .
```

**Optional PostgreSQL checkpointer:**
```bash
pip install prismlang[postgres]
```

---

## Quick Start

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node, BoundaryTranslator,
    JsonFileCheckpointer,
)
from langgraph.graph import StateGraph, END

# 1. Define your domain taxonomy
taxonomy = TaxonomyConfig(
    categories=[
        Category("risk",       "Risk",       ["risk", "exposure", "volatility"]),
        Category("market",     "Market",     ["price", "equity", "bond"]),
        Category("compliance", "Compliance", ["regulation", "audit", "kyc"]),
    ]
)

# 2. Create a tenant-isolated projector
projector = PrismProjector(taxonomy, tenant_id="my-org-prod", k=64)

# 3. Wrap your existing agent nodes — zero changes to agent logic
@prism_node(agent_id="analyst", projector=projector)
def analyst(state: PrismState) -> dict:
    # ... your existing LLM call here ...
    return {"raw_output": "Credit risk exposure elevated in EM bonds."}

@prism_node(agent_id="reviewer", projector=projector)
def reviewer(state: PrismState) -> dict:
    # Agents can read the category from the previous envelope
    prev_category = state["prism_sequence"][-1]["category_slug"]
    return {"raw_output": f"Reviewing {prev_category} findings for compliance."}

# 4. Boundary translator (human-readable output at graph exit)
translator = BoundaryTranslator()

# 5. Build the graph
graph = StateGraph(PrismState)
graph.add_node("analyst",    analyst)
graph.add_node("reviewer",   reviewer)
graph.add_node("translator", translator.as_langgraph_node())
graph.set_entry_point("analyst")
graph.add_edge("analyst",    "reviewer")
graph.add_edge("reviewer",   "translator")
graph.add_edge("translator", END)

app = graph.compile(checkpointer=JsonFileCheckpointer())

# 6. Run
result = app.invoke({
    "prism_sequence": [],
    "raw_output": "",
    "tenant_id": "my-org-prod",
})
print(result["raw_output"])
```

---

## Key Properties

| Property | Description |
|---|---|
| **No retraining** | Uses pre-trained ONNX encoder (`all-MiniLM-L6-v2`). No fine-tuning required. |
| **Deterministic** | Same input + same tenant → identical vector, every time. |
| **Auditable** | Every vector carries a `rule_chain` tracing the full projection path. |
| **Tenant-isolated** | SHA-256(tenant_id) seeds the JL matrix. Cross-tenant vectors are geometrically incompatible. |
| **Model-agnostic** | Works with any LLM (Gemini, Claude, GPT, Llama). Agents call whatever model they want. |
| **Middleware-only** | Drop `@prism_node` on existing nodes. No agent refactoring. |

---

## Architecture

```
C:\code\PrismLang\
├── prismlang/
│   ├── encoder.py       # ONNX all-MiniLM-L6-v2 → 384-d unit vector
│   ├── taxonomy.py      # TaxonomyConfig + Category direction vectors
│   ├── projector.py     # PrismProjector: spherical blend + JL reduction
│   ├── envelope.py      # PrismEnvelope TypedDict
│   ├── state.py         # PrismState (LangGraph channel)
│   ├── middleware.py    # @prism_node decorator
│   ├── checkpointer.py  # JsonFileCheckpointer + PostgresCheckpointer
│   └── translator.py    # BoundaryTranslator
├── benchmarks/
│   ├── domains/
│   │   ├── healthcare.py
│   │   ├── finance.py
│   │   └── trade_market.py
│   └── run_all.py
└── tests/               # 34 unit tests, 0 failures
```

---

## Citation

```bibtex
@techreport{parva2026prismlang,
  title   = {PrismLang: A Deterministic Vector Language Protocol for Multi-Agent AI Orchestration},
  author  = {Parva, Amin},
  year    = {2026},
  institution = {Insight IT Solutions LLC},
  email   = {prismrag@insightits.com}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

*Built on PrismRAG by Insight IT Solutions LLC.*
