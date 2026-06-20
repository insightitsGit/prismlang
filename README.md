<div align="center">

<img src="https://img.shields.io/badge/PrismLang-v0.1.0-6366f1?style=for-the-badge&labelColor=0f0f23" alt="PrismLang"/>

# PrismLang

### Deterministic Vector Language Protocol for LangGraph Multi-Agent AI

*Stop paying the token tax on every agent hop. Start routing with math.*

[![PyPI](https://img.shields.io/pypi/v/prismlang?color=06b6d4&label=PyPI&style=flat-square)](https://pypi.org/project/prismlang/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/prismlang?style=flat-square&color=06b6d4)](https://pypi.org/project/prismlang/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-22c55e?style=flat-square)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-f97316?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![CI](https://img.shields.io/github/actions/workflow/status/insightitsGit/prismlang/ci.yml?branch=master&label=CI&style=flat-square)](https://github.com/insightitsGit/prismlang/actions)
[![Tests](https://img.shields.io/badge/tests-34%20passed-22c55e?style=flat-square)](tests/)
[![Security](https://img.shields.io/badge/security-hardened-6366f1?style=flat-square)](docs/SECURITY.md)

<br/>

**[📖 Docs](https://www.insightits.com/prismlang)** · **[🚀 Quickstart](#quick-start)** · **[📊 Benchmarks](#benchmark-results)** · **[🏢 Insight IT Solutions](https://www.insightits.com)**

<br/>

```
pip install prismlang
```

</div>

---

## What Problem Does PrismLang Solve?

If you are building multi-agent AI systems with LangGraph, you are likely running into one or more of these problems right now:

---

### Problem 1 — You are paying for the same tokens over and over

Every node in a LangGraph pipeline reads **the entire message history** as prompt tokens. Each agent re-reads everything every prior agent wrote — even context it doesn't need.

```
Turn 1 →  800 B   agent A pays for its own output
Turn 2 → 1,600 B  agent B pays for A + B
Turn 3 → 2,400 B  agent C pays for A + B + C   ← 3× cost for the same data
Turn N → N×800 B  cost grows linearly with graph depth
```

In a 10-node production graph running thousands of times a day, this is not a rounding error — it is a significant and avoidable infrastructure cost.

**PrismLang fixes this** by replacing growing text history with compact 64-number vectors. Each agent turn costs ~414 bytes regardless of graph depth.

---

### Problem 2 — You have no audit trail for routing decisions

When your pipeline misroutes a request — sends a compliance question to the market-data agent, or a triage case to the wrong specialist — you have no structured record of *why*. You are debugging with logs or nothing.

Regulators in healthcare (HIPAA), finance (SOX, Basel III), and legal (privilege review) are increasingly asking: *show us how your AI made this decision*.

**PrismLang fixes this** by attaching a `rule_chain` to every agent output — a full trace of the encoding, category inference, and projection steps. Every routing decision is reproducible and explainable from first principles.

```python
envelope["rule_chain"]
# ['text -> encoder(all-MiniLM-L6-v2, d=384)',
#  "category_inference -> slug='risk'",
#  'spherical_blend(alpha=0.300) -> v_prime',
#  "JL_reduction(seed=sha256('acme-finance'), k=64) -> p"]
```

---

### Problem 3 — Multi-tenant AI has no safe isolation layer

In SaaS AI platforms, multiple clients share the same agents and graph infrastructure. One misconfigured node, one wrong state key, and Tenant A's reasoning context is visible to Tenant B's inference call.

Text payloads don't enforce isolation — they rely entirely on your application layer getting it right, every time.

**PrismLang fixes this** by making isolation *mathematical*. Each tenant gets a unique Johnson-Lindenstrauss projection matrix derived from `SHA-256(tenant_id)`. The same input text produces geometrically incompatible vectors under different tenant keys. Cross-tenant leakage is provably impossible at the vector level.

---

### Who is PrismLang for?

| If you are building... | PrismLang helps you... |
|---|---|
| Multi-agent LangGraph pipelines | Cut token costs 57–62% without changing agent logic |
| Multi-tenant SaaS AI products | Add cryptographic tenant isolation at the protocol layer |
| Healthcare or finance AI systems | Produce a full audit trail on every routing decision |
| AI platforms with compliance requirements | Satisfy regulators with structured, reproducible decision records |
| Any LangGraph graph with 3+ nodes | Reduce state size linearly — the deeper the graph, the bigger the saving |

---

## The Solution

PrismLang replaces growing text payloads with **64-number deterministic vectors** — one per agent turn. A single decorator on your existing nodes. No agent refactoring. No LLM retraining.

```
[Your Agent]  →  "Credit risk elevated in EM bonds."  (text, 400+ tokens)
                              ↓  @prism_node
              →  PrismEnvelope { vector[64], slug="risk", rule_chain }  (~414 bytes)
```

The math guarantees that **the same input always produces the same vector**, that **different tenants produce incompatible vectors**, and that **every routing decision is traceable back to a taxonomy rule**.

---

## How It Works

PrismLang applies two equations on every agent output:

**Step 1 — Spherical Blend** *(pulls the embedding toward its category direction)*
```
v' = normalize( (1 − α) · v  +  α · ‖v‖ · eᵢ )
```

**Step 2 — JL Reduction** *(compresses to k=64 dims, isolated per tenant)*
```
p = normalize( P · v' )
```

Where `P` is a `(64 × 384)` Gaussian matrix seeded from `SHA-256(tenant_id)`. A vector stolen from Tenant A is **geometrically meaningless** to any model operating under Tenant B's projection.

```
                    ┌─────────────────────────────────────────────┐
                    │           Your LangGraph Graph              │
                    │                                             │
  [researcher] ──→ [summarizer] ──→ [reviewer] ──→ [translator]  │
       │                │               │               │         │
  @prism_node      @prism_node     @prism_node     (boundary)     │
       │                │               │               │         │
  PrismEnvelope    PrismEnvelope   PrismEnvelope   Human text     │
  {64-d vector}    {64-d vector}   {64-d vector}                  │
  {rule_chain}     {rule_chain}    {rule_chain}                    │
                    │                                             │
                    │  prism_sequence  ─────────  append-only     │
                    └─────────────────────────────────────────────┘
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
taxonomy = TaxonomyConfig(categories=[
    Category("risk",       "Market Risk",   ["risk", "exposure", "volatility"]),
    Category("market",     "Market Data",   ["price", "equity", "bond"]),
    Category("compliance", "Compliance",    ["regulation", "audit", "kyc"]),
])

# 2. One projector per tenant — cryptographically isolated
projector = PrismProjector(taxonomy, tenant_id="acme-finance-prod", k=64)

# 3. Decorate your existing nodes — zero changes to agent logic
@prism_node(agent_id="analyst", projector=projector)
def analyst(state: PrismState) -> dict:
    return {"raw_output": "Credit risk exposure elevated in EM bonds."}

@prism_node(agent_id="reviewer", projector=projector)
def reviewer(state: PrismState) -> dict:
    prev = state["prism_sequence"][-1]["category_slug"]
    return {"raw_output": f"Reviewing {prev} findings for compliance sign-off."}

# 4. Build and run — exactly like any LangGraph graph
translator = BoundaryTranslator()
graph = StateGraph(PrismState)
graph.add_node("analyst",    analyst)
graph.add_node("reviewer",   reviewer)
graph.add_node("translator", translator.as_langgraph_node())
graph.set_entry_point("analyst")
graph.add_edge("analyst", "reviewer")
graph.add_edge("reviewer", "translator")
graph.add_edge("translator", END)

app = graph.compile(checkpointer=JsonFileCheckpointer())
result = app.invoke({
    "prism_sequence": [], "raw_output": "", "tenant_id": "acme-finance-prod"
})

# Inspect the audit envelope
envelope = result["prism_sequence"][0]
print(envelope["category_slug"])   # "risk"
print(len(envelope["vector"]))     # 64
print(envelope["rule_chain"])
# ['text -> encoder(all-MiniLM-L6-v2, d=384)',
#  "category_inference -> slug='risk'",
#  'spherical_blend(alpha=0.300) -> v_prime',
#  "JL_reduction(seed=sha256('acme-finance-prod'), k=64) -> p"]
```

---

## Benchmark Results

> Measured against standard LangGraph text-state across three enterprise domains.  
> Full methodology in [`docs/BENCHMARK.md`](docs/BENCHMARK.md). Results stored in PostgreSQL.

<table>
<tr>
  <th>Domain</th>
  <th>Metric</th>
  <th>Standard LangGraph</th>
  <th>PrismLang</th>
  <th>Change</th>
</tr>
<tr>
  <td rowspan="2"><b>🏥 Healthcare</b><br/><sub>ICU triage pipeline</sub></td>
  <td>Prompt tokens (3 turns)</td>
  <td>391</td>
  <td>148</td>
  <td><b>−62.1%</b></td>
</tr>
<tr>
  <td>State size (turn 3)</td>
  <td>1,928 B</td>
  <td>960 B</td>
  <td><b>−50.2%</b></td>
</tr>
<tr>
  <td rowspan="2"><b>💹 Finance</b><br/><sub>Risk / portfolio pipeline</sub></td>
  <td>Prompt tokens (3 turns)</td>
  <td>407</td>
  <td>175</td>
  <td><b>−57.0%</b></td>
</tr>
<tr>
  <td>State size (turn 3)</td>
  <td>1,760 B</td>
  <td>960 B</td>
  <td><b>−45.5%</b></td>
</tr>
<tr>
  <td rowspan="2"><b>📈 Trade Market</b><br/><sub>Signal / execution pipeline</sub></td>
  <td>Prompt tokens (3 turns)</td>
  <td>435</td>
  <td>180</td>
  <td><b>−58.6%</b></td>
</tr>
<tr>
  <td>State size (turn 3)</td>
  <td>1,867 B</td>
  <td>960 B</td>
  <td><b>−48.6%</b></td>
</tr>
</table>

> **LLM inference latency: unchanged.** PrismLang reduces state transport, not compute.  
> Encoding overhead per turn: ~31–35 ms CPU-only (no GPU required).

---

## Key Properties

| Property | Detail |
|---|---|
| **Zero agent refactoring** | Agents return `{"raw_output": "..."}` — nothing else changes |
| **Deterministic** | Same text + same tenant = identical vector, always |
| **Full audit trail** | Every envelope carries a `rule_chain` tracing the full decision path |
| **Tenant isolation** | `SHA-256(tenant_id)` seeds the JL matrix — cross-tenant vectors are incompatible |
| **No GPU** | ONNX Runtime CPU inference — runs on any standard server |
| **No external API** | Encoder is fully local — no network call per token |
| **Model-agnostic** | Works with GPT-4, Claude, Gemini, Llama, or any LLM |
| **Async native** | `@async_prism_node` for async LangGraph nodes |
| **Two checkpointers** | `JsonFileCheckpointer` (zero deps) + `PostgresCheckpointer` |

---

## Installation Options

```bash
# Core (local JSON checkpointing)
pip install prismlang

# PostgreSQL checkpointing
pip install "prismlang[postgres]"

# Async support (asyncpg + aiofiles)
pip install "prismlang[async-postgres,async-files]"

# Full development environment
pip install "prismlang[dev]"
```

---

## Run the Benchmarks

```bash
git clone https://github.com/insightitsGit/prismlang
cd prismlang
pip install -e ".[dev]"

# Runs all 3 domain benchmarks and prints comparison table
python -m benchmarks.run_all
```

Requires a running PostgreSQL instance. Set `DATABASE_URL` or use the default:  
`postgresql://insight_admin:...@localhost/prismLangDB`

---

## Project Structure

```
prismlang/
├── prismlang/
│   ├── encoder.py        # ONNX all-MiniLM-L6-v2 → 384-d unit vector
│   ├── taxonomy.py       # TaxonomyConfig + Category direction vectors (eᵢ)
│   ├── projector.py      # PrismProjector: spherical blend + JL reduction
│   ├── middleware.py     # @prism_node + @async_prism_node decorators
│   ├── checkpointer.py   # JsonFile + Postgres + Async variants
│   ├── exceptions.py     # Typed exception hierarchy (17 classes)
│   ├── envelope.py       # PrismEnvelope TypedDict
│   ├── state.py          # PrismState (LangGraph append-only channel)
│   └── translator.py     # BoundaryTranslator (structural reconstruction)
├── benchmarks/
│   └── domains/          # Healthcare · Finance · Trade Market
├── demo/
│   └── graph.py          # Runnable 3-node LangGraph demo
├── tests/                # 34 tests · 0 failures
└── docs/
    ├── ARCHITECTURE.md
    ├── BENCHMARK.md
    └── SECURITY.md
```

---

## Security

PrismLang's tenant isolation is a **geometric property** guaranteed by the Johnson-Lindenstrauss lemma — not an access-control system. For production deployments, see [`docs/SECURITY.md`](docs/SECURITY.md) which covers:

- What the JL matrix does and does not protect
- Overlay encryption for PII in `raw_output`
- Dependency security notes (onnxruntime, psycopg2, asyncpg)
- NumPy PRNG stability across version upgrades

To report a vulnerability: **prismrag@insightits.com** — do not open a public GitHub issue.

---

## Citation

```bibtex
@techreport{parva2026prismlang,
  title       = {PrismLang: A Deterministic Vector Language Protocol
                 for Auditable Multi-Agent AI Orchestration},
  author      = {Parva, Amin},
  year        = {2026},
  institution = {Insight IT Solutions LLC},
  url         = {https://www.insightits.com/prismlang}
}
```

---

## License

[Apache 2.0](LICENSE) — free for commercial and personal use.

---

<div align="center">

**Built by [Insight IT Solutions LLC](https://www.insightits.com)**

*Enterprise AI systems · LangGraph architecture · Vector search · Production deployment*

[🌐 Website](https://www.insightits.com) · [📧 Contact](mailto:prismrag@insightits.com) · [🔒 Security](mailto:prismrag@insightits.com)

</div>
