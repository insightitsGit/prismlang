# PrismLang — Code Examples

> Reference file for agents and developers integrating PrismLang into LangGraph pipelines.
> All examples are self-contained and runnable after `pip install prismlang`.

---

## Table of Contents

1. [Basic Single-Node Graph](#1-basic-single-node-graph)
2. [Multi-Node Pipeline](#2-multi-node-pipeline)
3. [Multi-Tenant Isolation Proof](#3-multi-tenant-isolation-proof)
4. [Healthcare Domain](#4-healthcare-domain)
5. [Finance Domain](#5-finance-domain)
6. [Async Nodes](#6-async-nodes)
7. [PostgreSQL Checkpointer](#7-postgresql-checkpointer)
8. [Custom Taxonomy](#8-custom-taxonomy)
9. [Boundary Translator](#9-boundary-translator)
10. [Reading the Audit Trail](#10-reading-the-audit-trail)
11. [Encoder Artifact Id & Shared Session](#11-encoder-artifact-id--shared-session)

---

## 1. Basic Single-Node Graph

The simplest possible integration — one agent, one decorator.

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node,
)
from langgraph.graph import StateGraph, END

# Define what categories your domain has
taxonomy = TaxonomyConfig(categories=[
    Category("general", "General", ["the", "is", "a"]),  # catch-all
])

projector = PrismProjector(taxonomy, tenant_id="my-org")

@prism_node(agent_id="assistant", projector=projector)
def assistant(state: PrismState) -> dict:
    return {"raw_output": "Hello! I can help you with that."}

graph = StateGraph(PrismState)
graph.add_node("assistant", assistant)
graph.set_entry_point("assistant")
graph.set_finish_point("assistant")
app = graph.compile()

result = app.invoke({
    "tenant_id": "my-org",
    "prism_sequence": [],
    "raw_output": "",
})

envelope = result["prism_sequence"][0]
print(f"Agent:    {envelope['agent_id']}")        # assistant
print(f"Category: {envelope['category_slug']}")   # general
print(f"Vector:   {len(envelope['vector'])}d")    # 64
```

---

## 2. Multi-Node Pipeline

Three agents chaining through a graph. Each agent can read the previous agent's category from `prism_sequence`.

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node, BoundaryTranslator,
    JsonFileCheckpointer,
)
from langgraph.graph import StateGraph, END

taxonomy = TaxonomyConfig(categories=[
    Category("research",    "Research",    ["data", "analysis", "findings", "study"]),
    Category("summary",     "Summary",     ["summary", "overview", "conclusion"]),
    Category("action",      "Action",      ["recommend", "action", "next", "step"]),
])

projector = PrismProjector(taxonomy, tenant_id="acme-corp", k=64)

@prism_node(agent_id="researcher", projector=projector)
def researcher(state: PrismState) -> dict:
    # Simulate an LLM call
    return {"raw_output": "Data analysis shows a 23% increase in user engagement this quarter."}

@prism_node(agent_id="summarizer", projector=projector)
def summarizer(state: PrismState) -> dict:
    # Read what the previous agent found
    prev = state["prism_sequence"][-1]
    return {"raw_output": f"Summary: {prev['category_slug']} findings reviewed. Engagement up 23%."}

@prism_node(agent_id="planner", projector=projector)
def planner(state: PrismState) -> dict:
    return {"raw_output": "Recommend increasing content frequency by 2x next month."}

translator = BoundaryTranslator()

graph = StateGraph(PrismState)
graph.add_node("researcher",  researcher)
graph.add_node("summarizer",  summarizer)
graph.add_node("planner",     planner)
graph.add_node("translator",  translator.as_langgraph_node())
graph.set_entry_point("researcher")
graph.add_edge("researcher", "summarizer")
graph.add_edge("summarizer", "planner")
graph.add_edge("planner",    "translator")
graph.add_edge("translator", END)

app = graph.compile(checkpointer=JsonFileCheckpointer())

result = app.invoke(
    {"tenant_id": "acme-corp", "prism_sequence": [], "raw_output": ""},
    config={"configurable": {"thread_id": "run-001"}},
)

# Inspect the full sequence
for env in result["prism_sequence"]:
    print(f"[{env['agent_id']:12}] category={env['category_slug']}, turn={env['turn_id']}")

# [researcher  ] category=research, turn=0
# [summarizer  ] category=summary,  turn=1
# [planner     ] category=action,   turn=2
```

---

## 3. Multi-Tenant Isolation Proof

The same input text produces incompatible vectors under different tenant keys.
Cross-tenant cosine similarity is provably near-orthogonal (< 0.20).

```python
import numpy as np
from prismlang import Category, TaxonomyConfig, PrismProjector

taxonomy = TaxonomyConfig(categories=[
    Category("risk", "Risk", ["risk", "exposure"]),
])

# Two different tenants
projector_a = PrismProjector(taxonomy, tenant_id="hospital-a")
projector_b = PrismProjector(taxonomy, tenant_id="hospital-b")

text = "Patient exhibits elevated cardiac risk markers."

_, vector_a, _ = projector_a.project(text)
_, vector_b, _ = projector_b.project(text)

cosine_similarity = float(np.dot(vector_a, vector_b))
print(f"Same text, different tenants → cosine similarity: {cosine_similarity:.4f}")
# → cosine similarity: 0.1523   (near-zero, provably isolated)

# Same tenant always produces identical vectors
_, vector_a2, _ = projector_a.project(text)
print(f"Same tenant, repeated call   → cosine similarity: {float(np.dot(vector_a, vector_a2)):.4f}")
# → cosine similarity: 1.0000   (perfectly deterministic)
```

---

## 4. Healthcare Domain

ICU triage pipeline with HIPAA-relevant taxonomy categories.

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node, JsonFileCheckpointer,
)
from langgraph.graph import StateGraph, END

healthcare_taxonomy = TaxonomyConfig(categories=[
    Category("triage",     "Triage",            ["urgent", "critical", "triage", "emergency", "vitals"]),
    Category("diagnosis",  "Clinical Diagnosis", ["diagnosis", "symptoms", "condition", "disease", "labs"]),
    Category("treatment",  "Treatment Plan",     ["treatment", "medication", "dosage", "therapy", "procedure"]),
    Category("compliance", "Regulatory",         ["hipaa", "consent", "privacy", "disclosure", "audit"]),
], alpha=0.3)

# Tenant = hospital system — each hospital gets its own projection space
projector = PrismProjector(healthcare_taxonomy, tenant_id="mercy-general-hospital")

@prism_node(agent_id="triage_nurse", projector=projector)
def triage_nurse(state: PrismState) -> dict:
    return {"raw_output": "Patient presents with critical chest pain. Vitals: BP 180/110, HR 112. Urgent cardiac workup required."}

@prism_node(agent_id="cardiologist", projector=projector)
def cardiologist(state: PrismState) -> dict:
    prev_category = state["prism_sequence"][-1]["category_slug"]
    return {"raw_output": f"Following {prev_category} assessment: STEMI diagnosis confirmed via ECG. Immediate cath lab required."}

@prism_node(agent_id="pharmacist", projector=projector)
def pharmacist(state: PrismState) -> dict:
    return {"raw_output": "Administering aspirin 325mg, heparin 4000 IU IV bolus. Contraindications reviewed."}

graph = StateGraph(PrismState)
graph.add_node("triage",       triage_nurse)
graph.add_node("cardiologist", cardiologist)
graph.add_node("pharmacist",   pharmacist)
graph.set_entry_point("triage")
graph.add_edge("triage",       "cardiologist")
graph.add_edge("cardiologist", "pharmacist")
graph.set_finish_point("pharmacist")

app = graph.compile(checkpointer=JsonFileCheckpointer(".prismlang_healthcare"))
result = app.invoke(
    {"tenant_id": "mercy-general-hospital", "prism_sequence": [], "raw_output": ""},
    config={"configurable": {"thread_id": "patient-case-8821"}},
)

# Audit trail — fully reproducible, traceable to taxonomy rules
for env in result["prism_sequence"]:
    print(f"\n[{env['agent_id']}]")
    print(f"  Category : {env['category_slug']}")
    print(f"  Audit    : {env['rule_chain'][-1]}")
```

---

## 5. Finance Domain

Risk / portfolio / compliance pipeline with Basel III-aligned categories.

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node,
)
from langgraph.graph import StateGraph, END

finance_taxonomy = TaxonomyConfig(categories=[
    Category("risk",       "Market Risk",    ["risk", "volatility", "exposure", "var", "drawdown"]),
    Category("portfolio",  "Portfolio",      ["portfolio", "allocation", "rebalance", "weight", "equity"]),
    Category("compliance", "Compliance",     ["regulation", "sec", "finra", "kyc", "aml", "audit"]),
    Category("market",     "Market Data",    ["price", "index", "yield", "spread", "benchmark"]),
], alpha=0.25)

# Tenant = fund — each fund's vectors are isolated from other funds
projector = PrismProjector(finance_taxonomy, tenant_id="blackrock-fund-xk7")

@prism_node(agent_id="risk_analyst", projector=projector)
def risk_analyst(state: PrismState) -> dict:
    return {"raw_output": "Portfolio VaR at 99% confidence: $4.2M. EM exposure elevated. Volatility up 18% vs. benchmark."}

@prism_node(agent_id="portfolio_manager", projector=projector)
def portfolio_manager(state: PrismState) -> dict:
    return {"raw_output": "Reducing EM allocation from 22% to 15%. Rotating into investment-grade fixed income."}

@prism_node(agent_id="compliance_officer", projector=projector)
def compliance_officer(state: PrismState) -> dict:
    return {"raw_output": "Rebalance approved. SEC 13F disclosure required within 45 days. KYC records updated."}

graph = StateGraph(PrismState)
graph.add_node("risk",       risk_analyst)
graph.add_node("portfolio",  portfolio_manager)
graph.add_node("compliance", compliance_officer)
graph.set_entry_point("risk")
graph.add_edge("risk",      "portfolio")
graph.add_edge("portfolio", "compliance")
graph.set_finish_point("compliance")

app = graph.compile()
result = app.invoke({
    "tenant_id": "blackrock-fund-xk7",
    "prism_sequence": [],
    "raw_output": "",
})

categories = [e["category_slug"] for e in result["prism_sequence"]]
print(f"Category flow: {' → '.join(categories)}")
# Category flow: risk → portfolio → compliance
```

---

## 6. Async Nodes

Use `@async_prism_node` for async LangGraph nodes. ONNX inference runs in a thread pool — the event loop is never blocked.

```python
import asyncio
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, async_prism_node,
)
from langgraph.graph import StateGraph, END

taxonomy = TaxonomyConfig(categories=[
    Category("analysis", "Analysis", ["analysis", "data", "insight"]),
    Category("report",   "Report",   ["report", "summary", "output"]),
])

projector = PrismProjector(taxonomy, tenant_id="async-tenant")

@async_prism_node(agent_id="async_analyst", projector=projector)
async def async_analyst(state: PrismState) -> dict:
    # await your async LLM call here
    await asyncio.sleep(0.01)  # simulating async API call
    return {"raw_output": "Async analysis complete. Data shows strong upward trend."}

@async_prism_node(agent_id="async_reporter", projector=projector)
async def async_reporter(state: PrismState) -> dict:
    await asyncio.sleep(0.01)
    return {"raw_output": "Report generated. Analysis findings documented for review."}

graph = StateGraph(PrismState)
graph.add_node("analyst",  async_analyst)
graph.add_node("reporter", async_reporter)
graph.set_entry_point("analyst")
graph.add_edge("analyst", "reporter")
graph.set_finish_point("reporter")

app = graph.compile()

# Run async graph
result = asyncio.run(app.ainvoke({
    "tenant_id": "async-tenant",
    "prism_sequence": [],
    "raw_output": "",
}))

print(f"Envelopes: {len(result['prism_sequence'])}")  # 2
```

---

## 7. PostgreSQL Checkpointer

Persist graph state across runs using PostgreSQL. Compatible with existing PostgreSQL + pgvector setups.

```python
import os
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node, PostgresCheckpointer,
)
from langgraph.graph import StateGraph, END

taxonomy = TaxonomyConfig(categories=[
    Category("task", "Task", ["task", "work", "complete"]),
])
projector = PrismProjector(taxonomy, tenant_id="my-org")

@prism_node(agent_id="worker", projector=projector)
def worker(state: PrismState) -> dict:
    return {"raw_output": "Task complete. All items processed successfully."}

graph = StateGraph(PrismState)
graph.add_node("worker", worker)
graph.set_entry_point("worker")
graph.set_finish_point("worker")

# Connect to PostgreSQL — use environment variable for credentials
checkpointer = PostgresCheckpointer(
    dsn=os.environ.get("DATABASE_URL", "postgresql://user:pass@localhost/mydb")
)

app = graph.compile(checkpointer=checkpointer)

# First run — creates checkpoint
result = app.invoke(
    {"tenant_id": "my-org", "prism_sequence": [], "raw_output": ""},
    config={"configurable": {"thread_id": "session-abc"}},
)

# Resume same thread — state is loaded from PostgreSQL
result2 = app.invoke(
    {"tenant_id": "my-org", "prism_sequence": [], "raw_output": ""},
    config={"configurable": {"thread_id": "session-abc"}},
)

print(f"Checkpoints persisted in: prismlang_checkpoint table")
```

---

## 8. Custom Taxonomy

Design a taxonomy for any domain. Categories are defined by keyword lists — PrismLang embeds the keywords and uses the mean as the category direction vector.

```python
from prismlang import Category, TaxonomyConfig, PrismProjector

# Legal document review taxonomy
legal_taxonomy = TaxonomyConfig(
    categories=[
        Category(
            slug="discovery",
            label="Discovery",
            keywords=["discovery", "document", "evidence", "subpoena", "deposition", "witness"],
        ),
        Category(
            slug="privilege",
            label="Attorney-Client Privilege",
            keywords=["privilege", "confidential", "attorney", "counsel", "protected", "waiver"],
        ),
        Category(
            slug="contract",
            label="Contract Review",
            keywords=["contract", "clause", "agreement", "liability", "indemnity", "breach"],
        ),
        Category(
            slug="ip",
            label="Intellectual Property",
            keywords=["patent", "trademark", "copyright", "ip", "infringement", "license"],
        ),
    ],
    alpha=0.35,  # higher alpha = stronger taxonomy enforcement
)

projector = PrismProjector(legal_taxonomy, tenant_id="law-firm-xyz", k=64)

# Pre-compute direction vectors at startup (avoids latency on first request)
legal_taxonomy.warm_up()

# Test category inference
test_texts = [
    "The defendant must produce all emails related to the merger by Friday.",
    "This communication is protected by attorney-client privilege and is not discoverable.",
    "Section 12.3 of the agreement limits liability to direct damages only.",
]

for text in test_texts:
    slug, vector, rule_chain = projector.project(text)
    print(f"Category: {slug:12} | Text: {text[:55]}...")

# Category: discovery     | Text: The defendant must produce all emails related to the...
# Category: privilege     | Text: This communication is protected by attorney-client p...
# Category: contract      | Text: Section 12.3 of the agreement limits liability to di...
```

---

## 9. Boundary Translator

`BoundaryTranslator` converts the `prism_sequence` back to human-readable output at the graph exit.

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, prism_node, BoundaryTranslator,
)
from langgraph.graph import StateGraph, END

taxonomy = TaxonomyConfig(categories=[
    Category("risk",   "Risk",   ["risk", "exposure"]),
    Category("action", "Action", ["recommend", "action"]),
])
projector = PrismProjector(taxonomy, tenant_id="demo")

@prism_node(agent_id="agent_a", projector=projector)
def agent_a(state: PrismState) -> dict:
    return {"raw_output": "Elevated risk detected in eastern region portfolio."}

@prism_node(agent_id="agent_b", projector=projector)
def agent_b(state: PrismState) -> dict:
    return {"raw_output": "Recommend reducing exposure by 20% over next two weeks."}

# Structural translator — no LLM needed, reconstructs from rule_chain
translator = BoundaryTranslator(mode="structural")

graph = StateGraph(PrismState)
graph.add_node("a",          agent_a)
graph.add_node("b",          agent_b)
graph.add_node("translator", translator.as_langgraph_node())
graph.set_entry_point("a")
graph.add_edge("a",          "b")
graph.add_edge("b",          "translator")
graph.add_edge("translator", END)

app = graph.compile()
result = app.invoke({
    "tenant_id": "demo", "prism_sequence": [], "raw_output": ""
})

# Final human-readable output assembled from envelopes
print(result["raw_output"])
```

---

## 10. Reading the Audit Trail

Every `PrismEnvelope` contains a `rule_chain` — an ordered list of strings that fully describes every decision made during projection.

```python
from prismlang import Category, TaxonomyConfig, PrismProjector

taxonomy = TaxonomyConfig(categories=[
    Category("compliance", "Compliance", ["regulation", "audit", "kyc", "aml"]),
    Category("risk",       "Risk",       ["risk", "exposure", "volatility"]),
])

projector = PrismProjector(taxonomy, tenant_id="regulated-bank-001", k=64)

text = "KYC review flagged three high-risk accounts for AML investigation."
slug, vector, rule_chain = projector.project(text)

print(f"Category : {slug}")
print(f"Vector   : [{vector[0]:.4f}, {vector[1]:.4f}, ... {vector[-1]:.4f}] (64-d)")
print(f"\nAudit Rule Chain:")
for i, rule in enumerate(rule_chain, 1):
    print(f"  {i}. {rule}")

# Category : compliance
# Vector   : [0.1823, -0.0421, ... 0.0912] (64-d)
#
# Audit Rule Chain:
#   1. text -> encoder(all-MiniLM-L6-v2, d=384)
#   2. category_inference -> slug='compliance'
#   3. spherical_blend(alpha=0.300) -> v_prime
#   4. JL_reduction(seed=sha256('regulated-bank-001'), k=64) -> p

# Matrix fingerprint — stable identifier for this tenant's projection space
print(f"\nMatrix fingerprint : {projector.matrix_fingerprint()}")
print(f"Tenant seed        : {projector._seed}")
```

---

## 11. Encoder Artifact Id & Shared Session

*New in 0.1.2.* PrismLang keeps exactly one ONNX encoder session per process — `encode`, `encode_batch`, their async variants, and `get_session()` all share the same lazily-initialised singleton. Host processes (e.g. verification layers that compare vectors produced at different times) can stamp stored vectors with `model_id()` and detect encoder-artifact mismatches on read.

```python
from prismlang import model_id, get_session

# Stable identifier of the loaded model artifact — cached for the process lifetime.
# Format: "{hf_repo}@{revision}:{sha256(model.onnx)[:12]}"
print(model_id())
# sentence-transformers/all-MiniLM-L6-v2@1110a243...:6fd5d72fe458

# The process-wide onnxruntime.InferenceSession (loads the model on first call).
session = get_session()
assert session is get_session()   # always the same object — no second model load

# Also available on the submodule:
from prismlang import encoder
assert encoder.model_id() == model_id()
assert encoder.get_session() is session
```

---

## Error Handling

PrismLang raises typed exceptions — catch specific types or the base class.

```python
from prismlang import (
    TaxonomyConfig, Category, PrismProjector,
    PrismLangError, ZeroVectorError,
    DuplicateCategoryError, MissingTenantError,
    CheckpointerConnectionError,
)

# Catch specific errors
try:
    taxonomy = TaxonomyConfig(categories=[
        Category("a", "A", ["apple"]),
        Category("a", "A duplicate", ["banana"]),  # duplicate slug!
    ])
except DuplicateCategoryError as e:
    print(f"Taxonomy error: {e}")
    # Taxonomy error: Duplicate category slug: 'a'. All slugs must be unique.

# Catch all PrismLang errors with one handler
try:
    taxonomy = TaxonomyConfig(categories=[
        Category("general", "General", ["the"]),
    ])
    projector = PrismProjector(taxonomy, tenant_id="test")
    projector.project("")  # empty input
except ZeroVectorError as e:
    print(f"Projection error: {e}")
except PrismLangError as e:
    print(f"PrismLang error: {e}")
```

---

## Full Exception Hierarchy

```
PrismLangError
├── EncoderError
│   ├── ModelDownloadError      # ONNX model could not be downloaded
│   ├── ModelNotFoundError      # ONNX file missing from cache
│   └── TokenizerNotFoundError  # tokenizer.json missing
├── TaxonomyError
│   ├── DuplicateCategoryError  # two categories share a slug
│   ├── UnknownCategoryError    # slug not in taxonomy
│   └── EmptyTaxonomyError      # no categories provided
├── ProjectionError
│   ├── ZeroVectorError         # empty or whitespace-only input
│   └── DimensionMismatchError  # encoder/JL matrix shape conflict
├── CheckpointerError
│   ├── CheckpointerConnectionError  # cannot connect to backend
│   └── CheckpointerSchemaError      # schema creation failed
└── TenantError
    └── MissingTenantError      # tenant_id absent from PrismState
```
