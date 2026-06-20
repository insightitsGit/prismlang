# PrismLang Architecture

## Protocol Overview

PrismLang is a **transport-layer protocol**, not a context-window compressor. The distinction matters:

- **Context-window compressors** (LLMLingua, ACON) reduce what the LLM *reads* on each call. They operate on the prompt before it reaches the model.
- **PrismLang** reduces what agents *transmit to each other* between turns. The LLM call itself is unchanged; only the inter-agent state envelope is compressed.

This means PrismLang is **composable with** LLMLingua — you can run both simultaneously.

---

## Component Map

```
┌─────────────────────────────────────────────────────────┐
│                   LangGraph Graph                        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ Agent A  │    │ Agent B  │    │ Agent C  │           │
│  │ (any LLM)│    │ (any LLM)│    │ (any LLM)│           │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘           │
│       │               │               │                  │
│  ┌────▼───────────────▼───────────────▼──────────────┐  │
│  │              @prism_node decorator                  │  │
│  │   text → encoder → taxonomy → blend → JL → envelope│  │
│  └────────────────────────┬───────────────────────────┘  │
│                            │                             │
│  ┌─────────────────────────▼───────────────────────────┐ │
│  │                 PrismState channel                   │ │
│  │  prism_sequence: [Env0, Env1, Env2] (append-only)   │ │
│  └─────────────────────────┬───────────────────────────┘ │
│                            │                             │
│  ┌─────────────────────────▼───────────────────────────┐ │
│  │              BoundaryTranslator                      │ │
│  │  envelope sequence → human-readable report           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow Per Turn

```
1. Agent node function runs → returns {"raw_output": "some text"}

2. @prism_node decorator intercepts:
   a. encoder.encode(text)
      → 384-d float32 unit vector  v
   b. taxonomy.infer_category(text)
      → slug: str  (e.g. "risk")
   c. taxonomy.direction_vector(slug)
      → 384-d unit vector  eᵢ
   d. Spherical blend:
      v' = normalize((1-α)·v + α·‖v‖·eᵢ)
   e. JL reduction (tenant-unique matrix P):
      p = normalize(P·v')   →  64-d float32 unit vector

3. PrismEnvelope created:
   {
     turn_id:       int,
     agent_id:      str,
     category_slug: str,
     vector:        [float × 64],    ← 256 bytes on the wire
     rule_chain:    [str × 4],       ← full audit trail
   }

4. PrismEnvelope appended to prism_sequence via operator.add
   → downstream agents see growing sequence of envelopes, NOT growing text

5. raw_output kept in state for in-memory readability only
   (never transmitted between agents in a distributed deployment)
```

---

## The PrismEnvelope Wire Format

```python
class PrismEnvelope(TypedDict):
    turn_id:       int            # 8 bytes
    agent_id:      str            # variable, ~20 bytes avg
    category_slug: str            # variable, ~10 bytes avg
    vector:        List[float]    # k × 4 bytes  (k=64 → 256 bytes)
    rule_chain:    List[str]      # 4 strings, ~120 bytes total
```

**Total wire size per envelope: ~414 bytes** at k=64.

Compare to a typical agent turn in text state: 400–800 words = 2,000–4,000 bytes.

**Compression ratio improves with each turn** because text state accumulates while vector state stays fixed per envelope.

---

## Tenant Isolation Mechanism

The JL matrix `P ∈ ℝ^(k×384)` is seeded deterministically from `SHA-256(tenant_id)`:

```python
seed = int(hashlib.sha256(tenant_id.encode()).hexdigest(), 16) % 2**32
rng  = np.random.default_rng(seed)
P    = rng.standard_normal((k, 384))
P   /= np.linalg.norm(P, axis=1, keepdims=True)
```

**Security properties:**
- Two tenants with different `tenant_id` values always produce different matrices
- A vector `p_A = normalize(P_A · v')` is geometrically incompatible with `P_B`
- Measured cross-tenant cosine similarity: **0.14–0.17** (near-zero, near-orthogonal)
- The `tenant_id` is the shared secret — it never leaves the server; only `p` is transmitted
- Uniqueness: `2^32` possible seeds from the 32-bit truncation of SHA-256 output

This is **not encryption** — it is geometric isolation. Recovering `v` from `p` would require knowing `P`, which requires knowing `tenant_id`. For adversarial hardness requirements, a full encryption layer should be added on top.

---

## Checkpointer Backends

### JsonFileCheckpointer (default)

```
.prismlang_checkpoints/
  {thread_id}/
    {checkpoint_id}.json
```

Zero dependencies beyond Python stdlib. Suitable for development and single-server deployments.

### PostgresCheckpointer

Creates `prismlang_checkpoint` table:

```sql
CREATE TABLE prismlang_checkpoint (
    thread_id      TEXT,
    checkpoint_id  TEXT,
    tenant_id      TEXT,
    checkpoint     JSONB,
    metadata       JSONB,
    parent_config  JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

Can share a database with PrismRAG. Point at any PostgreSQL instance via:

```python
from prismlang import PostgresCheckpointer
checkpointer = PostgresCheckpointer(dsn="postgresql://user:pass@host/db")
```

---

## Taxonomy Design Guidelines

A well-designed taxonomy is the single most important configuration decision.

| Guideline | Why |
|---|---|
| Use 3–7 categories | Too few = no routing signal. Too many = category overlap reduces projection quality. |
| Keywords should be domain-specific | Generic words ("the", "and") add noise to direction vectors. |
| Keywords should be mutually exclusive across categories | Overlap pulls direction vectors toward each other, reducing inter-category angle. |
| `alpha` between 0.2–0.4 | Lower alpha = more semantic richness, less taxonomy enforcement. Higher = stronger domain routing. |
| `k` between 32–128 | k=64 is the recommended default. Higher k = better distance preservation, larger payload. |

```python
# Good taxonomy design
TaxonomyConfig([
    Category("risk",       "Risk",       ["var", "exposure", "drawdown", "default", "margin"]),
    Category("portfolio",  "Portfolio",  ["allocation", "rebalance", "sharpe", "return", "pnl"]),
    Category("compliance", "Compliance", ["regulation", "kyc", "aml", "disclosure", "audit"]),
], alpha=0.3)

# Avoid: overlapping keywords between categories
# "risk" and "portfolio" both having "loss" reduces their directional separation
```
