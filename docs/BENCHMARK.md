# PrismLang Benchmark Documentation

## Test Environment

| Component | Version |
|---|---|
| Python | 3.12.10 |
| LangGraph | 0.2+ |
| ONNX Runtime | 1.17+ |
| Encoder model | `sentence-transformers/all-MiniLM-L6-v2` (ONNX export) |
| LLM (agent calls) | Gemini 2.0 Flash / offline fallback |
| Database | PostgreSQL 16 + pgvector extension |
| OS | Windows 11 / Linux-compatible |
| Hardware | CPU only (no GPU required) |

---

## Test Methodology

### What We Measure

Each benchmark run compares two modes of the **same** 3-node LangGraph pipeline:

**Standard mode** — agents exchange full text in state. State grows with every turn as the full message history accumulates.

**PrismLang mode** — agents exchange PrismEnvelopes. State carries a fixed-size vector per turn regardless of text length.

Both modes make **identical LLM calls** with **identical prompts** for the first agent turn. The key difference is what gets stored and re-transmitted in state between agents.

### Metrics Collected

| Metric | How measured |
|---|---|
| Wall-clock time (ms) | `time.perf_counter()` start/stop around full graph invocation |
| LLM call time (ms) | `time.perf_counter()` around each `generate_content()` call |
| Encode time (ms) | `time.perf_counter()` around `projector.project()` |
| Prompt tokens | `usage_metadata.prompt_token_count` from Gemini response (or word-count estimate in offline mode) |
| Output tokens | `usage_metadata.candidates_token_count` from Gemini response |
| State payload (bytes) | `len(json.dumps(state).encode('utf-8'))` at the end of each turn |
| Vector payload (bytes) | `k × 4 bytes` (float32) × number of turns — theoretical minimum wire size |
| Peak memory (MB) | `tracemalloc.get_traced_memory()` peak during graph execution |
| Compression ratio | `state_bytes / vector_bytes` — stored as a generated column in PostgreSQL |
| Category flow | Ordered list of `category_slug` values across all PrismEnvelopes |
| Tenant matrix fingerprint | First 16 hex chars of `SHA-256(JL_matrix.tobytes())` |

### What the Benchmark Does NOT Measure (and Why)

- **LLM quality degradation** — PrismLang does not change what the LLM sees or outputs in agent turn 1. For downstream agents, they receive the category slug rather than full text history. Quality testing under various prompt strategies is planned for v0.2.
- **Network latency** — benchmarks run on localhost. PrismLang's bandwidth advantage is most pronounced in distributed deployments.
- **GPU inference** — ONNX encoder runs CPU-only in this benchmark. GPU would reduce encode latency to <1ms.

---

## Domain Test Cases

### Healthcare: Critical ICU Admission

**Data source:** `healthcare.patients`, `healthcare.diagnoses`, `healthcare.lab_results`, `healthcare.clinical_notes`

**Patient:** Thomas Okafor (MRN-005) — COPD exacerbation + decompensated heart failure

**Agent pipeline:**
```
triage_agent → clinical_agent → compliance_agent
```

**What each agent does:**
- `triage_agent`: reads full patient context from DB, assesses urgency, flags critical SpO2 88% and BNP 980
- `clinical_agent`: reviews prior assessment category, recommends treatment (BiPAP, IV furosemide, allergy check)
- `compliance_agent`: verifies HIPAA consent, DNR status, documentation requirements, allergy safety

**Why this case:** Multiple active critical alerts, allergy constraint, dual diagnosis — tests whether category inference correctly routes `clinical → lab → compliance`.

**Result:** Category flow `clinical → lab → compliance` — correct. Allergy safety step lands in `compliance` category as expected.

---

### Finance: Hedge Fund Under Active Risk Events

**Data source:** `finance.accounts`, `finance.positions`, `finance.risk_events`, `finance.transactions`

**Account:** Apex Macro Hedge Fund (ACC-F003, AUM $1.2B) — simultaneous VaR breach + margin call

**Agent pipeline:**
```
risk_agent → portfolio_agent → compliance_agent
```

**What each agent does:**
- `risk_agent`: loads positions (SPY short, TLT long, EURUSD), risk events (VaR $22.4M vs $18M limit, margin call $8.7M), computes exposure
- `portfolio_agent`: receives risk category signal, recommends rebalancing (cover 50% SPY short, cut TLT)
- `compliance_agent`: checks position limit rules, KYC status, SEC Rule 15c3-1, pre-trade sign-off

**Why this case:** Multi-event crisis scenario with cascading risk (VaR breach → margin call). Tests category routing `risk → portfolio → compliance` under high-context load.

**Result:** Category flow `risk → portfolio → compliance` — correct. Standard state grew to 1,760 B; PrismLang held at 960 B.

---

### Trade Market: Momentum Trade Signal Analysis

**Data source:** `trade_market.instruments`, `trade_market.ohlcv`, `trade_market.order_book_snapshots`, `trade_market.signals`

**Instrument:** NVDA — 5-day momentum breakout above $151 resistance

**Agent pipeline:**
```
signal_agent → execution_agent → risk_agent
```

**What each agent does:**
- `signal_agent`: reads OHLCV (5 days), order book snapshot, existing signal — confirms momentum (RSI 65, volume +28%)
- `execution_agent`: receives signal category, analyses bid/ask spread (6.5bps), recommends VWAP limit entry
- `risk_agent`: receives execution category, sizes position (12,900 shares), validates against 20% single-name limit

**Why this case:** Quantitative multi-factor decision chain requiring context from all three agents. Tests sequential category flow `signal → execution → risk`.

**Result:** Category flow `signal → execution → risk` — correct. Prompt token reduction −58.6% vs standard.

---

## Results Summary

All results stored in `bench.run_results` table in `prismLangDB`. Reproducible via:

```bash
python -m benchmarks.run_all
```

### Token Reduction (Prompt Tokens)

| Domain | Standard | PrismLang | Reduction |
|---|---|---|---|
| Healthcare | 391 | 148 | **−62.1%** |
| Finance | 407 | 175 | **−57.0%** |
| Trade Market | 435 | 180 | **−58.6%** |
| **Average** | **411** | **168** | **−59.2%** |

### State Payload at Turn 3

| Domain | Standard | PrismLang Vector | Ratio |
|---|---|---|---|
| Healthcare | 1,928 B | 960 B | **2.0× smaller** |
| Finance | 1,760 B | 960 B | **1.8× smaller** |
| Trade Market | 1,867 B | 960 B | **1.9× smaller** |

> Note: PrismLang's vector payload is **fixed at `k × 4 bytes × turns`** regardless of text length. As agent output text grows longer (realistic in production), the ratio improves further. At 10 turns with 500-word outputs per turn, expected reduction is 8–12×.

### Latency Impact

| Component | Standard | PrismLang | Delta |
|---|---|---|---|
| LLM calls | ~151 ms | ~151 ms | **0%** |
| PrismLang encode | 0 ms | ~0 ms (ONNX cached) | negligible |
| Total overhead | — | +20–21 ms | graph orchestration overhead |

The +20ms overhead is **LangGraph's internal checkpoint and state management** cost, not PrismLang encoding. ONNX inference for a single text on CPU is <5ms after model warmup.

---

## Reproducibility

All tests are deterministic given the same inputs:
- JL matrices are seeded from `SHA-256(tenant_id)` — always identical for the same tenant
- ONNX encoder is deterministic for the same model weights
- Fallback agent responses are fixed strings when `GEMINI_API_KEY` is not set

To reproduce with real Gemini API:
```bash
set GEMINI_API_KEY=your_key
python -m benchmarks.run_all
```

To view stored results:
```sql
SELECT domain, mode, prompt_tokens, state_bytes, vector_bytes, compression_ratio, category_flow
FROM bench.run_results
ORDER BY created_at DESC;
```
