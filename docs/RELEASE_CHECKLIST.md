# PrismLang Release Checklist

## Current Status: v0.1.0 — Internal / Demo Ready

### ✅ Completed (v0.1.0)

- [x] Core protocol implementation (encoder, taxonomy, projector, envelope, state, middleware)
- [x] `@prism_node` decorator — zero-change agent integration
- [x] `BoundaryTranslator` — structural + LLM-assisted modes
- [x] `JsonFileCheckpointer` — zero-dependency persistence
- [x] `PostgresCheckpointer` — production-grade persistence
- [x] 34 unit tests, 0 failures — encoder, projector, state, tenant isolation
- [x] 3 domain benchmark examples with PostgreSQL-backed results (healthcare, finance, trade market)
- [x] Benchmark metrics: time, tokens, payload, memory, audit trail
- [x] PrismLang paper (PDF) — mathematical foundation
- [x] README, ARCHITECTURE, BENCHMARK docs
- [x] `prismLangDB` with all domain schemas and seed data

---

## 🔶 Required Before Public Release (v0.2.0)

### Code Quality
- [ ] `pyproject.toml` (replace `setup.py`) with proper metadata
- [ ] Add `py.typed` marker for type-checker compatibility
- [ ] Async support: `async_encode()`, async `@prism_node` for async LangGraph graphs
- [ ] Error handling: custom exceptions (`PrismLangEncoderError`, `TaxonomyError`)
- [ ] Logging module: structured logs per turn (optional debug mode)
- [ ] `__all__` exports audit — ensure no internal symbols leak

### Testing
- [ ] Integration tests: full LangGraph graph with real checkpointer
- [ ] Property-based tests (hypothesis): JL matrix seeding properties
- [ ] Load test: 100-turn graph, measure state growth curve
- [ ] Test with real Gemini + Claude API keys in CI
- [ ] Coverage report target: >90%

### Security
- [ ] Threat model document: what `tenant_id` as shared secret can/cannot protect
- [ ] Recommend overlay encryption for high-security deployments
- [ ] Pin ONNX model hash to prevent supply-chain model substitution

### Distribution
- [ ] `LICENSE` file (Apache 2.0)
- [ ] PyPI package: `pip install prismlang`
- [ ] GitHub Actions CI: test on Python 3.10, 3.11, 3.12
- [ ] GitHub Actions: auto-publish to PyPI on tag

### Documentation
- [ ] Sphinx or MkDocs site with API reference
- [ ] Tutorial: "Migrate an existing LangGraph graph to PrismLang in 10 minutes"
- [ ] Domain packs guide: how to design a custom taxonomy
- [ ] FAQ: "Does PrismLang change what my LLM sees?" (No — only state transport)
- [ ] CHANGELOG.md

---

## 🔷 Enterprise Features (v1.0.0)

- [ ] Multi-tenant management dashboard (FastAPI + React)
- [ ] Pre-built domain taxonomy packs (healthcare HIPAA, finance Basel III, legal discovery)
- [ ] PrismLang Studio: visual graph builder with category flow preview
- [ ] OpenTelemetry integration: emit spans per encoder call
- [ ] Kubernetes sidecar deployment guide
- [ ] SOC 2 / HIPAA alignment documentation
- [ ] SLA-backed enterprise support tier

---

## Estimated Timeline

| Milestone | Estimate |
|---|---|
| v0.2.0 — Public release ready | 3–4 weeks |
| v0.3.0 — PyPI + CI/CD + async | +2 weeks |
| v1.0.0 — Enterprise features | +6–8 weeks |
