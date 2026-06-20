# Changelog

All notable changes to PrismLang are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Sphinx/MkDocs hosted documentation site
- Dynamic taxonomy adaptation (per-tenant alpha tuning)
- OpenTelemetry span emission per encoder call
- Kubernetes sidecar deployment guide

---

## [0.1.0] — 2026-06-19

### Added
- **Core protocol**: `PrismProjector` implementing spherical blend (eq. 1) and JL reduction (eq. 2)
- **ONNX encoder**: `all-MiniLM-L6-v2` local inference, no external API required
- **Taxonomy system**: `TaxonomyConfig` + `Category` with keyword-based direction vectors
- **`@prism_node` decorator**: zero-change LangGraph node integration
- **`PrismState`**: append-only LangGraph state channel for envelope sequences
- **`BoundaryTranslator`**: structural + LLM-assisted decoding at graph exit
- **`JsonFileCheckpointer`**: zero-dependency checkpoint persistence
- **`PostgresCheckpointer`**: PostgreSQL + pgvector compatible persistence
- **Async support**: `async_prism_node`, `AsyncJsonFileCheckpointer`, `AsyncPostgresCheckpointer`
- **Exception hierarchy**: typed errors for all failure modes
- **34 unit tests**: encoder, projector, state, tenant isolation — all passing
- **3 domain benchmarks**: healthcare, finance, trade market with PostgreSQL-backed metrics
- **Benchmark database**: `prismLangDB` schema + seed data for all domains
- **Mathematical paper**: "PrismLang: A Deterministic Vector Language Protocol for Multi-Agent AI Orchestration" (Amin Parva, Insight IT Solutions LLC, June 2026)

### Performance (benchmark v0.1.0)
- Prompt token reduction: −57% to −62% across all domains
- State payload reduction: −45% to −50% at turn 3 vs standard text state
- LLM latency impact: 0% (encoding adds <5ms after warmup)
- Cross-tenant cosine similarity: 0.14–0.17 (near-orthogonal isolation)

---

[Unreleased]: https://github.com/insightits/prismlang/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/insightits/prismlang/releases/tag/v0.1.0
