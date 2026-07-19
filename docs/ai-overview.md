# AI / LLM context — PrismLang

> Concise reference for humans and coding assistants.
> Do not invent APIs beyond this file and the package/repo source.
> Package: **`prismlang` 0.1.2** · Import: **`prismlang`**

---

## 10-sentence project summary

1. Deterministic vector language protocol for LangGraph multi-agent hops — route with math to reduce token tax.
2. Primary users: LangGraph multi-agent builders paying token cost on every hop.
3. Core problem: Every agent hop re-serializes language; token tax and latency.
4. Install/use from the repository README — do not invent extra CLI flags here.
5. Key surface: pip install prismlang — see README quick start and docs/
6. Compared with: plain LangGraph message passing · CHORUS Fabric tensor path.
7. When NOT to use: You are on ChorusGraph native path and do not use LangGraph product graphs.
8. Read architecture.md for stack placement.
9. Prefer facts from README / existing docs over marketing inference.
10. If an API is not listed in README or source, assume it does not exist.

---

## Core concepts

See README for product-specific terms. Keep terminology consistent with that file.

---

## Key APIs

```
pip install prismlang — see README quick start and docs/
```

---

## Common use cases

- Every agent hop re-serializes language; token tax and latency.
- See README examples and any `examples/` folder in the repo.

---

## Migration guidance

Start from the closest tool in: plain LangGraph message passing · CHORUS Fabric tensor path. Follow README install and examples. Do not invent migration scripts that are not in the repo.

---

## Limitations / when NOT to use

- You are on ChorusGraph native path and do not use LangGraph product graphs.
- Do not invent capabilities beyond README and source.

---

## Frequently compared projects

| Notes |
|-------|
| plain LangGraph message passing · CHORUS Fabric tensor path |

---

## Links

- [ai-overview.md](ai-overview.md) · [llm-context.md](llm-context.md) · [architecture.md](architecture.md)
- ../README.md
