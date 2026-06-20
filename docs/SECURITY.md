# Security Policy & Threat Model

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Email: **prismrag@insightits.com**  
Subject: `[SECURITY] PrismLang — <brief description>`

We aim to acknowledge within 48 hours and provide a fix or mitigation within 14 days. We follow responsible disclosure: you will be credited in the release notes unless you request anonymity.

---

## What PrismLang Protects

### Tenant Isolation via JL Projection

The Johnson-Lindenstrauss (JL) matrix `P` is seeded from `SHA-256(tenant_id)`.

**What this guarantees:**
- Two tenants encoding the *same* text produce vectors with cosine similarity ≈ 0.14–0.17 (near-orthogonal).
- A vector intercepted from Tenant A's traffic is geometrically meaningless to a model operating under Tenant B's JL matrix.
- The mapping `tenant_id → JL matrix` is deterministic: the same `tenant_id` always produces the same matrix across runs, machines, and library versions (provided the numpy PRNG seeding API is stable — see note below).

**What this does NOT guarantee:**
- PrismLang is **not a cryptographic access control system**. Tenant isolation is a geometric property, not an authentication/authorisation mechanism.
- If an adversary knows the `tenant_id`, they can reconstruct the JL matrix and partially reverse the projection.
- PrismLang vectors are stored as cleartext JSON/JSONB. An attacker with read access to your checkpoint storage can observe the category slugs and vector components.

### Rule Chain Auditability

The `rule_chain` field in every `PrismEnvelope` records the full derivation path:

```
text -> encoder(all-MiniLM-L6-v2, d=384)
category_inference -> slug='risk'
spherical_blend(alpha=0.300) -> v_prime
JL_reduction(seed=sha256('acme-finance'), k=64) -> p
```

This chain is append-only (via `operator.add` on `prism_sequence`) and is never modified once written. It provides a complete audit trail for compliance workflows.

---

## Threat Model

### In Scope

| Threat | Mitigation |
|--------|-----------|
| Cross-tenant vector leakage | Near-orthogonal JL projection per tenant |
| Checkpoint tampering | rule_chain is immutable in PrismState |
| Model supply chain (ONNX) | SHA-256 verified snapshot download from HuggingFace |
| Empty / adversarial input | `ZeroVectorError` raised before projection |

### Out of Scope

| Threat | Notes |
|--------|-------|
| Authentication/authorisation | Handled by your application layer — PrismLang does not authenticate callers |
| Checkpoint encryption at rest | Use Postgres TDE or filesystem encryption — PrismLang stores plaintext JSON |
| Network transport security | Use TLS for your Postgres DSN — PrismLang passes the DSN as-is |
| LLM prompt injection | PrismLang does not parse or execute LLM output; the `BoundaryTranslator` uses structural reconstruction by default |
| Side-channel timing attacks | ONNX inference time varies with input length; PrismLang does not pad timing |

---

## Sensitive Configuration

### Database Credentials

PrismLang accepts a DSN string for `PostgresCheckpointer` and `AsyncPostgresCheckpointer`. **Never hard-code credentials in source code.** Use environment variables:

```python
import os
from prismlang import PostgresCheckpointer

checkpointer = PostgresCheckpointer(dsn=os.environ["DATABASE_URL"])
```

Set `DATABASE_URL` in your deployment environment (`.env`, secret manager, Kubernetes secret).

### Tenant ID

The `tenant_id` is the seed for the JL matrix. Treat it as a **sensitive configuration value**:
- Use opaque identifiers (UUIDs, hashed org IDs) rather than human-readable strings that expose business logic.
- Rotate `tenant_id` values periodically for high-security deployments — old vectors become incompatible after rotation, which is the intended behaviour.

---

## Dependency Security

PrismLang uses a minimal dependency set. Known security-sensitive dependencies:

| Package | Purpose | Notes |
|---------|---------|-------|
| `onnxruntime` | ONNX inference | Keep updated; ONNX Runtime has had CVEs for malformed model files. Only load models from trusted sources. |
| `huggingface-hub` | Model download | Downloads are cached; verify the `all-MiniLM-L6-v2` model hash after first download in regulated environments. |
| `psycopg2-binary` | PostgreSQL | Use `psycopg2` (not binary) in production for better control over OpenSSL linkage. |
| `asyncpg` | Async PostgreSQL | Pin to a known-good version; asyncpg has had TLS connection handling bugs in older releases. |

Run `pip audit` or `safety check` as part of your CI pipeline.

---

## Overlay Encryption (Recommended for PII)

If the `raw_output` field in your `PrismState` contains personally identifiable information (PII), encrypt it before writing to the checkpointer:

```python
from cryptography.fernet import Fernet

key = Fernet.generate_key()  # store this in your secret manager
cipher = Fernet(key)

# In your node:
encrypted_output = cipher.encrypt(raw_output.encode()).decode()
state["raw_output"] = encrypted_output

# At graph exit (BoundaryTranslator):
raw = cipher.decrypt(state["raw_output"].encode()).decode()
```

The PrismLang vector and category slug do not contain the original text — they are projections. Encrypting `raw_output` ensures that even if checkpoint storage is compromised, the original text is not recoverable without the Fernet key.

---

## NumPy PRNG Stability Note

The JL matrix is seeded via `numpy.random.default_rng(seed)` with `standard_normal`. NumPy guarantees that `default_rng` produces identical bitstreams for the same seed across NumPy ≥ 1.17 with the same OS and CPU — but the stream is **not guaranteed across major NumPy versions**. If you upgrade NumPy, validate that cross-tenant cosine similarity remains < 0.20 before deploying.

To pin the PRNG version explicitly, add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.26.0,<3.0.0",
    ...
]
```

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |
| < 0.1   | ❌ No     |
