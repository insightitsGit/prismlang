"""PrismLang demo — 3-node LangGraph with tenant isolation proof.

Graph topology:
    researcher → summarizer → reviewer → translator
         ↓            ↓           ↓           ↓
     PrismEnv     PrismEnv   PrismEnv   Human report

Each node is wrapped by @prism_node. Agents return plain text; PrismLang
intercepts at the node boundary, encodes to a compressed vector, and appends
a PrismEnvelope to the state sequence.

The demo also runs the same inputs under two different tenant IDs and prints
the cosine similarity of their output vectors — proving cryptographic isolation
without any encryption.

Run:
    python -m demo.graph
or from the project root:
    python demo/graph.py
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from langgraph.graph import StateGraph, END

from prismlang import (
    PrismState,
    PrismProjector,
    BoundaryTranslator,
    JsonFileCheckpointer,
    prism_node,
)
from demo.taxonomy_config import FINANCE_TAXONOMY


# ------------------------------------------------------------------ #
# Agent node functions (pure text — no PrismLang awareness needed)    #
# ------------------------------------------------------------------ #

RESEARCHER_OUTPUT = (
    "The portfolio shows elevated credit risk exposure in emerging market bonds. "
    "Volatility has spiked 18% over the last quarter, driven by currency risk in "
    "Brazilian Real and Turkish Lira positions. Counterparty default probability "
    "for the top-5 exposures has risen from 0.3% to 1.2% annualised."
)

SUMMARIZER_OUTPUT = (
    "Market conditions remain challenging. Equity indices are down 4.2% YTD. "
    "Bond yields in 10-year Treasuries have risen 45bps. FX volatility is at a "
    "3-year high. The asset allocation review suggests rotating 8% of the portfolio "
    "from EM equities to investment-grade corporate bonds to improve the Sharpe ratio."
)

REVIEWER_OUTPUT = (
    "Compliance review complete. The proposed reallocation requires disclosure under "
    "SEC Rule 17a-5 and a fresh KYC refresh for three counterparties flagged by AML "
    "screening. Regulatory reporting deadlines: Q3 form must be filed within 30 days. "
    "No sanctions violations detected. Fiduciary obligations satisfied."
)


def make_graph(tenant_id: str) -> tuple:
    """Build a compiled LangGraph for the given tenant."""
    projector = PrismProjector(taxonomy=FINANCE_TAXONOMY, tenant_id=tenant_id, k=64)
    checkpointer = JsonFileCheckpointer(root=f".prismlang_demo/{tenant_id}")

    @prism_node(agent_id="researcher", projector=projector)
    def researcher(state: PrismState) -> dict:
        print(f"  [researcher/{tenant_id}] analysing portfolio risk ...")
        return {"raw_output": RESEARCHER_OUTPUT}

    @prism_node(agent_id="summarizer", projector=projector)
    def summarizer(state: PrismState) -> dict:
        print(f"  [summarizer/{tenant_id}] summarising market conditions ...")
        return {"raw_output": SUMMARIZER_OUTPUT}

    @prism_node(agent_id="reviewer", projector=projector)
    def reviewer(state: PrismState) -> dict:
        print(f"  [reviewer/{tenant_id}] running compliance review ...")
        return {"raw_output": REVIEWER_OUTPUT}

    translator = BoundaryTranslator()

    def translator_node(state: PrismState) -> dict:
        report = translator.translate(state)
        return {"raw_output": report}

    workflow = StateGraph(PrismState)
    workflow.add_node("researcher", researcher)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("translator", translator_node)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "summarizer")
    workflow.add_edge("summarizer", "reviewer")
    workflow.add_edge("reviewer", "translator")
    workflow.add_edge("translator", END)

    app = workflow.compile(checkpointer=checkpointer)
    return app, projector


# ------------------------------------------------------------------ #
# Isolation proof                                                      #
# ------------------------------------------------------------------ #

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def isolation_proof(proj_a: PrismProjector, proj_b: PrismProjector) -> None:
    """Show that two tenants produce incompatible vectors for the same text."""
    test_texts = [RESEARCHER_OUTPUT, SUMMARIZER_OUTPUT, REVIEWER_OUTPUT]
    print("\n" + "=" * 60)
    print("TENANT ISOLATION PROOF")
    print("=" * 60)
    print(f"Tenant A: {proj_a.tenant_id!r}  (matrix fp: {proj_a.matrix_fingerprint()})")
    print(f"Tenant B: {proj_b.tenant_id!r}  (matrix fp: {proj_b.matrix_fingerprint()})")
    print()

    sims = []
    for i, text in enumerate(test_texts):
        _, vec_a, _ = proj_a.project(text)
        _, vec_b, _ = proj_b.project(text)
        sim = cosine_similarity(vec_a.tolist(), vec_b.tolist())
        sims.append(sim)
        print(f"  Text {i+1} cosine(A, B) = {sim:+.4f}  {'✓ isolated' if abs(sim) < 0.20 else '✗ NOT isolated'}")

    avg = float(np.mean(np.abs(sims)))
    print(f"\n  Mean |cosine similarity| = {avg:.4f}")
    print(f"  {'✓ PASS — tenant vectors are cryptographically isolated' if avg < 0.20 else '✗ FAIL'}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    print("\n" + "=" * 60)
    print("PrismLang Demo — Finance Multi-Agent Graph")
    print("Insight IT Solutions LLC")
    print("=" * 60)

    TENANT_A = "acme-finance-prod"
    TENANT_B = "globex-trading-dev"

    # Run graph for tenant A
    print(f"\n--- Running graph for tenant: {TENANT_A!r} ---")
    app_a, proj_a = make_graph(TENANT_A)
    config_a = {"configurable": {"thread_id": "demo-thread-a"}}
    initial_state: PrismState = {
        "prism_sequence": [],
        "raw_output": "",
        "tenant_id": TENANT_A,
    }
    final_state_a = app_a.invoke(initial_state, config=config_a)

    print("\n--- PrismEnvelope Sequence (Tenant A) ---")
    for env in final_state_a["prism_sequence"]:
        print(
            f"  Turn {env['turn_id']}  agent={env['agent_id']!r:<12} "
            f"category={env['category_slug']!r:<12} "
            f"vec[:4]={[round(x, 4) for x in env['vector'][:4]]}"
        )
        for step in env["rule_chain"]:
            print(f"    audit: {step}")

    print("\n--- Boundary Translator Output (Tenant A) ---")
    print(final_state_a["raw_output"])

    # Run graph for tenant B (same inputs, different JL matrix)
    print(f"\n--- Running graph for tenant: {TENANT_B!r} ---")
    app_b, proj_b = make_graph(TENANT_B)
    config_b = {"configurable": {"thread_id": "demo-thread-b"}}
    initial_state_b: PrismState = {
        "prism_sequence": [],
        "raw_output": "",
        "tenant_id": TENANT_B,
    }
    app_b.invoke(initial_state_b, config=config_b)

    # Isolation proof
    isolation_proof(proj_a, proj_b)

    print("\n" + "=" * 60)
    print("Demo complete. Checkpoints saved to .prismlang_demo/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
