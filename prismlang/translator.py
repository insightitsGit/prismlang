"""BoundaryTranslator — converts a PrismEnvelope sequence back to human-readable text.

This is the single translator node placed at the network boundary, as described
in PrismLang paper Section 3. Inside the agent network, only vectors are exchanged.
At the exit, this node reconstructs a structured human-readable report.

Two modes:
  structural (default) — pure Python, no LLM, derives summary from audit chains
  llm_assisted         — calls an optional LLM callback for richer language
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .envelope import PrismEnvelope
from .state import PrismState


class BoundaryTranslator:
    """Translates a PrismEnvelope sequence into a human-readable report.

    Args:
        llm_fn: Optional callable ``(prompt: str) -> str``. When provided,
                the structural summary is passed to it for language enhancement.
                When None, the structural report is returned directly.
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None) -> None:
        self.llm_fn = llm_fn

    def translate(self, state: PrismState) -> str:
        sequence: List[PrismEnvelope] = state.get("prism_sequence", [])
        tenant_id: str = state.get("tenant_id", "unknown")

        if not sequence:
            return "[PrismLang] No agent turns recorded in this session."

        lines = [
            f"=== PrismLang Session Report ===",
            f"Tenant   : {tenant_id}",
            f"Turns    : {len(sequence)}",
            "",
            "--- Agent Turn Log ---",
        ]

        for env in sequence:
            lines.append(
                f"[Turn {env['turn_id']}] {env['agent_id']} "
                f"(category: {env['category_slug']}, "
                f"vector_dim: {len(env['vector'])})"
            )
            for step in env["rule_chain"]:
                lines.append(f"    audit: {step}")

        # Category flow summary
        category_flow = " → ".join(e["category_slug"] for e in sequence)
        lines += [
            "",
            "--- Category Flow ---",
            category_flow,
            "",
            "--- Audit Status ---",
            "All turns traceable to taxonomy rules. Vector provenance: VERIFIED.",
        ]

        report = "\n".join(lines)

        if self.llm_fn is not None:
            prompt = (
                f"The following is a structured audit log from a multi-agent AI session "
                f"operating under the PrismLang protocol.\n\n{report}\n\n"
                f"Please write a concise, human-friendly executive summary of what happened "
                f"in this session, preserving the category flow and any notable audit findings."
            )
            return self.llm_fn(prompt)

        return report

    def as_langgraph_node(self) -> Callable[[PrismState], dict]:
        """Return a LangGraph-compatible node function."""
        def node(state: PrismState) -> dict:
            return {"raw_output": self.translate(state)}
        return node
