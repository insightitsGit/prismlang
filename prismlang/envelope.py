"""PrismEnvelope — the atomic unit transmitted over the PrismLang wire.

Each envelope represents one agent turn. The vector is the sole payload
that crosses agent boundaries; rule_chain provides the full audit trace.
"""

from __future__ import annotations

from typing import List
from typing_extensions import TypedDict


class PrismEnvelope(TypedDict):
    turn_id: int           # chronological index within the thread
    agent_id: str          # node/agent that generated this envelope
    category_slug: str     # taxonomy category inferred for this turn
    vector: List[float]    # k-dimensional PrismLang payload (JSON-serialisable)
    rule_chain: List[str]  # immutable audit trail: step-by-step projection log
