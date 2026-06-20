"""PrismState — the LangGraph state schema for PrismLang graphs.

prism_sequence uses operator.add as its reducer, which natively appends
each new PrismEnvelope to the chronological sequence without overwriting.

raw_output is the last agent's text output kept in-memory for intra-graph
readability. It is NOT transmitted between agents at the wire level —
only the encoded PrismEnvelopes cross agent boundaries.
"""

from __future__ import annotations

import operator
from typing import Annotated, List

from typing_extensions import TypedDict

from .envelope import PrismEnvelope


class PrismState(TypedDict):
    # Ordered sequence of compressed agent outputs — the PrismLang wire payload
    prism_sequence: Annotated[List[PrismEnvelope], operator.add]

    # In-memory text for nodes that need to read the previous agent's output
    raw_output: str

    # Tenant identifier — controls which JL matrix is active for this thread
    tenant_id: str
