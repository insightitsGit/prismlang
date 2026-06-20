"""PrismLang middleware decorator for LangGraph nodes.

Usage
-----
    projector = PrismProjector(taxonomy, tenant_id="acme-finance")

    @prism_node(agent_id="researcher", projector=projector)
    def researcher(state: PrismState) -> dict:
        # ... call LLM, do work ...
        return {"raw_output": "My analysis of the market risk is ..."}

The decorator intercepts the node's return value, encodes the raw_output
text into a PrismEnvelope, and returns both the text (for downstream nodes
to read in-memory) and the new envelope (appended to prism_sequence).

Agents require zero refactoring — they only need to return {"raw_output": "..."}.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Callable

from .envelope import PrismEnvelope
from .exceptions import MissingTenantError
from .projector import PrismProjector
from .state import PrismState


def prism_node(agent_id: str, projector: PrismProjector) -> Callable:
    """Decorator factory that wraps a LangGraph node with PrismLang encoding.

    Args:
        agent_id:  Unique identifier for this node, stored in each envelope.
        projector: PrismProjector bound to the current tenant.

    Returns:
        A decorator that wraps the node function.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: PrismState) -> dict:
            if not state.get("tenant_id"):
                raise MissingTenantError()

            result = fn(state)

            text: str = result.get("raw_output", "")

            # Empty output: use a sentinel so projection always succeeds and the
            # sequence stays contiguous and auditable.
            slug, vector, rule_chain = projector.project(text if text.strip() else "[no output]")

            envelope = PrismEnvelope(
                turn_id=len(state.get("prism_sequence", [])),
                agent_id=agent_id,
                category_slug=slug,
                vector=vector.tolist(),
                rule_chain=rule_chain,
            )

            # Merge with whatever else the node returned (e.g. custom fields)
            return {**result, "prism_sequence": [envelope]}

        return wrapper
    return decorator


def async_prism_node(agent_id: str, projector: PrismProjector) -> Callable:
    """Async variant of prism_node for use with async LangGraph nodes.

    The wrapped node must be an async function returning ``{"raw_output": "..."}``.
    ONNX inference runs in a thread pool so the event loop is never blocked.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(state: PrismState) -> dict:
            if not state.get("tenant_id"):
                raise MissingTenantError()

            result = await fn(state)

            text: str = result.get("raw_output", "")
            loop = asyncio.get_event_loop()
            _text = text if text.strip() else "[no output]"
            slug, vector, rule_chain = await loop.run_in_executor(
                None, projector.project, _text
            )

            envelope = PrismEnvelope(
                turn_id=len(state.get("prism_sequence", [])),
                agent_id=agent_id,
                category_slug=slug,
                vector=vector.tolist(),
                rule_chain=rule_chain,
            )

            return {**result, "prism_sequence": [envelope]}

        return wrapper
    return decorator
