"""Tests for PrismState, PrismEnvelope, and the middleware decorator."""

import operator
import pytest
import numpy as np

from prismlang import (
    Category,
    PrismEnvelope,
    PrismProjector,
    PrismState,
    TaxonomyConfig,
    prism_node,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture()
def taxonomy():
    return TaxonomyConfig(
        categories=[
            Category("risk", "Risk", ["risk", "exposure"]),
            Category("market", "Market", ["market", "price"]),
        ],
        alpha=0.3,
    )


@pytest.fixture()
def mock_encoder(monkeypatch):
    import prismlang.encoder as enc_mod

    def _encode(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % 2**32)
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)

    monkeypatch.setattr(enc_mod, "encode", _encode)
    monkeypatch.setattr(enc_mod, "encode_batch", lambda ts: np.stack([_encode(t) for t in ts]))


@pytest.fixture()
def projector(taxonomy, mock_encoder):
    return PrismProjector(taxonomy=taxonomy, tenant_id="state-test-tenant", k=32)


def _empty_state(tenant_id: str = "test") -> PrismState:
    return {"prism_sequence": [], "raw_output": "", "tenant_id": tenant_id}


# ------------------------------------------------------------------ #
# PrismEnvelope structure                                              #
# ------------------------------------------------------------------ #

def test_envelope_fields():
    env = PrismEnvelope(
        turn_id=0,
        agent_id="agent-x",
        category_slug="risk",
        vector=[0.1, 0.2, 0.3],
        rule_chain=["step1", "step2"],
    )
    assert env["turn_id"] == 0
    assert env["agent_id"] == "agent-x"
    assert env["category_slug"] == "risk"
    assert env["vector"] == [0.1, 0.2, 0.3]
    assert env["rule_chain"] == ["step1", "step2"]


# ------------------------------------------------------------------ #
# prism_sequence uses operator.add (append, not overwrite)            #
# ------------------------------------------------------------------ #

def test_sequence_appends_correctly():
    env_a = PrismEnvelope(turn_id=0, agent_id="a", category_slug="risk",
                          vector=[1.0], rule_chain=[])
    env_b = PrismEnvelope(turn_id=1, agent_id="b", category_slug="market",
                          vector=[2.0], rule_chain=[])
    seq = operator.add([env_a], [env_b])
    assert len(seq) == 2
    assert seq[0]["turn_id"] == 0
    assert seq[1]["turn_id"] == 1


# ------------------------------------------------------------------ #
# @prism_node decorator                                                #
# ------------------------------------------------------------------ #

def test_prism_node_appends_envelope(projector):
    @prism_node(agent_id="researcher", projector=projector)
    def researcher(state: PrismState) -> dict:
        return {"raw_output": "market price equity analysis"}

    state = _empty_state()
    result = researcher(state)

    assert "prism_sequence" in result
    assert len(result["prism_sequence"]) == 1
    env = result["prism_sequence"][0]
    assert env["agent_id"] == "researcher"
    assert env["turn_id"] == 0
    assert isinstance(env["vector"], list)
    assert len(env["vector"]) == 32  # k=32
    assert len(env["rule_chain"]) == 4


def test_prism_node_preserves_raw_output(projector):
    @prism_node(agent_id="x", projector=projector)
    def node(state):
        return {"raw_output": "some text here"}

    result = node(_empty_state())
    assert result["raw_output"] == "some text here"


def test_prism_node_turn_id_increments(projector):
    @prism_node(agent_id="a", projector=projector)
    def node_a(state):
        return {"raw_output": "risk analysis text"}

    @prism_node(agent_id="b", projector=projector)
    def node_b(state):
        return {"raw_output": "market price text"}

    state = _empty_state()
    r1 = node_a(state)
    state = {**state, "prism_sequence": state["prism_sequence"] + r1["prism_sequence"]}
    r2 = node_b(state)

    assert r1["prism_sequence"][0]["turn_id"] == 0
    assert r2["prism_sequence"][0]["turn_id"] == 1


def test_prism_node_empty_output_handled(projector):
    @prism_node(agent_id="silent", projector=projector)
    def node(state):
        return {"raw_output": ""}

    result = node(_empty_state())
    assert len(result["prism_sequence"]) == 1


def test_prism_node_preserves_extra_keys(projector):
    @prism_node(agent_id="custom", projector=projector)
    def node(state):
        return {"raw_output": "text", "custom_field": 42}

    result = node(_empty_state())
    assert result["custom_field"] == 42
