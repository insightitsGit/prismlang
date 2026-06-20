"""Finance benchmark domain.

Three-agent LangGraph pipeline:
  risk_agent       → reads positions & risk events from DB, computes exposure
  portfolio_agent  → analyses P&L, suggests rebalancing actions
  compliance_agent → checks regulatory limits, KYC, reporting requirements

Runs twice: standard text state vs PrismLang compressed vector state.
"""

from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

from prismlang import (
    Category, PrismProjector, PrismState, TaxonomyConfig, prism_node,
)
from benchmarks import BENCH_DSN
from benchmarks.metrics import BenchmarkResult, MetricsCollector
from benchmarks import gemini_client

# ------------------------------------------------------------------ #
# Taxonomy                                                             #
# ------------------------------------------------------------------ #

FINANCE_TAXONOMY = TaxonomyConfig(
    categories=[
        Category("risk", "Risk Management", [
            "risk", "var", "exposure", "breach", "margin", "volatility",
            "drawdown", "stress", "default", "credit", "counterparty", "hedging",
        ]),
        Category("portfolio", "Portfolio & P&L", [
            "portfolio", "position", "pnl", "return", "allocation", "rebalance",
            "equity", "bond", "yield", "price", "asset", "valuation", "sharpe",
        ]),
        Category("compliance", "Regulatory Compliance", [
            "compliance", "regulatory", "kyc", "aml", "report", "disclosure",
            "limit", "rule", "requirement", "audit", "sec", "finra", "mandate",
        ]),
    ],
    alpha=0.3,
)

# ------------------------------------------------------------------ #
# DB helpers                                                           #
# ------------------------------------------------------------------ #

def _fetch_account_context(account_id: str) -> dict:
    conn = psycopg2.connect(BENCH_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT client_name, account_type, aum_usd, risk_profile, manager, kyc_status "
            "FROM finance.accounts WHERE account_id=%s", (account_id,)
        )
        row = cur.fetchone()
        account = {
            "account_id": account_id,
            "client": row[0], "type": row[1], "aum_usd": row[2],
            "risk_profile": row[3], "manager": row[4], "kyc_status": row[5],
        }
        cur.execute(
            "SELECT ticker, asset_class, quantity, avg_cost, current_price, unrealised_pnl, currency "
            "FROM finance.positions WHERE account_id=%s", (account_id,)
        )
        account["positions"] = [
            {"ticker": r[0], "asset_class": r[1], "qty": r[2],
             "avg_cost": r[3], "price": r[4], "pnl": r[5], "ccy": r[6]}
            for r in cur.fetchall()
        ]
        cur.execute(
            "SELECT event_type, severity, description, triggered_at "
            "FROM finance.risk_events WHERE account_id=%s ORDER BY triggered_at DESC", (account_id,)
        )
        account["risk_events"] = [
            {"type": r[0], "severity": r[1], "desc": r[2], "at": str(r[3])}
            for r in cur.fetchall()
        ]
        return account
    finally:
        conn.close()

# ------------------------------------------------------------------ #
# Fallback responses                                                   #
# ------------------------------------------------------------------ #

_RISK_FALLBACK = (
    "RISK ASSESSMENT — APEX MACRO HEDGE FUND (ACC-F003)\n"
    "CRITICAL ALERTS: (1) VaR breach: 1-day 99% VaR at $22.4M vs $18M limit. (2) Margin call: $8.7M "
    "due by COB. (3) SPY short position down $2.5M (100K shares shorted at $520, now $545). "
    "TLT long position down $1.45M (yield pressure). EURUSD position +$270K (small offset). "
    "Total unrealised loss: approx -$3.68M. Risk score: 9.2/10 — CRITICAL. Immediate action required."
)

_PORTFOLIO_FALLBACK = (
    "PORTFOLIO ANALYSIS — ACC-F003\n"
    "Current positioning is high-risk and directionally exposed to rate and equity risk simultaneously. "
    "SPY short ($100K shares) is losing on equity rally. TLT long ($500K shares) is losing on rate rise. "
    "These are correlated losses — both driven by risk-on environment. "
    "RECOMMENDATION: (1) Cover 50% of SPY short immediately ($25M notional). "
    "(2) Reduce TLT by 200K shares to cut bond duration exposure. "
    "(3) Add rate hedges via Eurodollar futures. (4) Rebalance EURUSD to 5M EUR. "
    "Post-rebalance VaR estimate: $14.2M (within $18M limit). Sharpe ratio improves from -0.8 to +0.4."
)

_COMPLIANCE_FALLBACK = (
    "COMPLIANCE REVIEW — ACC-F003\n"
    "MARGIN CALL: $8.7M margin call from Goldwick Prime must be disclosed to compliance officer within 1 hour. "
    "SEC Rule 15c3-1: net capital requirement review triggered by VaR breach. "
    "AML: No suspicious activity flagged in recent transactions. KYC: current and verified. "
    "POSITION LIMITS: SPY short currently at 12.1% of NAV; mandate allows max 15% single-name short. "
    "TLT position at 38.5% NAV — approaching 40% fixed income concentration limit. "
    "REPORTING: VaR breach must be reported to prime broker risk committee within 24h. "
    "Rebalancing trades require pre-trade compliance sign-off. ALL CLEAR except margin call disclosure."
)

# ------------------------------------------------------------------ #
# Standard LangGraph                                                   #
# ------------------------------------------------------------------ #

class FinStandardState(TypedDict):
    messages: List[str]
    account_context: dict

def run_standard(account_context: dict, mc: MetricsCollector) -> None:
    def risk(state: FinStandardState) -> dict:
        ctx = json.dumps(state["account_context"], indent=2)
        prompt = f"You are a risk management AI for a hedge fund. Assess risk exposure:\n{ctx}"
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _RISK_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[risk]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(0, "risk_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    def portfolio(state: FinStandardState) -> dict:
        history = "\n".join(state["messages"])
        prompt = f"You are a portfolio management AI. Risk assessment:\n{history}\nRecommend rebalancing."
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _PORTFOLIO_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[portfolio]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(1, "portfolio_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    def compliance(state: FinStandardState) -> dict:
        history = "\n".join(state["messages"])
        prompt = f"You are a financial compliance AI. Review:\n{history}\nCheck regulations and limits."
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _COMPLIANCE_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[compliance]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(2, "compliance_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    g = StateGraph(FinStandardState)
    g.add_node("risk", risk)
    g.add_node("portfolio", portfolio)
    g.add_node("compliance", compliance)
    g.set_entry_point("risk")
    g.add_edge("risk", "portfolio")
    g.add_edge("portfolio", "compliance")
    g.add_edge("compliance", END)
    g.compile().invoke({"messages": [], "account_context": account_context})

# ------------------------------------------------------------------ #
# PrismLang LangGraph                                                  #
# ------------------------------------------------------------------ #

def run_prismlang(account_context: dict, projector: PrismProjector, mc: MetricsCollector) -> list:
    K = projector.k
    BYTES_PER_ENV = K * 4 + 64
    ctx_str = json.dumps(account_context, indent=2)

    def _make_node(agent_id, prompt_fn, fallback, turn_id):
        @prism_node(agent_id=agent_id, projector=projector)
        def node(state: PrismState) -> dict:
            prompt = prompt_fn(state)
            mc.start_llm()
            text, pt, ot, _ = gemini_client.call(prompt, fallback)
            llm_ms = mc.end_llm(pt, ot)
            mc.start_encode()
            result = {"raw_output": text}
            encode_ms = mc.end_encode()
            n_envs = turn_id + 1
            vec_bytes = n_envs * BYTES_PER_ENV
            state_bytes = len(json.dumps({"prism_sequence": state.get("prism_sequence", [])}).encode())
            mc.record_turn(turn_id, agent_id, state_bytes, vec_bytes, llm_ms=llm_ms, encode_ms=encode_ms, prompt_tokens=pt, output_tokens=ot)
            return result
        return node

    risk_node = _make_node(
        "risk_agent",
        lambda s: f"You are a risk AI. Assess exposure for:\n{ctx_str}",
        _RISK_FALLBACK, 0,
    )
    portfolio_node = _make_node(
        "portfolio_agent",
        lambda s: f"You are a portfolio AI. Risk category: {s['prism_sequence'][-1]['category_slug'] if s.get('prism_sequence') else 'unknown'}. Recommend rebalancing.",
        _PORTFOLIO_FALLBACK, 1,
    )
    compliance_node = _make_node(
        "compliance_agent",
        lambda s: f"You are a compliance AI. Category flow: {[e['category_slug'] for e in s.get('prism_sequence', [])]}. Review regulatory requirements.",
        _COMPLIANCE_FALLBACK, 2,
    )

    g = StateGraph(PrismState)
    g.add_node("risk", risk_node)
    g.add_node("portfolio", portfolio_node)
    g.add_node("compliance", compliance_node)
    g.set_entry_point("risk")
    g.add_edge("risk", "portfolio")
    g.add_edge("portfolio", "compliance")
    g.add_edge("compliance", END)
    final = g.compile().invoke({"prism_sequence": [], "raw_output": "", "tenant_id": projector.tenant_id})
    return [e["category_slug"] for e in final["prism_sequence"]]

# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #

TENANT_ID = "apex-macro-hedge-fund"

def benchmark() -> tuple:
    account_context = _fetch_account_context("ACC-F003")  # Hedge fund with active risk events
    projector = PrismProjector(FINANCE_TAXONOMY, tenant_id=TENANT_ID, k=64)

    with MetricsCollector() as mc_std:
        run_standard(account_context, mc_std)
    std_result = mc_std.snapshot("finance", "standard", "gemini-2.0-flash", TENANT_ID)

    with MetricsCollector() as mc_prism:
        category_flow = run_prismlang(account_context, projector, mc_prism)
    prism_result = mc_prism.snapshot(
        "finance", "prismlang", "gemini-2.0-flash", TENANT_ID,
        category_flow=category_flow,
        tenant_matrix_fp=projector.matrix_fingerprint(),
    )

    return std_result, prism_result
