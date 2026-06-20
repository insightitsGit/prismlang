"""Trade Market benchmark domain.

Three-agent LangGraph pipeline:
  signal_agent   → reads OHLCV + signals from DB, identifies trade opportunities
  execution_agent → analyses order book, determines entry/exit levels
  risk_agent      → position sizing, stop-loss, portfolio impact

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
from benchmarks.metrics import MetricsCollector
from benchmarks import gemini_client

# ------------------------------------------------------------------ #
# Taxonomy                                                             #
# ------------------------------------------------------------------ #

TRADE_TAXONOMY = TaxonomyConfig(
    categories=[
        Category("signal", "Trade Signal", [
            "signal", "momentum", "breakout", "trend", "rsi", "macd",
            "volume", "sentiment", "indicator", "bullish", "bearish",
            "resistance", "support", "crossover", "divergence",
        ]),
        Category("execution", "Trade Execution", [
            "order", "bid", "ask", "spread", "entry", "exit", "fill",
            "slippage", "execution", "market", "limit", "stop", "vwap",
            "liquidity", "depth", "book", "price", "level",
        ]),
        Category("risk", "Position Risk", [
            "risk", "position", "sizing", "stop_loss", "drawdown", "exposure",
            "var", "portfolio", "hedge", "correlation", "beta", "volatility",
            "pnl", "loss", "allocation", "concentration",
        ]),
    ],
    alpha=0.3,
)

# ------------------------------------------------------------------ #
# DB helpers                                                           #
# ------------------------------------------------------------------ #

def _fetch_market_context(symbol: str) -> dict:
    conn = psycopg2.connect(BENCH_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, asset_class, exchange, sector FROM trade_market.instruments WHERE symbol=%s",
            (symbol,)
        )
        row = cur.fetchone()
        ctx = {"symbol": symbol, "name": row[0], "asset_class": row[1],
               "exchange": row[2], "sector": row[3]}

        cur.execute(
            "SELECT trade_date, open_price, high_price, low_price, close_price, volume, vwap "
            "FROM trade_market.ohlcv WHERE symbol=%s ORDER BY trade_date DESC LIMIT 5",
            (symbol,)
        )
        ctx["ohlcv"] = [
            {"date": str(r[0]), "open": r[1], "high": r[2], "low": r[3],
             "close": r[4], "volume": r[5], "vwap": r[6]}
            for r in cur.fetchall()
        ]

        cur.execute(
            "SELECT bid_price, ask_price, bid_size, ask_size, spread_bps "
            "FROM trade_market.order_book_snapshots WHERE symbol=%s ORDER BY snapshot_ts DESC LIMIT 1",
            (symbol,)
        )
        ob = cur.fetchone()
        if ob:
            ctx["order_book"] = {"bid": ob[0], "ask": ob[1], "bid_size": ob[2],
                                  "ask_size": ob[3], "spread_bps": ob[4]}

        cur.execute(
            "SELECT signal_type, direction, confidence, description "
            "FROM trade_market.signals WHERE symbol=%s ORDER BY signal_date DESC LIMIT 1",
            (symbol,)
        )
        sig = cur.fetchone()
        if sig:
            ctx["signal"] = {"type": sig[0], "direction": sig[1],
                              "confidence": sig[2], "desc": sig[3]}
        return ctx
    finally:
        conn.close()

# ------------------------------------------------------------------ #
# Fallback responses                                                   #
# ------------------------------------------------------------------ #

_SIGNAL_FALLBACK = (
    "TRADE SIGNAL ANALYSIS — NVDA\n"
    "MOMENTUM SIGNAL CONFIRMED: NVDA breaking 5-day resistance at $151 with volume 28% above 20-day average. "
    "5-day price action: $143.90 -> $147.80 -> $151.20 -> $149.50 -> $154.90 (+7.6% move). "
    "RSI(14) = 65: elevated but not overbought. MACD positive crossover 2 sessions ago. "
    "Post-earnings catalyst still in play (beat by $0.18 EPS). "
    "Signal: LONG with confidence 0.82. Target zone: $163-$168. "
    "Volume confirmation: 67M shares on breakout day vs 52M average. STRONG SIGNAL."
)

_EXECUTION_FALLBACK = (
    "EXECUTION ANALYSIS — NVDA\n"
    "ORDER BOOK (14:30 UTC): Bid $154.85 x 12,500 / Ask $154.95 x 9,800. Spread: 6.5bps (tight). "
    "VWAP: $152.80 — current price $154.90 is 1.4% above VWAP, suggesting some intraday premium. "
    "ENTRY RECOMMENDATION: Use VWAP-based limit order at $154.50-$154.75 to avoid chasing. "
    "If momentum continues, market order acceptable up to $155.20 (within 20bps of ask). "
    "TARGET EXIT: $163.00 primary, $168.00 stretch target. "
    "STOP LOSS: $148.00 (below last support). Risk/Reward: 1:1.9 at entry $154.75. "
    "Suggested order: Limit buy $154.75, GTC. Tranche in 2 fills to reduce market impact."
)

_RISK_FALLBACK = (
    "POSITION RISK ANALYSIS — NVDA\n"
    "POSITION SIZING: For a $10M portfolio with 2% risk per trade (max loss $200K): "
    "Stop at $148.00 from entry $154.75 = $6.75 risk per share. Max shares: $200K / $6.75 = 29,629 shares. "
    "Notional at entry: 29,629 x $154.75 = $4.58M (45.8% of portfolio — EXCEEDS 20% single-name limit). "
    "ADJUSTED SIZE: 12,900 shares ($1.99M, 19.9% of portfolio — within limits). "
    "PORTFOLIO IMPACT: NVDA beta = 1.82 vs S&P. Adding 12,900 shares increases portfolio beta from 1.05 to 1.19. "
    "CORRELATION: NVDA correlates 0.71 with QQQ — check existing tech exposure before entering. "
    "FINAL RECOMMENDATION: Buy 12,900 NVDA limit $154.75 with hard stop $148.00. Monitor beta daily."
)

# ------------------------------------------------------------------ #
# Standard LangGraph                                                   #
# ------------------------------------------------------------------ #

class TMStandardState(TypedDict):
    messages: List[str]
    market_context: dict

def run_standard(market_context: dict, mc: MetricsCollector) -> None:
    def signal(state: TMStandardState) -> dict:
        ctx = json.dumps(state["market_context"], indent=2)
        prompt = f"You are a trade signal AI. Analyse this market data and identify opportunities:\n{ctx}"
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _SIGNAL_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[signal]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(0, "signal_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    def execution(state: TMStandardState) -> dict:
        history = "\n".join(state["messages"])
        prompt = f"You are a trade execution AI. Prior analysis:\n{history}\nDetermine optimal entry and exit levels."
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _EXECUTION_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[execution]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(1, "execution_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    def risk(state: TMStandardState) -> dict:
        history = "\n".join(state["messages"])
        prompt = f"You are a position risk AI. Execution plan:\n{history}\nCalculate position size and risk metrics."
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _RISK_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[risk]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(2, "risk_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    g = StateGraph(TMStandardState)
    g.add_node("signal", signal)
    g.add_node("execution", execution)
    g.add_node("risk", risk)
    g.set_entry_point("signal")
    g.add_edge("signal", "execution")
    g.add_edge("execution", "risk")
    g.add_edge("risk", END)
    g.compile().invoke({"messages": [], "market_context": market_context})

# ------------------------------------------------------------------ #
# PrismLang LangGraph                                                  #
# ------------------------------------------------------------------ #

def run_prismlang(market_context: dict, projector: PrismProjector, mc: MetricsCollector) -> list:
    K = projector.k
    BYTES_PER_ENV = K * 4 + 64
    ctx_str = json.dumps(market_context, indent=2)

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

    signal_node = _make_node(
        "signal_agent",
        lambda s: f"You are a signal AI. Analyse:\n{ctx_str}",
        _SIGNAL_FALLBACK, 0,
    )
    execution_node = _make_node(
        "execution_agent",
        lambda s: f"You are an execution AI. Signal category: {s['prism_sequence'][-1]['category_slug'] if s.get('prism_sequence') else 'unknown'}. Determine entry/exit levels.",
        _EXECUTION_FALLBACK, 1,
    )
    risk_node = _make_node(
        "risk_agent",
        lambda s: f"You are a risk AI. Category flow: {[e['category_slug'] for e in s.get('prism_sequence', [])]}. Calculate position size and risk.",
        _RISK_FALLBACK, 2,
    )

    g = StateGraph(PrismState)
    g.add_node("signal", signal_node)
    g.add_node("execution", execution_node)
    g.add_node("risk", risk_node)
    g.set_entry_point("signal")
    g.add_edge("signal", "execution")
    g.add_edge("execution", "risk")
    g.add_edge("risk", END)
    final = g.compile().invoke({"prism_sequence": [], "raw_output": "", "tenant_id": projector.tenant_id})
    return [e["category_slug"] for e in final["prism_sequence"]]

# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #

TENANT_ID = "ironwood-quant-trading"

def benchmark() -> tuple:
    market_context = _fetch_market_context("NVDA")
    projector = PrismProjector(TRADE_TAXONOMY, tenant_id=TENANT_ID, k=64)

    with MetricsCollector() as mc_std:
        run_standard(market_context, mc_std)
    std_result = mc_std.snapshot("trade_market", "standard", "gemini-2.0-flash", TENANT_ID)

    with MetricsCollector() as mc_prism:
        category_flow = run_prismlang(market_context, projector, mc_prism)
    prism_result = mc_prism.snapshot(
        "trade_market", "prismlang", "gemini-2.0-flash", TENANT_ID,
        category_flow=category_flow,
        tenant_matrix_fp=projector.matrix_fingerprint(),
    )

    return std_result, prism_result
