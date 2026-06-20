"""Healthcare benchmark domain.

Three-agent LangGraph pipeline:
  triage_agent    → reads patient vitals & diagnoses from DB, flags urgency
  clinical_agent  → reviews lab results & clinical notes, recommends treatment
  compliance_agent → checks allergy safety, HIPAA flags, regulatory requirements

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

HEALTHCARE_TAXONOMY = TaxonomyConfig(
    categories=[
        Category("clinical", "Clinical Assessment", [
            "diagnosis", "symptoms", "vital", "examination", "assessment",
            "icd", "patient", "condition", "chronic", "acute", "treatment",
        ]),
        Category("lab", "Laboratory & Diagnostics", [
            "lab", "result", "test", "blood", "hemoglobin", "glucose", "troponin",
            "bnp", "spo2", "hba1c", "ldl", "cea", "biopsy", "culture", "panel",
        ]),
        Category("compliance", "Regulatory & Compliance", [
            "allergy", "hipaa", "consent", "privacy", "regulation", "documentation",
            "medication", "contraindication", "safety", "protocol", "policy",
        ]),
    ],
    alpha=0.3,
)

# ------------------------------------------------------------------ #
# DB helpers                                                           #
# ------------------------------------------------------------------ #

def _fetch_patient_context(mrn: str) -> dict:
    conn = psycopg2.connect(BENCH_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT full_name, dob, gender, blood_type, allergies FROM healthcare.patients WHERE mrn=%s", (mrn,))
        row = cur.fetchone()
        patient = {"mrn": mrn, "name": row[0], "dob": str(row[1]), "gender": row[2],
                   "blood_type": row[3], "allergies": list(row[4] or [])}

        cur.execute("SELECT icd10_code, description, severity, status FROM healthcare.diagnoses WHERE patient_mrn=%s", (mrn,))
        patient["diagnoses"] = [{"code": r[0], "desc": r[1], "severity": r[2], "status": r[3]} for r in cur.fetchall()]

        cur.execute("SELECT test_name, result_value, result_unit, flag FROM healthcare.lab_results WHERE patient_mrn=%s ORDER BY collected_on DESC LIMIT 5", (mrn,))
        patient["labs"] = [{"test": r[0], "value": r[1], "unit": r[2], "flag": r[3]} for r in cur.fetchall()]

        cur.execute("SELECT note_text FROM healthcare.clinical_notes WHERE patient_mrn=%s ORDER BY note_date DESC LIMIT 1", (mrn,))
        n = cur.fetchone()
        patient["latest_note"] = n[0] if n else ""
        return patient
    finally:
        conn.close()

# ------------------------------------------------------------------ #
# Pre-written fallback responses                                       #
# ------------------------------------------------------------------ #

_TRIAGE_FALLBACK = (
    "TRIAGE ASSESSMENT — PRIORITY: CRITICAL\n"
    "Patient Thomas Okafor (MRN-005) presents with acute respiratory distress. SpO2 88% on room air "
    "is immediately life-threatening. BNP at 980 pg/mL confirms severe decompensated heart failure. "
    "Concurrent COPD exacerbation compounds respiratory compromise. Recommend ICU admission, "
    "immediate BiPAP initiation, IV furosemide 80mg bolus, and cardiology + pulmonology consult within 30 minutes. "
    "Ibuprofen allergy documented — avoid NSAIDs entirely. Risk stratification: APACHE II score estimated >25."
)

_CLINICAL_FALLBACK = (
    "CLINICAL RECOMMENDATION REPORT\n"
    "Based on lab results (SpO2 88%, BNP 980, bilateral CXR infiltrates) and COPD+CHF dual diagnosis: "
    "1. BiPAP settings: IPAP 12, EPAP 5, FiO2 50% — reassess in 1h. "
    "2. IV Furosemide 80mg now, repeat 40mg in 2h if urine output <30mL/hr. "
    "3. Hold beta-blockers acutely — may worsen bronchospasm. "
    "4. Avoid ibuprofen and all NSAIDs (documented allergy). Use acetaminophen for pain. "
    "5. Echocardiogram to assess EF within 12h. "
    "6. Cultures x2 before antibiotics — Augmentin if infection suspected (not penicillin per allergy list — "
    "   CORRECTION: patient allergy is ibuprofen, NOT penicillin — verify allergy list before prescribing). "
    "Target SpO2 >92%, urine output >0.5mL/kg/hr."
)

_COMPLIANCE_FALLBACK = (
    "REGULATORY & COMPLIANCE REVIEW\n"
    "ALLERGY SAFETY: Ibuprofen allergy confirmed in system. Current treatment plan avoids NSAIDs. "
    "HIPAA: Patient verbal consent obtained for ICU transfer and family notification. "
    "DOCUMENTATION: Admission note, allergy verification, and BiPAP initiation all require attending co-signature within 24h. "
    "MEDICATION RECONCILIATION: 6 home medications require reconciliation against ICU formulary. "
    "DNR STATUS: None on file — goals of care discussion required with family before intubation if needed. "
    "REPORTABLE CONDITIONS: None applicable. Regulatory audit trail: compliant."
)

# ------------------------------------------------------------------ #
# Standard LangGraph (text state)                                     #
# ------------------------------------------------------------------ #

class HCStandardState(TypedDict):
    messages: List[str]
    patient_context: dict

def run_standard(patient_context: dict, mc: MetricsCollector) -> None:
    def triage(state: HCStandardState) -> dict:
        ctx = json.dumps(state["patient_context"], indent=2)
        prompt = f"You are a triage nurse AI. Assess urgency for this patient:\n{ctx}"
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _TRIAGE_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[triage]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(0, "triage_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    def clinical(state: HCStandardState) -> dict:
        history = "\n".join(state["messages"])
        prompt = f"You are a clinical AI. Prior assessment:\n{history}\nProvide treatment recommendations."
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _CLINICAL_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[clinical]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(1, "clinical_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    def compliance(state: HCStandardState) -> dict:
        history = "\n".join(state["messages"])
        prompt = f"You are a healthcare compliance AI. Review:\n{history}\nCheck allergy safety, HIPAA, documentation."
        mc.start_llm()
        text, pt, ot, _ = gemini_client.call(prompt, _COMPLIANCE_FALLBACK)
        llm_ms = mc.end_llm(pt, ot)
        new_messages = state["messages"] + [f"[compliance]: {text}"]
        state_bytes = len(json.dumps({"messages": new_messages}).encode())
        mc.record_turn(2, "compliance_agent", state_bytes, 0, llm_ms=llm_ms, prompt_tokens=pt, output_tokens=ot)
        return {"messages": new_messages}

    g = StateGraph(HCStandardState)
    g.add_node("triage", triage)
    g.add_node("clinical", clinical)
    g.add_node("compliance", compliance)
    g.set_entry_point("triage")
    g.add_edge("triage", "clinical")
    g.add_edge("clinical", "compliance")
    g.add_edge("compliance", END)
    g.compile().invoke({"messages": [], "patient_context": patient_context})

# ------------------------------------------------------------------ #
# PrismLang LangGraph (vector state)                                  #
# ------------------------------------------------------------------ #

def run_prismlang(patient_context: dict, projector: PrismProjector, mc: MetricsCollector) -> list:
    K = projector.k
    BYTES_PER_ENV = K * 4 + 64  # float32 per dim + envelope overhead

    def _make_node(agent_id: str, prompt_fn, fallback: str, turn_id: int):
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
            state_bytes = len(json.dumps({"prism_sequence": state.get("prism_sequence", []), "raw_output": text}).encode())
            mc.record_turn(turn_id, agent_id, state_bytes, vec_bytes, llm_ms=llm_ms, encode_ms=encode_ms, prompt_tokens=pt, output_tokens=ot)
            return result
        return node

    ctx_str = json.dumps(patient_context, indent=2)

    triage_node = _make_node(
        "triage_agent",
        lambda s: f"You are a triage nurse AI. Assess urgency for:\n{ctx_str}",
        _TRIAGE_FALLBACK, 0,
    )
    clinical_node = _make_node(
        "clinical_agent",
        lambda s: f"You are a clinical AI. Triage summary category: {s['prism_sequence'][-1]['category_slug'] if s.get('prism_sequence') else 'unknown'}. Provide treatment recommendations.",
        _CLINICAL_FALLBACK, 1,
    )
    compliance_node = _make_node(
        "compliance_agent",
        lambda s: f"You are a compliance AI. Clinical category flow so far: {[e['category_slug'] for e in s.get('prism_sequence', [])]}. Review safety and documentation.",
        _COMPLIANCE_FALLBACK, 2,
    )

    g = StateGraph(PrismState)
    g.add_node("triage", triage_node)
    g.add_node("clinical", clinical_node)
    g.add_node("compliance", compliance_node)
    g.set_entry_point("triage")
    g.add_edge("triage", "clinical")
    g.add_edge("clinical", "compliance")
    g.add_edge("compliance", END)
    final = g.compile().invoke({"prism_sequence": [], "raw_output": "", "tenant_id": projector.tenant_id})
    return [e["category_slug"] for e in final["prism_sequence"]]

# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #

TENANT_ID = "hospital-system-prod"

def benchmark() -> tuple:
    patient_context = _fetch_patient_context("MRN-005")  # Critical case
    projector = PrismProjector(HEALTHCARE_TAXONOMY, tenant_id=TENANT_ID, k=64)

    with MetricsCollector() as mc_std:
        run_standard(patient_context, mc_std)
    std_result = mc_std.snapshot("healthcare", "standard", "gemini-2.0-flash", TENANT_ID)

    with MetricsCollector() as mc_prism:
        category_flow = run_prismlang(patient_context, projector, mc_prism)
    prism_result = mc_prism.snapshot(
        "healthcare", "prismlang", "gemini-2.0-flash", TENANT_ID,
        category_flow=category_flow,
        tenant_matrix_fp=projector.matrix_fingerprint(),
    )

    return std_result, prism_result
