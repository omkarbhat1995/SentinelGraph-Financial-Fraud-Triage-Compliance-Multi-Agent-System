from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.llm_factory import invoke_structured_with_failover
from src.schemas import (
    SARDraft,
    TriageAgentDecision,
    TriageOutcome,
)
from src.scorer import DEFAULT_DB_PATH, DEFAULT_POLICY_PATH, DeterministicScorer

load_dotenv(override=True)

TRIAGE_SYSTEM_PROMPT = (
    "You are SentinelGraph Triage Agent. "
    "Use ONLY the precomputed metrics, triggered_signals, and allowed_policy_ids in INPUT_JSON. "
    "Do NOT recalculate or invent numbers, balances, z-scores, speeds, or counts. "
    "Every claim in justification MUST reference exact signal_id values from triggered_signals "
    "and policy_id values from allowed_policy_ids. "
    "Respect deterministic recommended_outcome unless policy rules strictly require otherwise."
)

SAR_SYSTEM_PROMPT = (
    "You are SentinelGraph AML Compliance Drafting Agent. "
    "Draft a factual Suspicious Activity Report narrative using ONLY facts in INPUT_JSON. "
    "Every claim MUST explicitly cite exact signal_id and policy_id identifiers present in INPUT_JSON. "
    "Do NOT invent unstated dates, counterparties, balances, or metrics."
)


def _persist_triage_decision(
    db_path: Path,
    alert_id: str,
    deterministic_score: int,
    triggered_signals: List[Dict[str, Any]],
    cited_policies: List[str],
    final_outcome: TriageOutcome,
    analyst_decision: str,
    sar_draft_text: Optional[str],
) -> None:
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signals_json = json.dumps(triggered_signals, separators=(",", ":"))
    policies_json = json.dumps(cited_policies, separators=(",", ":"))

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO triage_decisions (
                alert_id, deterministic_score, triggered_signals_json,
                cited_policies_json, final_outcome, analyst_decision,
                sar_draft_text, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                deterministic_score = excluded.deterministic_score,
                triggered_signals_json = excluded.triggered_signals_json,
                cited_policies_json = excluded.cited_policies_json,
                final_outcome = excluded.final_outcome,
                analyst_decision = excluded.analyst_decision,
                sar_draft_text = COALESCE(excluded.sar_draft_text, triage_decisions.sar_draft_text),
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                alert_id,
                deterministic_score,
                signals_json,
                policies_json,
                final_outcome,
                analyst_decision,
                sar_draft_text,
                updated_at,
            ),
        )
        conn.execute(
            "UPDATE alerts SET initial_status = 'TRIAGED' WHERE alert_id = ?",
            (alert_id,),
        )
        conn.commit()


def _fallback_triage_decision(
    alert_id: str,
    scoring_dict: Dict[str, Any],
) -> TriageAgentDecision:
    signals = scoring_dict.get("triggered_signals", [])
    signal_ids = [s["signal_id"] for s in signals if "signal_id" in s]
    policies = scoring_dict.get("cited_policies", [])
    primary_policy = policies[0] if policies else "POL-BEN-006"
    outcome: TriageOutcome = scoring_dict.get("recommended_outcome", "STEP_UP_MFA")
    score = scoring_dict.get("deterministic_score", 0)
    zscore = scoring_dict.get("amount_zscore", 0.0)
    speed = scoring_dict.get("travel_speed_kmh")
    vel = scoring_dict.get("tx_count_24h", 1)

    sig_ref = ", ".join(signal_ids) if signal_ids else "NONE"
    pol_ref = ", ".join(policies) if policies else primary_policy
    speed_part = f" travel_speed_kmh={speed}," if speed is not None else ""
    justification = (
        f"Deterministic score={score} (zscore={zscore},{speed_part} tx_count_24h={vel}) "
        f"evaluated under signals [{sig_ref}] and policies [{pol_ref}]."
    )
    return TriageAgentDecision(
        alert_id=alert_id,
        proposed_outcome=outcome,
        confidence_score=0.9,
        primary_policy_cited=primary_policy,
        justification=justification,
        cited_signal_ids=signal_ids,
    )


def _fallback_sar_draft(
    alert_id: str,
    customer_id: str,
    tx_id: str,
    scoring_dict: Dict[str, Any],
    tx_dict: Dict[str, Any],
) -> SARDraft:
    signals = scoring_dict.get("triggered_signals", [])
    signal_ids = [s["signal_id"] for s in signals if "signal_id" in s]
    policies = scoring_dict.get("cited_policies", [])
    sig_str = ", ".join(signal_ids) if signal_ids else "N/A"
    pol_str = ", ".join(policies) if policies else "N/A"
    amount = tx_dict.get("amount", 0.0)
    currency = tx_dict.get("currency", "USD")
    ts = tx_dict.get("timestamp_utc", "")

    narrative = (
        f"Alert {alert_id} for customer {customer_id} on transaction {tx_id} "
        f"({amount} {currency} at {ts}) triggered signals [{sig_str}] under policies [{pol_str}] "
        f"with deterministic_score={scoring_dict.get('deterministic_score', 0)}. "
        f"Held for human compliance analyst approval prior to filing."
    )
    return SARDraft(
        alert_id=alert_id,
        customer_id=customer_id,
        tx_id=tx_id,
        narrative=narrative,
        cited_policies=policies,
        cited_signal_ids=signal_ids,
        recommended_regulatory_action="Hold case at HITL checkpoint for analyst review and SAR filing.",
        prepared_timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def score_and_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load context from SQLite and compute deterministic risk score."""
    db_path = Path(state.get("db_path") or os.getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)))
    policy_path = Path(
        state.get("policy_path") or os.getenv("AML_POLICY_PATH", str(DEFAULT_POLICY_PATH))
    )
    alert_id = state["alert_id"]

    scorer = DeterministicScorer(db_path=db_path, policy_path=policy_path)
    alert, customer, tx = scorer.fetch_alert_context(alert_id)
    scoring_result = scorer.evaluate(alert, customer, tx)

    trace = list(state.get("execution_trace", []))
    trace.append(
        f"score_and_context: score={scoring_result.deterministic_score}, "
        f"recommended={scoring_result.recommended_outcome}, "
        f"signals={[s.signal_id for s in scoring_result.triggered_signals]}"
    )

    return {
        "customer": customer.model_dump(),
        "transaction": tx.model_dump(),
        "scoring_result": scoring_result.model_dump(),
        "execution_trace": trace,
    }


def triage_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Grounded triage reasoning node using compact JSON and LLM failover."""
    alert_id = state["alert_id"]
    scoring_dict = state["scoring_result"]
    tx_dict = state["transaction"]
    cust_dict = state["customer"]

    compact_payload = {
        "alert_id": alert_id,
        "customer_id": cust_dict["customer_id"],
        "tx_id": tx_dict["tx_id"],
        "amount": tx_dict["amount"],
        "currency": tx_dict["currency"],
        "channel": tx_dict["channel"],
        "country": tx_dict["country"],
        "deterministic_score": scoring_dict["deterministic_score"],
        "recommended_outcome": scoring_dict["recommended_outcome"],
        "has_sar_trigger": scoring_dict["has_sar_trigger"],
        "amount_zscore": scoring_dict["amount_zscore"],
        "travel_speed_kmh": scoring_dict.get("travel_speed_kmh"),
        "tx_count_24h": scoring_dict["tx_count_24h"],
        "triggered_signals": scoring_dict["triggered_signals"],
        "allowed_policy_ids": scoring_dict["cited_policies"],
    }

    try:
        decision, provider_label = invoke_structured_with_failover(
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            compact_payload=compact_payload,
            schema=TriageAgentDecision,
        )
        # Enforce grounding: only allow signal_ids and policy_ids actually produced by scorer
        valid_signal_ids = {s["signal_id"] for s in scoring_dict.get("triggered_signals", [])}
        valid_policies = set(scoring_dict.get("cited_policies", []))
        decision.cited_signal_ids = [
            sid for sid in decision.cited_signal_ids if sid in valid_signal_ids
        ] or list(valid_signal_ids)
        if valid_policies and decision.primary_policy_cited not in valid_policies:
            decision.primary_policy_cited = sorted(valid_policies)[0]
    except Exception as exc:
        decision = _fallback_triage_decision(alert_id, scoring_dict)
        provider_label = f"deterministic-fallback ({type(exc).__name__})"

    # Final outcome aligns with deterministic SAR/threshold guardrails
    final_outcome: TriageOutcome = decision.proposed_outcome
    if scoring_dict.get("has_sar_trigger") or scoring_dict.get("recommended_outcome") == "ESCALATE_SAR":
        final_outcome = "ESCALATE_SAR"

    db_path = Path(state.get("db_path") or os.getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)))
    analyst_decision = "PENDING" if final_outcome == "ESCALATE_SAR" else "APPROVED"

    _persist_triage_decision(
        db_path=db_path,
        alert_id=alert_id,
        deterministic_score=scoring_dict["deterministic_score"],
        triggered_signals=scoring_dict.get("triggered_signals", []),
        cited_policies=scoring_dict.get("cited_policies", []),
        final_outcome=final_outcome,
        analyst_decision=analyst_decision,
        sar_draft_text=None,
    )

    trace = list(state.get("execution_trace", []))
    trace.append(
        f"triage_agent ({provider_label}): outcome={final_outcome}, "
        f"policy={decision.primary_policy_cited}, signals={decision.cited_signal_ids}"
    )

    return {
        "agent_decision": decision.model_dump(),
        "final_outcome": final_outcome,
        "analyst_decision": analyst_decision,
        "execution_trace": trace,
    }


def generate_sar_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executed after HITL approval/override for ESCALATE_SAR cases.
    Drafts a signal/policy-cited SAR narrative and persists it to SQLite.
    """
    alert_id = state["alert_id"]
    scoring_dict = state["scoring_result"]
    tx_dict = state["transaction"]
    cust_dict = state["customer"]

    analyst_decision = state.get("analyst_decision") or "APPROVED"
    final_outcome: TriageOutcome = state.get("final_outcome") or "ESCALATE_SAR"

    compact_payload = {
        "alert_id": alert_id,
        "customer_id": cust_dict["customer_id"],
        "risk_tier": cust_dict["risk_tier"],
        "tx_id": tx_dict["tx_id"],
        "timestamp_utc": tx_dict["timestamp_utc"],
        "amount": tx_dict["amount"],
        "currency": tx_dict["currency"],
        "channel": tx_dict["channel"],
        "country": tx_dict["country"],
        "city": tx_dict["city"],
        "deterministic_score": scoring_dict["deterministic_score"],
        "amount_zscore": scoring_dict["amount_zscore"],
        "travel_speed_kmh": scoring_dict.get("travel_speed_kmh"),
        "tx_count_24h": scoring_dict["tx_count_24h"],
        "triggered_signals": scoring_dict["triggered_signals"],
        "cited_policies": scoring_dict["cited_policies"],
        "analyst_decision": analyst_decision,
        "analyst_notes": state.get("analyst_notes"),
    }

    try:
        sar_draft, provider_label = invoke_structured_with_failover(
            system_prompt=SAR_SYSTEM_PROMPT,
            compact_payload=compact_payload,
            schema=SARDraft,
        )
        valid_signal_ids = {s["signal_id"] for s in scoring_dict.get("triggered_signals", [])}
        valid_policies = set(scoring_dict.get("cited_policies", []))
        sar_draft.cited_signal_ids = [
            sid for sid in sar_draft.cited_signal_ids if sid in valid_signal_ids
        ] or list(valid_signal_ids)
        sar_draft.cited_policies = [
            pid for pid in sar_draft.cited_policies if pid in valid_policies
        ] or sorted(valid_policies)
    except Exception as exc:
        sar_draft = _fallback_sar_draft(
            alert_id=alert_id,
            customer_id=cust_dict["customer_id"],
            tx_id=tx_dict["tx_id"],
            scoring_dict=scoring_dict,
            tx_dict=tx_dict,
        )
        provider_label = f"deterministic-fallback ({type(exc).__name__})"

    db_path = Path(state.get("db_path") or os.getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)))
    _persist_triage_decision(
        db_path=db_path,
        alert_id=alert_id,
        deterministic_score=scoring_dict["deterministic_score"],
        triggered_signals=scoring_dict.get("triggered_signals", []),
        cited_policies=sar_draft.cited_policies,
        final_outcome=final_outcome,
        analyst_decision=analyst_decision,
        sar_draft_text=sar_draft.narrative,
    )

    trace = list(state.get("execution_trace", []))
    trace.append(
        f"generate_sar ({provider_label}): analyst_decision={analyst_decision}, "
        f"cited_signals={sar_draft.cited_signal_ids}"
    )

    return {
        "sar_draft": sar_draft.model_dump(),
        "execution_trace": trace,
    }