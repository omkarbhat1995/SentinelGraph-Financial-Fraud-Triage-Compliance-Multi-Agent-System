from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graph import open_sentinel_graph, resume_hitl_escalation, run_alert_triage
from src.scorer import DEFAULT_DB_PATH, DEFAULT_POLICY_PATH

st.set_page_config(
    page_title="SentinelGraph — Fraud Triage & HITL Workbench",
    layout="wide",
)


def _load_alerts_table(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                a.alert_id,
                a.tx_id,
                a.customer_id,
                a.expected_outcome,
                a.initial_status,
                a.scenario_notes,
                c.full_name AS customer_name,
                c.risk_tier,
                t.amount,
                t.currency,
                t.channel,
                t.merchant_name,
                t.city,
                t.country,
                td.deterministic_score,
                td.final_outcome,
                td.analyst_decision,
                td.sar_draft_text
            FROM alerts a
            JOIN customers c ON a.customer_id = c.customer_id
            JOIN transactions t ON a.tx_id = t.tx_id
            LEFT JOIN triage_decisions td ON a.alert_id = td.alert_id
            ORDER BY a.alert_id ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _get_checkpoint_snapshot(
    thread_id: str,
    db_path: Path,
    policy_path: Path,
) -> tuple[Dict[str, Any], bool, List[str]]:
    if not db_path.exists():
        return {}, False, []
    config = {"configurable": {"thread_id": thread_id}}
    with open_sentinel_graph(db_path=db_path, policy_path=policy_path) as (app, _):
        snap = app.get_state(config)
        next_nodes = list(snap.next) if snap.next else []
        paused = "generate_sar" in next_nodes
        values = dict(snap.values) if snap.values else {}
        return values, paused, next_nodes


def main() -> None:
    db_path = Path(os.getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)))
    policy_path = Path(os.getenv("AML_POLICY_PATH", str(DEFAULT_POLICY_PATH)))

    st.title("SentinelGraph — Financial Fraud Triage & HITL Compliance Workbench")
    st.caption(
        "Deterministic SQL/statistical scoring + stateful LangGraph agents + SqliteSaver HITL checkpoints"
    )

    if not db_path.exists():
        st.error(
            f"Database `{db_path}` not found. Run `python src/setup_db.py` first."
        )
        return

    alerts = _load_alerts_table(db_path)
    if not alerts:
        st.warning("No alerts found in database.")
        return

    with st.sidebar:
        st.header("Alert Queue")
        outcome_filter = st.selectbox(
            "Filter by Expected / Final Outcome",
            options=["ALL", "AUTO_CLOSE", "STEP_UP_MFA", "ESCALATE_SAR"],
            index=0,
        )
        filtered = [
            a
            for a in alerts
            if outcome_filter == "ALL"
            or a["expected_outcome"] == outcome_filter
            or a["final_outcome"] == outcome_filter
        ]
        if not filtered:
            filtered = alerts

        alert_labels = [
            f"{a['alert_id']} — {a['customer_name']} ({a['amount']:.0f} {a['currency']} | exp:{a['expected_outcome']})"
            for a in filtered
        ]
        selected_idx = st.selectbox(
            "Select Alert",
            options=range(len(filtered)),
            format_func=lambda i: alert_labels[i],
        )
        selected_alert = filtered[selected_idx]
        alert_id = selected_alert["alert_id"]
        thread_id = st.text_input("LangGraph Thread ID", value=f"thread-{alert_id}")

    snap_values, paused_for_hitl, next_nodes = _get_checkpoint_snapshot(
        thread_id=thread_id,
        db_path=db_path,
        policy_path=policy_path,
    )

    col_top1, col_top2, col_top3, col_top4 = st.columns(4)
    scoring_data = snap_values.get("scoring_result") or {}
    det_score = scoring_data.get(
        "deterministic_score", selected_alert.get("deterministic_score", "—")
    )
    final_outcome = snap_values.get(
        "final_outcome", selected_alert.get("final_outcome") or "UNTRIAGED"
    )
    analyst_dec = snap_values.get(
        "analyst_decision", selected_alert.get("analyst_decision") or "—"
    )
    hitl_status = "PAUSED (HITL)" if paused_for_hitl else ("RESUMED / DONE" if snap_values else "IDLE")

    col_top1.metric("Deterministic Score", det_score)
    col_top2.metric("Final Outcome", final_outcome)
    col_top3.metric("HITL Checkpoint", hitl_status)
    col_top4.metric("Analyst Decision", analyst_dec)

    st.divider()

    left_col, right_col = st.columns([1.15, 1.0])

    with left_col:
        st.subheader(f"Alert & Transaction Context — `{alert_id}`")
        st.write(
            f"**Customer:** {selected_alert['customer_name']} (`{selected_alert['customer_id']}`) "
            f"| **Risk Tier:** `{selected_alert['risk_tier']}`"
        )
        st.write(
            f"**Transaction:** `{selected_alert['tx_id']}` — "
            f"**{selected_alert['amount']:.2f} {selected_alert['currency']}** via `{selected_alert['channel']}` "
            f"at *{selected_alert['merchant_name']}* ({selected_alert['city']}, {selected_alert['country']})"
        )
        st.info(f"**Scenario Note:** {selected_alert['scenario_notes']}")

        if st.button("Run Triage Pipeline", type="primary", use_container_width=True):
            with st.spinner(f"Running deterministic scorer & triage agent for {alert_id}..."):
                state_vals, now_paused, _ = run_alert_triage(
                    alert_id=alert_id,
                    thread_id=thread_id,
                    db_path=db_path,
                    policy_path=policy_path,
                )
            snap_values = state_vals
            paused_for_hitl = now_paused
            next_nodes = ["generate_sar"] if now_paused else []
            st.rerun()

        if scoring_data:
            st.subheader("Deterministic Signals & Cited Policies")
            st.write(
                f"**Amount Z-Score:** `{scoring_data.get('amount_zscore')}` | "
                f"**24h Tx Count:** `{scoring_data.get('tx_count_24h')}` | "
                f"**Travel Speed (km/h):** `{scoring_data.get('travel_speed_kmh')}` | "
                f"**SAR Trigger:** `{scoring_data.get('has_sar_trigger')}`"
            )
            signals = scoring_data.get("triggered_signals", [])
            if signals:
                for sig in signals:
                    st.markdown(
                        f"- **`{sig['signal_id']}`** (`{sig['policy_id']}`, `{sig['severity']}`, "
                        f"weight `{sig['score_weight']}`): {sig['title']}"
                    )
            else:
                st.write("No risk signals triggered.")

        agent_dec = snap_values.get("agent_decision")
        if agent_dec:
            st.subheader("Triage Agent Output")
            st.write(
                f"**Proposed Outcome:** `{agent_dec.get('proposed_outcome')}` | "
                f"**Confidence:** `{agent_dec.get('confidence_score')}` | "
                f"**Primary Policy:** `{agent_dec.get('primary_policy_cited')}`"
            )
            st.write(agent_dec.get("justification", ""))

    with right_col:
        st.subheader("Human-in-the-Loop (HITL) Escalation Gate")
        if paused_for_hitl:
            st.warning(
                f"Graph is **paused before `generate_sar`** on thread `{thread_id}`. "
                "Review the signals and choose whether to approve SAR generation or override the outcome."
            )
            decision_choice = st.radio(
                "Analyst Action",
                options=["APPROVED", "OVERRIDDEN"],
                horizontal=True,
                key=f"dec-{thread_id}",
            )
            override_outcome = None
            if decision_choice == "OVERRIDDEN":
                override_outcome = st.selectbox(
                    "Override Final Outcome",
                    options=["STEP_UP_MFA", "AUTO_CLOSE", "ESCALATE_SAR"],
                    key=f"ovr-{thread_id}",
                )
            analyst_notes = st.text_area(
                "Compliance Analyst Notes",
                placeholder="Document customer verification, travel context, or override rationale...",
                key=f"notes-{thread_id}",
            )

            if st.button("Submit Analyst Decision & Resume", type="primary", use_container_width=True):
                with st.spinner("Resuming graph and generating SAR record..."):
                    resumed_vals = resume_hitl_escalation(
                        thread_id=thread_id,
                        analyst_decision=decision_choice,  # type: ignore[arg-type]
                        analyst_notes=analyst_notes,
                        override_outcome=override_outcome,  # type: ignore[arg-type]
                        db_path=db_path,
                        policy_path=policy_path,
                    )
                snap_values = resumed_vals
                paused_for_hitl = False
                st.rerun()
        else:
            st.write(
                f"Current checkpoint next nodes: `{next_nodes if next_nodes else ['__end__']}`"
            )

        sar_draft = snap_values.get("sar_draft")
        db_sar_text = selected_alert.get("sar_draft_text")
        if sar_draft:
            st.subheader("Generated SAR Draft")
            st.success(
                f"**Analyst Decision:** `{snap_values.get('analyst_decision')}` | "
                f"**Final Outcome:** `{snap_values.get('final_outcome')}`"
            )
            st.write(sar_draft.get("narrative", ""))
            st.caption(
                f"Cited Policies: {sar_draft.get('cited_policies')} | "
                f"Cited Signals: {sar_draft.get('cited_signal_ids')}"
            )
        elif db_sar_text:
            st.subheader("Persisted SAR Narrative (SQLite)")
            st.write(db_sar_text)

        trace = snap_values.get("execution_trace") or []
        if trace:
            with st.expander("Execution Trace", expanded=False):
                for entry in trace:
                    st.code(entry, language="text")


if __name__ == "__main__":
    main()