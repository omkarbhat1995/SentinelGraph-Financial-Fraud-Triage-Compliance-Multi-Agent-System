from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from src.graph import resume_hitl_escalation, run_alert_triage
from src.schemas import SARDraft, TriageAgentDecision
from src.setup_db import init_database

TEST_POLICY_PATH = Path("data/aml_policy_rules.json")


@pytest.fixture()
def isolated_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_graph_fraud.sqlite"
    init_database(db_path=db_file, policy_path=TEST_POLICY_PATH)
    return db_file


def test_auto_close_flow_does_not_pause(isolated_db: Path):
    state, paused, thread_id = run_alert_triage(
        alert_id="ALT-1001",
        thread_id="test-thread-1001",
        db_path=isolated_db,
        policy_path=TEST_POLICY_PATH,
    )
    assert paused is False
    assert thread_id == "test-thread-1001"
    assert state["final_outcome"] == "AUTO_CLOSE"
    assert state["analyst_decision"] == "APPROVED"
    assert state.get("sar_draft") is None

    with closing(sqlite3.connect(isolated_db)) as conn:
        row = conn.execute(
            "SELECT final_outcome, analyst_decision FROM triage_decisions WHERE alert_id = 'ALT-1001'"
        ).fetchone()
    assert row == ("AUTO_CLOSE", "APPROVED")


def test_step_up_mfa_flow_does_not_pause(isolated_db: Path):
    state, paused, _ = run_alert_triage(
        alert_id="ALT-1003",
        thread_id="test-thread-1003",
        db_path=isolated_db,
        policy_path=TEST_POLICY_PATH,
    )
    assert paused is False
    assert state["final_outcome"] == "STEP_UP_MFA"
    assert "SIG-VEL-TX-9003" in state["scoring_result"]["cited_policies"] or "POL-VEL-001" in state["scoring_result"]["cited_policies"]


def test_escalate_sar_pauses_and_resumes_with_approval(isolated_db: Path):
    state, paused, thread_id = run_alert_triage(
        alert_id="ALT-1005",
        thread_id="test-thread-1005",
        db_path=isolated_db,
        policy_path=TEST_POLICY_PATH,
    )
    assert paused is True
    assert state["final_outcome"] == "ESCALATE_SAR"
    assert state["analyst_decision"] == "PENDING"
    assert state.get("sar_draft") is None

    resumed = resume_hitl_escalation(
        thread_id=thread_id,
        analyst_decision="APPROVED",
        analyst_notes="Confirmed impossible travel between Sao Paulo and London.",
        db_path=isolated_db,
        policy_path=TEST_POLICY_PATH,
    )
    assert resumed["analyst_decision"] == "APPROVED"
    assert resumed["final_outcome"] == "ESCALATE_SAR"
    sar = resumed.get("sar_draft")
    assert sar is not None
    assert "SIG-GEO-TX-9005" in sar["cited_signal_ids"]
    assert "POL-GEO-003" in sar["cited_policies"]
    assert "SIG-GEO-TX-9005" in sar["narrative"]

    with closing(sqlite3.connect(isolated_db)) as conn:
        row = conn.execute(
            "SELECT analyst_decision, sar_draft_text FROM triage_decisions WHERE alert_id = 'ALT-1005'"
        ).fetchone()
    assert row is not None
    assert row[0] == "APPROVED"
    assert row[1] is not None and "SIG-GEO-TX-9005" in row[1]


def test_escalate_sar_override_outcome(isolated_db: Path):
    _, paused, thread_id = run_alert_triage(
        alert_id="ALT-1006",
        thread_id="test-thread-1006",
        db_path=isolated_db,
        policy_path=TEST_POLICY_PATH,
    )
    assert paused is True

    resumed = resume_hitl_escalation(
        thread_id=thread_id,
        analyst_decision="OVERRIDDEN",
        analyst_notes="Pre-authorized corporate treasury batch.",
        override_outcome="STEP_UP_MFA",
        db_path=isolated_db,
        policy_path=TEST_POLICY_PATH,
    )
    assert resumed["analyst_decision"] == "OVERRIDDEN"
    assert resumed["final_outcome"] == "STEP_UP_MFA"
    assert resumed.get("sar_draft") is None

    with closing(sqlite3.connect(isolated_db)) as conn:
        row = conn.execute(
            "SELECT final_outcome, analyst_decision FROM triage_decisions WHERE alert_id = 'ALT-1006'"
        ).fetchone()
    assert row == ("STEP_UP_MFA", "OVERRIDDEN")


def test_llm_failover_invoked_when_configured(isolated_db: Path):
    def fake_failover(system_prompt: str, compact_payload: dict, schema: type):
        if schema is TriageAgentDecision:
            return (
                TriageAgentDecision(
                    alert_id=compact_payload["alert_id"],
                    proposed_outcome="STEP_UP_MFA",
                    confidence_score=0.95,
                    primary_policy_cited="POL-VEL-001",
                    justification="Velocity signal SIG-VEL-TX-9003 under POL-VEL-001.",
                    cited_signal_ids=["SIG-VEL-TX-9003"],
                ),
                "mock:failover-test",
            )
        return (
            SARDraft(
                alert_id=compact_payload["alert_id"],
                customer_id=compact_payload["customer_id"],
                tx_id=compact_payload["tx_id"],
                narrative="Narrative citing SIG-VEL-TX-9003 and POL-VEL-001.",
                cited_policies=["POL-VEL-001"],
                cited_signal_ids=["SIG-VEL-TX-9003"],
                recommended_regulatory_action="Review",
                prepared_timestamp_utc="2026-11-26T00:00:00Z",
            ),
            "mock:failover-test",
        )

    with patch("src.nodes.invoke_structured_with_failover", side_effect=fake_failover):
        state, paused, _ = run_alert_triage(
            alert_id="ALT-1003",
            thread_id="test-thread-mock-1003",
            db_path=isolated_db,
            policy_path=TEST_POLICY_PATH,
        )
    assert paused is False
    assert state["agent_decision"]["primary_policy_cited"] == "POL-VEL-001"
    assert any("mock:failover-test" in entry for entry in state["execution_trace"])