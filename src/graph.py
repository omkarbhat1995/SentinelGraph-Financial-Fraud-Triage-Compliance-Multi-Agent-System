from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.nodes import generate_sar_node, score_and_context_node, triage_agent_node
from src.scorer import DEFAULT_DB_PATH, DEFAULT_POLICY_PATH

load_dotenv(override=True)


class SentinelGraphState(TypedDict, total=False):
    alert_id: str
    db_path: str
    policy_path: str
    customer: Dict[str, Any]
    transaction: Dict[str, Any]
    scoring_result: Dict[str, Any]
    agent_decision: Dict[str, Any]
    final_outcome: Literal["AUTO_CLOSE", "STEP_UP_MFA", "ESCALATE_SAR"]
    sar_draft: Dict[str, Any]
    analyst_decision: Literal["APPROVED", "OVERRIDDEN", "PENDING"]
    analyst_notes: Optional[str]
    execution_trace: list[str]


def _route_after_triage(state: SentinelGraphState) -> Literal["generate_sar", "__end__"]:
    if state.get("final_outcome") == "ESCALATE_SAR":
        return "generate_sar"
    return END


def build_state_graph() -> StateGraph:
    builder = StateGraph(SentinelGraphState)

    builder.add_node("score_and_context", score_and_context_node)
    builder.add_node("triage_agent", triage_agent_node)
    builder.add_node("generate_sar", generate_sar_node)

    builder.add_edge(START, "score_and_context")
    builder.add_edge("score_and_context", "triage_agent")
    builder.add_conditional_edges(
        "triage_agent",
        _route_after_triage,
        {
            "generate_sar": "generate_sar",
            END: END,
        },
    )
    builder.add_edge("generate_sar", END)

    return builder


@contextmanager
def open_sentinel_graph(
    db_path: Path = DEFAULT_DB_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """
    Context manager yielding a compiled LangGraph app backed by SqliteSaver
    with `interrupt_before=["generate_sar"]` for HITL compliance review.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        app = build_state_graph().compile(
            checkpointer=checkpointer,
            interrupt_before=["generate_sar"],
        )
        yield app, {"db_path": str(db_path), "policy_path": str(policy_path)}
    finally:
        conn.close()


def run_alert_triage(
    alert_id: str,
    thread_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> tuple[Dict[str, Any], bool, str]:
    """
    Run triage for `alert_id`.
    Returns `(state_values, paused_for_hitl, thread_id)`.
    """
    resolved_thread_id = thread_id or f"thread-{alert_id}"
    config = {"configurable": {"thread_id": resolved_thread_id}}

    with open_sentinel_graph(db_path=db_path, policy_path=policy_path) as (app, base_input):
        initial_state: SentinelGraphState = {
            "alert_id": alert_id,
            "db_path": base_input["db_path"],
            "policy_path": base_input["policy_path"],
            "execution_trace": [],
        }
        app.invoke(initial_state, config=config)
        snapshot = app.get_state(config)
        paused_for_hitl = bool(snapshot.next) and ("generate_sar" in snapshot.next)
        return dict(snapshot.values), paused_for_hitl, resolved_thread_id


def resume_hitl_escalation(
    thread_id: str,
    analyst_decision: Literal["APPROVED", "OVERRIDDEN"],
    analyst_notes: Optional[str] = None,
    override_outcome: Optional[Literal["AUTO_CLOSE", "STEP_UP_MFA", "ESCALATE_SAR"]] = None,
    db_path: Path = DEFAULT_DB_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> Dict[str, Any]:
    """
    Resume a paused HITL checkpoint (`interrupt_before=["generate_sar"]`) after
    recording the compliance analyst's decision.
    """
    config = {"configurable": {"thread_id": thread_id}}

    with open_sentinel_graph(db_path=db_path, policy_path=policy_path) as (app, _):
        state_update: Dict[str, Any] = {
            "analyst_decision": analyst_decision,
            "analyst_notes": analyst_notes or "",
        }
        if override_outcome is not None:
            state_update["final_outcome"] = override_outcome

        app.update_state(config, state_update, as_node="triage_agent")
        app.invoke(None, config=config)
        snapshot = app.get_state(config)
        return dict(snapshot.values)