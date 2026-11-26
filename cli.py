from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graph import open_sentinel_graph, resume_hitl_escalation, run_alert_triage
from src.scorer import DEFAULT_DB_PATH, DEFAULT_POLICY_PATH

TriageOutcomeArg = str


def _fetch_alerts(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. Run `python src/setup_db.py` first."
        )
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                a.alert_id,
                a.tx_id,
                a.customer_id,
                a.initial_status,
                a.expected_outcome,
                a.scenario_notes,
                t.amount,
                t.currency,
                t.channel,
                t.country,
                t.city,
                td.deterministic_score,
                td.final_outcome,
                td.analyst_decision,
                td.updated_at_utc
            FROM alerts a
            JOIN transactions t ON a.tx_id = t.tx_id
            LEFT JOIN triage_decisions td ON a.alert_id = td.alert_id
            ORDER BY a.alert_id ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _get_checkpoint_info(
    thread_id: str,
    db_path: Path,
    policy_path: Path,
) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    with open_sentinel_graph(db_path=db_path, policy_path=policy_path) as (app, _):
        snapshot = app.get_state(config)
        next_nodes = list(snapshot.next) if snapshot.next else []
        paused_for_hitl = "generate_sar" in next_nodes
        values = dict(snapshot.values) if snapshot.values else {}
        return {
            "thread_id": thread_id,
            "next_nodes": next_nodes,
            "paused_for_hitl": paused_for_hitl,
            "values": values,
        }


def cmd_list(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    rows = _fetch_alerts(db_path)
    if args.json:
        print(json.dumps(rows, separators=(",", ":"), ensure_ascii=True))
        return 0

    print(f"Alerts in {db_path} ({len(rows)} total):")
    for r in rows:
        score_str = r["deterministic_score"] if r["deterministic_score"] is not None else "-"
        outcome_str = r["final_outcome"] or f"(exp:{r['expected_outcome']})"
        hitl_str = r["analyst_decision"] or r["initial_status"]
        print(
            f"  {r['alert_id']} | {r['tx_id']} | {r['customer_id']} | "
            f"{r['amount']:.2f} {r['currency']} ({r['channel']}, {r['city']}/{r['country']}) | "
            f"score={score_str} | outcome={outcome_str} | status={hitl_str}"
        )
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    policy_path = Path(args.policy)
    thread_id = args.thread_id or f"thread-{args.alert_id}"

    state_values, paused_for_hitl, resolved_tid = run_alert_triage(
        alert_id=args.alert_id,
        thread_id=thread_id,
        db_path=db_path,
        policy_path=policy_path,
    )

    payload = {
        "alert_id": args.alert_id,
        "thread_id": resolved_tid,
        "paused_for_hitl": paused_for_hitl,
        "next_nodes": ["generate_sar"] if paused_for_hitl else [],
        "final_outcome": state_values.get("final_outcome"),
        "analyst_decision": state_values.get("analyst_decision"),
        "scoring_result": state_values.get("scoring_result"),
        "agent_decision": state_values.get("agent_decision"),
        "sar_draft": state_values.get("sar_draft"),
        "execution_trace": state_values.get("execution_trace", []),
    }

    if args.json:
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
        return 0

    scoring = payload["scoring_result"] or {}
    print(f"Alert: {args.alert_id} (thread: {resolved_tid})")
    print(f"Deterministic Score: {scoring.get('deterministic_score')} | Recommended: {scoring.get('recommended_outcome')}")
    print(f"Final Outcome: {payload['final_outcome']} | Analyst Decision: {payload['analyst_decision']}")
    print(f"Paused for HITL: {paused_for_hitl}")
    for sig in scoring.get("triggered_signals", []):
        print(
            f"  - Signal {sig['signal_id']} [{sig['policy_id']}] "
            f"({sig['severity']}, weight={sig['score_weight']}): {sig['title']}"
        )
    if paused_for_hitl:
        print(
            f"\nCheckpoint paused before generate_sar. Resume with:\n"
            f"  python cli.py resume --thread-id {resolved_tid} --decision APPROVED --notes \"Confirmed anomalous activity\""
        )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    policy_path = Path(args.policy)
    info = _get_checkpoint_info(args.thread_id, db_path=db_path, policy_path=policy_path)

    if not info["paused_for_hitl"] and not args.force:
        print(
            f"Thread '{args.thread_id}' is not currently paused at generate_sar "
            f"(next={info['next_nodes']}). Use --force to update/invoke anyway.",
            file=sys.stderr,
        )
        return 1

    override_outcome: Optional[str] = args.override_outcome
    if args.decision == "OVERRIDDEN" and not override_outcome:
        print(
            "Error: --override-outcome is required when --decision OVERRIDDEN.",
            file=sys.stderr,
        )
        return 1

    final_state = resume_hitl_escalation(
        thread_id=args.thread_id,
        analyst_decision=args.decision,
        analyst_notes=args.notes,
        override_outcome=override_outcome,  # type: ignore[arg-type]
        db_path=db_path,
        policy_path=policy_path,
    )

    result = {
        "thread_id": args.thread_id,
        "alert_id": final_state.get("alert_id"),
        "analyst_decision": final_state.get("analyst_decision"),
        "final_outcome": final_state.get("final_outcome"),
        "sar_draft": final_state.get("sar_draft"),
        "execution_trace": final_state.get("execution_trace", []),
    }

    if args.json:
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=True))
        return 0

    print(f"Resumed thread: {args.thread_id} (alert: {result['alert_id']})")
    print(f"Analyst Decision: {result['analyst_decision']} | Final Outcome: {result['final_outcome']}")
    sar = result.get("sar_draft")
    if sar:
        print("\n--- SAR Draft ---")
        print(sar.get("narrative", ""))
        print(f"Cited Policies: {sar.get('cited_policies')}")
        print(f"Cited Signals:  {sar.get('cited_signal_ids')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinelgraph-cli",
        description="SentinelGraph Financial Fraud Triage & HITL Compliance CLI",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--policy",
        default=os.getenv("AML_POLICY_PATH", str(DEFAULT_POLICY_PATH)),
        help="Path to AML policy JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit compact JSON output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_p = subparsers.add_parser("list", help="List seeded/triaged alerts")
    list_p.set_defaults(func=cmd_list)

    triage_p = subparsers.add_parser("triage", help="Run triage pipeline for an alert_id")
    triage_p.add_argument("--alert-id", required=True, help="Alert ID (e.g. ALT-1005)")
    triage_p.add_argument("--thread-id", default=None, help="Optional LangGraph thread_id")
    triage_p.set_defaults(func=cmd_triage)

    resume_p = subparsers.add_parser("resume", help="Resume a paused HITL checkpoint")
    resume_p.add_argument("--thread-id", required=True, help="LangGraph thread_id (e.g. thread-ALT-1005)")
    resume_p.add_argument(
        "--decision",
        required=True,
        choices=["APPROVED", "OVERRIDDEN"],
        help="Compliance analyst decision",
    )
    resume_p.add_argument(
        "--override-outcome",
        default=None,
        choices=["AUTO_CLOSE", "STEP_UP_MFA", "ESCALATE_SAR"],
        help="Required when --decision OVERRIDDEN",
    )
    resume_p.add_argument("--notes", default="", help="Analyst review notes")
    resume_p.add_argument("--force", action="store_true", help="Resume even if not paused at generate_sar")
    resume_p.set_defaults(func=cmd_resume)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())