from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv(override=True)
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graph import resume_hitl_escalation, run_alert_triage
from src.scorer import DEFAULT_POLICY_PATH
from src.setup_db import init_database


DEFAULT_GOLDEN_PATH = ROOT_DIR / "eval" / "golden_cases.json"
DEFAULT_POLICY_PATH = ROOT_DIR / "data" / "aml_policy_rules.json"

def _resolve_repo_path(raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute() or p.exists():
        return p
    candidate = ROOT_DIR / p
    return candidate if candidate.exists() else p

def _deterministic_only_failover(system_prompt: str, compact_payload: Dict[str, Any], schema: Any):
    raise RuntimeError("Offline deterministic evaluation mode")


def evaluate_golden_suite(
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
    resume_escalations: bool = True,
) -> Dict[str, Any]:
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden cases file not found: {golden_path}")

    with golden_path.open("r", encoding="utf-8") as f:
        suite = json.load(f)

    cases: List[Dict[str, Any]] = suite.get("cases", [])
    results: List[Dict[str, Any]] = []
    passed_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval_fraud_triage.sqlite"
        init_database(db_path=db_path, policy_path=policy_path)

        for idx, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            alert_id = case["alert_id"]
            expected_outcome = case["expected_outcome"]
            expected_paused = bool(case["expected_paused_for_hitl"])
            req_policies = set(case.get("required_policies", []))
            req_signals = set(case.get("required_signal_ids", []))
            thread_id = f"eval-{case_id}-{idx}"

            with patch("src.nodes.invoke_structured_with_failover", side_effect=_deterministic_only_failover):
                state_vals, paused, tid = run_alert_triage(
                    alert_id=alert_id,
                    thread_id=thread_id,
                    db_path=db_path,
                    policy_path=policy_path,
                )

                scoring = state_vals.get("scoring_result") or {}
                actual_outcome = state_vals.get("final_outcome")
                actual_policies = set(scoring.get("cited_policies", []))
                actual_signals = {s["signal_id"] for s in scoring.get("triggered_signals", [])}

                sar_valid = True
                if paused and resume_escalations:
                    resumed = resume_hitl_escalation(
                        thread_id=tid,
                        analyst_decision="APPROVED",
                        analyst_notes="Golden suite automated compliance approval",
                        db_path=db_path,
                        policy_path=policy_path,
                    )
                    sar_draft = resumed.get("sar_draft") or {}
                    narrative = sar_draft.get("narrative", "")
                    sar_valid = (
                        bool(narrative)
                        and req_policies.issubset(set(sar_draft.get("cited_policies", [])))
                        and req_signals.issubset(set(sar_draft.get("cited_signal_ids", [])))
                    )

            case_passed = (
                actual_outcome == expected_outcome
                and paused == expected_paused
                and req_policies.issubset(actual_policies)
                and req_signals.issubset(actual_signals)
                and sar_valid
            )
            if case_passed:
                passed_count += 1

            results.append(
                {
                    "case_id": case_id,
                    "alert_id": alert_id,
                    "passed": case_passed,
                    "expected_outcome": expected_outcome,
                    "actual_outcome": actual_outcome,
                    "expected_paused": expected_paused,
                    "actual_paused": paused,
                    "missing_policies": sorted(list(req_policies - actual_policies)),
                    "missing_signals": sorted(list(req_signals - actual_signals)),
                    "sar_valid": sar_valid,
                }
            )

    total = len(results)
    accuracy = round((passed_count / total) * 100.0, 2) if total else 0.0
    return {
        "suite_version": suite.get("suite_version", "unknown"),
        "passed": passed_count,
        "total": total,
        "accuracy_pct": accuracy,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SentinelGraph Golden Evaluation Suite")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_PATH), help="Path to golden_cases.json")
    parser.add_argument("--policy", default=os.getenv("AML_POLICY_PATH", str(DEFAULT_POLICY_PATH)), help="Path to policy JSON")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume paused ESCALATE_SAR checkpoints")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON report")
    args = parser.parse_args()

    report = evaluate_golden_suite(
        golden_path=_resolve_repo_path(args.golden),
        policy_path=_resolve_repo_path(args.policy),
        resume_escalations=not args.no_resume,
    )

    if args.json:
        print(json.dumps(report, separators=(",", ":"), ensure_ascii=True))
    else:
        print(
            f"Golden Suite v{report['suite_version']}: "
            f"{report['passed']}/{report['total']} passed ({report['accuracy_pct']}%)"
        )
        for item in report["results"]:
            status = "PASS" if item["passed"] else "FAIL"
            print(
                f"  [{status}] {item['case_id']} ({item['alert_id']}): "
                f"outcome={item['actual_outcome']} (exp {item['expected_outcome']}), "
                f"paused={item['actual_paused']} (exp {item['expected_paused']})"
            )

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())