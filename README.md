# SentinelGraph — Financial Fraud Triage & Compliance Multi-Agent System

<!-- SentinelGraph Tech Stack & CI Badges -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF4B4B?style=flat-square&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/Testing-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![SentinelGraph CI](https://github.com/omkarbhat1995/SentinelGraph-Financial-Fraud-Triage-Compliance-Multi-Agent-System/actions/workflows/test.yaml/badge.svg)](https://github.com/omkarbhat1995/SentinelGraph-Financial-Fraud-Triage-Compliance-Multi-Agent-System/actions/workflows/test.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**SentinelGraph** is a standalone Python + LangGraph financial fraud triage and AML compliance pipeline that combines **deterministic statistical/SQL scoring**, **structured multi-agent reasoning with automatic provider failover**, and **Human-in-the-Loop (HITL) `SqliteSaver` checkpointing** before generating Suspicious Activity Report (SAR) narratives.

## Architecture & Triage Outcomes

Every alert flows through a stateful LangGraph `StateGraph`:

1. **`score_and_context`**: Loads customer, transaction, and alert context from SQLite (`data/fraud_triage.sqlite`) and computes exact Z-scores, Haversine distances, implied travel speed (km/h), and 24-hour velocity in Python against `data/aml_policy_rules.json`.
2. **`triage_agent`**: Receives a whitespace-stripped compact JSON payload (`separators=(",", ":")`) and produces a validated `TriageAgentDecision` grounded strictly in deterministic `signal_id` and `policy_id` identifiers, with automatic failover across Gemini (`gemini-3.5-flash-lite`) and Groq (`openai/gpt-oss-120b`).
3. **Conditional Routing & HITL Gate (`interrupt_before=["generate_sar"]`)**:
   - **`AUTO_CLOSE`**: Benign behavioral drift within baseline bounds; persists `APPROVED` auto-closure.
   - **`STEP_UP_MFA`**: Elevated risk (velocity, device/VPN, or Z-score anomaly); routes for step-up authentication.
   - **`ESCALATE_SAR`**: Critical fraud/AML trigger (impossible travel or structuring); pauses at a `SqliteSaver` checkpoint before `generate_sar` until a compliance analyst approves (`APPROVED`) or overrides (`OVERRIDDEN`) via CLI or Streamlit.
4. **`generate_sar`**: Drafts a structured `SARDraft` narrative citing exact `signal_id` and `policy_id` values and persists it to `triage_decisions`.

## Repository Layout

- `data/aml_policy_rules.json` — Versioned AML/fraud policy rules and score weights
- `src/setup_db.py` — SQLite schema initialization and 10 benchmark alerts
- `src/schemas.py` — Pydantic v2 schemas and compact JSON serialization
- `src/scorer.py` — Deterministic Z-score, Haversine travel speed, and policy evaluator
- `src/llm_factory.py` — Gemini + Groq structured-output failover factory
- `src/nodes.py` — LangGraph nodes (`score_and_context`, `triage_agent`, `generate_sar`)
- `src/graph.py` — `StateGraph`, `SqliteSaver` checkpointer, and HITL resume helpers
- `cli.py` — Command-line interface (`list`, `triage`, `resume`)
- `app.py` — Streamlit HITL Analyst Workbench
- `eval/golden_cases.json` & `eval/run_eval.py` — Golden benchmark evaluation runner
- `tests/test_scorer.py` & `tests/test_graph.py` — `pytest` test suite

## Quickstart (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# Seed database & run tests/eval
python src/setup_db.py
python -m pytest tests/ -v
python eval/run_eval.py
```
## CLI Usage

```powershell
python cli.py list
python cli.py triage --alert-id ALT-1005
python cli.py resume --thread-id thread-ALT-1005 --decision APPROVED --notes "Verified impossible travel"
```

## Streamlit Workbench
```powershell
streamlit run app.py
```

---

### Verification Commands (PowerShell)

```powershell
# Run full test suite and golden evaluation
python -m pytest tests/ -v
python eval/run_eval.py

# Commit Phase 5
git add eval/golden_cases.json eval/run_eval.py tests/test_graph.py .github/workflows/test.yaml README.md
git commit -m "Phase 5: add golden evaluation suite, graph tests, CI workflow, and README"
```

Verification: python -m pytest tests/ -v passes all scorer and graph tests, and python eval/run_eval.py reports 10/10 passed (100.0%)