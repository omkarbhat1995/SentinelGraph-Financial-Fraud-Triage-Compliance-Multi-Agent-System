from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.scorer import (
    DeterministicScorer,
    compute_zscore,
    haversine_distance_km,
    calculate_travel_speed,
)
from src.setup_db import init_database

TEST_DB_PATH = Path("data/test_fraud_triage.sqlite")
TEST_POLICY_PATH = Path("data/aml_policy_rules.json")


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_database(db_path=TEST_DB_PATH, policy_path=TEST_POLICY_PATH)
    yield
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)

@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db_file = tmp_path_factory.mktemp("db") / "test_fraud_triage.sqlite"
    init_database(db_path=db_file, policy_path=TEST_POLICY_PATH)
    return db_file

def test_haversine_and_speed():
    # Sao Paulo (-23.5505, -46.6333) to London (51.5072, -0.1276) ~ 9,400 km
    dist = haversine_distance_km(-23.5505, -46.6333, 51.5072, -0.1276)
    assert 9300 < dist < 9600

    speed = calculate_travel_speed(
        -23.5505,
        -46.6333,
        "2026-11-25T12:15:00Z",
        51.5072,
        -0.1276,
        "2026-11-25T13:00:00Z",
    )
    assert speed is not None
    assert speed > 10000.0  # ~12,500 km/h impossible travel speed


def test_compute_zscore():
    assert compute_zscore(100.0, 100.0, 20.0) == 0.0
    assert compute_zscore(160.0, 100.0, 20.0) == 3.0
    assert compute_zscore(50.0, 100.0, 0.0) == 0.0


def test_scorer_auto_close_scenario():
    scorer = DeterministicScorer(db_path=TEST_DB_PATH, policy_path=TEST_POLICY_PATH)
    alert, cust, tx = scorer.fetch_alert_context("ALT-1001")
    result = scorer.evaluate(alert, cust, tx)

    assert result.recommended_outcome == "AUTO_CLOSE"
    assert result.deterministic_score <= 24
    assert not result.has_sar_trigger
    assert "POL-BEN-006" in result.cited_policies


def test_scorer_step_up_mfa_scenario():
    scorer = DeterministicScorer(db_path=TEST_DB_PATH, policy_path=TEST_POLICY_PATH)
    alert, cust, tx = scorer.fetch_alert_context("ALT-1003")
    result = scorer.evaluate(alert, cust, tx)

    assert result.recommended_outcome == "STEP_UP_MFA"
    assert "POL-VEL-001" in result.cited_policies


def test_scorer_impossible_travel_escalate_sar():
    scorer = DeterministicScorer(db_path=TEST_DB_PATH, policy_path=TEST_POLICY_PATH)
    alert, cust, tx = scorer.fetch_alert_context("ALT-1005")
    result = scorer.evaluate(alert, cust, tx)

    assert result.recommended_outcome == "ESCALATE_SAR"
    assert result.has_sar_trigger
    assert "POL-GEO-003" in result.cited_policies
    assert result.travel_speed_kmh is not None and result.travel_speed_kmh > 800.0


def test_scorer_structuring_escalate_sar():
    scorer = DeterministicScorer(db_path=TEST_DB_PATH, policy_path=TEST_POLICY_PATH)
    alert, cust, tx = scorer.fetch_alert_context("ALT-1006")
    result = scorer.evaluate(alert, cust, tx)

    assert result.recommended_outcome == "ESCALATE_SAR"
    assert result.has_sar_trigger
    assert "POL-AML-005" in result.cited_policies