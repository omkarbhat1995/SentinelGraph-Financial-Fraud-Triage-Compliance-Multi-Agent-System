from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import closing

from dotenv import load_dotenv

from src.schemas import (
    AlertRecord,
    CustomerRecord,
    ScoringResult,
    TransactionRecord,
    TriggeredSignal,
    TriageOutcome,
)

load_dotenv(override=True)

DEFAULT_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "data/fraud_triage.sqlite"))
DEFAULT_POLICY_PATH = Path(os.getenv("AML_POLICY_PATH", "data/aml_policy_rules.json"))

AUTO_CLOSE_MAX_SCORE = int(os.getenv("AUTO_CLOSE_MAX_SCORE", "24"))
STEP_UP_MAX_SCORE = int(os.getenv("STEP_UP_MAX_SCORE", "64"))
HIGH_AMOUNT_ZSCORE_THRESHOLD = float(os.getenv("HIGH_AMOUNT_ZSCORE_THRESHOLD", "3.0"))
VELOCITY_24H_THRESHOLD = int(os.getenv("VELOCITY_24H_THRESHOLD", "5"))
IMPOSSIBLE_TRAVEL_KMH = float(os.getenv("IMPOSSIBLE_TRAVEL_KMH", "800.0"))


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Great-Circle distance between two geographical points in km."""
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def calculate_travel_speed(
    lat1: Optional[float],
    lon1: Optional[float],
    t1_utc: Optional[str],
    lat2: float,
    lon2: float,
    t2_utc: str,
) -> Optional[float]:
    """Calculate speed in km/h between two timestamped coordinates."""
    if lat1 is None or lon1 is None or t1_utc is None:
        return None

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        time1 = datetime.strptime(t1_utc, fmt).replace(tzinfo=timezone.utc)
        time2 = datetime.strptime(t2_utc, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None

    hours = (time2 - time1).total_seconds() / 3600.0
    if hours <= 0:
        return 99999.0  # Instantaneous jump indicator

    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    return dist_km / hours


def compute_zscore(amount: float, mean: float, std: float) -> float:
    """Compute exact amount Z-score against customer historical baseline."""
    if std <= 0:
        return 0.0
    return (amount - mean) / std


class DeterministicScorer:
    """Engine that computes exact risk metrics and evaluates AML policy rules."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        policy_path: Path = DEFAULT_POLICY_PATH,
    ):
        self.db_path = db_path
        self.policy_path = policy_path
        self.policies = self._load_policies()

    def _load_policies(self) -> Dict[str, Dict[str, Any]]:
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file missing: {self.policy_path}")
        with self.policy_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {rule["policy_id"]: rule for rule in data.get("rules", [])}

    def fetch_alert_context(
        self, alert_id: str
    ) -> Tuple[AlertRecord, CustomerRecord, TransactionRecord]:
        """Fetch alert, customer, and transaction records cleanly from SQLite."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            alert_row = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if not alert_row:
                raise ValueError(f"Alert {alert_id} not found.")

            tx_row = conn.execute(
                "SELECT * FROM transactions WHERE tx_id = ?", (alert_row["tx_id"],)
            ).fetchone()
            if not tx_row:
                raise ValueError(f"Transaction {alert_row['tx_id']} not found.")

            cust_row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (alert_row["customer_id"],)
            ).fetchone()
            if not cust_row:
                raise ValueError(f"Customer {alert_row['customer_id']} not found.")

        alert = AlertRecord(**dict(alert_row))
        customer = CustomerRecord(**dict(cust_row))
        transaction = TransactionRecord(
            tx_id=tx_row["tx_id"],
            customer_id=tx_row["customer_id"],
            timestamp_utc=tx_row["timestamp_utc"],
            amount=tx_row["amount"],
            currency=tx_row["currency"],
            merchant_name=tx_row["merchant_name"],
            merchant_category=tx_row["merchant_category"],
            channel=tx_row["channel"],
            device_id=tx_row["device_id"],
            is_new_device=bool(tx_row["is_new_device"]),
            is_vpn_or_proxy=bool(tx_row["is_vpn_or_proxy"]),
            country=tx_row["country"],
            city=tx_row["city"],
            lat=tx_row["lat"],
            lon=tx_row["lon"],
            prev_tx_timestamp_utc=tx_row["prev_tx_timestamp_utc"],
            prev_lat=tx_row["prev_lat"],
            prev_lon=tx_row["prev_lon"],
            tx_count_24h=tx_row["tx_count_24h"],
            volume_24h=tx_row["volume_24h"],
            account_balance_after=tx_row["account_balance_after"],
        )
        return alert, customer, transaction

    def evaluate(
        self,
        alert: AlertRecord,
        customer: CustomerRecord,
        tx: TransactionRecord,
    ) -> ScoringResult:
        """Run zero-hallucination deterministic scoring rules against context."""
        triggered_signals: List[TriggeredSignal] = []

        # 1. Math calculations
        amount_zscore = round(
            compute_zscore(tx.amount, customer.avg_tx_amount, customer.std_tx_amount), 2
        )
        travel_speed = calculate_travel_speed(
            tx.prev_lat, tx.prev_lon, tx.prev_tx_timestamp_utc, tx.lat, tx.lon, tx.timestamp_utc
        )

        # 2. Rule evaluation
        # Rule POL-VEL-001: 24h Velocity
        if tx.tx_count_24h >= VELOCITY_24H_THRESHOLD:
            rule = self.policies["POL-VEL-001"]
            triggered_signals.append(
                TriggeredSignal(
                    signal_id=f"SIG-VEL-{tx.tx_id}",
                    policy_id="POL-VEL-001",
                    title=rule["title"],
                    severity=rule["severity"],
                    score_weight=rule["score_weight"],
                    sar_trigger=rule["sar_trigger"],
                    metrics={"tx_count_24h": tx.tx_count_24h, "volume_24h": tx.volume_24h},
                )
            )

        # Rule POL-AMT-002: Amount Z-Score Anomaly
        if amount_zscore >= HIGH_AMOUNT_ZSCORE_THRESHOLD:
            rule = self.policies["POL-AMT-002"]
            triggered_signals.append(
                TriggeredSignal(
                    signal_id=f"SIG-AMT-{tx.tx_id}",
                    policy_id="POL-AMT-002",
                    title=rule["title"],
                    severity=rule["severity"],
                    score_weight=rule["score_weight"],
                    sar_trigger=rule["sar_trigger"],
                    metrics={"amount_zscore": amount_zscore, "amount": tx.amount},
                )
            )

        # Rule POL-GEO-003: Impossible Travel
        if travel_speed is not None and travel_speed >= IMPOSSIBLE_TRAVEL_KMH:
            rule = self.policies["POL-GEO-003"]
            triggered_signals.append(
                TriggeredSignal(
                    signal_id=f"SIG-GEO-{tx.tx_id}",
                    policy_id="POL-GEO-003",
                    title=rule["title"],
                    severity=rule["severity"],
                    score_weight=rule["score_weight"],
                    sar_trigger=rule["sar_trigger"],
                    metrics={"travel_speed_kmh": round(travel_speed, 2)},
                )
            )

        # Rule POL-DEV-004: Device & Proxy Risk
        if tx.is_new_device or tx.is_vpn_or_proxy:
            rule = self.policies["POL-DEV-004"]
            triggered_signals.append(
                TriggeredSignal(
                    signal_id=f"SIG-DEV-{tx.tx_id}",
                    policy_id="POL-DEV-004",
                    title=rule["title"],
                    severity=rule["severity"],
                    score_weight=rule["score_weight"],
                    sar_trigger=rule["sar_trigger"],
                    metrics={
                        "is_new_device": tx.is_new_device,
                        "is_vpn_or_proxy": tx.is_vpn_or_proxy,
                    },
                )
            )

        # Rule POL-AML-005: Structuring / Smurfing Pattern
        if tx.tx_count_24h >= 5 and tx.amount >= 9000.0 and tx.channel in ("WIRE", "ACH"):
            rule = self.policies["POL-AML-005"]
            triggered_signals.append(
                TriggeredSignal(
                    signal_id=f"SIG-AML-{tx.tx_id}",
                    policy_id="POL-AML-005",
                    title=rule["title"],
                    severity=rule["severity"],
                    score_weight=rule["score_weight"],
                    sar_trigger=rule["sar_trigger"],
                    metrics={
                        "amount": tx.amount,
                        "tx_count_24h": tx.tx_count_24h,
                        "volume_24h": tx.volume_24h,
                    },
                )
            )

        # Rule POL-BEN-006: Benign Spend (Applied only if no critical signals triggered)
        if (
            not triggered_signals
            and tx.device_id == customer.trusted_device_id
            and amount_zscore < 2.0
        ):
            rule = self.policies["POL-BEN-006"]
            triggered_signals.append(
                TriggeredSignal(
                    signal_id=f"SIG-BEN-{tx.tx_id}",
                    policy_id="POL-BEN-006",
                    title=rule["title"],
                    severity=rule["severity"],
                    score_weight=rule["score_weight"],
                    sar_trigger=rule["sar_trigger"],
                    metrics={"amount_zscore": amount_zscore},
                )
            )

        # Total deterministic score calculation
        total_score = max(0, sum(sig.score_weight for sig in triggered_signals))
        has_sar_trigger = any(sig.sar_trigger for sig in triggered_signals)

        # Outcome determination
        if has_sar_trigger or total_score > STEP_UP_MAX_SCORE:
            recommended_outcome: TriageOutcome = "ESCALATE_SAR"
        elif total_score > AUTO_CLOSE_MAX_SCORE:
            recommended_outcome = "STEP_UP_MFA"
        else:
            recommended_outcome = "AUTO_CLOSE"

        cited_policies = sorted(list({sig.policy_id for sig in triggered_signals}))

        return ScoringResult(
            alert_id=alert.alert_id,
            tx_id=tx.tx_id,
            customer_id=customer.customer_id,
            deterministic_score=total_score,
            recommended_outcome=recommended_outcome,
            has_sar_trigger=has_sar_trigger,
            triggered_signals=triggered_signals,
            cited_policies=cited_policies,
            amount_zscore=amount_zscore,
            travel_speed_kmh=round(travel_speed, 2) if travel_speed is not None else None,
            tx_count_24h=tx.tx_count_24h,
        )