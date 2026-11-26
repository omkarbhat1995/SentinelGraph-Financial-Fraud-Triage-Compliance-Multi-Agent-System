from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# Strict Enum Types
TriageOutcome = Literal["AUTO_CLOSE", "STEP_UP_MFA", "ESCALATE_SAR"]
RiskTier = Literal["LOW", "MEDIUM", "HIGH"]
SeverityLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
AnalystDecision = Literal["APPROVED", "OVERRIDDEN", "PENDING"]


class CompactBaseModel(BaseModel):
    """Base Pydantic model enforcing compact whitespace-stripped JSON serialization."""

    model_config = ConfigDict(populate_by_name=True)

    def to_compact_json(self) -> str:
        return json.dumps(self.model_dump(exclude_none=True), separators=(",", ":"))


class CustomerRecord(CompactBaseModel):
    customer_id: str
    full_name: str
    risk_tier: RiskTier
    avg_tx_amount: float
    std_tx_amount: float
    home_country: str
    home_lat: float
    home_lon: float
    trusted_device_id: str
    account_age_days: int


class TransactionRecord(CompactBaseModel):
    tx_id: str
    customer_id: str
    timestamp_utc: str
    amount: float
    currency: str = "USD"
    merchant_name: str
    merchant_category: str
    channel: str
    device_id: str
    is_new_device: bool
    is_vpn_or_proxy: bool
    country: str
    city: str
    lat: float
    lon: float
    prev_tx_timestamp_utc: Optional[str] = None
    prev_lat: Optional[float] = None
    prev_lon: Optional[float] = None
    tx_count_24h: int
    volume_24h: float
    account_balance_after: float


class AlertRecord(CompactBaseModel):
    alert_id: str
    tx_id: str
    customer_id: str
    created_at_utc: str
    initial_status: str
    expected_outcome: TriageOutcome
    scenario_notes: str


class TriggeredSignal(CompactBaseModel):
    signal_id: str
    policy_id: str
    title: str
    severity: SeverityLevel
    score_weight: int
    sar_trigger: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ScoringResult(CompactBaseModel):
    alert_id: str
    tx_id: str
    customer_id: str
    deterministic_score: int
    recommended_outcome: TriageOutcome
    has_sar_trigger: bool
    triggered_signals: List[TriggeredSignal]
    cited_policies: List[str]
    amount_zscore: float
    travel_speed_kmh: Optional[float] = None
    tx_count_24h: int


class TriageAgentDecision(CompactBaseModel):
    alert_id: str
    proposed_outcome: TriageOutcome
    confidence_score: float = Field(ge=0.0, le=1.0)
    primary_policy_cited: str
    justification: str
    cited_signal_ids: List[str]


class SARDraft(CompactBaseModel):
    alert_id: str
    customer_id: str
    tx_id: str
    narrative: str
    cited_policies: List[str]
    cited_signal_ids: List[str]
    recommended_regulatory_action: str
    prepared_timestamp_utc: str


class TriageState(CompactBaseModel):
    alert_id: str
    customer: Optional[CustomerRecord] = None
    transaction: Optional[TransactionRecord] = None
    scoring_result: Optional[ScoringResult] = None
    agent_decision: Optional[TriageAgentDecision] = None
    final_outcome: Optional[TriageOutcome] = None
    sar_draft: Optional[SARDraft] = None
    analyst_decision: AnalystDecision = "PENDING"
    analyst_notes: Optional[str] = None
    execution_trace: List[str] = Field(default_factory=list)