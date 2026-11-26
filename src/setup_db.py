from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from contextlib import closing

load_dotenv(override=True)

DEFAULT_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "data/fraud_triage.sqlite"))
DEFAULT_POLICY_PATH = Path(os.getenv("AML_POLICY_PATH", "data/aml_policy_rules.json"))

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    risk_tier TEXT NOT NULL CHECK (risk_tier IN ('LOW', 'MEDIUM', 'HIGH')),
    avg_tx_amount REAL NOT NULL CHECK (avg_tx_amount > 0),
    std_tx_amount REAL NOT NULL CHECK (std_tx_amount > 0),
    home_country TEXT NOT NULL,
    home_lat REAL NOT NULL,
    home_lon REAL NOT NULL,
    trusted_device_id TEXT NOT NULL,
    account_age_days INTEGER NOT NULL CHECK (account_age_days >= 0)
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    timestamp_utc TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount > 0),
    currency TEXT NOT NULL DEFAULT 'USD',
    merchant_name TEXT NOT NULL,
    merchant_category TEXT NOT NULL,
    channel TEXT NOT NULL CHECK (channel IN ('CARD_ONLINE', 'CARD_POS', 'ACH', 'WIRE')),
    device_id TEXT NOT NULL,
    is_new_device INTEGER NOT NULL CHECK (is_new_device IN (0, 1)),
    is_vpn_or_proxy INTEGER NOT NULL CHECK (is_vpn_or_proxy IN (0, 1)),
    country TEXT NOT NULL,
    city TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    prev_tx_timestamp_utc TEXT,
    prev_lat REAL,
    prev_lon REAL,
    tx_count_24h INTEGER NOT NULL DEFAULT 1 CHECK (tx_count_24h >= 1),
    volume_24h REAL NOT NULL DEFAULT 0.0 CHECK (volume_24h >= 0.0),
    account_balance_after REAL NOT NULL CHECK (account_balance_after >= 0.0)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    tx_id TEXT NOT NULL UNIQUE REFERENCES transactions(tx_id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    created_at_utc TEXT NOT NULL,
    initial_status TEXT NOT NULL DEFAULT 'OPEN' CHECK (initial_status IN ('OPEN', 'TRIAGED')),
    expected_outcome TEXT NOT NULL CHECK (expected_outcome IN ('AUTO_CLOSE', 'STEP_UP_MFA', 'ESCALATE_SAR')),
    scenario_notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triage_decisions (
    alert_id TEXT PRIMARY KEY REFERENCES alerts(alert_id) ON DELETE CASCADE,
    deterministic_score INTEGER NOT NULL,
    triggered_signals_json TEXT NOT NULL,
    cited_policies_json TEXT NOT NULL,
    final_outcome TEXT NOT NULL CHECK (final_outcome IN ('AUTO_CLOSE', 'STEP_UP_MFA', 'ESCALATE_SAR')),
    analyst_decision TEXT CHECK (analyst_decision IN ('APPROVED', 'OVERRIDDEN', 'PENDING')),
    sar_draft_text TEXT,
    updated_at_utc TEXT NOT NULL
);
"""

CUSTOMERS_SEED = [
    ("CUST-101", "Aarav Kulkarni", "LOW", 85.0, 28.0, "IN", 18.5204, 73.8567, "DEV-AK-01", 1420),
    ("CUST-102", "Hannah Lindqvist", "LOW", 140.0, 42.0, "US", 47.6062, -122.3321, "DEV-HL-09", 980),
    ("CUST-103", "Devon Brooks", "MEDIUM", 210.0, 65.0, "US", 41.8781, -87.6298, "DEV-DB-04", 540),
    ("CUST-104", "Priya Nair", "LOW", 95.0, 30.0, "IN", 19.0760, 72.8777, "DEV-PN-02", 1850),
    ("CUST-105", "Mateo Silva", "MEDIUM", 320.0, 95.0, "BR", -23.5505, -46.6333, "DEV-MS-07", 410),
    ("CUST-106", "Claire Moreau", "LOW", 160.0, 45.0, "FR", 48.8566, 2.3522, "DEV-CM-03", 1290),
    ("CUST-107", "Tariq Al-Mansoor", "HIGH", 1950.0, 420.0, "AE", 25.2048, 55.2708, "DEV-TA-05", 760),
    ("CUST-108", "Mei-Ling Zhou", "MEDIUM", 125.0, 35.0, "SG", 1.3521, 103.8198, "DEV-MZ-08", 630),
    ("CUST-109", "Liam O'Connor", "LOW", 110.0, 32.0, "IE", 53.3498, -6.2603, "DEV-LO-01", 1610),
    ("CUST-110", "Sonia Patel", "MEDIUM", 175.0, 50.0, "GB", 51.5072, -0.1276, "DEV-SP-06", 890),
]

TRANSACTIONS_SEED = [
    # 1. Benign recurring grocery purchase (AUTO_CLOSE)
    (
        "TX-9001", "CUST-101", "2026-11-25T09:15:00Z", 92.50, "USD",
        "Fresh Harvest Market", "GROCERY", "CARD_POS", "DEV-AK-01", 0, 0,
        "IN", "Pune", 18.5204, 73.8567,
        "2026-11-24T18:10:00Z", 18.5204, 73.8567,
        2, 168.00, 4820.50,
    ),
    # 2. Benign slight holiday spend drift on trusted device (AUTO_CLOSE)
    (
        "TX-9002", "CUST-102", "2026-11-25T10:05:00Z", 188.00, "USD",
        "Pacific Northwest Books", "BOOKS_MEDIA", "CARD_ONLINE", "DEV-HL-09", 0, 0,
        "US", "Seattle", 47.6062, -122.3321,
        "2026-11-23T14:20:00Z", 47.6062, -122.3321,
        1, 188.00, 9312.00,
    ),
    # 3. High 24h velocity + moderate amount bump (STEP_UP_MFA)
    (
        "TX-9003", "CUST-103", "2026-11-25T11:30:00Z", 365.00, "USD",
        "Metro Electronics Hub", "ELECTRONICS", "CARD_ONLINE", "DEV-DB-04", 0, 0,
        "US", "Chicago", 41.8781, -87.6298,
        "2026-11-25T11:02:00Z", 41.8781, -87.6298,
        6, 1490.00, 2740.25,
    ),
    # 4. New device + VPN + elevated amount z-score (STEP_UP_MFA)
    (
        "TX-9004", "CUST-104", "2026-11-25T12:12:00Z", 245.00, "USD",
        "CloudCompute Credits", "SAAS_SOFTWARE", "CARD_ONLINE", "DEV-UNKNOWN-77", 1, 1,
        "IN", "Mumbai", 19.0760, 72.8777,
        "2026-11-24T08:45:00Z", 19.0760, 72.8777,
        3, 510.00, 6190.80,
    ),
    # 5. Impossible travel: Sao Paulo to London in 45 minutes (ESCALATE_SAR)
    (
        "TX-9005", "CUST-105", "2026-11-25T13:00:00Z", 2450.00, "USD",
        "Luxury Watch Boutique", "JEWELRY_LUXURY", "CARD_POS", "DEV-MS-99", 1, 0,
        "GB", "London", 51.5072, -0.1276,
        "2026-11-25T12:15:00Z", -23.5505, -46.6333,
        4, 3120.00, 1450.00,
    ),
    # 6. Structuring / smurfing burst of round sub-$10k wires/transfers (ESCALATE_SAR)
    (
        "TX-9006", "CUST-106", "2026-11-25T13:40:00Z", 9500.00, "USD",
        "Apex Clearing Transfer", "TRANSFER", "WIRE", "DEV-CM-03", 0, 0,
        "FR", "Paris", 48.8566, 2.3522,
        "2026-11-25T13:10:00Z", 48.8566, 2.3522,
        7, 57000.00, 18400.00,
    ),
    # 7. Benign normal high-tier business wire in home city (AUTO_CLOSE)
    (
        "TX-9007", "CUST-107", "2026-11-25T14:05:00Z", 2150.00, "USD",
        "Gulf Logistics Services", "B2B_SERVICES", "ACH", "DEV-TA-05", 0, 0,
        "AE", "Dubai", 25.2048, 55.2708,
        "2026-11-22T16:00:00Z", 25.2048, 55.2708,
        1, 2150.00, 84200.00,
    ),
    # 8. Severe amount z-score spike + new device (STEP_UP_MFA)
    (
        "TX-9008", "CUST-108", "2026-11-25T14:50:00Z", 390.00, "USD",
        "Marina Bay Duty Free", "TRAVEL_RETAIL", "CARD_POS", "DEV-MZ-NEW2", 1, 0,
        "SG", "Singapore", 1.3521, 103.8198,
        "2026-11-24T19:30:00Z", 1.3521, 103.8198,
        2, 520.00, 3910.40,
    ),
    # 9. Impossible travel + VPN + high velocity (ESCALATE_SAR)
    (
        "TX-9009", "CUST-109", "2026-11-25T15:20:00Z", 4800.00, "USD",
        "Offshore Bullion Desk", "COMMODITIES", "WIRE", "DEV-LO-PROXY", 1, 1,
        "US", "New York", 40.7128, -74.0060,
        "2026-11-25T14:55:00Z", 53.3498, -6.2603,
        6, 14200.00, 950.00,
    ),
    # 10. Benign small transit/coffee charge (AUTO_CLOSE)
    (
        "TX-9010", "CUST-110", "2026-11-25T16:00:00Z", 168.00, "USD",
        "Thames Express Rail", "TRANSPORT", "CARD_POS", "DEV-SP-06", 0, 0,
        "GB", "London", 51.5072, -0.1276,
        "2026-11-24T09:10:00Z", 51.5072, -0.1276,
        2, 214.00, 5430.75,
    ),
]

ALERTS_SEED = [
    ("ALT-1001", "TX-9001", "CUST-101", "2026-11-25T09:15:05Z", "OPEN", "AUTO_CLOSE", "Routine local grocery purchase on trusted device."),
    ("ALT-1002", "TX-9002", "CUST-102", "2026-11-25T10:05:05Z", "OPEN", "AUTO_CLOSE", "Minor spend variance within normal baseline."),
    ("ALT-1003", "TX-9003", "CUST-103", "2026-11-25T11:30:05Z", "OPEN", "STEP_UP_MFA", "6 transactions in 24h with elevated electronics purchase."),
    ("ALT-1004", "TX-9004", "CUST-104", "2026-11-25T12:12:05Z", "OPEN", "STEP_UP_MFA", "New device + VPN on online software purchase."),
    ("ALT-1005", "TX-9005", "CUST-105", "2026-11-25T13:00:05Z", "OPEN", "ESCALATE_SAR", "Impossible travel from Sao Paulo to London in 45 minutes."),
    ("ALT-1006", "TX-9006", "CUST-106", "2026-11-25T13:40:05Z", "OPEN", "ESCALATE_SAR", "Repeated $9,500 wire transfers indicating potential structuring."),
    ("ALT-1007", "TX-9007", "CUST-107", "2026-11-25T14:05:05Z", "OPEN", "AUTO_CLOSE", "Normal B2B ACH within high-tier customer baseline."),
    ("ALT-1008", "TX-9008", "CUST-108", "2026-11-25T14:50:05Z", "OPEN", "STEP_UP_MFA", "Amount z-score spike above 3.0 on new device."),
    ("ALT-1009", "TX-9009", "CUST-109", "2026-11-25T15:20:05Z", "OPEN", "ESCALATE_SAR", "Impossible travel from Dublin to New York + VPN + high velocity."),
    ("ALT-1010", "TX-9010", "CUST-110", "2026-11-25T16:00:05Z", "OPEN", "AUTO_CLOSE", "Normal transit spend on trusted device."),
]


def validate_policy_rules(policy_path: Path = DEFAULT_POLICY_PATH) -> dict:
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    with policy_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rules = data.get("rules", [])
    if not rules:
        raise ValueError("Policy file contains no rules.")
    seen_ids = set()
    for rule in rules:
        pid = rule.get("policy_id")
        if not pid or pid in seen_ids:
            raise ValueError(f"Duplicate or missing policy_id: {pid}")
        seen_ids.add(pid)
    return data


def init_database(db_path: Path = DEFAULT_DB_PATH, policy_path: Path = DEFAULT_POLICY_PATH) -> None:
    policy_data = validate_policy_rules(policy_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            """
            INSERT OR REPLACE INTO customers (
                customer_id, full_name, risk_tier, avg_tx_amount, std_tx_amount,
                home_country, home_lat, home_lon, trusted_device_id, account_age_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            CUSTOMERS_SEED,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO transactions (
                tx_id, customer_id, timestamp_utc, amount, currency,
                merchant_name, merchant_category, channel, device_id,
                is_new_device, is_vpn_or_proxy, country, city, lat, lon,
                prev_tx_timestamp_utc, prev_lat, prev_lon,
                tx_count_24h, volume_24h, account_balance_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            TRANSACTIONS_SEED,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO alerts (
                alert_id, tx_id, customer_id, created_at_utc,
                initial_status, expected_outcome, scenario_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ALERTS_SEED,
        )
        conn.commit()

    print(
        f"Initialized {db_path} (policy v{policy_data['policy_version']}): "
        f"{len(CUSTOMERS_SEED)} customers, {len(TRANSACTIONS_SEED)} transactions, {len(ALERTS_SEED)} alerts."
    )


if __name__ == "__main__":
    init_database()