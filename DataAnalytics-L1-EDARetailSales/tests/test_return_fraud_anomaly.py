"""
Unit tests for Retail Return Merchandise Authorization (RMA) & Benford's Law Digital Surveillance Engine.
"""

import pytest
from src.return_fraud_anomaly import ReturnFraudAnomalyEngine


@pytest.fixture
def fraud_engine():
    return ReturnFraudAnomalyEngine()


def test_benford_law_analysis_basic(fraud_engine):
    res = fraud_engine.run_benford_law_analysis("Total Amount")
    assert res["target_column"] == "Total Amount"
    assert res["sample_size"] == 1000
    assert res["degrees_of_freedom"] == 8
    assert "chi_square_statistic" in res
    assert "p_value" in res
    assert res["mean_absolute_deviation_mad"] >= 0.0
    assert res["nigrini_conformity_status"] in [
        "Close Conformity",
        "Acceptable Conformity",
        "Marginally Acceptable",
        "Non-Conforming",
    ]
    assert len(res["benford_distribution_table"]) == 9


def test_benford_law_invalid_column(fraud_engine):
    with pytest.raises(ValueError, match="does not exist"):
        fraud_engine.run_benford_law_analysis("NonExistentColumn")


def test_profile_return_fraud_cohorts(fraud_engine):
    profile = fraud_engine.profile_return_fraud_cohorts()
    assert profile["total_transactions"] == 1000
    assert profile["total_returned_items"] > 0
    assert profile["total_returned_value_usd"] > 0
    assert profile["overall_dollar_return_rate_pct"] > 0

    wardrobing = profile["wardrobing_fraud"]
    assert wardrobing["incident_count"] >= 0
    assert wardrobing["dollar_loss_usd"] >= 0

    archetypes = profile["archetype_distribution"]
    assert "Legitimate Defect/Fit" in archetypes
    assert "Opportunistic Wardrober" in archetypes
    assert "Serial Return Abuser" in archetypes

    cats = profile["category_return_summary"]
    assert "Clothing" in cats
    assert "Electronics" in cats
    assert "Beauty" in cats


def test_score_customer_return_risk(fraud_engine):
    # Test valid customer
    cust_id = "CUST001"
    score_res = fraud_engine.score_customer_return_risk(cust_id)
    assert score_res["customer_id"] == cust_id
    assert 0.0 <= score_res["composite_abuse_risk_score"] <= 1.0
    assert "Risk" in score_res["risk_tier"]
    assert "authorization_protocol" in score_res
    assert "frontline_staff_action" in score_res


def test_score_customer_return_risk_not_found(fraud_engine):
    with pytest.raises(ValueError, match="not found in retail registry"):
        fraud_engine.score_customer_return_risk("INVALID_CUSTOMER_9999")


def test_simulate_policy_intervention(fraud_engine):
    sim = fraud_engine.simulate_policy_intervention(
        restocking_fee_pct=0.10,
        return_window_days=14,
        credit_only_for_high_risk=True,
    )
    fin = sim["financial_impact"]
    assert fin["prevented_fraud_and_wardrobing_usd"] > 0
    assert fin["restocking_fee_recovered_income_usd"] > 0
    assert fin["net_annual_profit_uplift_usd"] > 0
    assert "executive_verdict" in sim


def test_simulate_policy_intervention_validation(fraud_engine):
    with pytest.raises(ValueError, match="restocking_fee_pct must be between"):
        fraud_engine.simulate_policy_intervention(restocking_fee_pct=0.50)

    with pytest.raises(ValueError, match="return_window_days must be one of"):
        fraud_engine.simulate_policy_intervention(return_window_days=99)
