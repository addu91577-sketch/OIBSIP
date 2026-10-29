"""
Unit and integration tests for Customer Lifetime Value (CLV) & Churn Engine.
"""

import pytest
from src.clv_churn import CustomerLifetimeValueEngine


@pytest.fixture
def clv_engine():
    return CustomerLifetimeValueEngine()


def test_customer_df_shape_and_features(clv_engine):
    df = clv_engine.customer_df
    assert len(df) == 1000
    expected_cols = {
        "customer_id",
        "recency_days",
        "historical_spend",
        "prob_alive",
        "prob_churn",
        "churn_tier",
        "projected_12m_clv",
        "is_vip",
    }
    assert expected_cols.issubset(set(df.columns))


def test_churn_probability_decay(clv_engine):
    prob_0 = clv_engine.calculate_churn_probability(0)
    prob_90 = clv_engine.calculate_churn_probability(90)
    prob_365 = clv_engine.calculate_churn_probability(365)

    assert prob_0 == 0.0
    assert 0.40 < prob_90 < 0.60
    assert prob_365 > 0.90
    assert prob_0 < prob_90 < prob_365


def test_predict_customer_clv_tiers(clv_engine):
    res_recent = clv_engine.predict_customer_clv(recency_days=10, aov=500.0)
    assert res_recent["churn_tier"] == "Active (Low Risk)"
    assert res_recent["projected_clv"] > 150.0

    res_lapsed = clv_engine.predict_customer_clv(recency_days=300, aov=500.0)
    assert res_lapsed["churn_tier"] == "Lapsed (Defected)"
    assert res_lapsed["projected_clv"] < res_recent["projected_clv"]


def test_predict_customer_clv_invalid(clv_engine):
    with pytest.raises(ValueError, match="recency_days cannot be negative"):
        clv_engine.predict_customer_clv(recency_days=-5, aov=100.0)

    with pytest.raises(ValueError, match="aov must be strictly positive"):
        clv_engine.predict_customer_clv(recency_days=10, aov=0.0)


def test_audit_customer_equity(clv_engine):
    audit = clv_engine.audit_customer_equity()
    assert "customer_base" in audit
    assert audit["customer_base"]["total_customers"] == 1000
    assert audit["customer_base"]["total_historical_revenue"] == 456000.0
    assert audit["customer_base"]["total_projected_12m_customer_equity"] > 50000.0
    assert len(audit["churn_tier_distribution"]) == 4
    assert audit["vip_early_warning"]["total_vips"] == 202
    assert audit["vip_early_warning"]["vip_revenue_at_risk"] > 0


def test_get_at_risk_vips(clv_engine):
    vips = clv_engine.get_at_risk_vips(min_spend=1000.0, limit=10)
    assert len(vips) <= 10
    for vip in vips:
        assert vip["historical_spend"] >= 1000.0
        assert vip["prob_churn_pct"] >= 50.0
        assert "customer_id" in vip
