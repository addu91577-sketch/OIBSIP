"""
Unit tests for Causal Uplift Modeling & Next-Best-Action (NBA) Decision Engine.
"""

import pytest
from src.next_best_action import NextBestActionEngine


@pytest.fixture(scope="module")
def nba_engine():
    return NextBestActionEngine(seed=42)


def test_uplift_summary_structure(nba_engine):
    summary = nba_engine.get_uplift_summary()
    assert summary["total_customers"] > 0
    assert summary["persuadable_population"] > 0
    assert 0.0 < summary["persuadable_share_pct"] < 100.0
    assert len(summary["quadrant_breakdown"]) == 4

    q_names = [q["quadrant"] for q in summary["quadrant_breakdown"]]
    assert "Persuadables" in q_names
    assert "Sure Things" in q_names
    assert "Lost Causes" in q_names
    assert "Sleeping Dogs / Do Not Disturb" in q_names

    total_share = sum(q["population_share_pct"] for q in summary["quadrant_breakdown"])
    assert total_share == pytest.approx(100.0, rel=1e-1)


def test_customer_uplift_profile_existing(nba_engine):
    sample_id = nba_engine.cust_df["Customer ID"].iloc[0]
    profile = nba_engine.get_customer_uplift_profile(sample_id)
    assert profile["customer_id"] == sample_id
    assert profile["causal_quadrant"] in [
        "Persuadables",
        "Sure Things",
        "Lost Causes",
        "Sleeping Dogs / Do Not Disturb",
    ]
    assert -0.5 <= profile["individual_treatment_effect_tau"] <= 1.0
    assert 0.0 < profile["organic_conversion_prob"] < 1.0
    assert 0.0 < profile["treated_conversion_prob"] < 1.0
    assert profile["total_spend"] > 0


def test_customer_uplift_profile_nonexistent(nba_engine):
    with pytest.raises(ValueError, match="not found in portfolio"):
        nba_engine.get_customer_uplift_profile("CUST_NONEXISTENT_99999")


def test_recommend_next_best_action_persuadable(nba_engine):
    # Find a Persuadable customer
    persuadable = nba_engine.cust_df[nba_engine.cust_df["causal_quadrant"] == "Persuadables"].iloc[0]
    rec = nba_engine.recommend_next_best_action(
        customer_id=persuadable["Customer ID"],
        estimated_aov=300.0,
        gross_margin_pct=0.45,
    )
    assert rec["customer_id"] == persuadable["Customer ID"]
    assert rec["causal_quadrant"] == "Persuadables"
    assert rec["top_action_key"] in nba_engine.ACTION_PORTFOLIO
    assert len(rec["action_portfolio_ranking"]) == 5
    assert rec["projected_net_utility"] > 0


def test_recommend_next_best_action_cold_start(nba_engine):
    rec = nba_engine.recommend_next_best_action(
        customer_id=None,
        estimated_aov=200.0,
        gross_margin_pct=0.50,
        recent_touches_count=0,
    )
    assert rec["customer_id"] == "COHORT_SIMULATED"
    assert len(rec["action_portfolio_ranking"]) == 5


def test_recommend_next_best_action_validation(nba_engine):
    with pytest.raises(ValueError, match="estimated_aov must be positive"):
        nba_engine.recommend_next_best_action(estimated_aov=-10.0)

    with pytest.raises(ValueError, match="gross_margin_pct must be between"):
        nba_engine.recommend_next_best_action(gross_margin_pct=1.5)

    with pytest.raises(ValueError, match="recent_touches_count cannot be negative"):
        nba_engine.recommend_next_best_action(recent_touches_count=-2)


def test_optimize_campaign_budget(nba_engine):
    opt = nba_engine.optimize_campaign_budget(
        campaign_budget=5000.0,
        incentive_cost_per_touch=10.0,
        avg_basket_value=300.0,
        gross_margin_pct=0.45,
    )
    assert opt["campaign_budget_dollars"] == 5000.0
    assert opt["max_reach_capacity"] == 500
    assert "strategies" in opt
    assert "mass_marketing" in opt["strategies"]
    assert "propensity_targeting" in opt["strategies"]
    assert "causal_uplift_targeting" in opt["strategies"]

    uplift_strat = opt["strategies"]["causal_uplift_targeting"]
    assert uplift_strat["incremental_buyers"] > 0
    assert uplift_strat["marketing_roi_pct"] > 0
    assert opt["uplift_advantage_vs_propensity"]["incremental_profit_delta"] > 0
