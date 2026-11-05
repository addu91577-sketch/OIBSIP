"""
Unit tests for the Dynamic Real-Time Pricing & Multi-Armed Bandit (MAB) Rate Optimizer.
"""

import pytest
from src.dynamic_pricing import DynamicPricingEngine


@pytest.fixture(scope="module")
def pricing_engine():
    return DynamicPricingEngine(seed=42)


def test_get_candidate_arms_structure(pricing_engine):
    res = pricing_engine.get_candidate_arms()
    assert "categories" in res
    assert "algorithms_supported" in res
    assert len(res["algorithms_supported"]) == 3

    cats = res["categories"]
    for c in ["Beauty", "Clothing", "Electronics"]:
        assert c in cats
        cat_info = cats[c]
        assert "arms" in cat_info
        assert "oracle_optimal_price" in cat_info
        assert "oracle_expected_revenue" in cat_info
        assert len(cat_info["arms"]) == 5
        for arm in cat_info["arms"]:
            assert arm["price_arm"] > 0
            assert 0.0 < arm["conversion_probability"] < 1.0
            assert arm["expected_revenue_per_impression"] > 0


def test_simulate_bandit_learning_thompson(pricing_engine):
    sim = pricing_engine.simulate_bandit_learning(
        category="Electronics",
        algorithm="thompson_sampling",
        n_rounds=200,
    )
    assert sim["category"] == "Electronics"
    assert sim["algorithm"] == "thompson_sampling"
    assert sim["total_rounds"] == 200
    assert sim["cumulative_bandit_revenue"] > 0
    assert sim["cumulative_oracle_revenue"] >= sim["cumulative_bandit_revenue"] * 0.7
    assert sim["total_regret"] >= 0
    assert len(sim["arm_statistics"]) == 5
    assert len(sim["learning_curve"]["rounds"]) > 0


def test_simulate_bandit_learning_ucb1(pricing_engine):
    sim = pricing_engine.simulate_bandit_learning(
        category="Clothing",
        algorithm="ucb1",
        n_rounds=150,
    )
    assert sim["algorithm"] == "ucb1"
    assert sim["cumulative_bandit_revenue"] > 0
    assert sim["total_regret"] >= 0


def test_simulate_bandit_learning_epsilon_greedy(pricing_engine):
    sim = pricing_engine.simulate_bandit_learning(
        category="Beauty",
        algorithm="epsilon_greedy",
        n_rounds=100,
        epsilon=0.15,
    )
    assert sim["algorithm"] == "epsilon_greedy"
    assert sim["cumulative_bandit_revenue"] > 0


def test_simulate_bandit_invalid_parameters(pricing_engine):
    with pytest.raises(ValueError, match="Invalid category"):
        pricing_engine.simulate_bandit_learning(category="Automotive")

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        pricing_engine.simulate_bandit_learning(category="Electronics", algorithm="q_learning")

    with pytest.raises(ValueError, match="n_rounds must be between"):
        pricing_engine.simulate_bandit_learning(category="Electronics", n_rounds=5)


def test_calculate_dynamic_quote_scarcity_surge(pricing_engine):
    # Low stock below safety stock should trigger scarcity surge
    quote = pricing_engine.calculate_dynamic_quote(
        category="Electronics",
        base_price=300.0,
        current_stock=5,
        safety_stock=25,
        demand_velocity_ratio=1.3,
    )
    assert quote["category"] == "Electronics"
    assert quote["recommended_dynamic_price"] > 300.0
    assert quote["price_delta_dollars"] > 0
    assert quote["multipliers"]["scarcity_multiplier"] > 1.0
    assert quote["multipliers"]["demand_surge_multiplier"] > 1.0
    assert quote["stock_context"]["stock_status"] == "Depleted / Sub-Safety-Stock"


def test_calculate_dynamic_quote_surplus_discount(pricing_engine):
    # Abundant surplus stock should trigger clearance discount
    quote = pricing_engine.calculate_dynamic_quote(
        category="Clothing",
        base_price=50.0,
        current_stock=100,
        safety_stock=20,
        demand_velocity_ratio=0.8,
    )
    assert quote["category"] == "Clothing"
    assert quote["recommended_dynamic_price"] < 50.0
    assert quote["price_delta_dollars"] < 0
    assert quote["multipliers"]["scarcity_multiplier"] < 1.0
    assert quote["stock_context"]["stock_status"] == "Surplus Inventory / Markdown Pressure"


def test_calculate_dynamic_quote_competitor_clamping(pricing_engine):
    quote = pricing_engine.calculate_dynamic_quote(
        category="Beauty",
        base_price=50.0,
        current_stock=30,
        safety_stock=20,
        demand_velocity_ratio=1.0,
        competitor_price=42.0,
    )
    assert quote["guardrails"]["competitor_constrained"] is True
    assert quote["recommended_dynamic_price"] <= 42.0 * 1.15 + 0.01


def test_calculate_dynamic_quote_validation_errors(pricing_engine):
    with pytest.raises(ValueError, match="Invalid category"):
        pricing_engine.calculate_dynamic_quote(category="Luxury", base_price=100.0, current_stock=10)

    with pytest.raises(ValueError, match="base_price must be strictly positive"):
        pricing_engine.calculate_dynamic_quote(category="Beauty", base_price=-10.0, current_stock=10)

    with pytest.raises(ValueError, match="current_stock cannot be negative"):
        pricing_engine.calculate_dynamic_quote(category="Beauty", base_price=50.0, current_stock=-5)
