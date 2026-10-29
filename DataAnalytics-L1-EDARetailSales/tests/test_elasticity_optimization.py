"""
Unit and integration tests for Price Elasticity Optimization Engine.
"""

import pytest
from src.elasticity_optimization import PriceElasticityOptimizer


@pytest.fixture
def optimizer():
    return PriceElasticityOptimizer()


def test_overall_elasticity_summary_structure(optimizer):
    summary = optimizer.get_overall_elasticity_summary()
    assert "catalog_overall" in summary
    assert "categories" in summary
    assert "baseline_gross_margin_pct" in summary
    assert set(summary["categories"].keys()) == {"Beauty", "Clothing", "Electronics"}


def test_category_elasticity_computation(optimizer):
    for cat in ["Clothing", "Electronics", "Beauty"]:
        res = optimizer.compute_category_elasticity(cat)
        assert res["category"] == cat
        assert res["transaction_count"] > 0
        assert "ped_coefficient" in res
        assert "r_squared" in res
        assert "classification" in res
        assert len(res["price_tiers"]) == 5
        assert len(res["arc_elasticities"]) == 4
        assert len(res["breakeven_markdown_requirements"]) > 0


def test_invalid_category_raises(optimizer):
    with pytest.raises(ValueError, match="no transaction records"):
        optimizer.compute_category_elasticity("NonExistentCategory")


def test_markdown_simulation_value_accretive_or_dilutive(optimizer):
    # Inelastic product category with low volume lift should be margin dilutive
    res = optimizer.simulate_markdown_pricing(
        category="Electronics",
        discount_pct=0.15,
        expected_volume_lift_pct=0.05,
        gross_margin_pct=0.40,
    )
    assert res["financial_impact"]["status"] in [
        "Value Accretive",
        "Revenue Accretive / Margin Dilutive",
        "Value Destructive",
    ]
    assert res["financial_impact"]["profit_delta"] < 0  # Inelastic markdown destroys margin
    assert res["baseline"]["orders"] == 342


def test_markdown_simulation_validation_errors(optimizer):
    with pytest.raises(ValueError, match="discount_pct must be between"):
        optimizer.simulate_markdown_pricing("Beauty", discount_pct=1.2, expected_volume_lift_pct=0.2)

    with pytest.raises(ValueError, match="expected_volume_lift_pct cannot be less"):
        optimizer.simulate_markdown_pricing("Beauty", discount_pct=0.10, expected_volume_lift_pct=-1.5)


def test_optimize_price_points(optimizer):
    opt = optimizer.optimize_price_points("Clothing")
    assert opt["category"] == "Clothing"
    assert opt["profit_maximizing_tier"] in [25.0, 30.0, 50.0, 300.0, 500.0]
    assert len(opt["tier_evaluations"]) == 5


def test_compute_cohort_elasticity(optimizer):
    cohorts = optimizer.compute_cohort_elasticity()
    assert len(cohorts) >= 3
    for c in cohorts:
        assert "age_cohort" in c
        assert "ped_slope" in c
        assert "classification" in c
