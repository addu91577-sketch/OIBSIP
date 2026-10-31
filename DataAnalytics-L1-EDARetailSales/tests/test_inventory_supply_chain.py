"""
Unit tests validating supply chain & inventory optimization engine.
Covers daily demand velocity, safety stock calibration, EOQ, stockout risks, and ABC-FSN stratification.
"""

import pytest
from src.inventory_supply_chain import InventorySupplyChainEngine


@pytest.fixture(scope="module")
def inv_engine():
    return InventorySupplyChainEngine()


def test_daily_demand_metrics_catalog_and_category(inv_engine):
    """Verifies baseline daily demand aggregation across catalog and specific categories."""
    summary = inv_engine.get_daily_demand_metrics()
    assert summary["category"] == "All Categories"
    assert summary["total_units"] > 0
    assert summary["daily_mean_units"] > 0.0
    assert summary["daily_std_units"] > 0.0
    assert summary["daily_variance"] == pytest.approx(summary["daily_std_units"] ** 2, rel=1e-2)
    assert summary["active_days"] > 300

    # Test category specific
    cat_summary = inv_engine.get_daily_demand_metrics(category="Electronics")
    assert cat_summary["category"] == "Electronics"
    assert cat_summary["total_units"] < summary["total_units"]
    assert cat_summary["avg_unit_price"] > 0


def test_daily_demand_metrics_invalid_category(inv_engine):
    """Requesting non-existent category must raise ValueError."""
    with pytest.raises(ValueError, match="has no transaction records"):
        inv_engine.get_daily_demand_metrics(category="InvalidCategory")


def test_calculate_safety_stock_and_rop_math(inv_engine):
    """Higher service levels must monotonically require larger safety buffers."""
    res_90 = inv_engine.calculate_safety_stock_and_rop(service_level=0.90, category="Clothing")
    res_95 = inv_engine.calculate_safety_stock_and_rop(service_level=0.95, category="Clothing")
    res_99 = inv_engine.calculate_safety_stock_and_rop(service_level=0.99, category="Clothing")

    assert res_90["z_score"] < res_95["z_score"] < res_99["z_score"]
    assert res_90["safety_stock_units"] <= res_95["safety_stock_units"] <= res_99["safety_stock_units"]
    assert res_90["reorder_point_units"] <= res_95["reorder_point_units"] <= res_99["reorder_point_units"]
    assert res_95["safety_stock_capital"] > 0.0
    assert res_95["lead_time_demand_std"] > 0.0


def test_calculate_safety_stock_invalid_service_level(inv_engine):
    """Service level out of [0.50, 1.0) must raise ValueError."""
    with pytest.raises(ValueError, match="Service level must be in"):
        inv_engine.calculate_safety_stock_and_rop(service_level=0.40)

    with pytest.raises(ValueError, match="Service level must be in"):
        inv_engine.calculate_safety_stock_and_rop(service_level=1.0)


def test_calculate_eoq_math(inv_engine):
    """Verifies Economic Order Quantity formula and annual inventory costs."""
    eoq = inv_engine.calculate_eoq(order_cost=50.0, holding_cost_rate=0.20, category="Electronics")
    assert eoq["eoq_units"] > 0
    assert eoq["annual_order_frequency"] > 0.0
    assert eoq["order_cycle_interval_days"] > 0.0
    assert eoq["total_annual_inventory_cost"] > 0.0
    assert eoq["working_capital_in_cycle_stock"] > 0.0

    # At exact continuous EOQ, ordering cost approximately equals holding cost
    assert eoq["annual_ordering_cost"] == pytest.approx(eoq["annual_holding_cost"], rel=0.15)


def test_calculate_eoq_invalid_params(inv_engine):
    """Non-positive order cost or holding cost rate must raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        inv_engine.calculate_eoq(order_cost=-10.0)

    with pytest.raises(ValueError, match="strictly positive"):
        inv_engine.calculate_eoq(holding_cost_rate=0.0)


def test_simulate_stockout_risk_bounds(inv_engine):
    """Stockout simulation must calculate loss function G(k) and expected margin loss."""
    risk = inv_engine.simulate_stockout_risk(service_level=0.95, category="Beauty")
    assert 0.0 <= risk["stockout_probability_per_cycle"] <= 0.05
    assert 0.0 <= risk["cycle_fill_rate_percentage"] <= 100.0
    assert risk["loss_function_gk"] >= 0.0
    assert risk["annual_expected_unmet_units"] >= 0.0
    assert risk["annual_lost_gross_margin"] >= 0.0


def test_perform_abc_fsn_analysis(inv_engine):
    """Multi-criteria stratification must classify all catalog tiers into ABC and FSN categories."""
    res = inv_engine.perform_abc_fsn_analysis()
    assert res["total_sku_tiers"] > 0
    assert res["storewide_revenue"] > 0.0
    assert "Class A (High Value)" in res["abc_distribution"]
    assert "Fast-Moving (F)" in res["fsn_distribution"]

    for sku in res["sku_stratifications"]:
        assert sku["abc_classification"] in [
            "Class A (High Value)",
            "Class B (Moderate Value)",
            "Class C (Low Value)",
        ]
        assert sku["fsn_classification"] in [
            "Fast-Moving (F)",
            "Slow-Moving (S)",
            "Non-Moving / Dormant (N)",
        ]
        assert len(sku["governance_policy"]) > 10


def test_get_storewide_inventory_summary(inv_engine):
    """Storewide inventory summary must consolidate working capital across all categories."""
    summary = inv_engine.get_storewide_inventory_summary(service_level=0.95)
    assert summary["target_service_level"] == 0.95
    assert summary["total_working_capital_committed"] > 0.0
    assert summary["total_safety_stock_capital"] > 0.0
    assert summary["total_cycle_stock_capital"] > 0.0
    assert summary["total_annual_holding_cost"] > 0.0
    assert set(summary["categories"].keys()) == {"Beauty", "Clothing", "Electronics"}
