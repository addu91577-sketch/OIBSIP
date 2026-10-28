"""
Unit and integration tests for financial what-if scenario simulations.
"""

import pytest
from src.simulation import FinancialWhatIfSimulator


def test_simulator_baseline_initialization():
    """Verifies baseline store statistics match empirical dataset invariants."""
    sim = FinancialWhatIfSimulator()
    assert sim.baseline_revenue == 456000.0
    assert sim.baseline_orders == 1000
    assert sim.baseline_units == 2514
    assert sim.baseline_aov == pytest.approx(456.0, abs=0.01)
    assert sim.vip_orders == 202
    assert sim.vip_revenue == 286800.0


def test_simulate_bundle_expansion_financials():
    """Verifies bundle expansion generates positive revenue and profit uplift."""
    sim = FinancialWhatIfSimulator()
    res = sim.simulate_bundle_expansion(
        adoption_rate=0.25,
        target_units=3.20,
        bundle_discount_pct=0.10,
    )

    fin = res["financial_impact"]
    assert fin["net_revenue_uplift"] > 0
    assert fin["pct_revenue_lift"] > 0
    assert fin["projected_revenue"] > sim.baseline_revenue
    assert fin["adopting_orders_count"] == 250
    assert fin["discount_allowance_cost"] > 0
    assert fin["net_profit_uplift"] > 0
    assert fin["program_roi_pct"] > 0


def test_simulate_vip_retention_financials():
    """Verifies VIP retention generates incremental orders, revenue, and profit."""
    sim = FinancialWhatIfSimulator()
    res = sim.simulate_vip_retention(
        retention_lift_pct=0.15,
        loyalty_cost_pct=0.05,
    )

    fin = res["financial_impact"]
    assert fin["incremental_vip_orders"] == 30
    assert fin["incremental_revenue"] > 35000.0
    assert fin["incremental_profit"] > 0
    assert fin["loyalty_expense"] > 0
    assert fin["roi_pct"] > 100.0


def test_simulate_combined_strategy():
    """Verifies multi-lever strategy combines bundle and VIP lifts coherently."""
    sim = FinancialWhatIfSimulator()
    res = sim.simulate_combined_strategy(
        bundle_adoption=0.25,
        target_units=3.20,
        bundle_discount=0.10,
        vip_lift=0.15,
    )

    assert res["projected_total_revenue"] > sim.baseline_revenue
    assert res["total_incremental_revenue"] == pytest.approx(
        res["bundle_component"]["financial_impact"]["net_revenue_uplift"]
        + res["vip_component"]["financial_impact"]["incremental_revenue"],
        abs=0.05,
    )
    assert res["combined_roi_pct"] > 0


def test_simulation_input_validation():
    """Verifies out-of-range parameters raise ValueErrors."""
    sim = FinancialWhatIfSimulator()
    with pytest.raises(ValueError):
        sim.simulate_bundle_expansion(adoption_rate=1.5)  # > 1.0

    with pytest.raises(ValueError):
        sim.simulate_bundle_expansion(target_units=-1.0)  # negative

    with pytest.raises(ValueError):
        sim.simulate_vip_retention(retention_lift_pct=2.0)  # > 1.0
