"""
Unit tests for Retail Cash Flow, Working Capital Cycle & Monte Carlo Financial Risk Stress Testing.
"""

import pytest
from src.financial_risk import FinancialRiskStressEngine


@pytest.fixture(scope="module")
def financial_engine():
    return FinancialRiskStressEngine(seed=42)


def test_calculate_cash_conversion_cycle(financial_engine):
    res = financial_engine.calculate_cash_conversion_cycle()
    assert res["store_total_revenue"] == pytest.approx(456000.0, rel=1e-2)
    assert res["store_total_cogs"] > 0
    assert res["store_blended_dio"] > 0
    assert res["store_blended_dso"] > 0
    assert res["store_blended_dpo"] > 0
    assert res["store_blended_ccc_days"] == pytest.approx(
        res["store_blended_dio"] + res["store_blended_dso"] - res["store_blended_dpo"],
        abs=0.2,
    )
    assert res["net_working_capital_tied_up"] > 0
    assert len(res["category_metrics"]) == 3

    for cat in res["category_metrics"]:
        assert cat["category"] in ["Beauty", "Clothing", "Electronics"]
        assert cat["annual_revenue"] > 0
        assert cat["annual_cogs"] > 0
        assert cat["cash_conversion_cycle_days"] > 0


def test_run_monte_carlo_stress_test(financial_engine):
    res = financial_engine.run_monte_carlo_stress_test(
        n_simulations=500,
        demand_shock_mean_pct=-0.12,
        demand_shock_std_pct=0.06,
        cogs_inflation_mean_pct=0.05,
        cogs_inflation_std_pct=0.03,
        dpo_contraction_days=5.0,
        initial_cash_buffer=50000.0,
    )
    assert res["simulations_count"] == 500
    assert "stressed_outcomes" in res
    outcomes = res["stressed_outcomes"]

    assert outcomes["var_95_cash_flow"] <= outcomes["median_stressed_ocf"]
    assert outcomes["var_99_cash_flow"] <= outcomes["var_95_cash_flow"]
    assert outcomes["cvar_95_expected_shortfall"] <= outcomes["var_95_cash_flow"]
    assert outcomes["cvar_99_expected_shortfall"] <= outcomes["var_99_cash_flow"]

    liquidity = res["liquidity_and_solvency"]
    assert 0.0 <= liquidity["insolvency_risk_pct"] <= 100.0
    assert 0.0 < liquidity["average_solvency_runway_months"] <= 12.0
    assert liquidity["minimum_safe_treasury_buffer_required"] > 0

    assert len(res["cash_flow_distribution"]) == 30


def test_run_monte_carlo_stress_test_validation(financial_engine):
    with pytest.raises(ValueError, match="n_simulations must be between"):
        financial_engine.run_monte_carlo_stress_test(n_simulations=20)

    with pytest.raises(ValueError, match="initial_cash_buffer must be positive"):
        financial_engine.run_monte_carlo_stress_test(initial_cash_buffer=-5000.0)


def test_simulate_working_capital_optimization(financial_engine):
    res = financial_engine.simulate_working_capital_optimization(
        dio_reduction_days=10.0,
        dpo_extension_days=8.0,
        dso_reduction_days=1.5,
        short_term_borrowing_rate_pct=8.0,
    )
    outcomes = res["optimized_outcomes"]
    assert outcomes["ccc_reduction_days"] == pytest.approx(19.5, abs=0.1)
    assert outcomes["cash_freed_from_inventory"] > 0
    assert outcomes["cash_freed_from_supplier_terms"] > 0
    assert outcomes["cash_freed_from_ar_acceleration"] > 0
    assert outcomes["total_permanent_liquidity_unlocked"] > 0
    assert outcomes["annual_borrowing_interest_savings"] > 0


def test_simulate_working_capital_validation(financial_engine):
    with pytest.raises(ValueError, match="Day adjustment parameters cannot be negative"):
        financial_engine.simulate_working_capital_optimization(dio_reduction_days=-5.0)

    with pytest.raises(ValueError, match="borrowing rate cannot be negative"):
        financial_engine.simulate_working_capital_optimization(short_term_borrowing_rate_pct=-2.0)
