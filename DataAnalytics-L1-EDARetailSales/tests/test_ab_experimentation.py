"""
Unit and integration tests for Retail A/B Experimentation Engine.
"""

import pytest
from src.ab_experimentation import RetailABTestingEngine


@pytest.fixture
def ab_engine():
    return RetailABTestingEngine(random_state=42)


def test_continuous_ab_welch_and_mann_whitney(ab_engine):
    control = [100.0, 120.0, 95.0, 110.0, 105.0]
    treatment = [140.0, 155.0, 150.0, 145.0, 160.0]

    res = ab_engine.evaluate_continuous_ab(control, treatment, metric_name="AOV")
    assert res["metric_name"] == "AOV"
    assert res["lift"]["absolute_diff"] > 0
    assert res["frequentist_tests"]["welch_p_value"] < 0.01
    assert res["frequentist_tests"]["mann_whitney_p_value"] < 0.05
    assert res["frequentist_tests"]["is_significant"] is True
    assert "Statistically Significant Uplift" in res["conclusion"]


def test_continuous_ab_invalid_samples(ab_engine):
    with pytest.raises(ValueError, match="at least 2 observations"):
        ab_engine.evaluate_continuous_ab([10.0], [20.0, 30.0])


def test_proportion_ab_z_and_chi2(ab_engine):
    res = ab_engine.evaluate_proportion_ab(
        control_successes=100,
        control_total=1000,
        treatment_successes=150,
        treatment_total=1000,
    )
    assert res["control_conversion_rate"] == 0.10
    assert res["treatment_conversion_rate"] == 0.15
    assert res["lift"]["relative_lift_pct"] == 50.0
    assert res["z_test"]["p_value"] < 0.01
    assert res["chi2_test"]["p_value"] < 0.01
    assert res["is_significant"] is True


def test_proportion_ab_validation_errors(ab_engine):
    with pytest.raises(ValueError, match="positive integers"):
        ab_engine.evaluate_proportion_ab(10, 0, 20, 100)

    with pytest.raises(ValueError, match="between 0 and control_total"):
        ab_engine.evaluate_proportion_ab(150, 100, 20, 100)


def test_bayesian_conversion_test(ab_engine):
    res = ab_engine.bayesian_conversion_test(
        control_conversions=100,
        control_impressions=1000,
        treatment_conversions=150,
        treatment_impressions=1000,
        n_simulations=10000,
    )
    assert res["bayesian_metrics"]["prob_treatment_better_pct"] > 95.0
    assert res["recommendation"]["decision"] == "Deploy Variant (Treatment)"
    assert res["bayesian_metrics"]["expected_loss_treatment"] < 0.001


def test_bayesian_revenue_bootstrap(ab_engine):
    control = [25.0, 50.0, 30.0, 500.0, 100.0]
    treatment = [50.0, 100.0, 300.0, 500.0, 450.0]

    res = ab_engine.bayesian_revenue_test(control, treatment, n_simulations=5000)
    assert res["prob_treatment_superior_pct"] > 80.0
    assert res["expected_revenue_gain_per_order"] > 0


def test_sample_size_calculator(ab_engine):
    n = ab_engine.calculate_sample_size(baseline_rate=0.10, mde=0.10)
    assert n > 10000

    matrix = ab_engine.generate_power_matrix(baseline_rate=0.20)
    assert len(matrix) == 6
    assert matrix[0]["sample_size_per_arm"] > matrix[-1]["sample_size_per_arm"]
