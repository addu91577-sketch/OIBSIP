"""Unit tests for Day 10 Counterfactual Simulation, Price Elasticity & A/B Experimentation.

Validates:
1. Counterfactual 'What-If' sensitivity calculations and parameter adjustments.
2. Econometric price elasticity of demand estimation across catalog and categories.
3. Frequentist two-sample proportion Z-tests and Bayesian Beta-Binomial conjugate updating.
4. Statistical power and Minimum Detectable Effect (MDE) sample size calculations.
5. FastAPI REST API endpoints for simulation, experimentation, and elasticity analytics.
6. Publication artifact file generation and schema completeness.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.simulation_engine import (
    PlayStoreSimulationEngine,
    WhatIfScenarioResult,
    ABTestResult,
    PriceElasticityResult,
)
from src.production_api import app

_BASE_DIR = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


@pytest.fixture(scope="module")
def sim_engine():
    """Module-scoped fixture providing an initialized PlayStoreSimulationEngine."""
    return PlayStoreSimulationEngine(random_state=42)


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# Core Simulation Engine Tests
# ============================================================================

def test_simulation_engine_initialization(sim_engine):
    """Verify that PlayStoreSimulationEngine loads apps data and connects to ML engine."""
    assert sim_engine.df_apps is not None
    assert len(sim_engine.df_apps) >= 9000
    assert sim_engine.ml_engine is not None
    assert sim_engine.ml_engine.clf_rf is not None
    assert sim_engine.ml_engine.reg_rf is not None


def test_simulate_what_if_catalog_app(sim_engine):
    """Verify counterfactual simulation on an existing Play Store catalog application."""
    # Find an app present in catalog
    target_app = sim_engine.df_apps["App"].iloc[0]
    result = sim_engine.simulate_what_if(
        app_name=target_app,
        delta_size_pct=-0.25,
        delta_rating=0.20
    )

    assert isinstance(result, WhatIfScenarioResult)
    assert result.app_name == target_app
    assert result.base_size_mb > 0
    assert result.simulated_size_mb == pytest.approx(result.base_size_mb * 0.75, rel=1e-2)
    assert 0.0 <= result.base_scale_prob <= 1.0
    assert 0.0 <= result.simulated_scale_prob <= 1.0
    assert 1.0 <= result.base_expected_rating <= 5.0
    assert 1.0 <= result.simulated_expected_rating <= 5.0
    assert isinstance(result.recommendations, list)
    assert len(result.recommendations) > 0


def test_simulate_what_if_hypothetical_app(sim_engine):
    """Verify counterfactual simulation on hypothetical app parameters."""
    result = sim_engine.simulate_what_if(
        category="GAME",
        rating=4.10,
        reviews=25000,
        size_mb=60.0,
        price=2.99,
        delta_size_pct=-0.50,
        delta_price=-1.00,
        delta_rating=0.30
    )

    assert result.category == "GAME"
    assert result.base_size_mb == 60.0
    assert result.simulated_size_mb == 30.0
    assert result.base_price == 2.99
    assert result.simulated_price == 1.99
    assert result.simulated_scale_prob >= 0.0
    assert result.delta_revenue != 0 or result.base_expected_revenue >= 0


def test_price_elasticity_computation(sim_engine):
    """Verify econometric price elasticity calculations across catalog and categories."""
    res = sim_engine.analyze_price_elasticity(min_category_paid_apps=10)

    assert "overall_elasticity" in res
    assert "overall_r_squared" in res
    assert "category_benchmarks" in res
    assert res["total_paid_apps_analyzed"] > 100
    assert isinstance(res["overall_elasticity"], float)
    assert res["categories_analyzed"] >= 5

    # Check category benchmarks structure
    cat_item = res["category_benchmarks"][0]
    assert "category" in cat_item
    assert "elasticity_coefficient" in cat_item
    assert "optimal_price_recommendation" in cat_item
    assert cat_item["optimal_price_recommendation"] > 0


def test_ab_experiment_significant_win(sim_engine):
    """Verify A/B testing engine flags statistically significant winning treatment."""
    # Variant A: 500 / 10,000 (5.0%), Variant B: 620 / 10,000 (6.2%) -> strong positive uplift
    res = sim_engine.run_ab_experiment(
        visitors_a=10000,
        conversions_a=500,
        visitors_b=10000,
        conversions_b=620,
        alpha=0.05
    )

    assert isinstance(res, ABTestResult)
    assert res.decision == "ACCEPT_VARIANT_B"
    assert res.is_statistically_significant is True
    assert res.p_value < 0.05
    assert res.z_statistic > 2.0
    assert res.bayesian_prob_b_beats_a > 0.99
    assert res.expected_loss < 0.001
    assert "Deploy Variant B" in res.summary_verdict


def test_ab_experiment_significant_loss(sim_engine):
    """Verify A/B testing engine flags statistically significant degrading treatment."""
    # Variant A: 600 / 10,000 (6.0%), Variant B: 450 / 10,000 (4.5%)
    res = sim_engine.run_ab_experiment(
        visitors_a=10000,
        conversions_a=600,
        visitors_b=10000,
        conversions_b=450,
        alpha=0.05
    )

    assert res.decision == "REJECT_VARIANT_B"
    assert res.is_statistically_significant is True
    assert res.bayesian_prob_b_beats_a < 0.05
    assert "Abort experiment" in res.summary_verdict


def test_ab_experiment_inconclusive(sim_engine):
    """Verify A/B testing engine correctly detects inconclusive experiments."""
    # Negligible difference: 500 / 10,000 vs 505 / 10,000
    res = sim_engine.run_ab_experiment(
        visitors_a=10000,
        conversions_a=500,
        visitors_b=10000,
        conversions_b=505,
        alpha=0.05
    )

    assert res.decision == "INCONCLUSIVE"
    assert res.is_statistically_significant is False
    assert res.p_value > 0.50
    assert 0.40 <= res.bayesian_prob_b_beats_a <= 0.65


def test_sample_size_mde_calculation(sim_engine):
    """Verify statistical sample size formula for Minimum Detectable Effects."""
    # Baseline 5%, MDE 10% (i.e. detecting shift from 5.0% to 5.5%)
    n_10 = sim_engine.calculate_sample_size_mde(baseline_rate=0.05, mde_relative=0.10)
    # Detecting smaller MDE 5% (5.0% to 5.25%) must require larger sample size
    n_5 = sim_engine.calculate_sample_size_mde(baseline_rate=0.05, mde_relative=0.05)

    assert n_5 > n_10
    assert n_10 > 1000


def test_sample_size_matrix_monotonicity(sim_engine):
    """Verify sample size lookup matrix generation and monotonicity."""
    df_matrix = sim_engine.generate_sample_size_matrix()
    assert isinstance(df_matrix, pd.DataFrame)
    assert len(df_matrix) >= 5
    assert "MDE_5%" in df_matrix.columns
    assert "MDE_20%" in df_matrix.columns

    # Within any baseline row, larger MDE requires fewer samples
    for _, row in df_matrix.iterrows():
        assert int(row["MDE_5%"]) > int(row["MDE_20%"])


# ============================================================================
# Production REST API Integration Tests
# ============================================================================

def test_api_simulate_what_if_endpoint(api_client):
    """Test POST /api/v1/simulate/what-if endpoint."""
    payload = {
        "category": "FAMILY",
        "size_mb": 35.0,
        "price": 0.0,
        "delta_size_pct": -0.30,
        "delta_rating": 0.15
    }
    response = api_client.post("/api/v1/simulate/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["category"] == "FAMILY"
    assert data["simulated_size_mb"] == pytest.approx(24.5, rel=1e-2)
    assert "simulated_scale_prob" in data
    assert "pct_scale_prob_uplift" in data
    assert "recommendations" in data


def test_api_ab_test_endpoint(api_client):
    """Test POST /api/v1/experiment/ab-test endpoint."""
    payload = {
        "visitors_a": 12000,
        "conversions_a": 600,
        "visitors_b": 12000,
        "conversions_b": 720,
        "alpha": 0.05
    }
    response = api_client.post("/api/v1/experiment/ab-test", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] == "ACCEPT_VARIANT_B"
    assert data["is_statistically_significant"] is True
    assert data["p_value"] < 0.05
    assert len(data["credible_interval_a"]) == 2
    assert len(data["credible_interval_b"]) == 2


def test_api_ab_test_validation_error(api_client):
    """Test POST /api/v1/experiment/ab-test rejects invalid inputs."""
    # Conversions > visitors
    payload = {
        "visitors_a": 100,
        "conversions_a": 150,
        "visitors_b": 100,
        "conversions_b": 50,
    }
    response = api_client.post("/api/v1/experiment/ab-test", json=payload)
    assert response.status_code == 400


def test_api_price_elasticity_endpoint(api_client):
    """Test GET /api/v1/analytics/price-elasticity endpoint."""
    response = api_client.get("/api/v1/analytics/price-elasticity")
    assert response.status_code == 200
    data = response.json()

    assert data["total_paid_apps_analyzed"] > 100
    assert isinstance(data["overall_elasticity"], float)
    assert "category_benchmarks" in data
    assert len(data["category_benchmarks"]) > 0


# ============================================================================
# Artifact Verification Tests
# ============================================================================

def test_day10_reports_and_figures_existence(sim_engine):
    """Verify that Day 10 figures and CSV benchmarks exist and meet size thresholds."""
    fig20 = _FIGURES_DIR / "20_counterfactual_whatif_and_pricing_elasticity.png"
    fig21 = _FIGURES_DIR / "21_ab_testing_and_bayesian_power_analytics.png"
    elasticity_csv = _REPORTS_DIR / "category_price_elasticity_summary.csv"
    matrix_csv = _REPORTS_DIR / "ab_testing_sample_size_matrix.csv"
    benchmark_csv = _REPORTS_DIR / "counterfactual_simulation_benchmarks.csv"

    # If not already generated on disk, generate now
    if not (fig20.exists() and fig21.exists()):
        sim_engine.generate_reports_and_figures()

    assert fig20.exists()
    assert fig20.stat().st_size > 150_000
    assert fig21.exists()
    assert fig21.stat().st_size > 150_000

    assert elasticity_csv.exists()
    assert elasticity_csv.stat().st_size > 100
    assert matrix_csv.exists()
    assert matrix_csv.stat().st_size > 100
    assert benchmark_csv.exists()
    assert benchmark_csv.stat().st_size > 100
