"""Unit and integration test suite for Day 13 Retention, Churn Risk & LTV Engine.

Validates:
1. Engine initialization, dataset indexing, and category baselines.
2. Parametric retention curve modeling (Power-Law & Weibull) and milestone monotonicity.
3. Quality modifier behavior (ratings & sentiment boosting retention).
4. Customer Lifetime Value (LTV) forecasting, discounting, and CAC payback duration.
5. Developer maintenance lifecycle audit and abandonware categorization.
6. Composite 0-100 Churn Risk score calculation, sub-risk attribution, and prescriptive playbooks.
7. Markowitz Modern Portfolio Theory capital allocation and constraint satisfaction (sum(w) == 1.0).
8. Synthetic / unseen app fallback robustness.
9. Verification of analytical CSV reports and publication figures (Figures 26 & 27).
10. FastAPI REST API endpoints:
    - POST /api/v1/retention/predict-curve
    - POST /api/v1/retention/ltv-forecast
    - POST /api/v1/retention/churn-risk-score
    - GET  /api/v1/retention/developer-lifecycle
    - POST /api/v1/retention/portfolio-optimizer
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.retention_churn_ltv_engine import (
    RetentionChurnLTVEngine,
    RetentionMilestones,
    RetentionCurveResult,
    LTVProjectionResult,
    DeveloperLifecycleResult,
    ChurnRiskDiagnosticResult,
    PortfolioAllocationResult,
    CATEGORY_RETENTION_DEFAULTS
)
from src.production_api import app

_BASE_DIR = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


@pytest.fixture(scope="module")
def ret_engine():
    """Module-scoped fixture providing an initialized RetentionChurnLTVEngine."""
    return RetentionChurnLTVEngine()


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# 1. Engine Core & Dataset Verification
# ============================================================================

def test_engine_initialization(ret_engine):
    """Verify engine loads cleaned apps, user reviews, and precomputes category baselines."""
    assert len(ret_engine.df_apps) == 9659
    assert len(ret_engine.df_reviews) > 0
    assert "GAME" in ret_engine.category_baselines
    assert "FAMILY" in ret_engine.category_baselines
    assert len(ret_engine._apps_lookup) > 9000


def test_category_baselines_integrity(ret_engine):
    """Verify category baseline metrics are within realistic ranges."""
    for cat, base in ret_engine.category_baselines.items():
        assert 1.0 <= base["mean_rating"] <= 5.0
        assert base["median_installs"] >= 0
        assert base["mean_days_update"] >= 0
        assert base["app_count"] > 0


# ============================================================================
# 2. Parametric Retention Modeling
# ============================================================================

def test_retention_curve_power_law(ret_engine):
    """Verify power-law retention curve monotonicity and valid milestone values."""
    res = ret_engine.estimate_retention_curve(app_or_category="GAME", is_category=True, model_type="power_law")
    assert isinstance(res, RetentionCurveResult)
    assert res.model_type == "power_law"
    assert 0.10 <= res.baseline_d1 <= 0.65
    assert 0.10 <= res.decay_parameter <= 0.75

    m = res.milestones
    # Monotonic decay over time
    assert m.day_1 >= m.day_7 >= m.day_14 >= m.day_30 >= m.day_90 >= m.day_360
    assert res.retention_auc > 0.0


def test_retention_curve_weibull(ret_engine):
    """Verify Weibull survival model produces valid, strictly non-negative probabilities."""
    res = ret_engine.estimate_retention_curve(app_or_category="SOCIAL", is_category=True, model_type="weibull")
    assert res.model_type == "weibull"
    m = res.milestones
    assert 0.0 < m.day_360 <= m.day_1 <= 1.0
    assert m.day_1 >= m.day_7 >= m.day_30


def test_retention_quality_modifiers(ret_engine):
    """Verify that higher ratings and positive sentiment improve retention rates."""
    low_res = ret_engine.estimate_retention_curve(
        app_or_category="TOOLS",
        is_category=True,
        custom_rating=2.5,
        custom_polarity=-0.5
    )
    high_res = ret_engine.estimate_retention_curve(
        app_or_category="TOOLS",
        is_category=True,
        custom_rating=4.8,
        custom_polarity=0.6
    )
    assert high_res.baseline_d1 > low_res.baseline_d1
    assert high_res.milestones.day_30 > low_res.milestones.day_30
    assert high_res.retention_auc > low_res.retention_auc


# ============================================================================
# 3. Customer Lifetime Value (LTV) Forecasting
# ============================================================================

def test_ltv_forecasting_free_app(ret_engine):
    """Verify LTV calculation for a free ad-monetized app."""
    res = ret_engine.forecast_ltv("Subway Surfers")
    assert isinstance(res, LTVProjectionResult)
    assert res.price == 0.0
    assert res.ltv_30d <= res.ltv_90d <= res.ltv_180d <= res.ltv_360d
    assert res.ltv_360d > 0.0
    assert res.economic_tier in ["VIABLE_EXPANSION", "SUSTAINABLE", "BREAKEVEN", "UNSUSTAINABLE"]


def test_ltv_forecasting_paid_app(ret_engine):
    """Verify LTV calculation for a paid app incorporates net upfront revenue."""
    res = ret_engine.forecast_ltv("Minecraft", custom_price=6.99)
    assert res.price == 6.99
    # Net upfront is price * 0.95 = 6.64, so 30-day LTV must exceed net upfront
    assert res.ltv_30d >= (6.99 * 0.95)
    assert res.ltv_cac_ratio > 0.0


def test_ltv_payback_period_calculation(ret_engine):
    """Verify payback period is within horizon when cumulative revenue surpasses CAC."""
    # Low CAC guarantees quick breakeven
    res = ret_engine.forecast_ltv("Duolingo: Learn Languages Free", custom_cac=0.05)
    assert res.payback_period_days is not None
    assert 1 <= res.payback_period_days <= 30


# ============================================================================
# 4. Developer Maintenance Lifecycle & Abandonware Audit
# ============================================================================

def test_developer_lifecycle_audit_single_app(ret_engine):
    """Verify app maintenance lifecycle classification and recency calculation."""
    res = ret_engine.audit_developer_lifecycle("Duolingo: Learn Languages Free")
    assert isinstance(res, DeveloperLifecycleResult)
    assert res.days_since_update >= 0
    assert res.lifecycle_tier in ["VIBRANT_ACTIVE", "MATURE_SUSTAINED", "STAGNANT_RISK", "ZOMBIE_ABANDONWARE"]
    assert 0.0 <= res.abandonware_probability <= 1.0
    assert 0.0 <= res.maintenance_score <= 100.0


def test_developer_lifecycle_audit_catalog_summary(ret_engine):
    """Verify catalog-wide lifecycle audit generates valid distribution across all apps."""
    df_life = ret_engine.audit_developer_lifecycle()
    assert isinstance(df_life, pd.DataFrame)
    assert len(df_life) == 9659
    assert set(df_life["Lifecycle_Tier"].unique()).issubset({
        "VIBRANT_ACTIVE", "MATURE_SUSTAINED", "STAGNANT_RISK", "ZOMBIE_ABANDONWARE"
    })


# ============================================================================
# 5. Composite Churn Risk Diagnostic
# ============================================================================

def test_churn_risk_evaluation(ret_engine):
    """Verify 0-100 churn risk score, sub-risk attribution, and playbook generation."""
    res = ret_engine.evaluate_churn_risk("Subway Surfers")
    assert isinstance(res, ChurnRiskDiagnosticResult)
    assert 0.0 <= res.churn_risk_score <= 100.0
    assert res.churn_risk_tier in [
        "LOW_CHURN_DEFENSIVE", "MODERATE_CHURN_STABLE", "ELEVATED_CHURN_ATTENTION", "CRITICAL_CHURN_INTERVENTION"
    ]
    assert 0.0 <= res.stagnancy_risk <= 100.0
    assert 0.0 <= res.sentiment_risk <= 100.0
    assert 0.0 <= res.rating_risk <= 100.0
    assert len(res.prescriptive_playbook) >= 2


def test_unseen_app_churn_risk_fallback(ret_engine):
    """Verify synthetic / unseen app is handled gracefully without crashing."""
    res = ret_engine.evaluate_churn_risk("Totally Nonexistent Synthetic App 2026")
    assert isinstance(res, ChurnRiskDiagnosticResult)
    assert 0.0 <= res.churn_risk_score <= 100.0
    assert len(res.prescriptive_playbook) > 0


# ============================================================================
# 6. Modern Portfolio Theory (Markowitz) Capital Allocation
# ============================================================================

def test_portfolio_optimization_multi_asset(ret_engine):
    """Verify Markowitz optimizer yields weights summing to 1.0 with positive Sharpe ratio."""
    items = ["GAME", "FINANCE", "TOOLS", "HEALTH_AND_FITNESS", "EDUCATION"]
    port_res = ret_engine.optimize_app_portfolio(items, risk_free_rate=0.03)
    assert isinstance(port_res, PortfolioAllocationResult)
    assert len(port_res.optimal_weights) == len(items)

    total_weight = sum(port_res.optimal_weights.values())
    assert abs(total_weight - 1.0) < 1e-3
    assert port_res.expected_annual_return > 0.0
    assert port_res.portfolio_volatility > 0.0
    assert port_res.sharpe_ratio > 0.0


def test_portfolio_optimization_single_asset(ret_engine):
    """Verify single-asset portfolio allocates 100% weight cleanly."""
    port_res = ret_engine.optimize_app_portfolio(["GAME"])
    assert port_res.optimal_weights == {"GAME": 1.0}


# ============================================================================
# 7. Analytical Reports & Publication Figures
# ============================================================================

def test_analytical_reports_generation(ret_engine, tmp_path):
    """Verify generation of all 4 Day 13 summary CSV reports."""
    paths = ret_engine.generate_all_reports(output_dir=tmp_path)
    assert "category_retention_benchmarks" in paths
    assert "sample_ltv_projections" in paths
    assert "app_churn_risk_scorecard" in paths
    assert "developer_lifecycle_summary" in paths

    for name, p in paths.items():
        assert p.exists(), f"Report file {p} does not exist"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Report {name} is empty"


def test_publication_figures_generation(ret_engine, tmp_path):
    """Verify generation of Figures 26 and 27."""
    f26, f27 = ret_engine.generate_publication_figures(output_dir=tmp_path)
    assert f26.exists() and f26.stat().st_size > 50000
    assert f27.exists() and f27.stat().st_size > 50000


# ============================================================================
# 8. Production REST API Endpoints Verification
# ============================================================================

def test_api_retention_predict_curve(api_client):
    """Test POST /api/v1/retention/predict-curve endpoint."""
    payload = {
        "app_or_category": "GAME",
        "is_category": True,
        "model_type": "power_law",
        "custom_rating": 4.3,
        "custom_polarity": 0.25,
        "days": 180
    }
    response = api_client.post("/api/v1/retention/predict-curve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_name"] == "GAME"
    assert "milestones" in data
    assert data["milestones"]["day_1"] > 0.0
    assert len(data["daily_retention_sample"]) > 0


def test_api_retention_ltv_forecast(api_client):
    """Test POST /api/v1/retention/ltv-forecast endpoint."""
    payload = {
        "app_name": "Subway Surfers",
        "custom_price": 0.0,
        "custom_arpdau": 0.15,
        "custom_cac": 2.10,
        "annual_discount_rate": 0.10,
        "days": 360
    }
    response = api_client.post("/api/v1/retention/ltv-forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Subway Surfers"
    assert data["ltv_360d"] > 0.0
    assert "economic_tier" in data


def test_api_retention_churn_risk_score(api_client):
    """Test POST /api/v1/retention/churn-risk-score endpoint."""
    payload = {
        "app_name": "Duolingo: Learn Languages Free"
    }
    response = api_client.post("/api/v1/retention/churn-risk-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Duolingo: Learn Languages Free"
    assert 0.0 <= data["churn_risk_score"] <= 100.0
    assert len(data["prescriptive_playbook"]) >= 1


def test_api_retention_developer_lifecycle_overview(api_client):
    """Test GET /api/v1/retention/developer-lifecycle catalog overview."""
    response = api_client.get("/api/v1/retention/developer-lifecycle")
    assert response.status_code == 200
    data = response.json()
    assert data["total_apps"] == 9659
    assert "tier_counts" in data
    assert "VIBRANT_ACTIVE" in data["tier_counts"]


def test_api_retention_developer_lifecycle_single_app(api_client):
    """Test GET /api/v1/retention/developer-lifecycle for a single target app."""
    response = api_client.get("/api/v1/retention/developer-lifecycle?app_name=Subway+Surfers")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Subway Surfers"
    assert "days_since_update" in data
    assert "lifecycle_tier" in data


def test_api_retention_portfolio_optimizer(api_client):
    """Test POST /api/v1/retention/portfolio-optimizer endpoint."""
    payload = {
        "portfolio_items": ["GAME", "FINANCE", "TOOLS", "HEALTH_AND_FITNESS", "EDUCATION"],
        "risk_free_rate": 0.03
    }
    response = api_client.post("/api/v1/retention/portfolio-optimizer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_weights" in data
    assert abs(sum(data["optimal_weights"].values()) - 1.0) < 1e-3
    assert data["sharpe_ratio"] > 0.0
