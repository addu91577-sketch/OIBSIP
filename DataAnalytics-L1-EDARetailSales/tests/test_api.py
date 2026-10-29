"""
Integration tests validating the production FastAPI REST endpoints.
"""

import pytest


def test_root_endpoint(test_client):
    """GET / must return 200 OK with service metadata and KPI summary."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert data["version"] == "1.0.0"
    assert data["summary"]["total_revenue"] == 456000.0


def test_health_check_endpoint(test_client):
    """GET /health must return 200 OK and verified model readiness."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["database_records_loaded"] == 1000
    assert data["vip_model_ready"] is True
    assert data["spend_model_ready"] is True


def test_overview_metrics_endpoint(test_client):
    """GET /api/v1/metrics/overview must return complete executive scorecard."""
    response = test_client.get("/api/v1/metrics/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_revenue"] == 456000.0
    assert data["average_order_value"] == 456.0
    assert data["total_orders"] == 1000


def test_descriptive_stats_endpoint(test_client):
    """GET /api/v1/analytics/descriptive must profile all 4 numerical columns."""
    response = test_client.get("/api/v1/analytics/descriptive")
    assert response.status_code == 200
    data = response.json()
    for col in ["Age", "Quantity", "Price per Unit", "Total Amount"]:
        assert col in data
        assert "mean" in data[col]
        assert "median" in data[col]
        assert "skewness" in data[col]


def test_time_series_endpoint(test_client):
    """GET /api/v1/analytics/time-series must deliver monthly, quarterly, and weekly metrics."""
    response = test_client.get("/api/v1/analytics/time-series")
    assert response.status_code == 200
    data = response.json()
    assert "monthly_trends" in data
    assert "quarterly_trends" in data
    assert "day_of_week_dynamics" in data


def test_demographics_endpoint(test_client):
    """GET /api/v1/analytics/demographics must deliver cohorts and gender parity."""
    response = test_client.get("/api/v1/analytics/demographics")
    assert response.status_code == 200
    data = response.json()
    assert "age_cohorts" in data
    assert "gender_breakdown" in data


def test_products_endpoint(test_client):
    """GET /api/v1/analytics/products must return category and top-sellers analytics."""
    response = test_client.get("/api/v1/analytics/products")
    assert response.status_code == 200
    data = response.json()
    assert "department_summary" in data
    assert "catalog_price_tiers" in data
    assert "top_offerings_by_revenue" in data


def test_pareto_endpoint(test_client):
    """GET /api/v1/analytics/pareto must return Lorenz points and Gini index."""
    response = test_client.get("/api/v1/analytics/pareto")
    assert response.status_code == 200
    data = response.json()
    assert "gini_coefficient" in data
    assert "lorenz_curve" in data
    assert "spend_tier_polarization" in data


def test_predict_vip_success(test_client, sample_vip_payload):
    """POST /api/v1/predict/vip must return valid probability and badge."""
    response = test_client.post("/api/v1/predict/vip", json=sample_vip_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_vip"] is True
    assert 0.0 <= data["vip_probability"] <= 1.0
    assert "tier_badge" in data


def test_predict_spend_success(test_client, sample_budget_payload):
    """POST /api/v1/predict/spend must return predicted spend and range."""
    response = test_client.post("/api/v1/predict/spend", json=sample_budget_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_spend"] > 0
    assert data["expected_range_min"] <= data["predicted_spend"] <= data["expected_range_max"]


def test_predict_vip_validation_failure(test_client):
    """POST /api/v1/predict/vip with invalid input must return 422 Unprocessable Entity."""
    # Underage (<18)
    bad_payload = {
        "age": 12,
        "gender": "Male",
        "product_category": "Electronics",
        "quantity": 2,
        "price_per_unit": 50.0,
    }
    response = test_client.post("/api/v1/predict/vip", json=bad_payload)
    assert response.status_code == 422

    # Invalid Category
    bad_cat_payload = {
        "age": 30,
        "gender": "Female",
        "product_category": "Automotive",
        "quantity": 2,
        "price_per_unit": 50.0,
    }
    response2 = test_client.post("/api/v1/predict/vip", json=bad_cat_payload)
    assert response2.status_code == 422


def test_strategic_recommendations_endpoint(test_client):
    """GET /api/v1/recommendations must deliver all 5 strategic initiatives."""
    response = test_client.get("/api/v1/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert data["recommendations_count"] == 5


def test_bundle_recommendation_endpoint(test_client):
    """POST /api/v1/recommendations/bundle must generate synergy set."""
    payload = {"primary_category": "Electronics", "target_units": 3}
    response = test_client.post("/api/v1/recommendations/bundle", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["primary_category"] == "Electronics"
    assert data["items_count"] == 3
    assert data["discount_percentage"] == 10.0


# --- DAY 9 NEW REST ENDPOINT TESTS ---
def test_drift_audit_endpoint_stable(test_client):
    """POST /api/v1/mlops/drift with stable scenario must return 200 and HEALTHY_STABLE."""
    payload = {"scenario": "stable", "sample_size": 250}
    response = test_client.post("/api/v1/mlops/drift", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["global_status"] == "HEALTHY_STABLE"
    assert data["retrain_recommended"] is False
    assert "feature_audits" in data


def test_drift_audit_endpoint_surge(test_client):
    """POST /api/v1/mlops/drift with tech_surge must trigger CRITICAL_DRIFT."""
    payload = {"scenario": "tech_surge", "sample_size": 300}
    response = test_client.post("/api/v1/mlops/drift", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["global_status"] == "CRITICAL_DRIFT"
    assert data["retrain_recommended"] is True


def test_rfm_segments_endpoint(test_client):
    """GET /api/v1/analytics/rfm-segments must return 4 cluster profiles."""
    response = test_client.get("/api/v1/analytics/rfm-segments")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "profiles" in data
    assert len(data["profiles"]) == 4
    assert data["total_customers"] == 1000


def test_what_if_simulation_endpoint(test_client):
    """POST /api/v1/simulations/what-if must calculate projected revenue and uplift."""
    payload = {
        "bundle_adoption_rate": 0.25,
        "target_basket_units": 3.20,
        "bundle_discount_pct": 0.10,
        "vip_retention_lift_pct": 0.15,
    }
    response = test_client.post("/api/v1/simulations/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["projected_total_revenue"] > 456000.0
    assert data["overall_pct_revenue_lift"] > 0
    assert data["combined_roi_pct"] > 0


def test_what_if_simulation_validation_error(test_client):
    """POST /api/v1/simulations/what-if with out-of-bounds discount must return 422."""
    payload = {
        "bundle_adoption_rate": 0.25,
        "target_basket_units": 3.20,
        "bundle_discount_pct": 0.95,  # > 0.90 max
        "vip_retention_lift_pct": 0.15,
    }
    response = test_client.post("/api/v1/simulations/what-if", json=payload)
    assert response.status_code == 422


# --- DAY 10 API ENDPOINTS INTEGRATION TESTS ---
def test_elasticity_summary_endpoint(test_client):
    """GET /api/v1/elasticity/summary must return category & catalog PED coefficients."""
    response = test_client.get("/api/v1/elasticity/summary")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "Electronics" in data["categories"]
    assert "ped_coefficient" in data["categories"]["Electronics"]


def test_markdown_simulation_endpoint(test_client):
    """POST /api/v1/elasticity/simulate must return financial impact of markdown."""
    payload = {
        "category": "Clothing",
        "discount_pct": 0.10,
        "expected_volume_lift_pct": 0.20,
        "gross_margin_pct": 0.40,
    }
    response = test_client.post("/api/v1/elasticity/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Clothing"
    assert "financial_impact" in data
    assert "breakeven_volume_lift_required_pct" in data["financial_impact"]


def test_markdown_simulation_validation_error(test_client):
    """POST /api/v1/elasticity/simulate with invalid category must return 422."""
    payload = {
        "category": "Furniture",
        "discount_pct": 0.10,
        "expected_volume_lift_pct": 0.20,
    }
    response = test_client.post("/api/v1/elasticity/simulate", json=payload)
    assert response.status_code == 422


def test_tier_optimization_endpoint(test_client):
    """GET /api/v1/elasticity/tiers/Electronics must return tier gross margins."""
    response = test_client.get("/api/v1/elasticity/tiers/Electronics")
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Electronics"
    assert data["profit_maximizing_tier"] in [25.0, 30.0, 50.0, 300.0, 500.0]


def test_ab_proportions_endpoint(test_client):
    """POST /api/v1/experiments/ab-proportions must return Frequentist & Bayesian evaluation."""
    payload = {
        "control_successes": 50,
        "control_total": 500,
        "treatment_successes": 75,
        "treatment_total": 500,
        "metric_name": "VIP Conversion",
    }
    response = test_client.post("/api/v1/experiments/ab-proportions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "frequentist_analysis" in data
    assert "bayesian_analysis" in data
    assert data["bayesian_analysis"]["bayesian_metrics"]["prob_treatment_better_pct"] > 90


def test_ab_continuous_endpoint(test_client):
    """POST /api/v1/experiments/ab-continuous must return spend comparison."""
    payload = {
        "control_data": [50.0, 100.0, 150.0, 200.0],
        "treatment_data": [70.0, 120.0, 180.0, 250.0],
        "metric_name": "Basket Value",
    }
    response = test_client.post("/api/v1/experiments/ab-continuous", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "frequentist_analysis" in data
    assert "bayesian_bootstrap" in data


def test_power_matrix_endpoint(test_client):
    """GET /api/v1/experiments/power-matrix must return sample sizes across MDEs."""
    response = test_client.get("/api/v1/experiments/power-matrix?baseline_rate=0.20")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "sample_size_per_arm" in data[0]


def test_clv_overview_endpoint(test_client):
    """GET /api/v1/clv/overview must return customer equity audit & churn distribution."""
    response = test_client.get("/api/v1/clv/overview")
    assert response.status_code == 200
    data = response.json()
    assert "customer_base" in data
    assert "churn_tier_distribution" in data
    assert "vip_early_warning" in data


def test_predict_customer_clv_endpoint(test_client):
    """POST /api/v1/clv/predict-customer must score churn risk and calculate CLV."""
    payload = {
        "recency_days": 40.0,
        "aov": 450.0,
        "gross_margin_pct": 0.40,
    }
    response = test_client.post("/api/v1/clv/predict-customer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_tier"] in [
        "Active (Low Risk)",
        "Cooling (Medium Risk)",
        "At Risk (High Risk)",
        "Lapsed (Defected)",
    ]
    assert data["projected_clv"] > 0


def test_at_risk_vips_endpoint(test_client):
    """GET /api/v1/clv/at-risk-vips must return high-priority at-risk VIP shoppers."""
    response = test_client.get("/api/v1/clv/at-risk-vips?min_spend=1000&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5
    if len(data) > 0:
        assert data[0]["historical_spend"] >= 1000.0
