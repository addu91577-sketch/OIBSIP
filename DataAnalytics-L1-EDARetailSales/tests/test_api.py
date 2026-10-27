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
