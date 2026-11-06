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



# ==========================================
# DAY 11 API ENDPOINT TESTS
# ==========================================

def test_inventory_daily_metrics_endpoint(test_client):
    """GET /api/v1/inventory/daily-metrics must return velocity and demand metrics."""
    response = test_client.get("/api/v1/inventory/daily-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "daily_mean_units" in data
    assert "daily_std_units" in data
    assert data["total_units"] > 0


def test_inventory_safety_stock_endpoint(test_client):
    """POST /api/v1/inventory/safety-stock must calculate SS and ROP."""
    payload = {"service_level": 0.95, "lead_time_days": 7.0, "category": "Clothing"}
    response = test_client.post("/api/v1/inventory/safety-stock", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["safety_stock_units"] > 0
    assert data["reorder_point_units"] >= data["safety_stock_units"]
    assert data["z_score"] > 1.6


def test_inventory_safety_stock_validation_error(test_client):
    """POST /api/v1/inventory/safety-stock must reject invalid service level."""
    payload = {"service_level": 1.5}
    response = test_client.post("/api/v1/inventory/safety-stock", json=payload)
    assert response.status_code == 422


def test_inventory_eoq_endpoint(test_client):
    """POST /api/v1/inventory/eoq must calculate EOQ and cycle duration."""
    payload = {"order_cost": 60.0, "holding_cost_rate": 0.25, "category": "Electronics"}
    response = test_client.post("/api/v1/inventory/eoq", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["eoq_units"] > 0
    assert data["order_cycle_interval_days"] > 0.0


def test_inventory_stockout_risk_endpoint(test_client):
    """POST /api/v1/inventory/stockout-risk must simulate lost margin and fill rate."""
    payload = {"service_level": 0.95, "category": "Beauty"}
    response = test_client.post("/api/v1/inventory/stockout-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["cycle_fill_rate_percentage"] > 90.0
    assert data["annual_lost_gross_margin"] >= 0.0


def test_inventory_abc_fsn_endpoint(test_client):
    """GET /api/v1/inventory/abc-fsn must return multi-criteria classifications."""
    response = test_client.get("/api/v1/inventory/abc-fsn")
    assert response.status_code == 200
    data = response.json()
    assert "abc_distribution" in data
    assert "fsn_distribution" in data
    assert len(data["sku_stratifications"]) > 0


def test_inventory_storewide_summary_endpoint(test_client):
    """GET /api/v1/inventory/storewide-summary must aggregate working capital."""
    response = test_client.get("/api/v1/inventory/storewide-summary?service_level=0.95")
    assert response.status_code == 200
    data = response.json()
    assert data["total_working_capital_committed"] > 0.0
    assert "categories" in data


def test_market_basket_affinity_endpoint(test_client):
    """GET /api/v1/market-basket/affinity must return co-occurrence and lift."""
    response = test_client.get("/api/v1/market-basket/affinity")
    assert response.status_code == 200
    data = response.json()
    assert "rules" in data
    assert len(data["rules"]) > 0


def test_market_basket_rules_endpoint(test_client):
    """GET /api/v1/market-basket/rules must filter rules meeting thresholds."""
    response = test_client.get("/api/v1/market-basket/rules?min_support=0.01&min_lift=0.5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_market_basket_combos_endpoint(test_client):
    """POST /api/v1/market-basket/combos must generate bundles with discounts."""
    payload = {"target_category": "Electronics", "discount_rate": 0.10}
    response = test_client.post("/api/v1/market-basket/combos", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["customer_savings"] > 0.0


def test_market_basket_demographic_affinity_endpoint(test_client):
    """GET /api/v1/market-basket/demographic-affinity must break down cohorts."""
    response = test_client.get("/api/v1/market-basket/demographic-affinity")
    assert response.status_code == 200
    data = response.json()
    assert data["cohort_count"] > 0
    assert "cohort_breakdown" in data


def test_forecasting_predict_endpoint(test_client):
    """POST /api/v1/forecasting/predict must return forward projection and conformal bounds."""
    payload = {"horizon_days": 14, "category": "Clothing"}
    response = test_client.post("/api/v1/forecasting/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon_days"] == 14
    assert len(data["daily_projections"]) == 14
    assert data["total_projected_units"] > 0.0


def test_forecasting_accuracy_endpoint(test_client):
    """GET /api/v1/forecasting/accuracy must evaluate holdout metrics."""
    response = test_client.get("/api/v1/forecasting/accuracy?test_window_days=30")
    assert response.status_code == 200
    data = response.json()
    assert "mae_units" in data
    assert "rmse_units" in data
    assert "conformal_95_coverage_percentage" in data


def test_forecasting_anomalies_endpoint(test_client):
    """GET /api/v1/forecasting/anomalies must return velocity outlier alerts."""
    response = test_client.get("/api/v1/forecasting/anomalies?z_threshold=2.0")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# ==========================================
# DAY 12: OMNICHANNEL MARKETING ATTRIBUTION
# ==========================================

def test_attribution_models_endpoint(test_client):
    """GET /api/v1/attribution/models must compare 5 heuristic models."""
    response = test_client.get("/api/v1/attribution/models?half_life_days=7.0")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 1000
    assert "models" in data
    assert "First-Touch" in data["models"]
    assert "Last-Touch" in data["models"]
    assert "Time-Decay" in data["models"]


def test_attribution_markov_endpoint(test_client):
    """GET /api/v1/attribution/markov must return Markov removal effects."""
    response = test_client.get("/api/v1/attribution/markov")
    assert response.status_code == 200
    data = response.json()
    assert data["baseline_conversion_probability"] > 0
    assert len(data["channel_attribution"]) == 5
    assert data["channel_attribution"][0]["removal_effect"] >= 0


def test_attribution_shapley_endpoint(test_client):
    """GET /api/v1/attribution/shapley must return cooperative Shapley values."""
    response = test_client.get("/api/v1/attribution/shapley")
    assert response.status_code == 200
    data = response.json()
    assert "total_attributed_revenue" in data
    assert len(data["shapley_attribution"]) == 5


def test_attribution_optimize_budget_endpoint(test_client):
    """POST /api/v1/attribution/optimize-budget must execute Markowitz MPT allocation."""
    payload = {"total_budget": 75000.0, "risk_aversion": 0.5}
    response = test_client.post("/api/v1/attribution/optimize-budget", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_budget"] == 75000.0
    assert data["projected_optimal_revenue"] > 0
    assert len(data["channel_allocations"]) == 5


# ==========================================
# DAY 12: GMROI & ASSORTMENT RATIONALIZATION
# ==========================================

def test_gmroi_overview_endpoint(test_client):
    """GET /api/v1/gmroi/overview must return category GMROI benchmarks."""
    response = test_client.get("/api/v1/gmroi/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["store_blended_gmroi"] > 0
    assert len(data["categories"]) == 3


def test_gmroi_quadrant_matrix_endpoint(test_client):
    """GET /api/v1/gmroi/quadrant-matrix must classify SKUs into 4 quadrants."""
    response = test_client.get("/api/v1/gmroi/quadrant-matrix")
    assert response.status_code == 200
    data = response.json()
    assert "quadrant_summary" in data
    assert "Stars" in data["quadrant_summary"]
    assert "classified_skus" in data


def test_gmroi_rationalization_endpoint(test_client):
    """GET /api/v1/gmroi/rationalization must diagnose capital drag."""
    response = test_client.get("/api/v1/gmroi/rationalization")
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert data["total_capital_at_risk"] >= 0


def test_gmroi_reallocate_capital_endpoint(test_client):
    """POST /api/v1/gmroi/reallocate-capital must simulate capital shift to stars."""
    payload = {
        "reallocation_pct": 0.30,
        "source_quadrant": "Dead Stock / Capital Traps",
        "target_quadrant": "Stars"
    }
    response = test_client.post("/api/v1/gmroi/reallocate-capital", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["reallocation_pct"] == 30.0
    assert data["projected_store_gmroi"] >= data["baseline_store_gmroi"]


def test_gmroi_reallocate_capital_validation_error(test_client):
    """POST /api/v1/gmroi/reallocate-capital must reject out-of-bound allocation."""
    payload = {"reallocation_pct": 0.99}
    response = test_client.post("/api/v1/gmroi/reallocate-capital", json=payload)
    assert response.status_code == 422


# ==========================================
# DAY 12: CUSTOMER COHORT RETENTION & SURVIVAL
# ==========================================

def test_cohort_retention_matrix_endpoint(test_client):
    """GET /api/v1/cohorts/retention-matrix must return monthly cohort matrix."""
    response = test_client.get("/api/v1/cohorts/retention-matrix")
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 1000
    assert len(data["cohort_matrix"]) >= 12


def test_cohort_survival_analysis_endpoint(test_client):
    """GET /api/v1/cohorts/survival-analysis must return Kaplan-Meier curves."""
    response = test_client.get("/api/v1/cohorts/survival-analysis?segment_by=gender")
    assert response.status_code == 200
    data = response.json()
    assert "overall_survival" in data
    assert "segmented_curves" in data
    assert len(data["overall_survival"]["timeline"]) > 0


def test_cohort_log_rank_test_endpoint(test_client):
    """GET /api/v1/cohorts/log-rank-test must execute hypothesis test."""
    response = test_client.get("/api/v1/cohorts/log-rank-test?segment_col=gender")
    assert response.status_code == 200
    data = response.json()
    assert "chi_square_statistic" in data
    assert "p_value" in data
    assert data["degrees_of_freedom"] == 1


def test_cohort_hazard_cliffs_endpoint(test_client):
    """GET /api/v1/cohorts/hazard-cliffs must detect acute churn cliff windows."""
    response = test_client.get("/api/v1/cohorts/hazard-cliffs?interval_days=15")
    assert response.status_code == 200
    data = response.json()
    assert "hazard_curve" in data
    assert len(data["hazard_curve"]) > 0


def test_cohort_retention_campaign_endpoint(test_client):
    """POST /api/v1/cohorts/retention-campaign must project business case ROI."""
    payload = {
        "intervention_cost_per_user": 14.0,
        "expected_churn_reduction_pct": 0.20
    }
    response = test_client.post("/api/v1/cohorts/retention-campaign", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_population"] > 0
    assert data["projected_saved_customers"] > 0
    assert "campaign_roi_pct" in data


# --- DAY 13 API TESTS: DYNAMIC PRICING, CAUSAL UPLIFT & FINANCIAL RISK ---
def test_pricing_bandit_summary_endpoint(test_client):
    """GET /api/v1/pricing/bandit/summary returns candidate arms and algorithm list."""
    response = test_client.get("/api/v1/pricing/bandit/summary")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "algorithms_supported" in data
    assert "Electronics" in data["categories"]


def test_pricing_quote_endpoint(test_client):
    """POST /api/v1/pricing/quote calculates real-time quote with dynamic multipliers."""
    payload = {
        "category": "Electronics",
        "base_price": 300.0,
        "current_stock": 8,
        "safety_stock": 25,
        "demand_velocity_ratio": 1.25,
        "competitor_price": 320.0
    }
    response = test_client.post("/api/v1/pricing/quote", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Electronics"
    assert data["recommended_dynamic_price"] > 0
    assert "multipliers" in data
    assert "stock_context" in data
    assert "guardrails" in data


def test_pricing_quote_validation_error(test_client):
    """POST /api/v1/pricing/quote returns 422 for negative base_price."""
    payload = {
        "category": "Beauty",
        "base_price": -20.0,
        "current_stock": 10
    }
    response = test_client.post("/api/v1/pricing/quote", json=payload)
    assert response.status_code == 422


def test_pricing_simulate_stream_endpoint(test_client):
    """POST /api/v1/pricing/simulate-stream executes bandit simulation."""
    payload = {
        "category": "Clothing",
        "algorithm": "thompson_sampling",
        "n_rounds": 100,
        "epsilon": 0.10
    }
    response = test_client.post("/api/v1/pricing/simulate-stream", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Clothing"
    assert data["algorithm"] == "thompson_sampling"
    assert data["cumulative_bandit_revenue"] > 0
    assert len(data["arm_statistics"]) == 5


def test_pricing_arms_endpoint(test_client):
    """GET /api/v1/pricing/arms returns arm catalog and category filter."""
    response = test_client.get("/api/v1/pricing/arms?category=Beauty")
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Beauty"
    assert "details" in data
    assert len(data["details"]["arms"]) == 5


def test_uplift_summary_endpoint(test_client):
    """GET /api/v1/uplift/summary returns the 4 causal uplift quadrants."""
    response = test_client.get("/api/v1/uplift/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] > 0
    assert len(data["quadrant_breakdown"]) == 4
    quad_names = [q["quadrant"] for q in data["quadrant_breakdown"]]
    assert "Persuadables" in quad_names
    assert "Sure Things" in quad_names


def test_customer_uplift_profile_endpoint(test_client):
    """GET /api/v1/uplift/customer/{customer_id} returns individual uplift score."""
    # First get a valid customer id from analytics
    response_cust = test_client.get("/api/v1/uplift/summary")
    assert response_cust.status_code == 200

    response = test_client.get("/api/v1/uplift/customer/CUST001")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST001"
    assert "causal_quadrant" in data
    assert "individual_treatment_effect_tau" in data


def test_customer_uplift_profile_not_found(test_client):
    """GET /api/v1/uplift/customer/{customer_id} returns 404 for unknown customer."""
    response = test_client.get("/api/v1/uplift/customer/UNKNOWN_CUST_9999")
    assert response.status_code in [404, 422]


def test_recommend_next_best_action_endpoint(test_client):
    """POST /api/v1/uplift/nba/recommend returns ranked action portfolio."""
    payload = {
        "customer_id": "CUST001",
        "estimated_aov": 280.0,
        "gross_margin_pct": 0.45,
        "recent_touches_count": 1
    }
    response = test_client.post("/api/v1/uplift/nba/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_action" in data
    assert "top_action_key" in data
    assert len(data["action_portfolio_ranking"]) == 5


def test_optimize_campaign_budget_endpoint(test_client):
    """POST /api/v1/uplift/campaign/optimize compares targeting strategies."""
    payload = {
        "campaign_budget": 8000.0,
        "incentive_cost_per_touch": 10.0,
        "avg_basket_value": 320.0,
        "gross_margin_pct": 0.45
    }
    response = test_client.post("/api/v1/uplift/campaign/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert "causal_uplift_targeting" in data["strategies"]
    assert "uplift_advantage_vs_propensity" in data


def test_cash_conversion_cycle_endpoint(test_client):
    """GET /api/v1/finance/ccc/summary returns storewide and category CCC metrics."""
    response = test_client.get("/api/v1/finance/ccc/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["store_blended_ccc_days"] > 0
    assert data["net_working_capital_tied_up"] > 0
    assert len(data["category_metrics"]) == 3


def test_monte_carlo_stress_test_endpoint(test_client):
    """POST /api/v1/finance/stress-test calculates VaR 95/99 and solvency runway."""
    payload = {
        "n_simulations": 500,
        "demand_shock_mean_pct": -0.10,
        "demand_shock_std_pct": 0.05,
        "cogs_inflation_mean_pct": 0.05,
        "cogs_inflation_std_pct": 0.03,
        "dpo_contraction_days": 4.0,
        "initial_cash_buffer": 55000.0
    }
    response = test_client.post("/api/v1/finance/stress-test", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["simulations_count"] == 500
    assert "stressed_outcomes" in data
    assert "liquidity_and_solvency" in data
    assert len(data["cash_flow_distribution"]) == 30


def test_simulate_working_capital_endpoint(test_client):
    """POST /api/v1/finance/working-capital/simulate models liquidity unlocked."""
    payload = {
        "dio_reduction_days": 10.0,
        "dpo_extension_days": 8.0,
        "dso_reduction_days": 1.0,
        "short_term_borrowing_rate_pct": 7.0
    }
    response = test_client.post("/api/v1/finance/working-capital/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["optimized_outcomes"]["total_permanent_liquidity_unlocked"] > 0
    assert data["optimized_outcomes"]["annual_borrowing_interest_savings"] > 0


def test_liquidity_runway_baseline_endpoint(test_client):
    """GET /api/v1/finance/liquidity-runway returns baseline solvency runway."""
    response = test_client.get("/api/v1/finance/liquidity-runway?initial_cash_buffer=65000")
    assert response.status_code == 200
    data = response.json()
    assert "liquidity_and_solvency" in data


# ============================================================================
# DAY 14: OMNICHANNEL FULFILLMENT, GEOSPATIAL GRAVITY & RETURN FRAUD FORENSICS
# ============================================================================


def test_fulfillment_network_endpoint(test_client):
    """GET /api/v1/fulfillment/network returns network topology."""
    response = test_client.get("/api/v1/fulfillment/network")
    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] == 6
    assert "nodes_catalog" in data
    assert "carrier_zones_rate_card" in data


def test_route_order_endpoint(test_client):
    """POST /api/v1/fulfillment/route-order routes single order."""
    payload = {
        "category": "Clothing",
        "quantity": 2,
        "price_per_unit": 50.0,
        "destination_zone": 2,
        "channel_preference": "BOPIS",
        "expedited": False,
    }
    response = test_client.post("/api/v1/fulfillment/route-order", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["optimal_fulfillment"]["channel"] == "BOPIS"
    assert data["optimal_fulfillment"]["total_fulfillment_cost"] == 1.50
    assert "cost_breakdown" in data


def test_route_order_validation_error(test_client):
    """POST /api/v1/fulfillment/route-order returns 422 on invalid category."""
    payload = {
        "category": "InvalidCat",
        "quantity": 1,
        "price_per_unit": 50.0,
        "destination_zone": 1,
    }
    response = test_client.post("/api/v1/fulfillment/route-order", json=payload)
    assert response.status_code == 422


def test_simulate_network_fulfillment_endpoint(test_client):
    """POST /api/v1/fulfillment/simulate runs multi-channel simulation."""
    payload = {
        "bopis_adoption_rate": 0.20,
        "sfs_allocation_rate": 0.30,
        "expedited_share_rate": 0.10,
    }
    response = test_client.post("/api/v1/fulfillment/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["simulation_parameters"]["total_orders_simulated"] == 1000
    assert "financial_summary" in data
    assert "operational_metrics" in data


def test_delivery_tradeoff_endpoint(test_client):
    """POST /api/v1/fulfillment/delivery-tradeoff evaluates SLA tiers."""
    payload = {
        "order_value": 300.0,
        "category": "Electronics",
        "destination_zone": 3,
    }
    response = test_client.post("/api/v1/fulfillment/delivery-tradeoff", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["sla_tiers"]) == 3


def test_fulfillment_esg_carbon_endpoint(test_client):
    """GET /api/v1/fulfillment/esg-carbon returns Scope 3 carbon audit."""
    response = test_client.get("/api/v1/fulfillment/esg-carbon")
    assert response.status_code == 200
    data = response.json()
    assert data["storewide_scope3_carbon_kg"] > 0
    assert "green_logistics_initiatives" in data


def test_geospatial_network_endpoint(test_client):
    """GET /api/v1/geospatial/network returns store nodes and tracts."""
    response = test_client.get("/api/v1/geospatial/network")
    assert response.status_code == 200
    data = response.json()
    assert data["total_active_stores"] == 3
    assert data["total_demand_tracts"] == 10


def test_huff_market_shares_endpoint(test_client):
    """GET /api/v1/geospatial/huff-shares returns Huff gravity shares."""
    response = test_client.get("/api/v1/geospatial/huff-shares?beta=1.75&lambda_param=1.0")
    assert response.status_code == 200
    data = response.json()
    assert "store_market_shares" in data
    assert "STORE_A_MANHATTAN" in data["store_market_shares"]


def test_store_catchment_endpoint(test_client):
    """GET /api/v1/geospatial/catchment/{store_id} returns trade zones."""
    response = test_client.get("/api/v1/geospatial/catchment/STORE_A_MANHATTAN")
    assert response.status_code == 200
    data = response.json()
    assert "primary_trade_area_60pct" in data
    assert "secondary_trade_area_80pct" in data


def test_store_catchment_not_found(test_client):
    """GET /api/v1/geospatial/catchment/{store_id} returns 422 for invalid store."""
    response = test_client.get("/api/v1/geospatial/catchment/STORE_NONEXISTENT")
    assert response.status_code == 422


def test_analyze_store_cannibalization_endpoint(test_client):
    """POST /api/v1/geospatial/cannibalization simulates candidate store."""
    payload = {
        "new_store_name": "Brooklyn Heights Expansion",
        "new_store_lat": 40.6960,
        "new_store_lon": -73.9933,
        "new_store_sqft": 30000.0,
        "new_store_attractiveness": 85.0,
        "beta": 1.75,
    }
    response = test_client.post("/api/v1/geospatial/cannibalization", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "financial_impact_summary" in data
    assert "pairwise_cannibalization_breakdown" in data


def test_evaluate_expansion_candidates_endpoint(test_client):
    """GET /api/v1/geospatial/expansion-candidates ranks candidate sites."""
    response = test_client.get("/api/v1/geospatial/expansion-candidates")
    assert response.status_code == 200
    data = response.json()
    assert len(data["ranked_expansion_sites"]) == 3


def test_benford_law_analysis_endpoint(test_client):
    """GET /api/v1/fraud/benford-analysis returns Benford test."""
    response = test_client.get("/api/v1/fraud/benford-analysis?target_column=Total Amount")
    assert response.status_code == 200
    data = response.json()
    assert data["target_column"] == "Total Amount"
    assert "chi_square_statistic" in data
    assert len(data["benford_distribution_table"]) == 9


def test_return_fraud_summary_endpoint(test_client):
    """GET /api/v1/fraud/return-summary returns storewide returns profile."""
    response = test_client.get("/api/v1/fraud/return-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 1000
    assert "wardrobing_fraud" in data
    assert "archetype_distribution" in data


def test_customer_return_risk_endpoint(test_client):
    """GET /api/v1/fraud/customer/{customer_id} evaluates customer return risk."""
    response = test_client.get("/api/v1/fraud/customer/CUST001")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST001"
    assert "composite_abuse_risk_score" in data
    assert "authorization_protocol" in data


def test_customer_return_risk_not_found(test_client):
    """GET /api/v1/fraud/customer/{customer_id} returns 422 for unknown customer."""
    response = test_client.get("/api/v1/fraud/customer/CUST99999")
    assert response.status_code == 422


def test_simulate_return_policy_endpoint(test_client):
    """POST /api/v1/fraud/policy-simulate models policy tightening."""
    payload = {
        "restocking_fee_pct": 0.10,
        "return_window_days": 14,
        "credit_only_for_high_risk": True,
    }
    response = test_client.post("/api/v1/fraud/policy-simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "financial_impact" in data
    assert data["financial_impact"]["net_annual_profit_uplift_usd"] > 0


