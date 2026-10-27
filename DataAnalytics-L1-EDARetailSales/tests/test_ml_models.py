"""
Unit tests validating machine learning training, cross-validation, and inference reliability.
"""

import pytest


def test_vip_classifier_training_metrics(ml_manager):
    """Verifies that the VIP Whale classifier achieves high benchmark performance."""
    metrics = ml_manager.vip_metrics
    assert metrics["accuracy"] >= 0.85
    assert metrics["roc_auc"] >= 0.85
    assert metrics["cv_mean_roc_auc"] >= 0.80
    assert "Quantity" in metrics["feature_importances"] or "Price per Unit" in metrics["feature_importances"]


def test_spend_regressor_training_metrics(ml_manager):
    """Verifies that the basket spend regressor achieves R^2 > 0.90."""
    metrics = ml_manager.spend_metrics
    assert metrics["r2_score"] >= 0.90
    assert metrics["mae"] >= 0.0


def test_predict_vip_inference_elite(ml_manager):
    """A basket of 4 units of $500 items must be classified as VIP with high probability."""
    res = ml_manager.predict_vip(
        age=32,
        gender="Male",
        product_category="Electronics",
        quantity=4,
        price_per_unit=500.0,
    )
    assert res["is_vip"] is True
    assert res["vip_probability"] >= 0.80
    assert res["risk_tier"] == "VERY_HIGH"
    assert res["calculated_basket_spend"] == 2000.0


def test_predict_vip_inference_budget(ml_manager):
    """A basket of 1 unit of $25 clothing must be classified as Budget with low VIP probability."""
    res = ml_manager.predict_vip(
        age=45,
        gender="Female",
        product_category="Clothing",
        quantity=1,
        price_per_unit=25.0,
    )
    assert res["is_vip"] is False
    assert res["vip_probability"] < 0.20
    assert res["risk_tier"] == "LOW"
    assert res["calculated_basket_spend"] == 25.0


def test_predict_spend_inference_bounds(ml_manager):
    """Verifies predicted spend falls within expected range bounds."""
    res = ml_manager.predict_spend(
        age=28,
        gender="Female",
        product_category="Beauty",
        quantity=3,
        price_per_unit=50.0,
    )
    assert res["predicted_spend"] > 0
    assert res["expected_range_min"] <= res["predicted_spend"] <= res["expected_range_max"]
    assert res["deterministic_catalog_spend"] == 150.0


def test_cross_sell_bundle_generation():
    """Verifies tiered discount math for cross-department bundles."""
    from src.recommendations import BusinessRecommendationEngine

    # 3-unit bundle -> 10% discount
    bundle_3 = BusinessRecommendationEngine.generate_bundle("Electronics", target_units=3)
    assert bundle_3["items_count"] == 3
    assert bundle_3["discount_percentage"] == 10.0
    assert bundle_3["discounted_bundle_price"] < bundle_3["original_total_price"]

    # 4-unit bundle -> 15% discount
    bundle_4 = BusinessRecommendationEngine.generate_bundle("Clothing", target_units=4)
    assert bundle_4["items_count"] == 4
    assert bundle_4["discount_percentage"] == 15.0
    assert bundle_4["customer_savings"] > 0
