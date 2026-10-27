"""Unit and integration test suite for Day 9 Machine Learning & MLOps Drift Engine.

Validates:
1. Mathematical properties of Population Stability Index (PSI), KS test, and Wasserstein distance.
2. Supervised classification pipeline and performance bounds (ROC-AUC > 0.85).
3. Rating regression pipeline and error bounds (RMSE < 0.60).
4. Feature importance extraction and sanity.
5. Real-time inference routines (latency, probabilities, confidence intervals).
6. MLOps drift reporting and threshold verdicts.
7. Publication artifact generation (Figures 18, 19, and CSV reports).
"""

from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictive_modeling import (
    PlayStorePredictiveEngine,
    calculate_psi,
    calculate_ks_statistic,
    calculate_wasserstein,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def predictive_engine():
    """Module-scoped fixture to instantiate and train the predictive engine once."""
    engine = PlayStorePredictiveEngine()
    engine.train_models()
    return engine


def test_engine_initialization_and_schema(predictive_engine):
    """Test engine initialization, data loading, and feature matrix dimensions."""
    assert predictive_engine.df_apps is not None
    assert len(predictive_engine.df_apps) == 9659
    assert len(predictive_engine.feature_columns) >= 30
    assert predictive_engine.X_cls.shape[0] == 9659
    assert predictive_engine.y_cls.shape[0] == 9659
    assert predictive_engine.X_reg.shape[0] == 8196
    assert predictive_engine.y_reg.shape[0] == 8196


def test_psi_calculation_mathematical_properties():
    """Verify that PSI is 0 for identical distributions and strictly positive for shifted ones."""
    np.random.seed(42)
    dist_a = np.random.normal(loc=10.0, scale=2.0, size=2000)
    dist_identical = dist_a.copy()
    dist_shifted = np.random.normal(loc=14.0, scale=2.0, size=2000)

    psi_zero = calculate_psi(dist_a, dist_identical)
    assert psi_zero < 0.01, f"Expected PSI near 0 for identical distributions, got {psi_zero}"

    psi_shifted = calculate_psi(dist_a, dist_shifted)
    assert psi_shifted > 0.25, f"Expected high PSI (>0.25) for heavily shifted distributions, got {psi_shifted}"


def test_ks_and_wasserstein_metrics():
    """Verify two-sample Kolmogorov-Smirnov test and Wasserstein distance calculations."""
    np.random.seed(42)
    s1 = np.random.uniform(0, 10, size=1000)
    s2 = s1.copy()
    s3 = np.random.uniform(5, 15, size=1000)

    stat_same, pval_same = calculate_ks_statistic(s1, s2)
    assert stat_same == 0.0
    assert pval_same == 1.0

    stat_diff, pval_diff = calculate_ks_statistic(s1, s3)
    assert stat_diff > 0.3
    assert pval_diff < 0.001

    w_same = calculate_wasserstein(s1, s2)
    assert w_same == 0.0

    w_diff = calculate_wasserstein(s1, s3)
    assert w_diff > 2.0


def test_train_models_and_metrics(predictive_engine):
    """Verify that trained classifiers and regressors exceed minimum SLA performance thresholds."""
    cls_metrics = predictive_engine.classification_metrics
    assert "Random_Forest" in cls_metrics
    assert "Gradient_Boosting" in cls_metrics
    assert "Logistic_Regression" in cls_metrics

    rf_auc = cls_metrics["Random_Forest"]["roc_auc"]
    assert rf_auc >= 0.85, f"Random Forest ROC-AUC {rf_auc} fell below 0.85 SLA"
    assert cls_metrics["Random_Forest"]["accuracy"] >= 0.85

    reg_metrics = predictive_engine.regression_metrics
    assert "Random_Forest_Regressor" in reg_metrics
    rf_rmse = reg_metrics["Random_Forest_Regressor"]["rmse"]
    assert rf_rmse < 0.60, f"Random Forest Regressor RMSE {rf_rmse} exceeded 0.60 bound"


def test_feature_importances_attribution(predictive_engine):
    """Verify that feature importances are computed and sum to approximately 1.0."""
    importances = predictive_engine.feature_importances_
    assert len(importances) == len(predictive_engine.feature_columns)
    total_imp = sum(importances.values())
    assert 0.99 <= total_imp <= 1.01

    # log_reviews and size_mb should be among prominent features
    top_features = list(importances.keys())[:5]
    assert any("reviews" in f.lower() or "size" in f.lower() for f in top_features)


def test_predict_app_success_inference(predictive_engine):
    """Verify single app success prediction routine produces valid probabilities and risk assessment."""
    # Warm-up call
    _ = predictive_engine.predict_app_success(category="TOOLS", size_mb=10.0, price=0.0)

    res_game = predictive_engine.predict_app_success(category="GAME", size_mb=65.0, price=0.0)
    assert 0.0 <= res_game["success_probability"] <= 1.0
    assert res_game["predicted_tier"] in [
        "High Velocity Scale (>= 1,000,000 Installs)",
        "Moderate Scale Potential (100k - 1M Installs)",
        "Niche / Boutique Audience (< 100k Installs)",
    ]
    assert res_game["risk_assessment"] in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    assert res_game["latency_ms"] < 250.0  # responsive real-time inference
    assert len(res_game["key_drivers"]) >= 1


def test_predict_app_rating_inference(predictive_engine):
    """Verify single app rating valuation routine produces valid rating bounds and confidence interval."""
    res_rating = predictive_engine.predict_app_rating(category="EDUCATION", size_mb=25.0, price=1.99)
    assert 1.0 <= res_rating["predicted_rating"] <= 5.0
    ci = res_rating["confidence_interval_90"]
    assert len(ci) == 2
    assert ci[0] <= res_rating["predicted_rating"] <= ci[1]
    assert "Quality" in res_rating["quality_tier"] or "Mainstream" in res_rating["quality_tier"]


def test_generate_drift_report_and_thresholds(predictive_engine):
    """Verify MLOps distribution drift report structure, PSI evaluation, and health verdict."""
    rep = predictive_engine.generate_drift_report()
    assert "health_verdict" in rep
    assert rep["health_verdict"] in ["HEALTHY", "MODERATE_DRIFT", "ACTION_REQUIRED"]
    assert rep["overall_max_psi"] >= 0.0
    assert len(rep["metrics"]) >= 4

    for m in rep["metrics"]:
        assert "feature" in m
        assert "psi" in m
        assert "ks_statistic" in m
        assert "wasserstein_distance" in m
        assert m["status"].startswith(("PASS", "MONITOR", "ALERT"))


def test_day9_artifacts_generation(predictive_engine):
    """Verify that Figures 18 & 19 and summary CSVs exist and are non-empty."""
    fig18 = _PROJECT_ROOT / "reports" / "figures" / "18_predictive_modeling_benchmarks.png"
    fig19 = _PROJECT_ROOT / "reports" / "figures" / "19_rating_regression_and_drift_monitoring.png"
    bench_csv = _PROJECT_ROOT / "reports" / "ml_model_performance_benchmarks.csv"
    feat_csv = _PROJECT_ROOT / "reports" / "ml_feature_importances.csv"
    drift_csv = _PROJECT_ROOT / "reports" / "mlops_drift_metrics.csv"

    assert fig18.exists() and fig18.stat().st_size > 100_000
    assert fig19.exists() and fig19.stat().st_size > 100_000
    assert bench_csv.exists() and bench_csv.stat().st_size > 200
    assert feat_csv.exists() and feat_csv.stat().st_size > 200
    assert drift_csv.exists() and drift_csv.stat().st_size > 200
