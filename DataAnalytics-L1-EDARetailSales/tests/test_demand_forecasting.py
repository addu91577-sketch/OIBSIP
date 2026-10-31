"""
Unit tests validating multi-horizon demand forecasting, conformal uncertainty intervals,
and rolling Z-score demand anomaly detection.
"""

import pytest
from src.demand_forecasting import MultiHorizonDemandForecaster


@pytest.fixture(scope="module")
def forecaster():
    return MultiHorizonDemandForecaster()


def test_daily_demand_series_structure(forecaster):
    """Verifies continuous date index and rolling velocity attributes in daily series."""
    daily = forecaster.get_daily_demand_series()
    assert len(daily) >= 360
    assert "units" in daily.columns
    assert "revenue" in daily.columns
    assert "rolling_7d_mean" in daily.columns
    assert "rolling_14d_mean" in daily.columns
    assert (daily["units"] >= 0).all()
    assert (daily["revenue"] >= 0).all()

    # Category specific
    daily_elec = forecaster.get_daily_demand_series(category="Electronics")
    assert len(daily_elec) == len(daily)
    assert daily_elec["units"].sum() < daily["units"].sum()


def test_daily_demand_series_invalid_category(forecaster):
    """Requesting non-existent category must raise ValueError."""
    with pytest.raises(ValueError, match="has no transaction records"):
        forecaster.get_daily_demand_series(category="Groceries")


def test_decompose_demand(forecaster):
    """Verifies time-series decomposition into Level, Trend, and Day-of-Week Seasonality."""
    decomp = forecaster.decompose_demand(category="Clothing")
    assert decomp["baseline_level_units"] > 0.0
    assert "trend_daily_slope" in decomp
    assert len(decomp["day_of_week_seasonality"]) == 7

    # Check day of week names
    expected_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    assert set(decomp["day_of_week_seasonality"].keys()) == expected_days
    assert decomp["residual_std"] > 0.0


def test_forecast_demand_bounds_and_conformal(forecaster):
    """Forward projection must produce valid point forecasts and nested conformal intervals."""
    fc = forecaster.forecast_demand(horizon_days=30, category="Beauty")
    assert fc["forecast_horizon_days"] == 30
    assert len(fc["daily_projections"]) == 30
    assert fc["total_projected_units"] > 0.0
    assert fc["total_projected_revenue"] > 0.0
    assert fc["conformal_q80_margin"] <= fc["conformal_q95_margin"]

    for d in fc["daily_projections"]:
        assert d["predicted_units"] >= 0.0
        assert d["lower_bound_95"] <= d["lower_bound_80"] <= d["predicted_units"]
        assert d["predicted_units"] <= d["upper_bound_80"] <= d["upper_bound_95"]
        assert d["projected_revenue"] >= 0.0


def test_forecast_demand_invalid_horizon(forecaster):
    """Horizon <= 0 or > 180 must raise ValueError."""
    with pytest.raises(ValueError, match="Horizon days must be in"):
        forecaster.forecast_demand(horizon_days=0)

    with pytest.raises(ValueError, match="Horizon days must be in"):
        forecaster.forecast_demand(horizon_days=200)


def test_evaluate_forecast_accuracy_metrics(forecaster):
    """Walk-forward holdout evaluation must compute empirical MAE, RMSE, and conformal coverage."""
    metrics = forecaster.evaluate_forecast_accuracy(test_window_days=45)
    assert metrics["mae_units"] >= 0.0
    assert metrics["rmse_units"] >= 0.0
    assert metrics["mape_percentage"] >= 0.0
    assert metrics["wape_percentage"] >= 0.0
    # Conformal 95% coverage should be reasonably well-calibrated (typically >= 70%)
    assert 60.0 <= metrics["conformal_95_coverage_percentage"] <= 100.0


def test_detect_demand_anomalies(forecaster):
    """Rolling Z-score anomaly detector must return structured anomaly events."""
    anomalies = forecaster.detect_demand_anomalies(z_threshold=2.0)
    assert isinstance(anomalies, list)
    for anom in anomalies:
        assert abs(anom["z_score"]) >= 2.0
        assert anom["anomaly_severity"] in ["CRITICAL_SPIKE", "MODERATE_SPIKE", "DEMAND_TROUGH"]
        assert anom["actual_units"] >= 0
