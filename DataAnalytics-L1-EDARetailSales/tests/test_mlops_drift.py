"""
Unit and integration tests for MLOps drift monitoring and statistical tests.
"""

import numpy as np
import pandas as pd
import pytest
from src.mlops_drift import (
    calculate_psi,
    calculate_categorical_psi,
    DriftDetector,
)


def test_psi_identical_distributions_zero():
    """Verifies that PSI between two identical samples evaluates to 0.0."""
    sample = np.random.normal(loc=100.0, scale=15.0, size=500)
    psi = calculate_psi(sample, sample)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_significant_drift_detection():
    """Verifies that a substantial distribution shift produces PSI >= 0.25."""
    base = np.random.normal(loc=100.0, scale=15.0, size=1000)
    shifted = np.random.normal(loc=160.0, scale=25.0, size=1000)
    psi = calculate_psi(base, shifted)
    assert psi >= 0.25


def test_categorical_psi_and_tvd():
    """Tests PSI and Total Variation Distance for categorical distributions."""
    base_cat = pd.Series(["Clothing"] * 333 + ["Electronics"] * 334 + ["Beauty"] * 333)
    # Drifted: 80% Electronics
    shifted_cat = pd.Series(["Clothing"] * 100 + ["Electronics"] * 800 + ["Beauty"] * 100)

    psi, tvd = calculate_categorical_psi(base_cat, shifted_cat)
    assert psi >= 0.20
    assert 0.0 < tvd <= 1.0


def test_drift_detector_audit_stable():
    """Verifies that normal random sampling from baseline results in HEALTHY_STABLE status."""
    detector = DriftDetector()
    stable_batch = detector.generate_synthetic_batch(scenario="stable", n_samples=300, random_seed=42)
    audit = detector.run_drift_audit(stable_batch)

    assert audit["global_status"] == "HEALTHY_STABLE"
    assert audit["retrain_recommended"] is False
    assert audit["critical_features_count"] == 0
    assert audit["max_psi"] < 0.10


def test_drift_detector_audit_critical_tech_surge():
    """Verifies that tech surge triggers CRITICAL_DRIFT alert and retraining flag."""
    detector = DriftDetector()
    surge_batch = detector.generate_synthetic_batch(scenario="tech_surge", n_samples=300, random_seed=99)
    audit = detector.run_drift_audit(surge_batch)

    assert audit["global_status"] == "CRITICAL_DRIFT"
    assert audit["retrain_recommended"] is True
    assert audit["critical_features_count"] >= 1
    assert "retraining" in audit["action_plan"].lower()
