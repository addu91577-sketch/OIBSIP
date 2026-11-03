"""Unit and integration test suite for Day 14 App Store Fraud & Anomaly Surveillance Engine.

Validates:
1. Engine initialization, dataset indexing, and regularized covariance baseline calibration.
2. Benford's Law theoretical properties (monotonicity, sum to 1.0) and first-digit extraction.
3. Benford's Law Chi-Square and Kolmogorov-Smirnov goodness-of-fit testing on synthetic and real data.
4. Shannon lexical entropy calculation and repetition ratio for bot/spam detection.
5. Review sentiment vs star rating dissonance evaluation on real and fallback apps.
6. Multivariate Mahalanobis outlier detection and anomaly reason attribution.
7. Android dangerous permission and APK binary bloat security risk profiling.
8. Composite App Trust & Integrity Scorecard (ATIS 0-100), trust tiers, and compliance directives.
9. Synthetic / unlisted app trust auditing with custom parameters.
10. Category-level integrity benchmarks and top suspicious apps ranking.
11. Verification of analytical CSV reports and publication figures (Figures 28 & 29).
12. Production FastAPI REST microservice endpoints for fraud auditing and Benford testing.
"""

import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.fraud_anomaly_surveillance_engine import (
    PlayStoreFraudSurveillanceEngine,
    BenfordTestResult,
    ReviewDissonanceResult,
    MultivariateAnomalyResult,
    AppTrustScoreResult,
    CategoryIntegrityBenchmark,
    BENFORD_THEORETICAL
)
from src.production_api import app

_BASE_DIR = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


@pytest.fixture(scope="module")
def fraud_engine():
    """Module-scoped fixture providing an initialized PlayStoreFraudSurveillanceEngine."""
    return PlayStoreFraudSurveillanceEngine()


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped FastAPI test client."""
    return TestClient(app)


# ============================================================================
# 1. Engine Initialization & Dataset Indexing
# ============================================================================

def test_engine_initialization(fraud_engine):
    """Verify that the engine loads datasets and builds category baselines."""
    assert fraud_engine._apps_df is not None
    assert len(fraud_engine._apps_df) >= 9000
    assert "GLOBAL" in fraud_engine._category_models
    assert "GAME" in fraud_engine._category_models
    assert len(fraud_engine._app_name_index) > 0


# ============================================================================
# 2. Benford's Law Theoretical Properties & Digit Extraction
# ============================================================================

def test_benford_theoretical_properties():
    """Ensure theoretical Benford probabilities sum to 1.0 and decrease monotonically."""
    total_prob = sum(BENFORD_THEORETICAL.values())
    assert abs(total_prob - 1.0) < 1e-6
    probs = [BENFORD_THEORETICAL[d] for d in range(1, 10)]
    assert probs == sorted(probs, reverse=True)
    assert abs(BENFORD_THEORETICAL[1] - 0.30103) < 1e-4
    assert abs(BENFORD_THEORETICAL[9] - 0.04576) < 1e-4


def test_extract_first_digits():
    """Test leading significant digit extraction across integers, floats, and scientific notation."""
    samples = [123, 0.0456, 98765, 3.1415, 7.0e-5, -10, 0, "invalid"]
    digits = PlayStoreFraudSurveillanceEngine.extract_first_digits(samples)
    assert list(digits) == [1, 4, 9, 3, 7]


def test_benford_goodness_of_fit_synthetic(fraud_engine):
    """Verify that a synthetic Benford distribution conforms while uniform distribution diverges."""
    np.random.seed(42)
    # Generate conformant Benford distribution via inverse CDF sampling
    u = np.random.uniform(0, 1, 3000)
    benford_samples = 10 ** u

    conformant_res = fraud_engine.test_benfords_law(benford_samples, metric_name="Synthetic_Benford")
    assert conformant_res.sample_size == 3000
    assert conformant_res.p_value >= 0.01
    assert conformant_res.is_conformant is True

    # Generate uniform leading digit distribution (violates Benford)
    uniform_digits = np.random.choice(range(1, 10), size=3000)
    divergent_res = fraud_engine.test_benfords_law(uniform_digits, metric_name="Synthetic_Uniform")
    assert divergent_res.chi2_stat > 30.0
    assert divergent_res.conformance_tier == "CRITICAL_BENFORD_ANOMALY"


def test_benford_small_sample_fallback(fraud_engine):
    """Ensure small sample sizes (< 10) trigger safe fallback without throwing errors."""
    res = fraud_engine.test_benfords_law([12, 34], metric_name="Tiny_Sample")
    assert res.sample_size == 2
    assert res.is_conformant is True
    assert res.conformance_tier == "ORGANIC_BENFORD_CONFORMANT"


def test_category_benford_audit(fraud_engine):
    """Verify category-level Benford auditing for reviews and installs."""
    audit = fraud_engine.audit_category_benford("GAME")
    assert "reviews_benford" in audit
    assert "installs_benford" in audit
    assert isinstance(audit["reviews_benford"], BenfordTestResult)
    assert audit["reviews_benford"].sample_size > 100


# ============================================================================
# 3. Shannon Lexical Entropy & Review Dissonance
# ============================================================================

def test_lexical_entropy_and_repetition():
    """Verify that repetitive bot spam produces lower Shannon entropy than rich text."""
    spam_reviews = ["great app good great good app good great nice nice"] * 10
    rich_reviews = [
        "The interface design is remarkably intuitive with swift navigation and zero latency.",
        "Exceptional utility that seamlessly integrates with external productivity services."
    ]

    spam_entropy, spam_rep = PlayStoreFraudSurveillanceEngine.calculate_lexical_entropy(spam_reviews)
    rich_entropy, rich_rep = PlayStoreFraudSurveillanceEngine.calculate_lexical_entropy(rich_reviews)

    assert spam_entropy < rich_entropy
    assert spam_rep > rich_rep


def test_evaluate_review_dissonance_real_app(fraud_engine):
    """Verify review sentiment dissonance calculation on a known catalog app with reviews."""
    app_with_reviews = "Coloring book moana"
    res = fraud_engine.evaluate_review_dissonance(app_with_reviews)
    assert isinstance(res, ReviewDissonanceResult)
    assert res.review_count > 0
    assert 1.0 <= res.star_rating <= 5.0
    assert -1.0 <= res.mean_polarity <= 1.0
    assert 0.0 <= res.dissonance_score <= 1.0
    assert res.bot_suspicion_tier in ["ORGANIC_HUMAN", "SUSPECT_INCONGRUENT", "HIGH_BOT_BURST"]


def test_evaluate_review_dissonance_fallback(fraud_engine):
    """Verify graceful handling for apps lacking review text records."""
    res = fraud_engine.evaluate_review_dissonance("NonExistentApp12345")
    assert res.review_count == 0
    assert res.bot_suspicion_tier == "ORGANIC_HUMAN"
    assert res.dissonance_score == 0.15


# ============================================================================
# 4. Multivariate Mahalanobis Anomaly & Outlier Isolation
# ============================================================================

def test_multivariate_anomaly_normal_app(fraud_engine):
    """Ensure standard applications receive low anomaly scores and are not marked as outliers."""
    res = fraud_engine.evaluate_multivariate_anomaly(
        app_name="Typical Social App",
        category="SOCIAL",
        installs=100000,
        reviews=2500,
        rating=4.2,
        size_mb=25.0,
        price=0.0
    )
    assert isinstance(res, MultivariateAnomalyResult)
    assert res.anomaly_score < 0.8
    assert res.is_outlier is False


def test_multivariate_anomaly_extreme_outlier(fraud_engine):
    """Ensure abnormal configurations trigger high Mahalanobis distance and outlier flags."""
    res = fraud_engine.evaluate_multivariate_anomaly(
        app_name="Suspicious Extortion App",
        category="TOOLS",
        installs=5,
        reviews=200000,
        rating=4.9,
        size_mb=180.0,
        price=399.99
    )
    assert res.is_outlier is True
    assert res.mahalanobis_distance > 3.8
    assert len(res.outlier_reasons) > 0


# ============================================================================
# 5. Security & Permission Bloat Risk Profiling
# ============================================================================

def test_security_risk_profiling(fraud_engine):
    """Verify penalty calculation for heavy APK bloat and high-danger categories."""
    pen_safe, flags_safe = fraud_engine.evaluate_security_risk(category="EDUCATION", size_mb=15.0, price=0.0)
    pen_risky, flags_risky = fraud_engine.evaluate_security_risk(category="TOOLS", size_mb=95.0, price=150.0)

    assert pen_risky > pen_safe
    assert len(flags_risky) >= 2


# ============================================================================
# 6. Composite App Trust & Integrity Scorecard (ATIS)
# ============================================================================

def test_audit_app_trust_known_app(fraud_engine):
    """Verify composite ATIS scorecard evaluation for a known application."""
    res = fraud_engine.audit_app_trust("Subway Surfers")
    assert isinstance(res, AppTrustScoreResult)
    assert 5.0 <= res.trust_score <= 100.0
    assert res.trust_tier in [
        "TIER_A_VERIFIED_ORGANIC", "TIER_B_NORMAL_TRUST",
        "TIER_C_ELEVATED_SUSPICION", "TIER_D_CRITICAL_FRAUD_ALERT"
    ]
    assert res.compliance_action in [
        "VERIFIED_CLEAR", "ENHANCED_TELEMETRY",
        "TRUST_AND_SAFETY_AUDIT", "STORE_TAKEDOWN_RECOMMENDED"
    ]


def test_audit_app_trust_synthetic_fraudulent(fraud_engine):
    """Verify that an egregious fraudulent concept triggers critical fraud alert and takedown action."""
    res = fraud_engine.audit_app_trust(
        app_name="Bot Farm Clicker Scam",
        custom_category="TOOLS",
        custom_rating=4.9,
        custom_reviews=950000,
        custom_installs=10,
        custom_price=389.99,
        custom_size=190.0
    )
    assert res.trust_score < 60.0
    assert res.trust_tier in ["TIER_C_ELEVATED_SUSPICION", "TIER_D_CRITICAL_FRAUD_ALERT"]
    assert res.compliance_action in ["TRUST_AND_SAFETY_AUDIT", "STORE_TAKEDOWN_RECOMMENDED"]
    assert len(res.audit_triggers) > 0


def test_app_trust_to_dict(fraud_engine):
    """Verify dictionary serialization for REST API and JSON export."""
    res = fraud_engine.audit_app_trust("Pixel Draw")
    d = res.to_dict()
    assert isinstance(d, dict)
    assert "trust_score" in d
    assert "trust_tier" in d
    assert "compliance_action" in d


# ============================================================================
# 7. Batch Catalog Auditing & Benchmarks
# ============================================================================

def test_audit_catalog_sampling(fraud_engine):
    """Verify that batch catalog auditing returns structured DataFrame."""
    df_sample = fraud_engine.audit_catalog(sample_size=20)
    assert isinstance(df_sample, pd.DataFrame)
    assert len(df_sample) == 20
    assert "Trust_Score" in df_sample.columns
    assert "Compliance_Action" in df_sample.columns
    assert "Audit_Triggers" in df_sample.columns


def test_category_benchmarks(fraud_engine):
    """Verify category integrity benchmarks generation."""
    bench = fraud_engine.generate_category_benchmarks()
    assert isinstance(bench, pd.DataFrame)
    assert len(bench) >= 10
    assert "Category" in bench.columns
    assert "Mean_Trust_Score" in bench.columns
    assert "Anomaly_Rate_Pct" in bench.columns


def test_top_suspicious_apps(fraud_engine):
    """Verify that get_top_suspicious_apps retrieves ascending sorted list of lowest trust scores."""
    suspicious = fraud_engine.get_top_suspicious_apps(n=10)
    assert len(suspicious) == 10
    scores = suspicious["Trust_Score"].tolist()
    assert scores == sorted(scores)


# ============================================================================
# 8. Analytical Reports and Publication Figures Verification
# ============================================================================

def test_analytical_csv_reports_exist():
    """Verify that Day 14 analytical CSV reports are present and non-empty."""
    scorecard_path = _REPORTS_DIR / "app_fraud_anomaly_scorecard.csv"
    benchmarks_path = _REPORTS_DIR / "category_integrity_benchmarks.csv"
    suspicious_path = _REPORTS_DIR / "suspicious_apps_audit_report.csv"

    for p in [scorecard_path, benchmarks_path, suspicious_path]:
        assert p.exists(), f"Missing analytical CSV: {p}"
        assert p.stat().st_size > 0, f"Empty analytical CSV: {p}"


def test_publication_figures_exist():
    """Verify that Figures 28 and 29 are generated and non-empty."""
    fig28 = _FIGURES_DIR / "28_benford_law_and_review_manipulation.png"
    fig29 = _FIGURES_DIR / "29_multivariate_anomalies_and_risk_quadrant.png"

    for f in [fig28, fig29]:
        assert f.exists(), f"Missing publication figure: {f}"
        assert f.stat().st_size > 10000, f"Publication figure too small: {f}"


# ============================================================================
# 9. FastAPI REST Microservice Endpoints
# ============================================================================

def test_api_fraud_audit_endpoint(api_client):
    """Verify POST /api/v1/fraud/audit-app endpoint."""
    payload = {
        "app_name": "Test Security Auditor App",
        "category": "TOOLS",
        "rating": 4.1,
        "reviews": 1200,
        "installs": 50000,
        "price": 0.0,
        "size_mb": 25.0
    }
    resp = api_client.post("/api/v1/fraud/audit-app", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "trust_score" in data
    assert "trust_tier" in data
    assert "compliance_action" in data


def test_api_benford_test_endpoint(api_client):
    """Verify POST /api/v1/fraud/benford-test endpoint."""
    payload = {
        "values": [100, 200, 300, 400, 500, 150, 180, 220, 310, 105, 120, 250],
        "metric_name": "API_Test_Metric"
    }
    resp = api_client.post("/api/v1/fraud/benford-test", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "chi2_stat" in data
    assert "p_value" in data
    assert "conformance_tier" in data


def test_api_suspicious_apps_endpoint(api_client):
    """Verify GET /api/v1/fraud/suspicious-apps endpoint."""
    resp = api_client.get("/api/v1/fraud/suspicious-apps?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) <= 5
    if data:
        assert "Trust_Score" in data[0]
