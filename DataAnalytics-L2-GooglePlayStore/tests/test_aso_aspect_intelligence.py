"""Unit and integration test suite for Day 12 ASO & Review Aspect Intelligence Engine.

Validates:
1. Engine initialization, dataset indexing, and precompiled regex patterns.
2. Aspect-Based Sentiment Analysis (ABSA) across UI/UX, Performance, Ads, Features, Support.
3. Category-level ABSA aggregation and unseen app fallback logic.
4. Mathematical bounds of Net Sentiment Score (NSS) and Aspect Health Index.
5. Latent Dirichlet Allocation (LDA) review topic extraction, clustering, and caching.
6. Review authenticity auditing, bot brigading detection, duplicate hashing, and entropy.
7. Competitor ASO keyword gap mining, opportunity weighting, and metadata copy synthesis.
8. 360-degree App Health Index calculation, multi-pillar weighting, and health tier classification.
9. Verification of generated publication figures (Figures 24 & 25) and summary CSV reports.
10. FastAPI REST API endpoints:
    - POST /api/v1/aso/aspect-analysis
    - GET  /api/v1/aso/topics
    - POST /api/v1/aso/review-authenticity
    - POST /api/v1/aso/keyword-gap
    - POST /api/v1/aso/app-health-diagnostic
"""

import math
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.aso_intelligence_engine import (
    ASOReviewIntelligenceEngine,
    AspectScore,
    AppAspectAnalysisResult,
    TopicCluster,
    ReviewAuthenticityReport,
    ASOGapAnalysisResult,
    AppHealthDiagnosticResult,
    ASPECT_LEXICONS
)
from src.production_api import app

_BASE_DIR = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


@pytest.fixture(scope="module")
def aso_engine():
    """Module-scoped fixture providing an initialized ASOReviewIntelligenceEngine."""
    return ASOReviewIntelligenceEngine()


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# 1. Engine Core & Dataset Verification
# ============================================================================

def test_engine_initialization(aso_engine):
    """Verify datasets loaded and precompiled patterns ready."""
    assert len(aso_engine.df_apps) >= 9000
    assert len(aso_engine.df_reviews) >= 25000
    assert len(aso_engine.categories) >= 30
    assert set(aso_engine._aspect_patterns.keys()) == set(ASPECT_LEXICONS.keys())


# ============================================================================
# 2. Aspect-Based Sentiment Analysis (ABSA) Tests
# ============================================================================

def test_aspect_sentiment_single_app(aso_engine):
    """Test granular aspect-based sentiment assessment on a representative app."""
    res = aso_engine.analyze_aspect_sentiment("Duolingo: Learn Languages Free", is_category=False)
    assert isinstance(res, AppAspectAnalysisResult)
    assert res.target_name == "Duolingo: Learn Languages Free"
    assert res.total_reviews_analyzed > 0
    assert set(res.aspect_scores.keys()) == set(ASPECT_LEXICONS.keys())
    assert res.dominant_strength in ASPECT_LEXICONS
    assert res.dominant_vulnerability in ASPECT_LEXICONS
    assert 0.0 <= res.aggregate_aspect_health <= 100.0

    for aspect_name, score in res.aspect_scores.items():
        assert isinstance(score, AspectScore)
        assert score.mention_count >= 0
        assert -1.0 <= score.mean_polarity <= 1.0
        assert 0.0 <= score.mean_subjectivity <= 1.0
        assert -100.0 <= score.net_sentiment_score <= 100.0
        assert 0.0 <= score.aspect_health_index <= 100.0


def test_aspect_sentiment_category_level(aso_engine):
    """Test category-level ABSA aggregation."""
    res = aso_engine.analyze_aspect_sentiment("EDUCATION", is_category=True)
    assert isinstance(res, AppAspectAnalysisResult)
    assert res.total_reviews_analyzed > 0
    assert "ui_ux" in res.aspect_scores
    assert "performance" in res.aspect_scores


def test_aspect_sentiment_fallback_unseen_app(aso_engine):
    """Verify graceful handling for apps not present in review corpus."""
    res = aso_engine.analyze_aspect_sentiment("Completely Nonexistent Mock Application 2026", is_category=False)
    assert isinstance(res, AppAspectAnalysisResult)
    assert res.total_reviews_analyzed > 0
    assert 0.0 <= res.aggregate_aspect_health <= 100.0


def test_net_sentiment_score_bounds(aso_engine):
    """Verify Net Sentiment Score mathematical boundaries."""
    res = aso_engine.analyze_aspect_sentiment("Subway Surfers")
    for aspect_name, score in res.aspect_scores.items():
        if score.mention_count > 0:
            calc_nss = ((score.positive_mentions - score.negative_mentions) / score.mention_count) * 100.0
            assert math.isclose(score.net_sentiment_score, round(calc_nss, 2), abs_tol=0.1)


# ============================================================================
# 3. Latent Dirichlet Allocation (LDA) Topic Modeling Tests
# ============================================================================

def test_extract_lda_topics(aso_engine):
    """Verify LDA topic discovery returns coherent thematic clusters."""
    topics = aso_engine.extract_lda_topics(num_topics=5, max_reviews=5000)
    assert len(topics) == 5
    for topic in topics:
        assert isinstance(topic, TopicCluster)
        assert 1 <= topic.topic_id <= 5
        assert len(topic.top_words) >= 5
        assert topic.topic_weight > 0.0
        assert topic.prevalent_sentiment in {"Positive", "Negative", "Neutral"}


def test_lda_topic_caching(aso_engine):
    """Verify that cached LDA topics are returned instantaneously."""
    topics1 = aso_engine.extract_lda_topics(num_topics=5)
    topics2 = aso_engine.extract_lda_topics(num_topics=5)
    assert len(topics1) == len(topics2)
    assert topics1[0].top_words == topics2[0].top_words


# ============================================================================
# 4. Review Authenticity & Brigading Detection Tests
# ============================================================================

def test_review_authenticity_evaluation(aso_engine):
    """Audit review authenticity on a catalog application."""
    rep = aso_engine.evaluate_review_authenticity("Colorfy: Colouring Book for Adults")
    assert isinstance(rep, ReviewAuthenticityReport)
    assert rep.total_reviews > 0
    assert 0.0 <= rep.authenticity_index <= 100.0
    assert 0.0 <= rep.entropy_score <= 1.0
    assert rep.verdict in {"EXCELLENT_ORGANIC", "MODERATE_NOISE", "SUSPICIOUS_BRIGADING"}
    assert rep.suspicious_reviews_count >= 0


def test_review_authenticity_empty_fallback(aso_engine):
    """Verify authenticity reporting for an unseen app."""
    rep = aso_engine.evaluate_review_authenticity("UnseenAppMock9999")
    assert isinstance(rep, ReviewAuthenticityReport)
    assert rep.authenticity_index >= 0.0
    assert rep.verdict in {"EXCELLENT_ORGANIC", "ACCEPTABLE_ORGANIC", "MODERATE_NOISE", "SUSPICIOUS_BRIGADING"}


# ============================================================================
# 5. Competitor ASO Keyword Gap Mining Tests
# ============================================================================

def test_mine_aso_keyword_gap(aso_engine):
    """Verify competitor keyword gap miner extracts actionable keywords."""
    res = aso_engine.mine_aso_keyword_gap("Duolingo: Learn Languages Free", top_n_keywords=8)
    assert isinstance(res, ASOGapAnalysisResult)
    assert res.target_app == "Duolingo: Learn Languages Free"
    assert len(res.missing_high_value_keywords) <= 8
    assert len(res.missing_high_value_keywords) > 0
    assert 0.0 <= res.aso_optimization_score <= 100.0
    assert "Duolingo" in res.prescribed_title_copy
    assert len(res.prescribed_feature_bullets) >= 3


def test_aso_prescribed_copy_synthesis(aso_engine):
    """Verify title copy synthesis formatting."""
    res = aso_engine.mine_aso_keyword_gap("Subway Surfers", top_n_keywords=5)
    assert len(res.prescribed_title_copy) > len("Subway Surfers")
    assert any("Search visibility" in b or "Optimize" in b for b in res.prescribed_feature_bullets)


# ============================================================================
# 6. Holistic 360-Degree App Health Diagnostics Tests
# ============================================================================

def test_evaluate_app_health(aso_engine):
    """Verify composite 360-degree app health scorecard."""
    diag = aso_engine.evaluate_app_health("Duolingo: Learn Languages Free")
    assert isinstance(diag, AppHealthDiagnosticResult)
    assert diag.app_name == "Duolingo: Learn Languages Free"
    assert 0.0 <= diag.composite_health_index <= 100.0
    assert 0.0 <= diag.rating_score <= 100.0
    assert 0.0 <= diag.sentiment_score <= 100.0
    assert 0.0 <= diag.aspect_health_score <= 100.0
    assert 0.0 <= diag.authenticity_score <= 100.0
    assert 0.0 <= diag.aso_optimization_score <= 100.0
    assert diag.health_tier in {"CHAMPION", "SOLID_PERFORMER", "VULNERABLE", "CRITICAL_RENOVATION"}
    assert len(diag.prescriptions) >= 3


def test_app_health_weighting_consistency(aso_engine):
    """Verify that composite index accurately adheres to 30/25/20/15/10 weighting."""
    diag = aso_engine.evaluate_app_health("Subway Surfers")
    expected_composite = (
        0.30 * diag.rating_score +
        0.25 * diag.sentiment_score +
        0.20 * diag.aspect_health_score +
        0.15 * diag.authenticity_score +
        0.10 * diag.aso_optimization_score
    )
    assert math.isclose(diag.composite_health_index, round(expected_composite, 2), abs_tol=0.1)


# ============================================================================
# 7. Verification of Artifacts on Disk
# ============================================================================

def test_publication_figures_exist():
    """Verify Figure 24 and Figure 25 exist and meet publication size criteria."""
    fig24 = _FIGURES_DIR / "24_aspect_based_sentiment_and_topic_modeling.png"
    fig25 = _FIGURES_DIR / "25_aso_competitor_keyword_gap_and_app_health.png"
    assert fig24.exists(), f"Missing Figure 24 at {fig24}"
    assert fig25.exists(), f"Missing Figure 25 at {fig25}"
    assert fig24.stat().st_size > 50_000, "Figure 24 is unexpectedly small"
    assert fig25.stat().st_size > 50_000, "Figure 25 is unexpectedly small"


def test_summary_csv_reports_exist():
    """Verify all 4 Day 12 summary reports exist with valid data rows."""
    reports = [
        "aspect_sentiment_summary.csv",
        "lda_topic_clusters.csv",
        "app_health_scorecard.csv",
        "aso_keyword_opportunity_matrix.csv"
    ]
    for rep in reports:
        p = _REPORTS_DIR / rep
        assert p.exists(), f"Missing report: {rep}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Report {rep} is empty"


# ============================================================================
# 8. Production REST API Endpoints Tests
# ============================================================================

def test_api_aso_aspect_analysis(api_client):
    """Test POST /api/v1/aso/aspect-analysis endpoint."""
    response = api_client.post(
        "/api/v1/aso/aspect-analysis",
        json={"target_name": "Duolingo: Learn Languages Free", "is_category": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_name"] == "Duolingo: Learn Languages Free"
    assert "aspect_scores" in data
    assert "ui_ux" in data["aspect_scores"]
    assert "performance" in data["aspect_scores"]
    assert 0.0 <= data["aggregate_aspect_health"] <= 100.0


def test_api_aso_topics(api_client):
    """Test GET /api/v1/aso/topics endpoint."""
    response = api_client.get("/api/v1/aso/topics?num_topics=5")
    assert response.status_code == 200
    data = response.json()
    assert data["num_topics"] == 5
    assert len(data["topics"]) == 5
    assert "top_words" in data["topics"][0]


def test_api_aso_review_authenticity(api_client):
    """Test POST /api/v1/aso/review-authenticity endpoint."""
    response = api_client.post(
        "/api/v1/aso/review-authenticity",
        json={"app_name": "Duolingo: Learn Languages Free"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Duolingo: Learn Languages Free"
    assert 0.0 <= data["authenticity_index"] <= 100.0
    assert data["verdict"] in {"EXCELLENT_ORGANIC", "MODERATE_NOISE", "SUSPICIOUS_BRIGADING"}


def test_api_aso_keyword_gap(api_client):
    """Test POST /api/v1/aso/keyword-gap endpoint."""
    response = api_client.post(
        "/api/v1/aso/keyword-gap",
        json={"target_app": "Duolingo: Learn Languages Free", "top_n_keywords": 6}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_app"] == "Duolingo: Learn Languages Free"
    assert len(data["missing_high_value_keywords"]) <= 6
    assert "prescribed_title_copy" in data
    assert len(data["prescribed_feature_bullets"]) > 0


def test_api_aso_app_health(api_client):
    """Test POST /api/v1/aso/app-health-diagnostic endpoint."""
    response = api_client.post(
        "/api/v1/aso/app-health-diagnostic",
        json={"app_name": "Duolingo: Learn Languages Free"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Duolingo: Learn Languages Free"
    assert 0.0 <= data["composite_health_index"] <= 100.0
    assert data["health_tier"] in {"CHAMPION", "SOLID_PERFORMER", "VULNERABLE", "CRITICAL_RENOVATION"}
    assert len(data["prescriptions"]) >= 3
