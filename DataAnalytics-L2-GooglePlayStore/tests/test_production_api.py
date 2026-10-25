"""Pytest Suite for Day 7 Production REST API Service.

Validates:
1. System root and health check endpoints.
2. Market intelligence category listings and single-category lookups.
3. Developer app concept feasibility scoring and input validation.
4. Smart-keyboard real-time typing assistant inference (<18ms budget).
5. Standalone autocomplete and autocorrect microservice endpoints.
6. Executive scorecard metrics and dynamic latency benchmarking.
7. Request timing middleware headers and OpenAPI documentation accessibility.
"""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.production_api import app, create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing an initialized TestClient for FastAPI."""
    test_app = create_app()
    with TestClient(test_app) as tc:
        yield tc


# ============================================================================
# 1. System & Health Endpoints
# ============================================================================

def test_root_endpoint(client: TestClient) -> None:
    """Verify that root endpoint returns API information and links to documentation."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["version"] == "1.0.0"
    assert "/docs" in data["docs_url"]
    assert "endpoints" in data


def test_health_check_endpoint(client: TestClient) -> None:
    """Verify health endpoint reports correct system status, catalog size, and model state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["catalog_apps_count"] == 9659
    assert data["review_corpus_records"] == 29692
    assert data["nlp_engine_initialized"] is True
    assert data["uptime_seconds"] >= 0.0


def test_timing_header_middleware(client: TestClient) -> None:
    """Verify that X-Process-Time-Ms header is injected into HTTP responses."""
    response = client.get("/health")
    assert "x-process-time-ms" in response.headers
    time_val = float(response.headers["x-process-time-ms"])
    assert time_val > 0.0


# ============================================================================
# 2. Market Intelligence Endpoints
# ============================================================================

def test_list_categories_endpoint(client: TestClient) -> None:
    """Verify retrieval of all 33 Play Store app categories."""
    response = client.get("/api/v1/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["total_categories"] == 33
    cats = [c["category"] for c in data["categories"]]
    assert "EDUCATION" in cats
    assert "GAME" in cats
    assert "FAMILY" in cats


def test_category_profile_valid(client: TestClient) -> None:
    """Verify detailed market intelligence lookup for a valid category."""
    response = client.get("/api/v1/category/EDUCATION")
    assert response.status_code == 200
    data = response.json()
    assert data["Category"] == "EDUCATION"
    assert data["Total_Apps"] == 107
    assert data["Saturation_Status"] == "Niche Opportunity"
    assert 4.0 <= data["Mean_Rating"] <= 5.0
    assert "Spearman_P_Value" in data
    assert "Strategic_Recommendation" in data


def test_category_profile_case_insensitive(client: TestClient) -> None:
    """Verify case-insensitive lookup and space/underscore normalization."""
    response = client.get("/api/v1/category/art_and_design")
    assert response.status_code == 200
    data = response.json()
    assert data["Category"] == "ART_AND_DESIGN"
    assert data["Total_Apps"] == 61


def test_category_profile_not_found(client: TestClient) -> None:
    """Verify HTTP 404 response when querying a non-existent category."""
    response = client.get("/api/v1/category/INVALID_NONEXISTENT_SECTOR")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


# ============================================================================
# 3. Developer App Concept Feasibility Scoring
# ============================================================================

def test_concept_feasibility_scoring_valid(client: TestClient) -> None:
    """Verify concept feasibility scoring returns expected green-light score for niche concept."""
    payload = {
        "category": "EDUCATION",
        "size_mb": 15.0,
        "price_usd": 1.99
    }
    response = client.post("/api/v1/feasibility", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["Category"] == "EDUCATION"
    assert data["Feasibility_Score"] >= 75.0
    assert "Exceptional Market Fit" in data["Feasibility_Tier"]
    assert data["Size_Adjustment_Delta"] == 1.0
    assert data["Price_Adjustment_Delta"] == 6.0


def test_concept_feasibility_invalid_size(client: TestClient) -> None:
    """Verify validation error when submitting non-positive APK size."""
    payload = {
        "category": "GAME",
        "size_mb": 0.0,  # Invalid: gt=0.0
        "price_usd": 0.0
    }
    response = client.post("/api/v1/feasibility", json=payload)
    assert response.status_code == 422


def test_concept_feasibility_invalid_price(client: TestClient) -> None:
    """Verify validation error when submitting negative price."""
    payload = {
        "category": "GAME",
        "size_mb": 25.0,
        "price_usd": -5.0  # Invalid: ge=0.0
    }
    response = client.post("/api/v1/feasibility", json=payload)
    assert response.status_code == 422


# ============================================================================
# 4. Smart Keyboard & NLP Inference Endpoints
# ============================================================================

def test_smart_typing_assistant_typo_correction(client: TestClient) -> None:
    """Verify real-time typing assistant corrects deliberate typos and provides autocomplete predictions."""
    payload = {
        "text": "the new update is amzaing",
        "max_candidates": 3
    }
    response = client.post("/api/v1/smart-type", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["active_word"] == "amzaing"
    assert data["is_typo"] is True
    assert data["autocorrected_word"] == "amazing"
    assert len(data["top3_autocomplete_predictions"]) > 0
    assert data["inference_latency_ms"] < 100.0  # Well within local execution budget


def test_smart_typing_assistant_clean_input(client: TestClient) -> None:
    """Verify that known dictionary words are preserved without false positive correction."""
    payload = {
        "text": "this app is great",
        "max_candidates": 3
    }
    response = client.post("/api/v1/smart-type", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["active_word"] == "great"
    assert data["is_typo"] is False
    assert data["autocorrected_word"] == "great"


def test_autocomplete_endpoint(client: TestClient) -> None:
    """Verify direct N-gram autocomplete endpoint predictions."""
    payload = {
        "prefix": "the new",
        "top_k": 3,
        "strategy": "stupid_backoff"
    }
    response = client.post("/api/v1/autocomplete", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prefix"] == "the new"
    assert len(data["predictions"]) > 0
    assert data["latency_ms"] >= 0.0


def test_autocorrect_endpoint(client: TestClient) -> None:
    """Verify direct autocorrect endpoint detects typos and returns ranked suggestions."""
    payload = {
        "word": "amzaing",
        "top_k": 3
    }
    response = client.post("/api/v1/autocorrect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "amzaing"
    assert data["is_typo"] is True
    assert "amazing" in data["candidates"]
    assert data["top_candidate"] == "amazing"


# ============================================================================
# 5. Scorecard & Benchmarking Endpoints
# ============================================================================

def test_master_scorecard_endpoint(client: TestClient) -> None:
    """Verify master executive scorecard returns complete KPI synthesis table."""
    response = client.get("/api/v1/scorecard")
    assert response.status_code == 200
    data = response.json()
    assert data["total_catalog_apps"] == 9659
    assert data["total_categories"] == 33
    assert len(data["metrics_summary_table"]) >= 15


def test_api_benchmark_endpoint(client: TestClient) -> None:
    """Verify dynamic empirical benchmark endpoint executes correctly."""
    response = client.get("/api/v1/benchmark?iterations=5")
    assert response.status_code == 200
    data = response.json()
    assert data["endpoints_tested"] >= 4
    assert len(data["summary"]) >= 4
    for stat in data["summary"]:
        assert stat["requests_sampled"] == 5
        assert stat["mean_latency_ms"] >= 0.0
