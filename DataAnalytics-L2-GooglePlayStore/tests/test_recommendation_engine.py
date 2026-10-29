"""Unit tests for Day 11 App Recommendation & Discovery Engine.

Validates:
1. Multi-modal embedding space construction, dimensional shapes, and L2-normalization.
2. Content-based cosine similarity search and self-exclusion logic.
3. Carbonell & Goldstein Maximal Marginal Relevance (MMR) diversification.
4. Personalized portfolio recommendations and cold-start competitor benchmarking.
5. Inter-category cosine similarity matrix mathematical properties.
6. Quantitative ranking evaluation metrics (Hit Rate@K, Precision@K, NDCG@K, MRR, ILD).
7. FastAPI REST API endpoints:
   - POST /api/v1/recommend/similar-apps
   - POST /api/v1/recommend/personalized
   - POST /api/v1/recommend/cold-start
   - GET /api/v1/recommend/metrics
8. Verification of generated publication figures (Figures 22 & 23) and CSV benchmark reports.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.recommendation_engine import (
    PlayStoreRecommendationEngine,
    AppRecommendationItem,
    RecommendationBatchResult,
    ColdStartRecommendationResult
)
from src.production_api import app

_BASE_DIR = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


@pytest.fixture(scope="module")
def rec_engine():
    """Module-scoped fixture providing an initialized PlayStoreRecommendationEngine."""
    return PlayStoreRecommendationEngine()


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# Core Recommendation Engine Tests
# ============================================================================

def test_recommendation_engine_initialization(rec_engine):
    """Verify that PlayStoreRecommendationEngine loads dataset and builds embedding space."""
    assert rec_engine.apps_df is not None
    assert len(rec_engine.apps_df) >= 9000
    assert rec_engine.embedding_matrix is not None
    assert rec_engine.embedding_matrix.shape[0] == len(rec_engine.apps_df)
    assert rec_engine.embedding_matrix.shape[1] > 50


def test_embedding_vector_properties(rec_engine):
    """Verify that all embedding vectors are unit L2-normalized."""
    norms = np.linalg.norm(rec_engine.embedding_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "All embedding vectors must have unit L2 norm"

    # Dot product with self must equal 1.0
    self_dot = np.sum(rec_engine.embedding_matrix[0] * rec_engine.embedding_matrix[0])
    assert pytest.approx(self_dot, abs=1e-5) == 1.0


def test_get_similar_apps_greedy(rec_engine):
    """Verify greedy cosine similarity retrieval (diversity_weight=1.0)."""
    target_app = "Duolingo: Learn Languages Free"
    result = rec_engine.get_similar_apps(target_app, top_k=5, diversity_weight=1.0)

    assert isinstance(result, RecommendationBatchResult)
    assert result.query_target == target_app
    assert len(result.recommendations) == 5
    assert result.diversity_weight == 1.0
    assert result.inference_latency_ms > 0

    # Verify query app is NOT in its own recommendations
    rec_names = [r.app_name for r in result.recommendations]
    assert target_app not in rec_names

    # Verify ranks are ordered 1 to 5
    ranks = [r.rank for r in result.recommendations]
    assert ranks == [1, 2, 3, 4, 5]

    # Verify similarities are monotonically decreasing for greedy
    sims = [r.similarity_score for r in result.recommendations]
    assert sims == sorted(sims, reverse=True)


def test_get_similar_apps_mmr_diversification(rec_engine):
    """Verify that MMR with lower lambda yields higher intra-list diversity than pure greedy."""
    target_app = "Duolingo: Learn Languages Free"
    res_greedy = rec_engine.get_similar_apps(target_app, top_k=5, diversity_weight=1.0)
    res_mmr = rec_engine.get_similar_apps(target_app, top_k=5, diversity_weight=0.3)

    assert len(res_greedy.recommendations) == 5
    assert len(res_mmr.recommendations) == 5
    assert res_mmr.intra_list_diversity >= res_greedy.intra_list_diversity


def test_get_similar_apps_category_filter(rec_engine):
    """Verify category filtering restricts recommendations strictly to requested category."""
    target_app = "Subway Surfers"
    result = rec_engine.get_similar_apps(target_app, top_k=4, diversity_weight=0.7, category_filter="GAME")

    assert len(result.recommendations) == 4
    for r in result.recommendations:
        assert r.category.upper() == "GAME"


def test_get_similar_apps_not_found_raises(rec_engine):
    """Verify that querying a nonexistent app raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="not found in the catalog"):
        rec_engine.get_similar_apps("NonExistentApp123456789X")


def test_recommend_for_user_portfolio(rec_engine):
    """Verify personalized recommendation based on user portfolio centroid."""
    portfolio = ["Spotify Music", "Duolingo: Learn Languages Free"]
    result = rec_engine.recommend_for_user_portfolio(portfolio, top_k=5, diversity_weight=0.7)

    assert isinstance(result, RecommendationBatchResult)
    assert result.query_type == "personalized_portfolio"
    assert len(result.recommendations) == 5

    # Neither of the installed apps should be in recommendations
    rec_names = [r.app_name.lower() for r in result.recommendations]
    for app_name in portfolio:
        assert app_name.lower() not in rec_names


def test_recommend_for_user_portfolio_empty_raises(rec_engine):
    """Verify that empty portfolio raises a ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        rec_engine.recommend_for_user_portfolio([])


def test_recommend_cold_start(rec_engine):
    """Verify cold-start competitor benchmarking for an unreleased app."""
    result = rec_engine.recommend_cold_start(
        category="FINANCE",
        genres="Finance",
        size_mb=18.5,
        price=0.0,
        content_rating="Everyone",
        top_k=5
    )

    assert isinstance(result, ColdStartRecommendationResult)
    assert result.candidate_category == "FINANCE"
    assert len(result.top_competitors) == 5
    assert result.category_mean_rating > 0
    assert result.category_median_installs >= 0
    assert len(result.strategic_advice) >= 1
    assert result.inference_latency_ms > 0


def test_category_similarity_matrix(rec_engine):
    """Verify properties of inter-category similarity matrix."""
    cat_df = rec_engine.compute_category_similarity_matrix()

    assert cat_df.shape[0] == cat_df.shape[1]
    assert cat_df.shape[0] == len(rec_engine.apps_df["Category"].unique())

    # Diagonal must be 1.0 (self-similarity)
    diag = np.diag(cat_df.values)
    assert np.allclose(diag, 1.0, atol=1e-3)

    # Matrix must be symmetric
    assert np.allclose(cat_df.values, cat_df.values.T, atol=1e-3)


def test_evaluate_engine_metrics_sanity(rec_engine):
    """Verify quantitative offline benchmark metrics compute within expected bounds."""
    eval_res = rec_engine.evaluate_engine(test_samples=15, k_values=(3, 5))

    assert eval_res["test_sample_size"] == 15
    for lam, metrics in eval_res["benchmark_results"].items():
        assert 0.0 <= metrics["hit_rate_at_5"] <= 1.0
        assert 0.0 <= metrics["precision_at_5"] <= 1.0
        assert 0.0 <= metrics["ndcg_at_5"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert metrics["intra_list_diversity"] >= 0.0
        assert metrics["latency_p50_ms"] > 0.0


# ============================================================================
# FastAPI REST API Endpoint Tests
# ============================================================================

def test_api_similar_apps_endpoint_valid(api_client):
    """Verify POST /api/v1/recommend/similar-apps returns 200 and valid schema."""
    payload = {
        "app_name": "Duolingo: Learn Languages Free",
        "top_k": 3,
        "diversity_weight": 0.7
    }
    response = api_client.post("/api/v1/recommend/similar-apps", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query_target"] == "Duolingo: Learn Languages Free"
    assert data["top_k"] == 3
    assert len(data["recommendations"]) == 3
    assert "intra_list_diversity" in data
    assert "inference_latency_ms" in data


def test_api_similar_apps_endpoint_not_found(api_client):
    """Verify POST /api/v1/recommend/similar-apps returns 404 for unknown app."""
    payload = {
        "app_name": "NonExistentApp123456789X",
        "top_k": 3
    }
    response = api_client.post("/api/v1/recommend/similar-apps", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_api_personalized_endpoint_valid(api_client):
    """Verify POST /api/v1/recommend/personalized returns 200 for user portfolio."""
    payload = {
        "installed_apps": ["Spotify Music", "Duolingo: Learn Languages Free"],
        "top_k": 4,
        "diversity_weight": 0.6
    }
    response = api_client.post("/api/v1/recommend/personalized", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query_type"] == "personalized_portfolio"
    assert len(data["recommendations"]) == 4


def test_api_cold_start_endpoint_valid(api_client):
    """Verify POST /api/v1/recommend/cold-start returns 200 and competitor analysis."""
    payload = {
        "category": "GAME",
        "genres": "Action",
        "size_mb": 45.0,
        "price": 0.0,
        "content_rating": "Everyone 10+",
        "top_k": 3
    }
    response = api_client.post("/api/v1/recommend/cold-start", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["candidate_category"] == "GAME"
    assert len(data["top_competitors"]) == 3
    assert "strategic_advice" in data


def test_api_recommend_metrics_endpoint(api_client):
    """Verify GET /api/v1/recommend/metrics returns 200 and system benchmarks."""
    response = api_client.get("/api/v1/recommend/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "benchmark_results" in data
    assert len(data["benchmark_results"]) >= 1


# ============================================================================
# Publication Artifacts & Figures Verification
# ============================================================================

def test_day11_reports_and_figures_existence():
    """Verify that Day 11 benchmark reports and figures exist and are non-empty."""
    fig22 = _FIGURES_DIR / "22_recommendation_engine_benchmarks.png"
    fig23 = _FIGURES_DIR / "23_app_similarity_network_and_cold_start.png"
    rep_bench = _REPORTS_DIR / "recommendation_benchmarks.csv"
    rep_samples = _REPORTS_DIR / "sample_recommendations.csv"
    rep_cat_sim = _REPORTS_DIR / "category_similarity_matrix.csv"

    assert fig22.exists(), "Figure 22 must exist"
    assert fig22.stat().st_size > 50000, "Figure 22 must be a non-trivial image"

    assert fig23.exists(), "Figure 23 must exist"
    assert fig23.stat().st_size > 50000, "Figure 23 must be a non-trivial image"

    assert rep_bench.exists(), "recommendation_benchmarks.csv must exist"
    assert rep_bench.stat().st_size > 100

    assert rep_samples.exists(), "sample_recommendations.csv must exist"
    assert rep_samples.stat().st_size > 100

    assert rep_cat_sim.exists(), "category_similarity_matrix.csv must exist"
    assert rep_cat_sim.stat().st_size > 100
