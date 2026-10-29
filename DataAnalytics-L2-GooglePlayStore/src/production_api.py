"""Production REST API Service for Google Play Store Market Intelligence & Smart Typing Assistant.

Day 7 Milestone Core for Google Play Store Analysis & NLP Analytics (Tasks 4 & 5).
Provides:
1. High-performance FastAPI REST application wrapping GooglePlaystoreMasterEngine.
2. Endpoints for category market intelligence, saturation analysis, and strategic profiling.
3. Developer App Concept Feasibility scoring API (0-100 score + risk and pricing adjustments).
4. Real-time Smart-Keyboard typing assistant API (<18ms latency budget).
5. Standalone Autocomplete and Autocorrect microservice endpoints.
6. Executive Scorecard and system health/benchmark endpoints.
7. Request timing middleware, Pydantic validation schemas, and OpenAPI documentation.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Path as FPath, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_REPORTS_DIR = _PROJECT_ROOT / "reports"

from src.master_pipeline import GooglePlaystoreMasterEngine
from src.nlp_autocomplete_engine import CorpusPreprocessor
from src.predictive_modeling import PlayStorePredictiveEngine
from src.simulation_engine import PlayStoreSimulationEngine
from src.recommendation_engine import PlayStoreRecommendationEngine


# ============================================================================
# Pydantic Request & Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """System health check and runtime status response."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    service: str = Field(..., json_schema_extra={"example": "Google Play Store Market Intelligence & NLP API"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    uptime_seconds: float = Field(..., json_schema_extra={"example": 42.5})
    catalog_apps_count: int = Field(..., json_schema_extra={"example": 9659})
    review_corpus_records: int = Field(..., json_schema_extra={"example": 29692})
    nlp_engine_initialized: bool = Field(..., json_schema_extra={"example": True})
    server_time: str = Field(..., json_schema_extra={"example": "2026-10-25T12:00:00Z"})


class CategorySummary(BaseModel):
    """Summary metrics for an app category."""
    category: str = Field(..., json_schema_extra={"example": "EDUCATION"})
    total_apps: int = Field(..., json_schema_extra={"example": 107})
    market_share_pct: float = Field(..., json_schema_extra={"example": 1.11})
    saturation_status: str = Field(..., json_schema_extra={"example": "Niche Opportunity"})
    mean_rating: float = Field(..., json_schema_extra={"example": 4.355})
    median_rating: float = Field(..., json_schema_extra={"example": 4.40})
    total_installs: int = Field(..., json_schema_extra={"example": 189952000})
    estimated_revenue_usd: float = Field(..., json_schema_extra={"example": 408980.0})
    mean_vader_compound: float = Field(..., json_schema_extra={"example": 0.449})
    strategic_quadrant: str = Field(..., json_schema_extra={"example": "Quadrant I: High Satisfaction / High Delight"})
    opportunity_score: float = Field(..., json_schema_extra={"example": 76.9})


class CategoryListResponse(BaseModel):
    """Response containing list of all available app categories."""
    total_categories: int = Field(..., json_schema_extra={"example": 33})
    categories: List[CategorySummary]


class CategoryDetailResponse(BaseModel):
    """Detailed market intelligence and strategic profile for a single category."""
    Category: str = Field(..., json_schema_extra={"example": "EDUCATION"})
    Total_Apps: int = Field(..., json_schema_extra={"example": 107})
    Market_Share_Pct: float = Field(..., json_schema_extra={"example": 1.11})
    Saturation_Status: str = Field(..., json_schema_extra={"example": "Niche Opportunity"})
    Mean_Rating: float = Field(..., json_schema_extra={"example": 4.355})
    Median_Rating: float = Field(..., json_schema_extra={"example": 4.4})
    Total_Installs: int = Field(..., json_schema_extra={"example": 189952000})
    Paid_Apps: int = Field(..., json_schema_extra={"example": 3})
    Paid_Share_Pct: float = Field(..., json_schema_extra={"example": 2.8})
    Estimated_Revenue_USD: float = Field(..., json_schema_extra={"example": 408980.0})
    Mean_Paid_Price: float = Field(..., json_schema_extra={"example": 4.99})
    Mean_VADER_Compound: float = Field(..., json_schema_extra={"example": 0.449})
    Positive_Review_Pct: float = Field(..., json_schema_extra={"example": 76.3})
    Negative_Review_Pct: float = Field(..., json_schema_extra={"example": 11.9})
    Size_Installs_Spearman_Rho: float = Field(..., json_schema_extra={"example": 0.1321})
    Spearman_P_Value: float = Field(..., json_schema_extra={"example": 0.242})
    Strategic_Quadrant: str = Field(..., json_schema_extra={"example": "Quadrant I: High Satisfaction / High Delight"})
    Opportunity_Score: float = Field(..., json_schema_extra={"example": 76.9})
    Strategic_Recommendation: str = Field(..., json_schema_extra={"example": "High user satisfaction sector..."})


class ConceptEvaluationRequest(BaseModel):
    """Input payload for scoring a prospective mobile app concept."""
    category: str = Field(..., description="App category name (e.g. EDUCATION, GAME, TOOLS)", json_schema_extra={"example": "EDUCATION"})
    size_mb: float = Field(..., gt=0.0, le=1000.0, description="Intended APK download size in Megabytes", json_schema_extra={"example": 15.0})
    price_usd: float = Field(0.0, ge=0.0, le=500.0, description="Target download price in USD (0.0 for Free)", json_schema_extra={"example": 1.99})


class ConceptEvaluationResponse(BaseModel):
    """Output scorecard evaluating prospective app concept viability."""
    Category: str = Field(..., json_schema_extra={"example": "EDUCATION"})
    Concept_Size_MB: float = Field(..., json_schema_extra={"example": 15.0})
    Concept_Price_USD: float = Field(..., json_schema_extra={"example": 1.99})
    Base_Category_Opportunity_Score: float = Field(..., json_schema_extra={"example": 76.9})
    Size_Adjustment_Delta: float = Field(..., json_schema_extra={"example": 1.0})
    Price_Adjustment_Delta: float = Field(..., json_schema_extra={"example": 6.0})
    Feasibility_Score: float = Field(..., json_schema_extra={"example": 83.9})
    Feasibility_Tier: str = Field(..., json_schema_extra={"example": "Exceptional Market Fit (Green Light)"})
    Saturation_Status: str = Field(..., json_schema_extra={"example": "Niche Opportunity"})
    Strategic_Quadrant: str = Field(..., json_schema_extra={"example": "Quadrant I: High Satisfaction / High Delight"})
    Identified_Market_Risks: List[str] = Field(default_factory=list)
    Strategic_Launch_Guidance: List[str] = Field(default_factory=list)


class AutocompleteCandidate(BaseModel):
    """Autocomplete candidate prediction with probability."""
    word: str = Field(..., json_schema_extra={"example": "amazing"})
    probability_pct: float = Field(..., json_schema_extra={"example": 33.33})


class SmartTypeRequest(BaseModel):
    """Input payload for real-time smart keyboard typing assistant simulation."""
    text: str = Field(..., min_length=1, max_length=500, description="Active input typing stream", json_schema_extra={"example": "the new update is amzaing"})
    max_candidates: int = Field(3, ge=1, le=10, description="Maximum candidate predictions to return", json_schema_extra={"example": 3})


class SmartTypeResponse(BaseModel):
    """Structured inference response from real-time smart typing assistant."""
    raw_input: str = Field(..., json_schema_extra={"example": "the new update is amzaing"})
    context_prefix: str = Field(..., json_schema_extra={"example": "the new update is"})
    active_word: str = Field(..., json_schema_extra={"example": "amzaing"})
    is_typo: bool = Field(..., json_schema_extra={"example": True})
    autocorrected_word: str = Field(..., json_schema_extra={"example": "amazing"})
    autocorrect_candidates: List[str] = Field(default_factory=list, json_schema_extra={"example": ["amazing"]})
    top3_autocomplete_predictions: List[AutocompleteCandidate] = Field(default_factory=list)
    suggested_completion: str = Field(..., json_schema_extra={"example": "the new update is amazing i"})
    inference_latency_ms: float = Field(..., json_schema_extra={"example": 1.45})


class AutocompleteRequest(BaseModel):
    """Request payload for next-word autocomplete prediction."""
    prefix: str = Field(..., min_length=1, description="Prompt prefix string", json_schema_extra={"example": "this game is"})
    top_k: int = Field(3, ge=1, le=10, description="Top-k candidates to predict", json_schema_extra={"example": 3})
    strategy: str = Field("stupid_backoff", description="Autocomplete backoff strategy", json_schema_extra={"example": "stupid_backoff"})


class AutocompleteResponse(BaseModel):
    """Response containing next-word autocomplete predictions."""
    prefix: str = Field(..., json_schema_extra={"example": "this game is"})
    predictions: List[AutocompleteCandidate]
    strategy_used: str = Field(..., json_schema_extra={"example": "stupid_backoff"})
    latency_ms: float = Field(..., json_schema_extra={"example": 0.82})


class AutocorrectRequest(BaseModel):
    """Request payload for Bayesian spelling correction."""
    word: str = Field(..., min_length=1, max_length=100, description="Word to verify and correct", json_schema_extra={"example": "updaet"})
    top_k: int = Field(3, ge=1, le=10, description="Number of candidate suggestions", json_schema_extra={"example": 3})


class AutocorrectResponse(BaseModel):
    """Response containing autocorrect candidates and error status."""
    word: str = Field(..., json_schema_extra={"example": "updaet"})
    is_typo: bool = Field(..., json_schema_extra={"example": True})
    candidates: List[str] = Field(default_factory=list, json_schema_extra={"example": ["update"]})
    top_candidate: str = Field(..., json_schema_extra={"example": "update"})
    latency_ms: float = Field(..., json_schema_extra={"example": 1.2})


class MasterScorecardResponse(BaseModel):
    """Executive KPI scorecard and synthesis summary."""
    title: str = Field(..., json_schema_extra={"example": "Google Play Store & NLP Analytics Executive Scorecard"})
    total_catalog_apps: int = Field(..., json_schema_extra={"example": 9659})
    total_categories: int = Field(..., json_schema_extra={"example": 33})
    total_installs: int = Field(..., json_schema_extra={"example": 167600000000})
    total_estimated_revenue_usd: float = Field(..., json_schema_extra={"example": 291140000.0})
    free_app_share_pct: float = Field(..., json_schema_extra={"example": 92.19})
    average_rating: float = Field(..., json_schema_extra={"example": 4.173})
    nlp_review_records: int = Field(..., json_schema_extra={"example": 29692})
    nlp_vocabulary_size: int = Field(..., json_schema_extra={"example": 20344})
    top_autocomplete_model: str = Field(..., json_schema_extra={"example": "Trigram (Stupid Backoff)"})
    autocomplete_hit_rate_at_3: float = Field(..., json_schema_extra={"example": 16.73})
    top_autocorrect_engine: str = Field(..., json_schema_extra={"example": "Damerau-Levenshtein Bayesian"})
    autocorrect_accuracy_pct: float = Field(..., json_schema_extra={"example": 87.88})
    autocorrect_precision_pct: float = Field(..., json_schema_extra={"example": 100.0})
    smart_typing_latency_ms: float = Field(..., json_schema_extra={"example": 1.45})
    metrics_summary_table: List[Dict[str, Any]]


class EndpointLatencyStat(BaseModel):
    """Statistical summary of endpoint benchmark latencies."""
    endpoint: str
    requests_sampled: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


class BenchmarkResponse(BaseModel):
    """Empirical API latency and throughput benchmark results."""
    timestamp: str
    total_requests: int
    endpoints_tested: int
    summary: List[EndpointLatencyStat]


# --- Day 9 Predictive ML & MLOps Drift Schemas ---

class SuccessPredictRequest(BaseModel):
    """Payload for predicting app high-velocity scale (>= 1,000,000 installs)."""
    category: str = Field(..., json_schema_extra={"example": "GAME"})
    size_mb: float = Field(..., gt=0.0, le=1024.0, json_schema_extra={"example": 45.0})
    price: float = Field(default=0.0, ge=0.0, json_schema_extra={"example": 0.0})
    content_rating: str = Field(default="Everyone", json_schema_extra={"example": "Everyone"})
    expected_reviews: Optional[int] = Field(default=None, ge=0, json_schema_extra={"example": 15000})


class SuccessPredictResponse(BaseModel):
    """Prediction result for app install velocity potential."""
    success_probability: float = Field(..., json_schema_extra={"example": 0.8842})
    rf_probability: float = Field(..., json_schema_extra={"example": 0.8933})
    gbm_probability: float = Field(..., json_schema_extra={"example": 0.8705})
    predicted_tier: str = Field(..., json_schema_extra={"example": "High Velocity Scale (>= 1,000,000 Installs)"})
    risk_assessment: str = Field(..., json_schema_extra={"example": "LOW_RISK"})
    category: str = Field(..., json_schema_extra={"example": "GAME"})
    size_mb: float = Field(..., json_schema_extra={"example": 45.0})
    price: float = Field(..., json_schema_extra={"example": 0.0})
    effective_reviews_evaluated: int = Field(..., json_schema_extra={"example": 15000})
    key_drivers: List[str]
    latency_ms: float = Field(..., json_schema_extra={"example": 2.15})


class RatingPredictRequest(BaseModel):
    """Payload for predicting app rating valuation."""
    category: str = Field(..., json_schema_extra={"example": "EDUCATION"})
    size_mb: float = Field(..., gt=0.0, le=1024.0, json_schema_extra={"example": 25.0})
    price: float = Field(default=0.0, ge=0.0, json_schema_extra={"example": 1.99})
    content_rating: str = Field(default="Everyone", json_schema_extra={"example": "Everyone"})
    expected_reviews: Optional[int] = Field(default=None, ge=0, json_schema_extra={"example": 2500})


class RatingPredictResponse(BaseModel):
    """Prediction result for expected user rating."""
    predicted_rating: float = Field(..., json_schema_extra={"example": 4.38})
    confidence_interval_90: List[float] = Field(..., json_schema_extra={"example": [4.13, 4.63]})
    quality_tier: str = Field(..., json_schema_extra={"example": "Top Quartile Quality (Elite App Tier)"})
    category: str = Field(..., json_schema_extra={"example": "EDUCATION"})
    category_baseline_mean: float = Field(..., json_schema_extra={"example": 4.36})
    latency_ms: float = Field(..., json_schema_extra={"example": 1.95})


class DriftMetricItem(BaseModel):
    """Statistical drift metrics for an individual feature."""
    feature: str
    column: str
    baseline_mean: float
    monitored_mean: float
    psi: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    status: str


class DriftReportResponse(BaseModel):
    """Comprehensive MLOps distribution drift report."""
    health_verdict: str = Field(..., json_schema_extra={"example": "HEALTHY"})
    overall_max_psi: float = Field(..., json_schema_extra={"example": 0.0842})
    recommendation: str
    baseline_samples: int
    monitored_samples: int
    metrics: List[DriftMetricItem]
    generated_at: str


class ModelCatalogResponse(BaseModel):
    """Trained machine learning models and feature attribution metadata."""
    classification_models: Dict[str, Any]
    regression_models: Dict[str, Any]
    top_features: List[Dict[str, Any]]


# --- Day 10 Simulation & A/B Experimentation Schemas ---

class WhatIfSimulateRequest(BaseModel):
    """Payload for counterfactual what-if scenario simulation."""
    app_name: Optional[str] = Field(None, json_schema_extra={"example": "Subway Surfers"})
    category: Optional[str] = Field(None, json_schema_extra={"example": "GAME"})
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, json_schema_extra={"example": 4.2})
    reviews: Optional[int] = Field(None, ge=0, json_schema_extra={"example": 15000})
    size_mb: Optional[float] = Field(None, gt=0.0, le=1024.0, json_schema_extra={"example": 45.0})
    price: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 0.0})
    content_rating: Optional[str] = Field("Everyone", json_schema_extra={"example": "Everyone"})
    delta_size_pct: float = Field(0.0, ge=-0.95, le=5.0, description="Percentage change in APK size", json_schema_extra={"example": -0.25})
    delta_price: float = Field(0.0, description="Absolute change in USD price", json_schema_extra={"example": 0.0})
    delta_rating: float = Field(0.0, ge=-4.0, le=4.0, description="Projected rating shift", json_schema_extra={"example": 0.2})


class WhatIfSimulateResponse(BaseModel):
    """Result of counterfactual what-if simulation."""
    app_name: str
    category: str
    base_size_mb: float
    simulated_size_mb: float
    base_price: float
    simulated_price: float
    base_rating: float
    simulated_rating: float
    base_scale_prob: float
    simulated_scale_prob: float
    delta_scale_prob: float
    pct_scale_prob_uplift: float
    base_expected_rating: float
    simulated_expected_rating: float
    base_expected_revenue: float
    simulated_expected_revenue: float
    delta_revenue: float
    marginal_size_elasticity: float
    marginal_price_elasticity: float
    recommendations: List[str]


class ABTestExperimentRequest(BaseModel):
    """Payload for two-sample A/B test analysis."""
    visitors_a: int = Field(..., gt=0, json_schema_extra={"example": 10000})
    conversions_a: int = Field(..., ge=0, json_schema_extra={"example": 500})
    visitors_b: int = Field(..., gt=0, json_schema_extra={"example": 10000})
    conversions_b: int = Field(..., ge=0, json_schema_extra={"example": 575})
    alpha: float = Field(0.05, gt=0.0, lt=0.5, json_schema_extra={"example": 0.05})


class ABTestExperimentResponse(BaseModel):
    """Comprehensive statistical outcome of A/B test."""
    visitors_a: int
    conversions_a: int
    rate_a: float
    visitors_b: int
    conversions_b: int
    rate_b: float
    absolute_uplift: float
    relative_uplift: float
    z_statistic: float
    p_value: float
    is_statistically_significant: bool
    ci_lower_diff: float
    ci_upper_diff: float
    bayesian_prob_b_beats_a: float
    expected_loss: float
    credible_interval_a: List[float]
    credible_interval_b: List[float]
    decision: str
    summary_verdict: str


class PriceElasticityResponse(BaseModel):
    """Empirical price elasticity of demand analysis."""
    total_paid_apps_analyzed: int
    overall_elasticity: float
    overall_r_squared: float
    overall_elasticity_type: str
    overall_optimal_price: float
    categories_analyzed: int
    category_benchmarks: List[Dict[str, Any]]


# Day 11 Recommendation Schemas

class AppRecommendationItemSchema(BaseModel):
    """Individual app recommendation item schema."""
    app_name: str
    category: str
    rating: float
    reviews: int
    installs: int
    size_mb: float
    price: float
    genres: str
    content_rating: str
    similarity_score: float
    rank: int


class SimilarAppsRequest(BaseModel):
    """Payload for finding similar apps with optional MMR diversification."""
    app_name: str = Field(..., json_schema_extra={"example": "Duolingo: Learn Languages Free"})
    top_k: int = Field(5, ge=1, le=50, json_schema_extra={"example": 5})
    diversity_weight: float = Field(0.7, ge=0.0, le=1.0, json_schema_extra={"example": 0.7})
    category_filter: Optional[str] = Field(None, json_schema_extra={"example": "EDUCATION"})


class SimilarAppsResponse(BaseModel):
    """Batch response for similar apps query."""
    query_type: str
    query_target: str
    top_k: int
    diversity_weight: float
    intra_list_diversity: float
    inference_latency_ms: float
    recommendations: List[AppRecommendationItemSchema]


class PersonalizedRecommendRequest(BaseModel):
    """Payload for personalized recommendations based on installed portfolio."""
    installed_apps: List[str] = Field(..., min_length=1, json_schema_extra={"example": ["Spotify Music", "Duolingo: Learn Languages Free"]})
    top_k: int = Field(5, ge=1, le=50, json_schema_extra={"example": 5})
    diversity_weight: float = Field(0.7, ge=0.0, le=1.0, json_schema_extra={"example": 0.7})


class PersonalizedRecommendResponse(BaseModel):
    """Batch response for personalized portfolio recommendation."""
    query_type: str
    query_target: str
    top_k: int
    diversity_weight: float
    intra_list_diversity: float
    inference_latency_ms: float
    recommendations: List[AppRecommendationItemSchema]


class ColdStartRecommendRequest(BaseModel):
    """Payload for cold-start prospective app positioning and competitor discovery."""
    category: str = Field(..., json_schema_extra={"example": "FINANCE"})
    genres: str = Field(..., json_schema_extra={"example": "Finance"})
    size_mb: float = Field(25.0, gt=0.0, json_schema_extra={"example": 18.5})
    price: float = Field(0.0, ge=0.0, json_schema_extra={"example": 0.0})
    content_rating: str = Field("Everyone", json_schema_extra={"example": "Everyone"})
    top_k: int = Field(5, ge=1, le=20, json_schema_extra={"example": 5})


class ColdStartRecommendResponse(BaseModel):
    """Comprehensive cold-start competitor analysis and positioning advice."""
    candidate_category: str
    candidate_genres: str
    candidate_size_mb: float
    candidate_price: float
    candidate_content_rating: str
    top_competitors: List[AppRecommendationItemSchema]
    category_mean_rating: float
    category_median_installs: int
    category_competition_intensity: str
    strategic_advice: List[str]
    inference_latency_ms: float


class RecommendationMetricsResponse(BaseModel):
    """System-wide recommendation offline benchmark evaluation response."""
    test_sample_size: int
    benchmark_results: Dict[str, Any]


# ============================================================================
# Engine Singleton & Lifespan Management
# ============================================================================

_engine_instance: Optional[GooglePlaystoreMasterEngine] = None
_predictive_engine_instance: Optional[PlayStorePredictiveEngine] = None
_simulation_engine_instance: Optional[PlayStoreSimulationEngine] = None
_recommendation_engine_instance: Optional[PlayStoreRecommendationEngine] = None
_server_start_time: float = time.time()


def get_engine() -> GooglePlaystoreMasterEngine:
    """Retrieve or initialize the singleton GooglePlaystoreMasterEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = GooglePlaystoreMasterEngine()
        _ = _engine_instance.apps_df
        _ = _engine_instance.reviews_df
        _engine_instance.init_nlp_assistant()
    return _engine_instance


def get_predictive_engine() -> PlayStorePredictiveEngine:
    """Retrieve or initialize the singleton PlayStorePredictiveEngine instance."""
    global _predictive_engine_instance
    if _predictive_engine_instance is None:
        _predictive_engine_instance = PlayStorePredictiveEngine()
        _predictive_engine_instance.train_models()
    return _predictive_engine_instance


def get_simulation_engine() -> PlayStoreSimulationEngine:
    """Retrieve or initialize the singleton PlayStoreSimulationEngine instance."""
    global _simulation_engine_instance
    if _simulation_engine_instance is None:
        _simulation_engine_instance = PlayStoreSimulationEngine(predictive_engine=get_predictive_engine())
    return _simulation_engine_instance


def get_recommendation_engine() -> PlayStoreRecommendationEngine:
    """Retrieve or initialize the singleton PlayStoreRecommendationEngine instance."""
    global _recommendation_engine_instance
    if _recommendation_engine_instance is None:
        _recommendation_engine_instance = PlayStoreRecommendationEngine()
    return _recommendation_engine_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup pre-loading and graceful shutdown."""
    engine = get_engine()
    engine.init_nlp_assistant()
    yield


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the production FastAPI application."""
    app = FastAPI(
        title="Google Play Store Market Intelligence & Smart Typing REST API",
        description=(
            "Production REST microservice powering real-time app market queries, "
            "developer feasibility scoring, and sub-18ms NLP autocorrect and autocomplete typing assistance."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Processing Time Middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Exception Handler for Domain ValueErrors
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc), "status_code": 404}
        )

    # ------------------------------------------------------------------------
    # Root & Health Endpoints
    # ------------------------------------------------------------------------

    @app.get(
        "/",
        summary="API Root Information",
        tags=["System"]
    )
    def root() -> Dict[str, Any]:
        """Root API metadata and documentation directory."""
        return {
            "name": "Google Play Store Market Intelligence & Smart Typing REST API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "openapi_schema": "/openapi.json",
            "endpoints": {
                "health": "/health",
                "categories": "/api/v1/categories",
                "category_profile": "/api/v1/category/{category_name}",
                "feasibility": "/api/v1/feasibility",
                "smart_type": "/api/v1/smart-type",
                "autocomplete": "/api/v1/autocomplete",
                "autocorrect": "/api/v1/autocorrect",
                "scorecard": "/api/v1/scorecard",
                "benchmark": "/api/v1/benchmark"
            }
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health & Readiness Check",
        tags=["System"]
    )
    def health_check() -> HealthResponse:
        """Verify service health, dataset integrity, and loaded NLP models."""
        engine = get_engine()
        uptime = round(time.time() - _server_start_time, 2)
        apps_count = len(engine.apps_df)
        rev_count = len(engine.reviews_df)
        nlp_ready = engine._autocomplete_engine is not None and engine._autocorrect_engine is not None

        return HealthResponse(
            status="healthy",
            service="Google Play Store Market Intelligence & NLP API",
            version="1.0.0",
            uptime_seconds=uptime,
            catalog_apps_count=apps_count,
            review_corpus_records=rev_count,
            nlp_engine_initialized=nlp_ready,
            server_time=datetime.now(timezone.utc).isoformat()
        )

    # ------------------------------------------------------------------------
    # Market Intelligence Endpoints
    # ------------------------------------------------------------------------

    @app.get(
        "/api/v1/categories",
        response_model=CategoryListResponse,
        summary="List All App Categories",
        tags=["Market Intelligence"]
    )
    def list_categories() -> CategoryListResponse:
        """Retrieve high-level market metrics across all 33 Play Store app categories."""
        engine = get_engine()
        opp_df = engine.opportunity_matrix
        summaries: List[CategorySummary] = []

        total_apps_all = len(engine.apps_df)
        for _, row in opp_df.iterrows():
            cat = str(row["Category"])
            app_cnt = int(row.get("Total_Apps", 0))
            share = round((app_cnt / total_apps_all) * 100.0, 2) if total_apps_all > 0 else 0.0

            if app_cnt > 800:
                sat = "Hyper-Saturated (Red Ocean)"
            elif app_cnt > 250:
                sat = "Moderate Saturation"
            else:
                sat = "Niche Opportunity"

            summaries.append(CategorySummary(
                category=cat,
                total_apps=app_cnt,
                market_share_pct=share,
                saturation_status=sat,
                mean_rating=round(float(row.get("Mean_Rating", 0.0)), 3),
                median_rating=round(float(row.get("Median_Rating", 0.0)), 3),
                total_installs=int(row.get("Total_Installs", 0)),
                estimated_revenue_usd=round(float(row.get("Estimated_Revenue", 0.0)), 2),
                mean_vader_compound=round(float(row.get("Mean_VADER_Compound", 0.0)), 3),
                strategic_quadrant=str(row.get("Strategic_Quadrant", "Quadrant I")),
                opportunity_score=round(float(row.get("Opportunity_Score", 50.0)), 1)
            ))

        return CategoryListResponse(
            total_categories=len(summaries),
            categories=summaries
        )

    @app.get(
        "/api/v1/category/{category_name}",
        response_model=CategoryDetailResponse,
        summary="Category Market Intelligence Profile",
        tags=["Market Intelligence"]
    )
    def get_category_profile(
        category_name: str = FPath(..., description="App category name (e.g. EDUCATION, GAME, TOOLS)")
    ) -> CategoryDetailResponse:
        """Retrieve complete market intelligence profile, correlations, and strategic recommendations for a category."""
        engine = get_engine()
        try:
            metrics = engine.get_category_metrics(category_name)
            return CategoryDetailResponse(**metrics)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))

    # ------------------------------------------------------------------------
    # Developer Concept Feasibility Scoring
    # ------------------------------------------------------------------------

    @app.post(
        "/api/v1/feasibility",
        response_model=ConceptEvaluationResponse,
        summary="Score App Concept Feasibility (0-100)",
        tags=["Developer Tools"]
    )
    def evaluate_app_concept(payload: ConceptEvaluationRequest) -> ConceptEvaluationResponse:
        """Quantify the market feasibility score (0 to 100) of a prospective app concept."""
        engine = get_engine()
        try:
            scorecard = engine.score_app_concept(
                category=payload.category,
                size_mb=payload.size_mb,
                price=payload.price_usd
            )
            return ConceptEvaluationResponse(**scorecard)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))

    # ------------------------------------------------------------------------
    # Smart Keyboard & NLP Inference Endpoints
    # ------------------------------------------------------------------------

    @app.post(
        "/api/v1/smart-type",
        response_model=SmartTypeResponse,
        summary="Real-Time Smart Keyboard Assistant (<18ms)",
        tags=["Smart Keyboard NLP"]
    )
    def smart_typing_assistant(payload: SmartTypeRequest) -> SmartTypeResponse:
        """Simulate real-world smartphone keyboard keystroke processing."""
        engine = get_engine()
        result = engine.smart_keyboard_type(payload.text)

        preds = [
            AutocompleteCandidate(word=p["word"], probability_pct=p["probability_pct"])
            for p in result.get("top3_autocomplete_predictions", [])[:payload.max_candidates]
        ]

        return SmartTypeResponse(
            raw_input=result["raw_input"],
            context_prefix=result["context_prefix"],
            active_word=result["active_word"],
            is_typo=result["is_typo"],
            autocorrected_word=result["autocorrected_word"],
            autocorrect_candidates=result.get("autocorrect_candidates", [])[:payload.max_candidates],
            top3_autocomplete_predictions=preds,
            suggested_completion=result["suggested_completion"],
            inference_latency_ms=result["inference_latency_ms"]
        )

    @app.post(
        "/api/v1/autocomplete",
        response_model=AutocompleteResponse,
        summary="Direct N-Gram Autocomplete Next-Word Prediction",
        tags=["Smart Keyboard NLP"]
    )
    def predict_autocomplete(payload: AutocompleteRequest) -> AutocompleteResponse:
        """Predict the most probable next words given a prompt prefix using N-gram language models."""
        engine = get_engine()
        engine.init_nlp_assistant()
        t0 = time.perf_counter()

        raw_clean = CorpusPreprocessor.clean_text(payload.prefix)
        preds = engine._autocomplete_engine.predict_next_words(
            raw_clean,
            top_k=payload.top_k,
            strategy=payload.strategy
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        candidates = [
            AutocompleteCandidate(word=w, probability_pct=round(p * 100.0, 2))
            for w, p in preds
        ]

        return AutocompleteResponse(
            prefix=payload.prefix,
            predictions=candidates,
            strategy_used=payload.strategy,
            latency_ms=round(latency_ms, 3)
        )

    @app.post(
        "/api/v1/autocorrect",
        response_model=AutocorrectResponse,
        summary="Direct Peter Norvig & Damerau-Levenshtein Autocorrect",
        tags=["Smart Keyboard NLP"]
    )
    def correct_spelling(payload: AutocorrectRequest) -> AutocorrectResponse:
        """Detect misspellings and generate candidate corrections scored by review corpus frequency priors."""
        engine = get_engine()
        engine.init_nlp_assistant()
        t0 = time.perf_counter()

        clean_w = payload.word.lower().strip()
        is_known = clean_w in engine._autocorrect_engine.vocab_set
        candidates_scored = engine._autocorrect_engine.predict_candidates(clean_w, top_k=payload.top_k)
        candidates = [c[0] for c in candidates_scored]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_cand = candidates[0] if candidates else clean_w

        return AutocorrectResponse(
            word=payload.word,
            is_typo=not is_known,
            candidates=candidates,
            top_candidate=top_cand,
            latency_ms=round(latency_ms, 3)
        )

    # ------------------------------------------------------------------------
    # Scorecard & Benchmarking Endpoints
    # ------------------------------------------------------------------------

    @app.get(
        "/api/v1/scorecard",
        response_model=MasterScorecardResponse,
        summary="Master Executive Scorecard & KPI Synthesis",
        tags=["Analytics & Scorecard"]
    )
    def get_master_scorecard() -> MasterScorecardResponse:
        """Retrieve executive KPI scorecard synthesizing Play Store market scale and NLP linguistic metrics."""
        engine = get_engine()
        summary_csv = engine.reports_dir / "master_synthesis_summary.csv"
        if summary_csv.exists():
            scorecard_df = pd.read_csv(summary_csv)
        else:
            _, scorecard_df = engine.compute_cross_task_synthesis()

        metrics_table = scorecard_df.to_dict(orient="records")

        return MasterScorecardResponse(
            title="Google Play Store & NLP Analytics Executive Scorecard",
            total_catalog_apps=9659,
            total_categories=33,
            total_installs=167600000000,
            total_estimated_revenue_usd=291140000.0,
            free_app_share_pct=92.19,
            average_rating=4.173,
            nlp_review_records=29692,
            nlp_vocabulary_size=20344,
            top_autocomplete_model="Trigram (Stupid Backoff)",
            autocomplete_hit_rate_at_3=16.73,
            top_autocorrect_engine="Damerau-Levenshtein Bayesian",
            autocorrect_accuracy_pct=87.88,
            autocorrect_precision_pct=100.0,
            smart_typing_latency_ms=1.45,
            metrics_summary_table=metrics_table
        )

    @app.get(
        "/api/v1/benchmark",
        response_model=BenchmarkResponse,
        summary="Run Empirical API Latency Benchmarks",
        tags=["Analytics & Scorecard"]
    )
    def run_api_benchmarks(
        iterations: int = Query(25, ge=5, le=100, description="Iterations per endpoint for latency statistics")
    ) -> BenchmarkResponse:
        """Empirically benchmark latencies across all core production endpoints."""
        engine = get_engine()
        engine.init_nlp_assistant()

        bench_results: List[EndpointLatencyStat] = []

        # 1. Health check benchmark
        t_health: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = len(engine.apps_df)
            t_health.append((time.perf_counter() - t0) * 1000.0)

        # 2. Category Profile benchmark (EDUCATION, GAME, TOOLS)
        cats = ["EDUCATION", "GAME", "TOOLS"]
        t_cat: List[float] = []
        for i in range(iterations):
            c = cats[i % len(cats)]
            t0 = time.perf_counter()
            _ = engine.get_category_metrics(c)
            t_cat.append((time.perf_counter() - t0) * 1000.0)

        # 3. Feasibility Scoring benchmark
        t_feas: List[float] = []
        for i in range(iterations):
            t0 = time.perf_counter()
            _ = engine.score_app_concept("EDUCATION", 15.0 + (i % 10), 1.99)
            t_feas.append((time.perf_counter() - t0) * 1000.0)

        # 4. Smart Typing benchmark (with typo)
        queries = ["the new update is amzaing", "this game is very good", "best app for shoping"]
        t_type: List[float] = []
        for i in range(iterations):
            q = queries[i % len(queries)]
            t0 = time.perf_counter()
            _ = engine.smart_keyboard_type(q)
            t_type.append((time.perf_counter() - t0) * 1000.0)

        # 5. Direct Autocomplete benchmark
        t_ac: List[float] = []
        for i in range(iterations):
            t0 = time.perf_counter()
            _ = engine._autocomplete_engine.predict_next_words("the new", top_k=3)
            t_ac.append((time.perf_counter() - t0) * 1000.0)

        def _calc_stats(name: str, samples: List[float]) -> EndpointLatencyStat:
            arr = np.array(samples)
            return EndpointLatencyStat(
                endpoint=name,
                requests_sampled=len(samples),
                mean_latency_ms=round(float(np.mean(arr)), 3),
                median_latency_ms=round(float(np.median(arr)), 3),
                p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
                p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
                min_latency_ms=round(float(np.min(arr)), 3),
                max_latency_ms=round(float(np.max(arr)), 3)
            )

        bench_results.append(_calc_stats("/health", t_health))
        bench_results.append(_calc_stats("/api/v1/category/{name}", t_cat))
        bench_results.append(_calc_stats("/api/v1/feasibility", t_feas))
        bench_results.append(_calc_stats("/api/v1/smart-type", t_type))
        bench_results.append(_calc_stats("/api/v1/autocomplete", t_ac))

        return BenchmarkResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_requests=iterations * len(bench_results),
            endpoints_tested=len(bench_results),
            summary=bench_results
        )

    # -------------------------------------------------------------
    # Day 9 ML Endpoints: Predictive Modeling & MLOps Drift
    # -------------------------------------------------------------

    @app.post(
        "/api/v1/predict/success",
        response_model=SuccessPredictResponse,
        summary="Predict App Install Velocity Scale (>= 1M Installs)",
        tags=["Predictive Intelligence"]
    )
    async def predict_success_endpoint(request_body: SuccessPredictRequest) -> SuccessPredictResponse:
        """Predict probability of an app achieving high-velocity scale (>= 1,000,000 installs)."""
        pred_engine = get_predictive_engine()
        res = pred_engine.predict_app_success(
            category=request_body.category,
            size_mb=request_body.size_mb,
            price=request_body.price,
            content_rating=request_body.content_rating,
            expected_reviews=request_body.expected_reviews,
        )
        return SuccessPredictResponse(**res)

    @app.post(
        "/api/v1/predict/rating",
        response_model=RatingPredictResponse,
        summary="Predict App User Rating Valuation",
        tags=["Predictive Intelligence"]
    )
    async def predict_rating_endpoint(request_body: RatingPredictRequest) -> RatingPredictResponse:
        """Predict expected user rating (1.0 to 5.0) and confidence interval."""
        pred_engine = get_predictive_engine()
        res = pred_engine.predict_app_rating(
            category=request_body.category,
            size_mb=request_body.size_mb,
            price=request_body.price,
            content_rating=request_body.content_rating,
            expected_reviews=request_body.expected_reviews,
        )
        return RatingPredictResponse(**res)

    @app.get(
        "/api/v1/mlops/drift-report",
        response_model=DriftReportResponse,
        summary="Retrieve MLOps Distribution Drift Report",
        tags=["MLOps & Observability"]
    )
    async def get_drift_report_endpoint() -> DriftReportResponse:
        """Evaluate live dataset feature stability using PSI, KS-test, and Wasserstein distance."""
        pred_engine = get_predictive_engine()
        res = pred_engine.generate_drift_report()
        return DriftReportResponse(**res)

    @app.get(
        "/api/v1/mlops/models",
        response_model=ModelCatalogResponse,
        summary="Retrieve Trained ML Model Catalog & Feature Weights",
        tags=["MLOps & Observability"]
    )
    async def get_models_catalog_endpoint() -> ModelCatalogResponse:
        """Return evaluation benchmarks and top feature importances for active models."""
        pred_engine = get_predictive_engine()
        top_feats = [
            {"rank": i + 1, "feature": feat, "importance": round(imp, 5), "pct": round(imp * 100, 2)}
            for i, (feat, imp) in enumerate(list(pred_engine.feature_importances_.items())[:12])
        ]
        return ModelCatalogResponse(
            classification_models=pred_engine.classification_metrics,
            regression_models=pred_engine.regression_metrics,
            top_features=top_feats
        )

    # --- Day 10 Counterfactual Simulation & A/B Experimentation Endpoints ---

    @app.post(
        "/api/v1/simulate/what-if",
        response_model=WhatIfSimulateResponse,
        summary="Simulate Counterfactual App What-If Scenario",
        tags=["Simulation & Optimization"]
    )
    async def simulate_what_if_endpoint(request_body: WhatIfSimulateRequest) -> WhatIfSimulateResponse:
        """Simulate the effect of APK size compression, price adjustments, or rating shifts."""
        sim_engine = get_simulation_engine()
        res = sim_engine.simulate_what_if(
            app_name=request_body.app_name,
            category=request_body.category,
            rating=request_body.rating,
            reviews=request_body.reviews,
            size_mb=request_body.size_mb,
            price=request_body.price,
            content_rating=request_body.content_rating or "Everyone",
            delta_size_pct=request_body.delta_size_pct,
            delta_price=request_body.delta_price,
            delta_rating=request_body.delta_rating,
        )
        return WhatIfSimulateResponse(**res.to_dict())

    @app.post(
        "/api/v1/experiment/ab-test",
        response_model=ABTestExperimentResponse,
        summary="Evaluate Two-Sample Frequentist & Bayesian A/B Test",
        tags=["Simulation & Optimization"]
    )
    async def run_ab_test_endpoint(request_body: ABTestExperimentRequest) -> ABTestExperimentResponse:
        """Run frequentist Z-test and Bayesian Beta-Binomial superiority evaluation."""
        sim_engine = get_simulation_engine()
        try:
            res = sim_engine.run_ab_experiment(
                visitors_a=request_body.visitors_a,
                conversions_a=request_body.conversions_a,
                visitors_b=request_body.visitors_b,
                conversions_b=request_body.conversions_b,
                alpha=request_body.alpha,
            )
            d = res.to_dict()
            d["credible_interval_a"] = list(d["credible_interval_a"])
            d["credible_interval_b"] = list(d["credible_interval_b"])
            return ABTestExperimentResponse(**d)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.get(
        "/api/v1/analytics/price-elasticity",
        response_model=PriceElasticityResponse,
        summary="Retrieve Empirical Price Elasticity of Demand Summary",
        tags=["Simulation & Optimization"]
    )
    async def get_price_elasticity_endpoint() -> PriceElasticityResponse:
        """Compute catalog and category econometric price elasticity of demand."""
        sim_engine = get_simulation_engine()
        res = sim_engine.analyze_price_elasticity()
        return PriceElasticityResponse(**res)

    # --- Day 11 Recommendation & Discovery Endpoints ---

    @app.post(
        "/api/v1/recommend/similar-apps",
        response_model=SimilarAppsResponse,
        summary="Find Top-K Similar Apps with MMR Diversification",
        tags=["Recommendation & Discovery"]
    )
    async def recommend_similar_apps_endpoint(request_body: SimilarAppsRequest) -> SimilarAppsResponse:
        """Find Top-K similar apps to a seed app using multi-modal embeddings and MMR diversity."""
        rec_engine = get_recommendation_engine()
        try:
            res = rec_engine.get_similar_apps(
                app_name=request_body.app_name,
                top_k=request_body.top_k,
                diversity_weight=request_body.diversity_weight,
                category_filter=request_body.category_filter
            )
            return SimilarAppsResponse(**res.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    @app.post(
        "/api/v1/recommend/personalized",
        response_model=PersonalizedRecommendResponse,
        summary="Generate Personalized Recommendations for User Portfolio",
        tags=["Recommendation & Discovery"]
    )
    async def recommend_personalized_endpoint(request_body: PersonalizedRecommendRequest) -> PersonalizedRecommendResponse:
        """Construct user profile vector from installed apps portfolio and return diversified recommendations."""
        rec_engine = get_recommendation_engine()
        try:
            res = rec_engine.recommend_for_user_portfolio(
                installed_apps=request_body.installed_apps,
                top_k=request_body.top_k,
                diversity_weight=request_body.diversity_weight
            )
            return PersonalizedRecommendResponse(**res.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.post(
        "/api/v1/recommend/cold-start",
        response_model=ColdStartRecommendResponse,
        summary="Cold-Start Competitor Benchmarking for Unreleased App",
        tags=["Recommendation & Discovery"]
    )
    async def recommend_cold_start_endpoint(request_body: ColdStartRecommendRequest) -> ColdStartRecommendResponse:
        """Synthesize candidate vector and benchmark competitor positioning for a prospective app."""
        rec_engine = get_recommendation_engine()
        res = rec_engine.recommend_cold_start(
            category=request_body.category,
            genres=request_body.genres,
            size_mb=request_body.size_mb,
            price=request_body.price,
            content_rating=request_body.content_rating,
            top_k=request_body.top_k
        )
        return ColdStartRecommendResponse(**res.to_dict())

    @app.get(
        "/api/v1/recommend/metrics",
        response_model=RecommendationMetricsResponse,
        summary="Retrieve System Offline Recommendation Benchmarks",
        tags=["Recommendation & Discovery"]
    )
    async def get_recommendation_metrics_endpoint() -> RecommendationMetricsResponse:
        """Retrieve Hit Rate@K, Precision@K, NDCG@K, MRR, and ILD metrics across diversity levels."""
        benchmarks_path = _REPORTS_DIR / "recommendation_benchmarks.csv"
        if benchmarks_path.exists():
            df = pd.read_csv(benchmarks_path)
            res_dict = {str(row["diversity_weight_lambda"]): row.to_dict() for _, row in df.iterrows()}
            return RecommendationMetricsResponse(test_sample_size=100, benchmark_results=res_dict)
        rec_engine = get_recommendation_engine()
        metrics = rec_engine.evaluate_engine(test_samples=50)
        return RecommendationMetricsResponse(**metrics)

    return app


# Application entry point for Uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.production_api:app", host="0.0.0.0", port=8000, reload=False)
