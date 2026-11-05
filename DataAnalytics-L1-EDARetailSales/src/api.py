"""
Production FastAPI REST API for Retail Sales Analytics & Machine Learning Inference.

Exposes comprehensive endpoints for descriptive statistics, time-series velocity,
demographic segmentation, Pareto concentration profiling, real-time ML inference,
continuous MLOps drift monitoring, RFM customer clustering, and financial what-if simulation.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.analytics import RetailAnalyticsEngine
from src.ml_models import RetailMLManager
from src.recommendations import BusinessRecommendationEngine
from src.mlops_drift import DriftDetector
from src.segmentation import CustomerSegmentationEngine
from src.simulation import FinancialWhatIfSimulator
from src.elasticity_optimization import PriceElasticityOptimizer
from src.ab_experimentation import RetailABTestingEngine
from src.clv_churn import CustomerLifetimeValueEngine
from src.inventory_supply_chain import InventorySupplyChainEngine
from src.market_basket import MarketBasketAffinityEngine
from src.demand_forecasting import MultiHorizonDemandForecaster
from src.attribution_marketing import AttributionMarketingEngine
from src.gmroi_assortment import GMROIAssortmentEngine
from src.cohort_retention import CohortRetentionSurvivalEngine
from src.dynamic_pricing import DynamicPricingEngine
from src.next_best_action import NextBestActionEngine
from src.financial_risk import FinancialRiskStressEngine

# Global singletons
analytics_engine: Optional[RetailAnalyticsEngine] = None
ml_manager: Optional[RetailMLManager] = None
drift_detector: Optional[DriftDetector] = None
segmentation_engine: Optional[CustomerSegmentationEngine] = None
whatif_simulator: Optional[FinancialWhatIfSimulator] = None
elasticity_optimizer: Optional[PriceElasticityOptimizer] = None
ab_engine: Optional[RetailABTestingEngine] = None
clv_engine: Optional[CustomerLifetimeValueEngine] = None
inventory_engine: Optional[InventorySupplyChainEngine] = None
market_basket_engine: Optional[MarketBasketAffinityEngine] = None
forecasting_engine: Optional[MultiHorizonDemandForecaster] = None
attribution_engine: Optional[AttributionMarketingEngine] = None
gmroi_engine: Optional[GMROIAssortmentEngine] = None
cohort_engine: Optional[CohortRetentionSurvivalEngine] = None
pricing_engine: Optional[DynamicPricingEngine] = None
nba_engine: Optional[NextBestActionEngine] = None
financial_risk_engine: Optional[FinancialRiskStressEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes engines and loads/trains machine learning models on startup."""
    global analytics_engine, ml_manager, drift_detector, segmentation_engine, whatif_simulator
    global elasticity_optimizer, ab_engine, clv_engine
    global inventory_engine, market_basket_engine, forecasting_engine
    global attribution_engine, gmroi_engine, cohort_engine
    global pricing_engine, nba_engine, financial_risk_engine
    analytics_engine = RetailAnalyticsEngine()
    ml_manager = RetailMLManager()
    ml_manager.load_or_train_models()
    drift_detector = DriftDetector(baseline_df=analytics_engine.df)
    segmentation_engine = CustomerSegmentationEngine()
    whatif_simulator = FinancialWhatIfSimulator()
    elasticity_optimizer = PriceElasticityOptimizer()
    ab_engine = RetailABTestingEngine()
    clv_engine = CustomerLifetimeValueEngine()
    inventory_engine = InventorySupplyChainEngine()
    market_basket_engine = MarketBasketAffinityEngine()
    forecasting_engine = MultiHorizonDemandForecaster()
    attribution_engine = AttributionMarketingEngine()
    gmroi_engine = GMROIAssortmentEngine()
    cohort_engine = CohortRetentionSurvivalEngine()
    pricing_engine = DynamicPricingEngine()
    nba_engine = NextBestActionEngine()
    financial_risk_engine = FinancialRiskStressEngine()
    yield


app = FastAPI(
    title="Retail Sales Analytics & Predictive Intelligence API",
    description=(
        "Production-ready enterprise REST API delivering real-time statistical analytics, "
        "time-series forecasting, demographic profiling, Pareto concentration intelligence, "
        "machine learning inference, continuous MLOps drift monitoring, RFM behavioral clustering, "
        "and financial what-if scenario simulations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class VIPPredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years (18–100)")
    gender: str = Field(..., description="Customer gender ('Male' or 'Female')")
    product_category: str = Field(
        ..., description="Catalog department ('Electronics', 'Clothing', or 'Beauty')"
    )
    quantity: int = Field(..., ge=1, le=20, description="Item quantity in basket (1–20)")
    price_per_unit: float = Field(
        ..., gt=0, description="Price per unit in USD ($25, $30, $50, $300, $500)"
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v_title = v.strip().title()
        if v_title not in ["Male", "Female"]:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v_title

    @field_validator("product_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v_title = v.strip().title()
        if v_title not in ["Electronics", "Clothing", "Beauty"]:
            raise ValueError(
                "Product category must be 'Electronics', 'Clothing', or 'Beauty'"
            )
        return v_title


class SpendPredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years (18–100)")
    gender: str = Field(..., description="Customer gender ('Male' or 'Female')")
    product_category: str = Field(
        ..., description="Catalog department ('Electronics', 'Clothing', or 'Beauty')"
    )
    quantity: int = Field(..., ge=1, le=20, description="Item quantity in basket (1–20)")
    price_per_unit: float = Field(..., gt=0, description="Unit price in USD")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v_title = v.strip().title()
        if v_title not in ["Male", "Female"]:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v_title

    @field_validator("product_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v_title = v.strip().title()
        if v_title not in ["Electronics", "Clothing", "Beauty"]:
            raise ValueError(
                "Product category must be 'Electronics', 'Clothing', or 'Beauty'"
            )
        return v_title


class BundleRecommendationRequest(BaseModel):
    primary_category: str = Field(
        ..., description="Primary interested department ('Electronics', 'Clothing', 'Beauty')"
    )
    target_units: int = Field(
        default=3, ge=1, le=4, description="Target total basket unit size (1–4 items)"
    )

    @field_validator("primary_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v_title = v.strip().title()
        if v_title not in ["Electronics", "Clothing", "Beauty"]:
            raise ValueError(
                "Primary category must be 'Electronics', 'Clothing', or 'Beauty'"
            )
        return v_title


class DriftAuditRequest(BaseModel):
    scenario: str = Field(
        default="stable",
        description="Drift test scenario: 'stable', 'tech_surge', 'inflation', or 'youth_pivot'",
    )
    sample_size: int = Field(
        default=300, ge=10, le=1000, description="Batch sample size (10-1000)"
    )

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        valid = ["stable", "tech_surge", "inflation", "youth_pivot"]
        if v.lower() not in valid:
            raise ValueError(f"Scenario must be one of {valid}")
        return v.lower()


class WhatIfSimulationRequest(BaseModel):
    bundle_adoption_rate: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Bundle adoption fraction (0.0 - 1.0)"
    )
    target_basket_units: float = Field(
        default=3.20, gt=0.0, le=10.0, description="Target basket units (baseline: 2.51)"
    )
    bundle_discount_pct: float = Field(
        default=0.10, ge=0.0, le=0.90, description="Tiered bundle discount percentage (0.0 - 0.90)"
    )
    vip_retention_lift_pct: float = Field(
        default=0.15, ge=0.0, le=1.0, description="VIP repeat purchase rate increase (0.0 - 1.0)"
    )


# --- DAY 10 SCHEMAS ---
class MarkdownSimulationRequest(BaseModel):
    category: str = Field(..., description="Category ('Electronics', 'Clothing', or 'Beauty')")
    discount_pct: float = Field(..., ge=0.01, le=0.90, description="Promotional discount fraction (0.01 - 0.90)")
    expected_volume_lift_pct: float = Field(
        ..., ge=-0.90, le=5.0, description="Projected volume lift fraction (-0.90 to 5.0)"
    )
    gross_margin_pct: Optional[float] = Field(
        default=0.40, ge=0.05, le=0.95, description="Category gross margin fraction (default 0.40)"
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v_title = v.strip().title()
        if v_title not in ["Electronics", "Clothing", "Beauty"]:
            raise ValueError("Category must be 'Electronics', 'Clothing', or 'Beauty'")
        return v_title


class ABTestProportionRequest(BaseModel):
    control_successes: int = Field(..., ge=0, description="Control group successes (e.g. conversions)")
    control_total: int = Field(..., gt=0, description="Control group total sample size")
    treatment_successes: int = Field(..., ge=0, description="Treatment group successes")
    treatment_total: int = Field(..., gt=0, description="Treatment group total sample size")
    metric_name: Optional[str] = Field(default="Conversion Rate", description="Metric label")
    alpha: Optional[float] = Field(default=0.05, gt=0.0, lt=0.5, description="Significance level alpha")


class ABTestContinuousRequest(BaseModel):
    control_data: List[float] = Field(..., min_length=2, description="Continuous observations for Control")
    treatment_data: List[float] = Field(..., min_length=2, description="Continuous observations for Treatment")
    metric_name: Optional[str] = Field(default="Basket Spend", description="Metric label")
    alpha: Optional[float] = Field(default=0.05, gt=0.0, lt=0.5, description="Significance level alpha")


class CustomerCLVRequest(BaseModel):
    recency_days: float = Field(..., ge=0.0, description="Days elapsed since most recent transaction")
    aov: float = Field(..., gt=0.0, description="Average Order Value ($)")
    gross_margin_pct: Optional[float] = Field(
        default=0.40, ge=0.05, le=0.95, description="Product margin fraction (0.05 - 0.95)"
    )
    horizon_days: Optional[int] = Field(
        default=365, ge=30, le=1825, description="Forecasting horizon in days (30 - 1825)"
    )


# --- DAY 11 SCHEMAS ---
class SafetyStockRequest(BaseModel):
    service_level: float = Field(default=0.95, ge=0.50, lt=1.0, description="Target cycle service level")
    lead_time_days: Optional[float] = Field(default=None, gt=0.0, description="Supplier lead time in days")
    lead_time_std: Optional[float] = Field(default=None, ge=0.0, description="Lead time standard deviation")
    category: Optional[str] = Field(default=None, description="Merchandise category filter")


class EOQRequest(BaseModel):
    order_cost: Optional[float] = Field(default=50.0, gt=0.0, description="Fixed order setup cost ($)")
    holding_cost_rate: Optional[float] = Field(default=0.20, gt=0.0, le=1.0, description="Annual holding cost rate")
    category: Optional[str] = Field(default=None, description="Merchandise category filter")


class StockoutRiskRequest(BaseModel):
    service_level: float = Field(default=0.95, ge=0.50, lt=1.0, description="Target cycle service level")
    lead_time_days: Optional[float] = Field(default=7.0, gt=0.0, description="Supplier lead time in days")
    lead_time_std: Optional[float] = Field(default=1.5, ge=0.0, description="Lead time standard deviation")
    category: Optional[str] = Field(default=None, description="Merchandise category filter")
    gross_margin_pct: Optional[float] = Field(default=0.40, gt=0.0, lt=1.0, description="Gross profit margin percentage")


class ComboRecommendationRequest(BaseModel):
    target_category: str = Field(..., description="Target merchandise category")
    discount_rate: Optional[float] = Field(default=0.12, ge=0.01, le=0.40, description="Bundle promotional discount rate")
    gross_margin_pct: Optional[float] = Field(default=0.40, gt=0.0, lt=1.0, description="Gross profit margin percentage")


class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=30, ge=1, le=180, description="Forward forecast horizon in days")
    category: Optional[str] = Field(default=None, description="Merchandise category filter")
    alpha_conformal: Optional[float] = Field(default=0.05, gt=0.0, lt=0.50, description="Conformal significance level")


class BudgetOptimizationRequest(BaseModel):
    total_budget: Optional[float] = Field(default=None, gt=0.0, description="Total marketing budget to allocate")
    risk_aversion: float = Field(default=0.5, ge=0.0, le=2.0, description="Risk penalty coefficient")


class CapitalReallocationRequest(BaseModel):
    reallocation_pct: float = Field(default=0.35, ge=0.05, le=0.90, description="Fraction of capital to reallocate")
    source_quadrant: str = Field(default="Dead Stock / Capital Traps", description="Source quadrant to divest from")
    target_quadrant: str = Field(default="Stars", description="Target quadrant to reinvest in")


class RetentionCampaignRequest(BaseModel):
    intervention_cost_per_user: float = Field(default=12.0, gt=0.0, le=200.0, description="Cost per retention intervention")
    expected_churn_reduction_pct: float = Field(default=0.25, ge=0.01, le=0.80, description="Expected churn reduction fraction")



# --- DAY 13 REQUEST MODELS: DYNAMIC PRICING, CAUSAL UPLIFT & FINANCIAL RISK ---
class DynamicQuoteRequest(BaseModel):
    category: str = Field(..., description="Merchandise category: 'Beauty', 'Clothing', or 'Electronics'")
    base_price: float = Field(..., gt=0.0, description="Catalog base list price")
    current_stock: int = Field(..., ge=0, description="Current on-hand inventory count")
    safety_stock: Optional[int] = Field(default=25, gt=0, description="Calibrated safety stock threshold")
    demand_velocity_ratio: Optional[float] = Field(default=1.0, gt=0.0, description="Rolling sales velocity multiplier")
    competitor_price: Optional[float] = Field(default=None, gt=0.0, description="Optional competitor price benchmark")


class BanditSimulationRequest(BaseModel):
    category: str = Field(default="Electronics", description="Merchandise category")
    algorithm: str = Field(default="thompson_sampling", description="Algorithm: 'thompson_sampling', 'ucb1', or 'epsilon_greedy'")
    n_rounds: int = Field(default=1000, ge=10, le=50000, description="Number of simulation rounds")
    epsilon: Optional[float] = Field(default=0.10, ge=0.01, le=0.50, description="Exploration rate for epsilon")


class NBARecommendationRequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, description="Optional customer ID to personalize")
    age: Optional[int] = Field(default=None, ge=18, le=100)
    gender: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    estimated_aov: float = Field(default=250.0, gt=0.0, description="Estimated average order value")
    gross_margin_pct: float = Field(default=0.45, ge=0.05, le=0.95, description="Expected gross margin percentage")
    recent_touches_count: int = Field(default=1, ge=0, description="Number of recent marketing contacts")


class UpliftCampaignRequest(BaseModel):
    campaign_budget: float = Field(default=10000.0, gt=0.0, description="Total campaign budget")
    incentive_cost_per_touch: float = Field(default=8.0, gt=0.0, description="Unit cost per incentive touch")
    avg_basket_value: float = Field(default=350.0, gt=0.0, description="Average order value")
    gross_margin_pct: float = Field(default=0.45, ge=0.10, le=0.90, description="Gross margin percentage")


class MonteCarloStressRequest(BaseModel):
    n_simulations: int = Field(default=5000, ge=100, le=20000, description="Number of Monte Carlo paths")
    demand_shock_mean_pct: float = Field(default=-0.10, ge=-0.50, le=0.20, description="Mean demand shock percentage")
    demand_shock_std_pct: float = Field(default=0.08, ge=0.01, le=0.30, description="Standard deviation of demand")
    cogs_inflation_mean_pct: float = Field(default=0.06, ge=-0.10, le=0.30, description="Mean supplier price inflation")
    cogs_inflation_std_pct: float = Field(default=0.04, ge=0.01, le=0.20, description="Standard deviation of inflation")
    dpo_contraction_days: float = Field(default=5.0, ge=0.0, le=30.0, description="Days supplier terms are compressed")
    initial_cash_buffer: float = Field(default=60000.0, gt=0.0, description="Starting treasury cash reserves")


class WorkingCapitalOptimizationRequest(BaseModel):
    dio_reduction_days: float = Field(default=12.0, ge=0.0, le=60.0, description="DIO days reduction")
    dpo_extension_days: float = Field(default=10.0, ge=0.0, le=45.0, description="DPO supplier credit extension")
    dso_reduction_days: float = Field(default=1.0, ge=0.0, le=15.0, description="DSO settlement acceleration")
    short_term_borrowing_rate_pct: float = Field(default=7.5, ge=0.0, le=30.0, description="Annual credit borrowing rate")

def get_analytics() -> RetailAnalyticsEngine:
    global analytics_engine
    if analytics_engine is None:
        analytics_engine = RetailAnalyticsEngine()
    return analytics_engine


def get_ml() -> RetailMLManager:
    global ml_manager
    if ml_manager is None:
        ml_manager = RetailMLManager()
        ml_manager.load_or_train_models()
    return ml_manager


def get_drift() -> DriftDetector:
    global drift_detector
    if drift_detector is None:
        drift_detector = DriftDetector(baseline_df=get_analytics().df)
    return drift_detector


def get_segmentation() -> CustomerSegmentationEngine:
    global segmentation_engine
    if segmentation_engine is None:
        segmentation_engine = CustomerSegmentationEngine()
    return segmentation_engine


def get_simulation() -> FinancialWhatIfSimulator:
    global whatif_simulator
    if whatif_simulator is None:
        whatif_simulator = FinancialWhatIfSimulator()
    return whatif_simulator


def get_elasticity() -> PriceElasticityOptimizer:
    global elasticity_optimizer
    if elasticity_optimizer is None:
        elasticity_optimizer = PriceElasticityOptimizer()
    return elasticity_optimizer


def get_ab_engine() -> RetailABTestingEngine:
    global ab_engine
    if ab_engine is None:
        ab_engine = RetailABTestingEngine()
    return ab_engine


def get_clv() -> CustomerLifetimeValueEngine:
    global clv_engine
    if clv_engine is None:
        clv_engine = CustomerLifetimeValueEngine()
    return clv_engine


def get_inventory() -> InventorySupplyChainEngine:
    global inventory_engine
    if inventory_engine is None:
        inventory_engine = InventorySupplyChainEngine()
    return inventory_engine


def get_market_basket() -> MarketBasketAffinityEngine:
    global market_basket_engine
    if market_basket_engine is None:
        market_basket_engine = MarketBasketAffinityEngine()
    return market_basket_engine


def get_forecasting() -> MultiHorizonDemandForecaster:
    global forecasting_engine
    if forecasting_engine is None:
        forecasting_engine = MultiHorizonDemandForecaster()
    return forecasting_engine


def get_attribution() -> AttributionMarketingEngine:
    global attribution_engine
    if attribution_engine is None:
        attribution_engine = AttributionMarketingEngine()
    return attribution_engine


def get_gmroi() -> GMROIAssortmentEngine:
    global gmroi_engine
    if gmroi_engine is None:
        gmroi_engine = GMROIAssortmentEngine()
    return gmroi_engine


def get_cohort_retention() -> CohortRetentionSurvivalEngine:
    global cohort_engine
    if cohort_engine is None:
        cohort_engine = CohortRetentionSurvivalEngine()
    return cohort_engine


def get_pricing() -> DynamicPricingEngine:
    global pricing_engine
    if pricing_engine is None:
        pricing_engine = DynamicPricingEngine()
    return pricing_engine


def get_nba() -> NextBestActionEngine:
    global nba_engine
    if nba_engine is None:
        nba_engine = NextBestActionEngine()
    return nba_engine


def get_financial_risk() -> FinancialRiskStressEngine:
    global financial_risk_engine
    if financial_risk_engine is None:
        financial_risk_engine = FinancialRiskStressEngine()
    return financial_risk_engine


# --- ROOT & HEALTH ---
@app.get("/", tags=["System & Health"])
def root_status() -> Dict[str, Any]:
    """Root endpoint verifying API availability and active version."""
    analytics = get_analytics()
    kpis = analytics.get_kpi_overview()
    return {
        "status": "ONLINE",
        "version": "1.0.0",
        "service": "Retail Sales Analytics & Machine Learning Intelligence API",
        "summary": {
            "total_revenue": kpis["total_revenue"],
            "total_orders": kpis["total_orders"],
            "average_order_value": kpis["average_order_value"],
            "gini_coefficient": kpis["gini_coefficient"],
        },
    }


@app.get("/health", tags=["System & Health"])
def health_check() -> Dict[str, Any]:
    """Provides operational readiness and system diagnostics."""
    analytics = get_analytics()
    ml = get_ml()
    return {
        "status": "HEALTHY",
        "database_records_loaded": len(analytics.df),
        "vip_model_ready": ml.vip_pipeline is not None,
        "spend_model_ready": ml.spend_pipeline is not None,
        "gini_index": analytics.calculate_gini(),
    }


# --- ANALYTICAL ROUTES ---
@app.get("/api/v1/metrics/overview", tags=["Executive KPIs"])
def get_executive_overview() -> Dict[str, Any]:
    """Returns storewide executive performance indicators and baseline metrics."""
    return get_analytics().get_kpi_overview()


@app.get("/api/v1/analytics/descriptive", tags=["Statistical Profiling"])
def get_descriptive_stats() -> Dict[str, Any]:
    """Returns descriptive statistics (mean, median, mode, std dev, skewness, kurtosis, IQR) for all numerical columns."""
    return get_analytics().get_descriptive_statistics()


@app.get("/api/v1/analytics/time-series", tags=["Time-Series & Velocity"])
def get_time_series_analytics() -> Dict[str, Any]:
    """Returns monthly trends, quarterly cycles, and day-of-week sales velocity with category lift."""
    analytics = get_analytics()
    return {
        "monthly_trends": analytics.get_monthly_trends(),
        "quarterly_trends": analytics.get_quarterly_trends(),
        "day_of_week_dynamics": analytics.get_day_of_week_dynamics(),
    }


@app.get("/api/v1/analytics/demographics", tags=["Customer Demographics"])
def get_customer_demographics() -> Dict[str, Any]:
    """Returns demographic profiles across customer age cohorts and gender purchasing parity."""
    return get_analytics().get_demographics_breakdown()


@app.get("/api/v1/analytics/products", tags=["Merchandise & Catalog"])
def get_merchandise_analytics() -> Dict[str, Any]:
    """Returns departmental performance, catalog price tier contributions, and top revenue offerings."""
    return get_analytics().get_product_category_performance()


@app.get("/api/v1/analytics/pareto", tags=["Pareto & Concentration"])
def get_pareto_analytics() -> Dict[str, Any]:
    """Returns Lorenz curve coordinates, empirical Gini coefficient, and spend tier polarization."""
    return get_analytics().get_pareto_concentration()


# --- ML INFERENCE ROUTES ---
@app.post("/api/v1/predict/vip", tags=["Machine Learning Inference"])
def predict_vip_propensity(payload: VIPPredictionRequest) -> Dict[str, Any]:
    """
    Predicts the probability and risk classification that a customer transaction
    qualifies as a VIP / Whale order (Total Amount >= $1,000).
    """
    try:
        ml = get_ml()
        result = ml.predict_vip(
            age=payload.age,
            gender=payload.gender,
            product_category=payload.product_category,
            quantity=payload.quantity,
            price_per_unit=payload.price_per_unit,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/api/v1/predict/spend", tags=["Machine Learning Inference"])
def predict_transaction_spend(payload: SpendPredictionRequest) -> Dict[str, Any]:
    """
    Predicts total basket expenditure ($) using the Gradient Boosting spend regressor,
    with 95% empirical confidence intervals.
    """
    try:
        ml = get_ml()
        result = ml.predict_spend(
            age=payload.age,
            gender=payload.gender,
            product_category=payload.product_category,
            quantity=payload.quantity,
            price_per_unit=payload.price_per_unit,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Regression error: {str(e)}",
        )


@app.get("/api/v1/recommendations", tags=["Strategic Intelligence"])
def get_strategic_recommendations() -> Dict[str, Any]:
    """Returns the 5 data-driven strategic business recommendations formulated during EDA."""
    recs = BusinessRecommendationEngine.get_strategic_recommendations()
    return {"recommendations_count": len(recs), "recommendations": recs}


@app.post("/api/v1/recommendations/bundle", tags=["Strategic Intelligence"])
def generate_cross_sell_bundle(payload: BundleRecommendationRequest) -> Dict[str, Any]:
    """
    Generates a curated cross-department bundle with tiered discounting to expand
    basket unit volume past the 2.5-unit inelasticity threshold.
    """
    bundle = BusinessRecommendationEngine.generate_bundle(
        primary_category=payload.primary_category,
        target_units=payload.target_units,
    )
    return bundle


# --- DAY 9 NEW REST ENDPOINTS ---
@app.post("/api/v1/mlops/drift", tags=["MLOps & Data Drift"])
def run_drift_audit(payload: DriftAuditRequest) -> Dict[str, Any]:
    """
    Executes a real-time statistical drift audit comparing live production batch
    against the golden baseline dataset using Population Stability Index (PSI)
    and Kolmogorov-Smirnov (KS) hypothesis tests.
    """
    try:
        detector = get_drift()
        batch = detector.generate_synthetic_batch(
            scenario=payload.scenario,
            n_samples=payload.sample_size,
        )
        audit_results = detector.run_drift_audit(batch)
        audit_results["audit_scenario"] = payload.scenario
        return audit_results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift audit failed: {str(e)}",
        )


@app.get("/api/v1/analytics/rfm-segments", tags=["Customer Segmentation"])
def get_rfm_customer_segments() -> Dict[str, Any]:
    """
    Returns customer RFM metrics and unsupervised K-Means behavioral clustering profiles
    (VIP Whales, Core Lifestyle Shoppers, Premium Specialists, Casual Budget Explorers).
    """
    try:
        engine = get_segmentation()
        return engine.get_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}",
        )


@app.post("/api/v1/simulations/what-if", tags=["Strategic Simulations"])
def run_what_if_simulation(payload: WhatIfSimulationRequest) -> Dict[str, Any]:
    """
    Simulates commercial interventions (cross-department bundles & VIP retention),
    projecting net revenue uplift ($ and %), gross margin gains, and ROI.
    """
    try:
        sim = get_simulation()
        result = sim.simulate_combined_strategy(
            bundle_adoption=payload.bundle_adoption_rate,
            target_units=payload.target_basket_units,
            bundle_discount=payload.bundle_discount_pct,
            vip_lift=payload.vip_retention_lift_pct,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error: {str(e)}",
        )


# --- DAY 10 REST ENDPOINTS: ELASTICITY, A/B EXPERIMENTATION & CLV ---
@app.get("/api/v1/elasticity/summary", tags=["Price Elasticity & Optimization"])
def get_price_elasticity_summary() -> Dict[str, Any]:
    """
    Returns econometric log-log regression price elasticity of demand (PED)
    and arc elasticity breakdown across all catalog merchandise categories.
    """
    try:
        optimizer = get_elasticity()
        return optimizer.get_overall_elasticity_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Elasticity calculation failed: {str(e)}",
        )


@app.post("/api/v1/elasticity/simulate", tags=["Price Elasticity & Optimization"])
def simulate_markdown_promotion(payload: MarkdownSimulationRequest) -> Dict[str, Any]:
    """
    Simulates the commercial and margin impact of a promotional markdown discount
    evaluating breakeven volume lift and value-accretive viability.
    """
    try:
        optimizer = get_elasticity()
        return optimizer.simulate_markdown_pricing(
            category=payload.category,
            discount_pct=payload.discount_pct,
            expected_volume_lift_pct=payload.expected_volume_lift_pct,
            gross_margin_pct=payload.gross_margin_pct,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Markdown simulation failed: {str(e)}",
        )


@app.get("/api/v1/elasticity/tiers/{category}", tags=["Price Elasticity & Optimization"])
def get_tier_optimization(category: str) -> Dict[str, Any]:
    """
    Evaluates revenue and gross profit contribution across discrete pricing tiers
    for a given category and identifies optimal profit-maximizing tier.
    """
    cat_clean = category.strip().title()
    if cat_clean not in ["Electronics", "Clothing", "Beauty"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{category}' not found in catalog.",
        )
    try:
        optimizer = get_elasticity()
        return optimizer.optimize_price_points(category=cat_clean)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tier optimization failed: {str(e)}",
        )


@app.post("/api/v1/experiments/ab-proportions", tags=["A/B Experimentation"])
def evaluate_ab_proportions(payload: ABTestProportionRequest) -> Dict[str, Any]:
    """
    Executes dual Frequentist (Two-proportion z-test, Chi-square) and Bayesian
    Beta-Binomial conjugate updating for conversion / adoption experiment arms.
    """
    try:
        engine = get_ab_engine()
        freq_results = engine.evaluate_proportion_ab(
            control_successes=payload.control_successes,
            control_total=payload.control_total,
            treatment_successes=payload.treatment_successes,
            treatment_total=payload.treatment_total,
            metric_name=payload.metric_name or "Conversion Rate",
            alpha=payload.alpha or 0.05,
        )
        bayes_results = engine.bayesian_conversion_test(
            control_conversions=payload.control_successes,
            control_impressions=payload.control_total,
            treatment_conversions=payload.treatment_successes,
            treatment_impressions=payload.treatment_total,
        )
        return {
            "frequentist_analysis": freq_results,
            "bayesian_analysis": bayes_results,
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B evaluation failed: {str(e)}",
        )


@app.post("/api/v1/experiments/ab-continuous", tags=["A/B Experimentation"])
def evaluate_ab_continuous(payload: ABTestContinuousRequest) -> Dict[str, Any]:
    """
    Executes Welch's t-test, Mann-Whitney U test, and Bayesian Dirichlet bootstrap
    for continuous spend and basket value experimentation.
    """
    try:
        engine = get_ab_engine()
        freq_results = engine.evaluate_continuous_ab(
            control_data=payload.control_data,
            treatment_data=payload.treatment_data,
            metric_name=payload.metric_name or "Basket Spend",
            alpha=payload.alpha or 0.05,
        )
        bayes_results = engine.bayesian_revenue_test(
            control_spend=payload.control_data,
            treatment_spend=payload.treatment_data,
        )
        return {
            "frequentist_analysis": freq_results,
            "bayesian_bootstrap": bayes_results,
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Continuous A/B evaluation failed: {str(e)}",
        )


@app.get("/api/v1/experiments/power-matrix", tags=["A/B Experimentation"])
def get_sample_size_power_matrix(
    baseline_rate: float = Query(0.20, ge=0.01, le=0.99, description="Baseline conversion rate"),
    alpha: float = Query(0.05, ge=0.01, le=0.20, description="Significance level"),
    power: float = Query(0.80, ge=0.50, le=0.99, description="Target statistical power"),
) -> List[Dict[str, Any]]:
    """
    Generates required sample sizes per arm across retail Minimum Detectable Effects (MDE).
    """
    try:
        engine = get_ab_engine()
        return engine.generate_power_matrix(
            baseline_rate=baseline_rate,
            alpha=alpha,
            power=power,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Power matrix generation failed: {str(e)}",
        )


@app.get("/api/v1/clv/overview", tags=["Customer Lifetime Value & Churn"])
def get_clv_overview() -> Dict[str, Any]:
    """
    Returns storewide customer equity audit, churn tier risk distribution,
    and VIP whale early warning statistics.
    """
    try:
        engine = get_clv()
        return engine.audit_customer_equity()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CLV overview failed: {str(e)}",
        )


@app.post("/api/v1/clv/predict-customer", tags=["Customer Lifetime Value & Churn"])
def predict_individual_clv(payload: CustomerCLVRequest) -> Dict[str, Any]:
    """
    Predicts probabilistic churn risk and discounted future CLV for an individual shopper.
    """
    try:
        engine = get_clv()
        return engine.predict_customer_clv(
            recency_days=payload.recency_days,
            aov=payload.aov,
            gross_margin_pct=payload.gross_margin_pct,
            horizon_days=payload.horizon_days or 365,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Customer CLV prediction failed: {str(e)}",
        )


@app.get("/api/v1/clv/at-risk-vips", tags=["Customer Lifetime Value & Churn"])
def get_at_risk_vip_list(
    min_spend: float = Query(1000.0, ge=100.0, description="Minimum historical spend"),
    limit: int = Query(15, ge=1, le=100, description="Maximum records to return"),
) -> List[Dict[str, Any]]:
    """
    Retrieves highest-priority at-risk VIP whale customers for targeted retention campaigns.
    """
    try:
        engine = get_clv()
        return engine.get_at_risk_vips(min_spend=min_spend, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"At-risk VIP query failed: {str(e)}",
        )


# ==========================================
# DAY 11: SUPPLY CHAIN & INVENTORY OPTIMIZATION
# ==========================================

@app.get("/api/v1/inventory/daily-metrics", tags=["Supply Chain & Inventory"])
def get_daily_demand_metrics_endpoint(
    category: Optional[str] = Query(None, description="Optional product category filter")
) -> Dict[str, Any]:
    """Calculates daily demand statistics, velocity, and average unit price."""
    try:
        engine = get_inventory()
        return engine.get_daily_demand_metrics(category=category)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/inventory/safety-stock", tags=["Supply Chain & Inventory"])
def calculate_safety_stock_endpoint(payload: SafetyStockRequest) -> Dict[str, Any]:
    """Calculates Safety Stock (SS) and Reorder Point (ROP) under lead time uncertainty."""
    try:
        engine = get_inventory()
        return engine.calculate_safety_stock_and_rop(
            service_level=payload.service_level,
            lead_time_days=payload.lead_time_days,
            lead_time_std=payload.lead_time_std,
            category=payload.category,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/inventory/eoq", tags=["Supply Chain & Inventory"])
def calculate_eoq_endpoint(payload: EOQRequest) -> Dict[str, Any]:
    """Calculates Economic Order Quantity (EOQ), order cycles, and inventory holding costs."""
    try:
        engine = get_inventory()
        return engine.calculate_eoq(
            order_cost=payload.order_cost,
            holding_cost_rate=payload.holding_cost_rate,
            category=payload.category,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/inventory/stockout-risk", tags=["Supply Chain & Inventory"])
def simulate_stockout_risk_endpoint(payload: StockoutRiskRequest) -> Dict[str, Any]:
    """Simulates stockout probabilities and financial margin at risk using standard normal loss G(k)."""
    try:
        engine = get_inventory()
        return engine.simulate_stockout_risk(
            service_level=payload.service_level,
            lead_time_days=payload.lead_time_days,
            lead_time_std=payload.lead_time_std,
            category=payload.category,
            gross_margin_pct=payload.gross_margin_pct or 0.40,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/inventory/abc-fsn", tags=["Supply Chain & Inventory"])
def get_abc_fsn_analysis_endpoint() -> Dict[str, Any]:
    """Executes multi-criteria ABC-FSN strategic inventory stratification."""
    try:
        engine = get_inventory()
        return engine.perform_abc_fsn_analysis()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/inventory/storewide-summary", tags=["Supply Chain & Inventory"])
def get_storewide_inventory_summary_endpoint(
    service_level: float = Query(0.95, ge=0.50, lt=1.0, description="Target service level")
) -> Dict[str, Any]:
    """Consolidates storewide supply chain parameters and working capital investments."""
    try:
        engine = get_inventory()
        return engine.get_storewide_inventory_summary(service_level=service_level)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==========================================
# DAY 11: MARKET BASKET & CROSS-MERCHANDISING
# ==========================================

@app.get("/api/v1/market-basket/affinity", tags=["Market Basket & Merchandising"])
def get_market_basket_affinity_endpoint() -> Dict[str, Any]:
    """Computes cross-category co-occurrence, association rule metrics, and statistical affinity."""
    try:
        engine = get_market_basket()
        return engine.build_cooccurrence_and_affinity_matrix()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/market-basket/rules", tags=["Market Basket & Merchandising"])
def get_association_rules_endpoint(
    min_support: float = Query(0.05, ge=0.01, le=1.0, description="Minimum rule support"),
    min_confidence: float = Query(0.20, ge=0.01, le=1.0, description="Minimum rule confidence"),
    min_lift: float = Query(0.80, ge=0.0, le=10.0, description="Minimum rule lift"),
) -> List[Dict[str, Any]]:
    """Filters and ranks association rules meeting specified minimum thresholds."""
    try:
        engine = get_market_basket()
        return engine.mine_association_rules(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/market-basket/combos", tags=["Market Basket & Merchandising"])
def get_merchandising_combos_endpoint(payload: ComboRecommendationRequest) -> List[Dict[str, Any]]:
    """Generates optimal cross-department promotional bundle pairings for a target category."""
    try:
        engine = get_market_basket()
        return engine.recommend_merchandising_combos(
            target_category=payload.target_category,
            discount_rate=payload.discount_rate or 0.12,
            gross_margin_pct=payload.gross_margin_pct or 0.40,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/market-basket/demographic-affinity", tags=["Market Basket & Merchandising"])
def get_demographic_affinity_endpoint() -> Dict[str, Any]:
    """Cross-analyzes category preferences and joint affinities across demographic age cohorts."""
    try:
        engine = get_market_basket()
        return engine.demographic_affinity_breakdown()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==========================================
# DAY 11: MULTI-HORIZON DEMAND FORECASTING
# ==========================================

@app.post("/api/v1/forecasting/predict", tags=["Multi-Horizon Demand Forecasting"])
def predict_demand_forecast_endpoint(payload: ForecastRequest) -> Dict[str, Any]:
    """Generates forward demand forecasts with conformal uncertainty prediction intervals."""
    try:
        engine = get_forecasting()
        return engine.forecast_demand(
            horizon_days=payload.horizon_days,
            category=payload.category,
            alpha_conformal=payload.alpha_conformal or 0.05,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/forecasting/accuracy", tags=["Multi-Horizon Demand Forecasting"])
def get_forecast_accuracy_endpoint(
    test_window_days: int = Query(45, ge=7, le=90, description="Holdout test window in days"),
    category: Optional[str] = Query(None, description="Optional merchandise category filter"),
) -> Dict[str, Any]:
    """Executes holdout walk-forward backtesting to quantify empirical forecast accuracy (MAE, RMSE, MAPE)."""
    try:
        engine = get_forecasting()
        return engine.evaluate_forecast_accuracy(
            test_window_days=test_window_days,
            category=category,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/forecasting/anomalies", tags=["Multi-Horizon Demand Forecasting"])
def get_demand_anomalies_endpoint(
    category: Optional[str] = Query(None, description="Optional merchandise category filter"),
    z_threshold: float = Query(2.5, ge=1.5, le=5.0, description="Z-score outlier threshold"),
) -> List[Dict[str, Any]]:
    """Identifies demand spikes and operational velocity outliers using a rolling Z-score indicator."""
    try:
        engine = get_forecasting()
        return engine.detect_demand_anomalies(
            category=category,
            z_threshold=z_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==========================================
# DAY 12: OMNICHANNEL MARKETING ATTRIBUTION
# ==========================================

@app.get("/api/v1/attribution/models", tags=["Omnichannel Marketing Attribution"])
def get_heuristic_attribution_models_endpoint(
    half_life_days: float = Query(7.0, gt=0.0, le=90.0, description="Half-life decay constant in days for time-decay model"),
) -> Dict[str, Any]:
    """Compares 5 standard heuristic multi-touch attribution models (First, Last, Linear, Time-Decay, Position)."""
    try:
        engine = get_attribution()
        engine.half_life_days = half_life_days
        return engine.get_heuristic_attribution_summary()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/attribution/markov", tags=["Omnichannel Marketing Attribution"])
def get_markov_attribution_endpoint() -> Dict[str, Any]:
    """Computes algorithmic First-Order Markov Chain attribution with channel removal effects."""
    try:
        engine = get_attribution()
        return engine.compute_markov_attribution()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/attribution/shapley", tags=["Omnichannel Marketing Attribution"])
def get_shapley_attribution_endpoint() -> Dict[str, Any]:
    """Computes cooperative Game-Theoretic Shapley Value Attribution across channel coalitions."""
    try:
        engine = get_attribution()
        return engine.compute_shapley_attribution()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/attribution/optimize-budget", tags=["Omnichannel Marketing Attribution"])
def optimize_marketing_budget_endpoint(payload: BudgetOptimizationRequest) -> Dict[str, Any]:
    """Executes Markowitz Modern Portfolio Theory (MPT) quadratic optimization to maximize storewide ROAS."""
    try:
        engine = get_attribution()
        return engine.optimize_marketing_budget(
            total_budget=payload.total_budget,
            risk_aversion=payload.risk_aversion,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==========================================
# DAY 12: GMROI & ASSORTMENT RATIONALIZATION
# ==========================================

@app.get("/api/v1/gmroi/overview", tags=["GMROI & Assortment Rationalization"])
def get_category_gmroi_overview_endpoint() -> Dict[str, Any]:
    """Computes category and storewide Gross Margin Return on Investment (GMROI) and capital productivity."""
    try:
        engine = get_gmroi()
        return engine.calculate_category_gmroi()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/gmroi/quadrant-matrix", tags=["GMROI & Assortment Rationalization"])
def get_margin_turn_quadrant_matrix_endpoint() -> Dict[str, Any]:
    """Profiles catalog price tiers on the Margin-Turn 4-Quadrant Strategic Assortment Matrix."""
    try:
        engine = get_gmroi()
        return engine.compute_margin_turn_matrix()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/gmroi/rationalization", tags=["GMROI & Assortment Rationalization"])
def get_assortment_rationalization_endpoint() -> Dict[str, Any]:
    """Diagnoses SKU assortment productivity, capital drag, and identifies rationalization/delisting candidates."""
    try:
        engine = get_gmroi()
        return engine.evaluate_assortment_rationalization()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/gmroi/reallocate-capital", tags=["GMROI & Assortment Rationalization"])
def reallocate_working_capital_endpoint(payload: CapitalReallocationRequest) -> Dict[str, Any]:
    """Simulates commercial yield and profit uplift from shifting capital from dead stock to stars."""
    try:
        engine = get_gmroi()
        return engine.simulate_working_capital_reallocation(
            reallocation_pct=payload.reallocation_pct,
            source_quadrant=payload.source_quadrant,
            target_quadrant=payload.target_quadrant,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==========================================
# DAY 12: CUSTOMER COHORT RETENTION & SURVIVAL
# ==========================================

@app.get("/api/v1/cohorts/retention-matrix", tags=["Customer Cohort Retention & Survival"])
def get_cohort_retention_matrix_endpoint() -> Dict[str, Any]:
    """Builds the monthly customer acquisition cohort retention triangle matrix (Month 0 to Month 11)."""
    try:
        engine = get_cohort_retention()
        return engine.compute_cohort_retention_matrix()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/cohorts/survival-analysis", tags=["Customer Cohort Retention & Survival"])
def get_kaplan_meier_survival_endpoint(
    segment_by: Optional[str] = Query(None, description="Optional segment breakdown: gender, age_cohort, or category"),
) -> Dict[str, Any]:
    """Calculates non-parametric Kaplan-Meier customer survival curves with Greenwood 95% confidence bands."""
    try:
        engine = get_cohort_retention()
        return engine.compute_kaplan_meier_survival(segment_by=segment_by)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/cohorts/log-rank-test", tags=["Customer Cohort Retention & Survival"])
def get_log_rank_test_endpoint(
    segment_col: str = Query("gender", description="Feature column to compare: gender, age_cohort, or category"),
) -> Dict[str, Any]:
    """Executes Log-Rank statistical hypothesis test comparing survival trajectories between groups."""
    try:
        engine = get_cohort_retention()
        return engine.compute_log_rank_test(segment_col=segment_col)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/cohorts/hazard-cliffs", tags=["Customer Cohort Retention & Survival"])
def get_churn_hazard_cliffs_endpoint(
    interval_days: int = Query(15, ge=5, le=60, description="Chronological tenure interval in days"),
) -> Dict[str, Any]:
    """Profiles the empirical hazard rate function lambda(t) and pinpoints acute churn cliffs."""
    try:
        engine = get_cohort_retention()
        return engine.detect_churn_hazard_cliffs(interval_days=interval_days)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/cohorts/retention-campaign", tags=["Customer Cohort Retention & Survival"])
def calculate_retention_campaign_roi_endpoint(payload: RetentionCampaignRequest) -> Dict[str, Any]:
    """Calculates the financial business case and projected ROI of proactive retention interventions."""
    try:
        engine = get_cohort_retention()
        return engine.calculate_retention_campaign_roi(
            intervention_cost_per_user=payload.intervention_cost_per_user,
            expected_churn_reduction_pct=payload.expected_churn_reduction_pct,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- DAY 13 REST ENDPOINTS: DYNAMIC PRICING, CAUSAL UPLIFT & FINANCIAL RISK ---
@app.get("/api/v1/pricing/bandit/summary", tags=["Dynamic Pricing & Multi-Armed Bandit"])
def get_pricing_bandit_summary_endpoint() -> Dict[str, Any]:
    """Returns candidate price arms, conversion probabilities, and theoretical oracle price points."""
    try:
        engine = get_pricing()
        return engine.get_candidate_arms()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/pricing/quote", tags=["Dynamic Pricing & Multi-Armed Bandit"])
def get_dynamic_price_quote_endpoint(payload: DynamicQuoteRequest) -> Dict[str, Any]:
    """Computes real-time dynamic pricing quote with demand surge, inventory scarcity multiplier, and price guardrails."""
    try:
        engine = get_pricing()
        return engine.calculate_dynamic_quote(
            category=payload.category,
            base_price=payload.base_price,
            current_stock=payload.current_stock,
            safety_stock=payload.safety_stock or 25,
            demand_velocity_ratio=payload.demand_velocity_ratio or 1.0,
            competitor_price=payload.competitor_price,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/pricing/simulate-stream", tags=["Dynamic Pricing & Multi-Armed Bandit"])
def simulate_pricing_stream_endpoint(payload: BanditSimulationRequest) -> Dict[str, Any]:
    """Simulates multi-period customer transactions comparing Thompson Sampling, UCB1, or Epsilon-Greedy vs static baseline."""
    try:
        engine = get_pricing()
        return engine.simulate_bandit_learning(
            category=payload.category,
            algorithm=payload.algorithm,
            n_rounds=payload.n_rounds,
            epsilon=payload.epsilon or 0.10,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/pricing/arms", tags=["Dynamic Pricing & Multi-Armed Bandit"])
def get_pricing_arms_endpoint(
    category: Optional[str] = Query(None, description="Optional category filter: Beauty, Clothing, or Electronics"),
) -> Dict[str, Any]:
    """Returns candidate price arms and parameters for all or a specific category."""
    try:
        engine = get_pricing()
        arms_data = engine.get_candidate_arms()
        if category:
            if category not in arms_data["categories"]:
                raise ValueError(f"Category '{category}' not found in candidate arms catalog.")
            return {"category": category, "details": arms_data["categories"][category]}
        return arms_data
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/uplift/summary", tags=["Causal Uplift & Next-Best-Action"])
def get_uplift_summary_endpoint() -> Dict[str, Any]:
    """Summarizes the 4 causal uplift response quadrants (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)."""
    try:
        engine = get_nba()
        return engine.get_uplift_summary()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/uplift/customer/{customer_id}", tags=["Causal Uplift & Next-Best-Action"])
def get_customer_uplift_profile_endpoint(customer_id: str) -> Dict[str, Any]:
    """Retrieves individualized causal uplift score, individual treatment effect tau, and quadrant classification."""
    try:
        engine = get_nba()
        return engine.get_customer_uplift_profile(customer_id=customer_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/uplift/nba/recommend", tags=["Causal Uplift & Next-Best-Action"])
def recommend_next_best_action_endpoint(payload: NBARecommendationRequest) -> Dict[str, Any]:
    """Ranks and recommends the optimal Next-Best-Action maximizing net utility after channel fatigue and cannibalization."""
    try:
        engine = get_nba()
        return engine.recommend_next_best_action(
            customer_id=payload.customer_id,
            age=payload.age,
            gender=payload.gender,
            category=payload.category,
            estimated_aov=payload.estimated_aov,
            gross_margin_pct=payload.gross_margin_pct,
            recent_touches_count=payload.recent_touches_count,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/uplift/campaign/optimize", tags=["Causal Uplift & Next-Best-Action"])
def optimize_campaign_budget_endpoint(payload: UpliftCampaignRequest) -> Dict[str, Any]:
    """Benchmarks Mass Marketing vs Naive Propensity Targeting vs Causal Uplift Targeting."""
    try:
        engine = get_nba()
        return engine.optimize_campaign_budget(
            campaign_budget=payload.campaign_budget,
            incentive_cost_per_touch=payload.incentive_cost_per_touch,
            avg_basket_value=payload.avg_basket_value,
            gross_margin_pct=payload.gross_margin_pct,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/finance/ccc/summary", tags=["Working Capital & Financial Stress Testing"])
def get_cash_conversion_cycle_endpoint() -> Dict[str, Any]:
    """Calculates storewide and merchandise category Cash Conversion Cycle (DIO, DSO, DPO, CCC)."""
    try:
        engine = get_financial_risk()
        return engine.calculate_cash_conversion_cycle()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/finance/stress-test", tags=["Working Capital & Financial Stress Testing"])
def run_monte_carlo_stress_test_endpoint(payload: MonteCarloStressRequest) -> Dict[str, Any]:
    """Simulates 10,000 forward-looking macroeconomic stress scenarios, calculating VaR 95/99, CVaR, and solvency runway."""
    try:
        engine = get_financial_risk()
        return engine.run_monte_carlo_stress_test(
            n_simulations=payload.n_simulations,
            demand_shock_mean_pct=payload.demand_shock_mean_pct,
            demand_shock_std_pct=payload.demand_shock_std_pct,
            cogs_inflation_mean_pct=payload.cogs_inflation_mean_pct,
            cogs_inflation_std_pct=payload.cogs_inflation_std_pct,
            dpo_contraction_days=payload.dpo_contraction_days,
            initial_cash_buffer=payload.initial_cash_buffer,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/finance/working-capital/simulate", tags=["Working Capital & Financial Stress Testing"])
def simulate_working_capital_optimization_endpoint(payload: WorkingCapitalOptimizationRequest) -> Dict[str, Any]:
    """Simulates permanent liquidity unlocked and borrowing interest saved via DIO, DPO, and DSO rebalancing."""
    try:
        engine = get_financial_risk()
        return engine.simulate_working_capital_optimization(
            dio_reduction_days=payload.dio_reduction_days,
            dpo_extension_days=payload.dpo_extension_days,
            dso_reduction_days=payload.dso_reduction_days,
            short_term_borrowing_rate_pct=payload.short_term_borrowing_rate_pct,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/finance/liquidity-runway", tags=["Working Capital & Financial Stress Testing"])
def get_liquidity_runway_baseline_endpoint(
    initial_cash_buffer: float = Query(60000.0, gt=0.0, description="Starting cash buffer in dollars"),
) -> Dict[str, Any]:
    """Calculates baseline liquidity runway and risk profile under normal operational variance."""
    try:
        engine = get_financial_risk()
        return engine.run_monte_carlo_stress_test(
            n_simulations=1000,
            demand_shock_mean_pct=-0.05,
            demand_shock_std_pct=0.04,
            cogs_inflation_mean_pct=0.03,
            cogs_inflation_std_pct=0.02,
            dpo_contraction_days=0.0,
            initial_cash_buffer=initial_cash_buffer,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

