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

# Global singletons
analytics_engine: Optional[RetailAnalyticsEngine] = None
ml_manager: Optional[RetailMLManager] = None
drift_detector: Optional[DriftDetector] = None
segmentation_engine: Optional[CustomerSegmentationEngine] = None
whatif_simulator: Optional[FinancialWhatIfSimulator] = None
elasticity_optimizer: Optional[PriceElasticityOptimizer] = None
ab_engine: Optional[RetailABTestingEngine] = None
clv_engine: Optional[CustomerLifetimeValueEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes engines and loads/trains machine learning models on startup."""
    global analytics_engine, ml_manager, drift_detector, segmentation_engine, whatif_simulator
    global elasticity_optimizer, ab_engine, clv_engine
    analytics_engine = RetailAnalyticsEngine()
    ml_manager = RetailMLManager()
    ml_manager.load_or_train_models()
    drift_detector = DriftDetector(baseline_df=analytics_engine.df)
    segmentation_engine = CustomerSegmentationEngine()
    whatif_simulator = FinancialWhatIfSimulator()
    elasticity_optimizer = PriceElasticityOptimizer()
    ab_engine = RetailABTestingEngine()
    clv_engine = CustomerLifetimeValueEngine()
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
