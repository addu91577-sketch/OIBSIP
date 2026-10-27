"""
Production FastAPI REST API for Retail Sales Analytics & Machine Learning Inference.

Exposes comprehensive endpoints for descriptive statistics, time-series velocity,
demographic segmentation, Pareto concentration profiling, real-time ML inference,
and strategic cross-department bundle recommendations.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.analytics import RetailAnalyticsEngine
from src.ml_models import RetailMLManager
from src.recommendations import BusinessRecommendationEngine

# Global singletons
analytics_engine: Optional[RetailAnalyticsEngine] = None
ml_manager: Optional[RetailMLManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes engines and loads/trains machine learning models on startup."""
    global analytics_engine, ml_manager
    analytics_engine = RetailAnalyticsEngine()
    ml_manager = RetailMLManager()
    ml_manager.load_or_train_models()
    yield


app = FastAPI(
    title="Retail Sales Analytics & Predictive Intelligence API",
    description=(
        "Production-ready enterprise REST API delivering real-time statistical analytics, "
        "time-series forecasting, demographic profiling, Pareto concentration intelligence, "
        "and machine learning inference for retail transactions."
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


# Endpoints
@app.get("/", tags=["System & Health"])
def root_endpoint() -> Dict[str, Any]:
    """Returns root API metadata, operational status, and available route modules."""
    analytics = get_analytics()
    kpis = analytics.get_kpi_overview()
    return {
        "service": "Retail Sales Analytics & Machine Learning Intelligence API",
        "version": "1.0.0",
        "status": "ONLINE",
        "documentation": "/docs",
        "summary": {
            "total_revenue": kpis["total_revenue"],
            "total_transactions": kpis["total_orders"],
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
