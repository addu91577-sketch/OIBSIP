"""App Cohort Retention, Churn Risk & Lifetime Value (LTV) Forecasting Engine.

Day 13 Milestone for Google Play Store Market Intelligence & Decision Science Platform.
Provides:
1. Parametric User Retention Decay & Cohort Survival Modeling:
   - Power-law decay: R(t) = R_0 * (t + 1)^(-alpha)
   - Weibull survival model: S(t) = exp(-(t / lambda)^beta)
   - Benchmark milestone retention rates (D1, D7, D14, D30, D60, D90, D180, D360).
   - Instantaneous hazard rate and cumulative user-days (AUC).
2. Customer Lifetime Value (LTV) Forecasting & Unit Economics:
   - Category-calibrated daily ARPDAU (ad + IAP monetization) & upfront pricing yield.
   - Discounted cash flow capitalization: LTV = NetUpfront + sum( R(t) * ARPDAU / (1 + d/365)^t ).
   - Category CAC benchmarks, LTV/CAC ratio, payback period (days to breakeven), and economic tier.
3. Developer Maintenance Lifecycle & Abandonware Risk Audit:
   - App freshness and release recency (days elapsed since last update).
   - Stagnancy scoring, Android OS API deprecation risk, and lifecycle tier categorization:
     VIBRANT_ACTIVE, MATURE_SUSTAINED, STAGNANT_RISK, ZOMBIE_ABANDONWARE.
4. Composite App Churn Risk Diagnostic & Prescriptive Playbook:
   - 0-100 Churn Risk Score synthesizing Stagnancy, Sentiment, Rating, Volatility, and Binary Bloat.
   - Risk tiers: LOW_CHURN_DEFENSIVE, MODERATE_CHURN_STABLE, ELEVATED_CHURN_ATTENTION, CRITICAL_CHURN_INTERVENTION.
   - Prescriptive action playbook with targeted developer interventions.
5. Modern Portfolio Theory (Markowitz) Capital Allocation Optimizer:
   - Mean-variance quadratic optimization maximizing Sharpe ratio across multi-app portfolios.
   - Efficient frontier calculation, asset weights, portfolio volatility, and diversification analysis.
6. Publication Visualizations (Figures 26 & 27) and Analytical CSV Reports.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base project paths
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

_DATA_DIR = _BASE_DIR / "Data"
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


# ============================================================================
# Empirical Industry Benchmarks by Category
# ============================================================================

CATEGORY_RETENTION_DEFAULTS = {
    "GAME": {"d1": 0.32, "alpha": 0.52, "arpdau": 0.15, "cac": 2.10},
    "SOCIAL": {"d1": 0.42, "alpha": 0.28, "arpdau": 0.07, "cac": 1.05},
    "COMMUNICATION": {"d1": 0.44, "alpha": 0.26, "arpdau": 0.06, "cac": 0.95},
    "FINANCE": {"d1": 0.35, "alpha": 0.30, "arpdau": 0.18, "cac": 3.80},
    "BUSINESS": {"d1": 0.33, "alpha": 0.32, "arpdau": 0.14, "cac": 2.60},
    "PRODUCTIVITY": {"d1": 0.30, "alpha": 0.34, "arpdau": 0.05, "cac": 0.85},
    "TOOLS": {"d1": 0.26, "alpha": 0.38, "arpdau": 0.03, "cac": 0.60},
    "HEALTH_AND_FITNESS": {"d1": 0.31, "alpha": 0.42, "arpdau": 0.11, "cac": 1.95},
    "LIFESTYLE": {"d1": 0.28, "alpha": 0.40, "arpdau": 0.06, "cac": 1.10},
    "ENTERTAINMENT": {"d1": 0.34, "alpha": 0.44, "arpdau": 0.09, "cac": 1.30},
    "PHOTOGRAPHY": {"d1": 0.31, "alpha": 0.41, "arpdau": 0.07, "cac": 0.90},
    "EDUCATION": {"d1": 0.29, "alpha": 0.36, "arpdau": 0.08, "cac": 1.45},
    "BOOKS_AND_REFERENCE": {"d1": 0.28, "alpha": 0.37, "arpdau": 0.05, "cac": 0.75},
    "SHOPPING": {"d1": 0.38, "alpha": 0.34, "arpdau": 0.16, "cac": 2.40},
    "TRAVEL_AND_LOCAL": {"d1": 0.27, "alpha": 0.45, "arpdau": 0.08, "cac": 1.70},
    "DATING": {"d1": 0.36, "alpha": 0.48, "arpdau": 0.22, "cac": 2.80},
    "FAMILY": {"d1": 0.30, "alpha": 0.46, "arpdau": 0.08, "cac": 1.20},
    "MEDICAL": {"d1": 0.32, "alpha": 0.31, "arpdau": 0.15, "cac": 3.20},
    "DEFAULT": {"d1": 0.30, "alpha": 0.38, "arpdau": 0.07, "cac": 1.35}
}

DATASET_REFERENCE_DATE = "2018-08-08"


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class RetentionMilestones:
    """Milestone user retention rates over key lifecycle intervals."""
    day_1: float
    day_7: float
    day_14: float
    day_30: float
    day_60: float
    day_90: float
    day_180: float
    day_360: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RetentionCurveResult:
    """Parametric cohort retention trajectory and summary metrics."""
    target_name: str
    category: str
    model_type: str
    baseline_d1: float
    decay_parameter: float
    retention_auc: float
    milestones: RetentionMilestones
    daily_retention_sample: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["milestones"] = self.milestones.to_dict()
        return d


@dataclass
class LTVProjectionResult:
    """Comprehensive Lifetime Value forecasting, unit economics, and payback analysis."""
    app_name: str
    category: str
    price: float
    installs: int
    arpdau: float
    cac: float
    annual_discount_rate: float
    ltv_30d: float
    ltv_90d: float
    ltv_180d: float
    ltv_360d: float
    ltv_cac_ratio: float
    payback_period_days: Optional[int]
    economic_tier: str
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeveloperLifecycleResult:
    """App maintenance, release cadence, and abandonware risk assessment."""
    app_name: str
    category: str
    last_updated: str
    days_since_update: int
    lifecycle_tier: str
    abandonware_probability: float
    compatibility_risk: str
    maintenance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChurnRiskDiagnosticResult:
    """Holistic App Churn Risk diagnostic, sub-risk attribution, and prescriptive playbook."""
    app_name: str
    category: str
    rating: float
    installs: int
    churn_risk_score: float
    churn_risk_tier: str
    days_since_update: int
    sentiment_polarity: float
    stagnancy_risk: float
    sentiment_risk: float
    rating_risk: float
    volatility_risk: float
    size_risk: float
    primary_churn_driver: str
    prescriptive_playbook: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioAllocationResult:
    """Markowitz Modern Portfolio Theory capital allocation and risk-adjusted frontier."""
    portfolio_items: List[str]
    optimal_weights: Dict[str, float]
    expected_annual_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    diversification_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Core Retention, Churn & LTV Engine Class
# ============================================================================

class RetentionChurnLTVEngine:
    """Unified engine for cohort retention, churn prediction, LTV forecasting, and portfolio allocation."""

    def __init__(
        self,
        apps_csv_path: Optional[Union[str, Path]] = None,
        reviews_csv_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the RetentionChurnLTVEngine with datasets and precomputed lookups."""
        self.apps_path = Path(apps_csv_path) if apps_csv_path else _DATA_DIR / "cleaned_googleplaystore.csv"
        self.reviews_path = Path(reviews_csv_path) if reviews_csv_path else _DATA_DIR / "cleaned_user_reviews.csv"
        self.ref_date = pd.to_datetime(DATASET_REFERENCE_DATE)

        self._load_datasets()
        self._precompute_aggregates()
        logger.info(
            f"RetentionChurnLTVEngine initialized with {len(self.df_apps):,} apps and {len(self.df_reviews):,} reviews."
        )

    def _load_datasets(self) -> None:
        """Load and normalize apps and user review datasets."""
        if not self.apps_path.exists():
            raise FileNotFoundError(f"Cleaned apps dataset not found at {self.apps_path}")
        self.df_apps = pd.read_csv(self.apps_path)

        if not self.reviews_path.exists():
            raise FileNotFoundError(f"Cleaned reviews dataset not found at {self.reviews_path}")
        self.df_reviews = pd.read_csv(self.reviews_path)

        # Standardize and clean key columns
        self.df_apps["App_Clean"] = self.df_apps["App"].astype(str).str.strip()
        self.df_apps["Category_Clean"] = self.df_apps["Category"].astype(str).str.upper().str.strip()
        self.df_apps["Price_Num"] = pd.to_numeric(
            self.df_apps["Price"].astype(str).str.replace("$", "", regex=False),
            errors="coerce"
        ).fillna(0.0)
        self.df_apps["Installs_Num"] = pd.to_numeric(
            self.df_apps["Installs"].astype(str).str.replace("+", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0).astype(int)
        self.df_apps["Rating_Num"] = pd.to_numeric(self.df_apps["Rating"], errors="coerce")
        self.df_apps["Reviews_Num"] = pd.to_numeric(self.df_apps["Reviews"], errors="coerce").fillna(0).astype(int)
        self.df_apps["Parsed_Date"] = pd.to_datetime(self.df_apps["Last Updated"], errors="coerce").fillna(self.ref_date)
        self.df_apps["Days_Since_Update"] = (self.ref_date - self.df_apps["Parsed_Date"]).dt.days.clip(lower=0)

        # Build fast lookup dictionary for apps
        self._apps_lookup = {}
        for _, row in self.df_apps.iterrows():
            key = row["App_Clean"].lower()
            if key not in self._apps_lookup:
                self._apps_lookup[key] = row

        # Aggregate sentiment metrics per app
        rev_grouped = self.df_reviews.groupby(self.df_reviews["App"].astype(str).str.strip().str.lower()).agg(
            mean_polarity=("Sentiment_Polarity", "mean"),
            std_polarity=("Sentiment_Polarity", "std"),
            negative_review_ratio=("Sentiment_Polarity", lambda s: float((s < -0.05).mean())),
            total_reviews=("Sentiment_Polarity", "count")
        ).reset_index()
        self._reviews_sentiment_lookup = rev_grouped.set_index("App").to_dict(orient="index")

    def _precompute_aggregates(self) -> None:
        """Precompute category-level baselines for ratings, update recency, and sentiment."""
        self.category_baselines = {}
        for cat, group in self.df_apps.groupby("Category_Clean"):
            self.category_baselines[cat] = {
                "mean_rating": float(group["Rating_Num"].mean()) if not group["Rating_Num"].dropna().empty else 4.15,
                "median_installs": float(group["Installs_Num"].median()),
                "mean_days_update": float(group["Days_Since_Update"].mean()),
                "app_count": int(len(group))
            }

    # ------------------------------------------------------------------------
    # 1. Parametric Retention Modeling
    # ------------------------------------------------------------------------

    def estimate_retention_curve(
        self,
        app_or_category: str,
        is_category: bool = False,
        model_type: str = "power_law",
        custom_rating: Optional[float] = None,
        custom_polarity: Optional[float] = None,
        days: int = 360
    ) -> RetentionCurveResult:
        """Estimate day-by-day retention trajectory and milestone retention rates.

        Parameters
        ----------
        app_or_category : str
            Target application title or Google Play category name.
        is_category : bool
            True if querying category level; False for specific app lookup.
        model_type : str
            'power_law' (R0 * (t+1)^-alpha) or 'weibull' (exp(-(t/lambda)^beta)).
        custom_rating : Optional[float]
            Optional rating override for hypothetical simulation.
        custom_polarity : Optional[float]
            Optional sentiment polarity override (-1.0 to 1.0).
        days : int
            Total horizon to project (typically 360 days).
        """
        category = "DEFAULT"
        rating = 4.17
        polarity = 0.20

        if is_category:
            category = app_or_category.upper().strip()
            cat_info = self.category_baselines.get(category, {})
            rating = custom_rating if custom_rating is not None else cat_info.get("mean_rating", 4.17)
            polarity = custom_polarity if custom_polarity is not None else 0.20
        else:
            app_clean = app_or_category.strip()
            app_row = self._apps_lookup.get(app_clean.lower())
            if app_row is not None:
                category = app_row["Category_Clean"]
                rating = custom_rating if custom_rating is not None else (
                    app_row["Rating_Num"] if pd.notna(app_row["Rating_Num"]) else 4.17
                )
                rev_stat = self._reviews_sentiment_lookup.get(app_clean.lower(), {})
                polarity = custom_polarity if custom_polarity is not None else rev_stat.get("mean_polarity", 0.20)
            else:
                category = "DEFAULT"
                rating = custom_rating if custom_rating is not None else 4.17
                polarity = custom_polarity if custom_polarity is not None else 0.20

        # Retrieve category benchmark parameters
        bench = CATEGORY_RETENTION_DEFAULTS.get(category, CATEGORY_RETENTION_DEFAULTS["DEFAULT"])
        base_d1 = bench["d1"]
        base_alpha = bench["alpha"]

        # Quality modifier based on Rating and Review Polarity
        # Higher rating & positive sentiment boost Day 1 retention and flatten the decay
        rating_delta = (rating - 4.15) / 1.5  # Roughly -0.5 to +0.5
        polarity_delta = np.clip(polarity, -1.0, 1.0) * 0.15

        calibrated_d1 = float(np.clip(base_d1 + (rating_delta * 0.08) + polarity_delta, 0.10, 0.65))
        calibrated_alpha = float(np.clip(base_alpha - (rating_delta * 0.06) - (polarity_delta * 0.05), 0.18, 0.70))

        # Generate trajectory from t=1 to days
        t_arr = np.arange(1, days + 1, dtype=float)

        if model_type.lower() == "weibull":
            # Calibrate Weibull: S(t) = exp(-(t / lambda)^beta)
            # Find lambda such that S(1) = calibrated_d1
            beta = float(np.clip(0.40 - (rating_delta * 0.05), 0.25, 0.60))
            lambda_param = float(1.0 / ((-np.log(calibrated_d1)) ** (1.0 / beta))) if calibrated_d1 < 1.0 else 100.0
            daily_curve = np.exp(-((t_arr / lambda_param) ** beta))
            decay_param = beta
        else:
            # Power law: R(t) = calibrated_d1 * t^(-calibrated_alpha)
            # Notice at t=1, R(1) = calibrated_d1 * 1 = calibrated_d1
            daily_curve = calibrated_d1 * (t_arr ** (-calibrated_alpha))
            decay_param = calibrated_alpha

        daily_curve = np.clip(daily_curve, 0.001, 1.0)

        # Extract standard milestone markers
        def _get_val(day_idx: int) -> float:
            idx = min(day_idx - 1, len(daily_curve) - 1)
            return float(round(daily_curve[idx], 4))

        milestones = RetentionMilestones(
            day_1=_get_val(1),
            day_7=_get_val(7),
            day_14=_get_val(14),
            day_30=_get_val(30),
            day_60=_get_val(60),
            day_90=_get_val(90),
            day_180=_get_val(180),
            day_360=_get_val(360)
        )

        auc = float(round(np.sum(daily_curve), 2))
        sample_indices = [0, 6, 13, 29, 59, 89, 179, min(359, days - 1)]
        sample_points = [float(round(daily_curve[i], 4)) for i in sample_indices if i < len(daily_curve)]

        return RetentionCurveResult(
            target_name=app_or_category,
            category=category,
            model_type=model_type,
            baseline_d1=round(calibrated_d1, 4),
            decay_parameter=round(decay_param, 4),
            retention_auc=auc,
            milestones=milestones,
            daily_retention_sample=sample_points
        )

    # ------------------------------------------------------------------------
    # 2. Lifetime Value (LTV) Forecasting & Unit Economics
    # ------------------------------------------------------------------------

    def forecast_ltv(
        self,
        app_name: str,
        custom_price: Optional[float] = None,
        custom_arpdau: Optional[float] = None,
        custom_cac: Optional[float] = None,
        annual_discount_rate: float = 0.10,
        days: int = 360
    ) -> LTVProjectionResult:
        """Forecast dynamic LTV, payback duration, and economic viability.

        Parameters
        ----------
        app_name : str
            Application title.
        custom_price : Optional[float]
            Optional price override ($).
        custom_arpdau : Optional[float]
            Optional Average Revenue Per Daily Active User ($).
        custom_cac : Optional[float]
            Optional Customer Acquisition Cost ($).
        annual_discount_rate : float
            Annualized discount rate (default 10%).
        days : int
            LTV projection horizon (default 360 days).
        """
        app_clean = app_name.strip()
        app_row = self._apps_lookup.get(app_clean.lower())

        if app_row is not None:
            category = app_row["Category_Clean"]
            price = custom_price if custom_price is not None else float(app_row["Price_Num"])
            installs = int(app_row["Installs_Num"])
            rating = float(app_row["Rating_Num"]) if pd.notna(app_row["Rating_Num"]) else 4.15
        else:
            category = "DEFAULT"
            price = custom_price if custom_price is not None else 0.0
            installs = 100000
            rating = 4.15

        bench = CATEGORY_RETENTION_DEFAULTS.get(category, CATEGORY_RETENTION_DEFAULTS["DEFAULT"])
        arpdau = custom_arpdau if custom_arpdau is not None else bench["arpdau"]
        cac = custom_cac if custom_cac is not None else bench["cac"]

        # Retention curve for daily active user weighting
        ret_res = self.estimate_retention_curve(app_or_category=app_name, days=days)
        t_arr = np.arange(1, days + 1, dtype=float)
        # Power-law daily retention
        daily_ret = ret_res.baseline_d1 * (t_arr ** (-ret_res.decay_parameter))

        # Net upfront revenue per acquired user (price minus 5% refunds/chargebacks)
        net_upfront = price * 0.95

        # Discount factor per day: (1 + d/365)^t
        daily_discount = (1.0 + annual_discount_rate / 365.0) ** t_arr
        daily_discounted_ad_iap = (daily_ret * arpdau) / daily_discount

        # Cumulative cash flow array
        cumulative_ltv = net_upfront + np.cumsum(daily_discounted_ad_iap)

        ltv_30d = float(round(cumulative_ltv[min(29, days - 1)], 2))
        ltv_90d = float(round(cumulative_ltv[min(89, days - 1)], 2))
        ltv_180d = float(round(cumulative_ltv[min(179, days - 1)], 2))
        ltv_360d = float(round(cumulative_ltv[min(359, days - 1)], 2))

        # Payback period determination (first day where cumulative LTV >= CAC)
        payback_days: Optional[int] = None
        if cac <= 0:
            payback_days = 0
        else:
            breakeven_indices = np.where(cumulative_ltv >= cac)[0]
            if len(breakeven_indices) > 0:
                payback_days = int(breakeven_indices[0] + 1)

        ltv_cac_ratio = float(round(ltv_360d / max(cac, 0.01), 2))

        # Economic tier categorization
        if ltv_cac_ratio >= 3.0:
            economic_tier = "VIABLE_EXPANSION"
            verdict = "Exceptional unit economics; aggressively scale paid acquisition campaigns."
        elif ltv_cac_ratio >= 1.5:
            economic_tier = "SUSTAINABLE"
            verdict = "Healthy margin profile; optimize ad monetization to accelerate payback."
        elif ltv_cac_ratio >= 1.0:
            economic_tier = "BREAKEVEN"
            verdict = "Marginal unit economics; high vulnerability to CAC inflation or churn."
        else:
            economic_tier = "UNSUSTAINABLE"
            verdict = "Negative ROI; CAC exceeds 360-day LTV. Urgent retention overhaul required."

        return LTVProjectionResult(
            app_name=app_name,
            category=category,
            price=round(price, 2),
            installs=installs,
            arpdau=round(arpdau, 4),
            cac=round(cac, 2),
            annual_discount_rate=round(annual_discount_rate, 4),
            ltv_30d=ltv_30d,
            ltv_90d=ltv_90d,
            ltv_180d=ltv_180d,
            ltv_360d=ltv_360d,
            ltv_cac_ratio=ltv_cac_ratio,
            payback_period_days=payback_days,
            economic_tier=economic_tier,
            verdict=verdict
        )

    # ------------------------------------------------------------------------
    # 3. Developer Maintenance Lifecycle & Abandonware Audit
    # ------------------------------------------------------------------------

    def audit_developer_lifecycle(self, app_name: Optional[str] = None) -> Union[DeveloperLifecycleResult, pd.DataFrame]:
        """Audit app update freshness, abandonware probability, and maintenance health.

        If app_name is provided, returns DeveloperLifecycleResult for that app.
        If app_name is None, returns summary DataFrame across all apps.
        """
        if app_name is not None:
            app_clean = app_name.strip()
            app_row = self._apps_lookup.get(app_clean.lower())
            if app_row is None:
                # Fallback for synthetic / unseen app
                return DeveloperLifecycleResult(
                    app_name=app_name,
                    category="DEFAULT",
                    last_updated=DATASET_REFERENCE_DATE,
                    days_since_update=0,
                    lifecycle_tier="VIBRANT_ACTIVE",
                    abandonware_probability=0.05,
                    compatibility_risk="LOW",
                    maintenance_score=95.0
                )

            days = int(app_row["Days_Since_Update"])
            last_up = str(app_row["Last Updated"])
            category = str(app_row["Category_Clean"])

            # Classify lifecycle tier based on update recency
            if days <= 90:
                tier = "VIBRANT_ACTIVE"
                p_abandon = round(min(days / 900.0, 0.15), 3)
                compat_risk = "LOW"
                score = round(max(100.0 - (days * 0.2), 80.0), 1)
            elif days <= 365:
                tier = "MATURE_SUSTAINED"
                p_abandon = round(0.15 + ((days - 90) / 275.0) * 0.35, 3)
                compat_risk = "MEDIUM"
                score = round(max(80.0 - ((days - 90) * 0.1), 50.0), 1)
            elif days <= 730:
                tier = "STAGNANT_RISK"
                p_abandon = round(0.50 + ((days - 365) / 365.0) * 0.35, 3)
                compat_risk = "HIGH"
                score = round(max(50.0 - ((days - 365) * 0.08), 20.0), 1)
            else:
                tier = "ZOMBIE_ABANDONWARE"
                p_abandon = round(min(0.85 + ((days - 730) / 1000.0) * 0.14, 0.99), 3)
                compat_risk = "CRITICAL"
                score = round(max(20.0 - ((days - 730) * 0.01), 0.0), 1)

            return DeveloperLifecycleResult(
                app_name=app_name,
                category=category,
                last_updated=last_up,
                days_since_update=days,
                lifecycle_tier=tier,
                abandonware_probability=p_abandon,
                compatibility_risk=compat_risk,
                maintenance_score=score
            )

        # Catalog-wide summary dataframe
        df_summary = self.df_apps[["App_Clean", "Category_Clean", "Last Updated", "Days_Since_Update"]].copy()
        df_summary.columns = ["App", "Category", "Last_Updated", "Days_Since_Update"]

        conditions = [
            df_summary["Days_Since_Update"] <= 90,
            (df_summary["Days_Since_Update"] > 90) & (df_summary["Days_Since_Update"] <= 365),
            (df_summary["Days_Since_Update"] > 365) & (df_summary["Days_Since_Update"] <= 730),
            df_summary["Days_Since_Update"] > 730
        ]
        choices = ["VIBRANT_ACTIVE", "MATURE_SUSTAINED", "STAGNANT_RISK", "ZOMBIE_ABANDONWARE"]
        df_summary["Lifecycle_Tier"] = np.select(conditions, choices, default="ZOMBIE_ABANDONWARE")
        return df_summary

    # ------------------------------------------------------------------------
    # 4. Composite Churn Risk Scoring & Prescriptive Playbook
    # ------------------------------------------------------------------------

    def evaluate_churn_risk(self, app_name: str) -> ChurnRiskDiagnosticResult:
        """Synthesize multi-dimensional churn risks into a 0-100 diagnostic index.

        Dimensions:
        - Stagnancy Risk (30%): Days since last developer update
        - Sentiment Risk (25%): Review polarity & negative comment density
        - Rating Degradation Risk (20%): App rating vs category benchmark
        - Volatility Risk (15%): Review-to-install engagement velocity
        - Size Bloat Risk (10%): App binary weight vs category tolerance
        """
        app_clean = app_name.strip()
        app_row = self._apps_lookup.get(app_clean.lower())

        if app_row is not None:
            category = app_row["Category_Clean"]
            rating = float(app_row["Rating_Num"]) if pd.notna(app_row["Rating_Num"]) else 4.15
            installs = int(app_row["Installs_Num"])
            reviews = int(app_row["Reviews_Num"])
            days = int(app_row["Days_Since_Update"])
            size_mb = float(app_row["Size_MB_Imputed"]) if "Size_MB_Imputed" in app_row and pd.notna(app_row["Size_MB_Imputed"]) else 25.0
        else:
            category = "DEFAULT"
            rating = 4.15
            installs = 50000
            reviews = 500
            days = 120
            size_mb = 25.0

        # Review sentiment stats
        rev_stats = self._reviews_sentiment_lookup.get(app_clean.lower(), {})
        mean_polarity = rev_stats.get("mean_polarity", 0.20)
        neg_ratio = rev_stats.get("negative_review_ratio", 0.15)

        # 1. Stagnancy Risk (0 - 100)
        # 0-30 days: 5-15, 31-90 days: 16-30, 91-365 days: 31-65, >365 days: 66-100
        stagnancy_risk = float(np.clip((days / 365.0) * 65.0 + 10.0, 5.0, 100.0))

        # 2. Sentiment Risk (0 - 100)
        # Polarity spans -1 to +1. Invert: negative polarity -> high risk
        sentiment_risk = float(np.clip((1.0 - mean_polarity) * 40.0 + (neg_ratio * 40.0), 5.0, 100.0))

        # 3. Rating Risk (0 - 100)
        # Category baseline comparison
        cat_rating = self.category_baselines.get(category, {}).get("mean_rating", 4.17)
        rating_gap = cat_rating - rating  # positive means worse than category
        rating_risk = float(np.clip(35.0 + (rating_gap * 70.0), 5.0, 100.0))

        # 4. Volatility Risk (0 - 100)
        # Low engagement or high review standard deviation
        engagement_ratio = reviews / max(installs, 100)
        volatility_risk = float(np.clip(60.0 - (engagement_ratio * 400.0), 10.0, 90.0))

        # 5. Size Bloat Risk (0 - 100)
        # Large size in non-game categories
        if category in ["GAME", "FAMILY"]:
            size_risk = float(np.clip((size_mb / 100.0) * 35.0, 5.0, 60.0))
        else:
            size_risk = float(np.clip((size_mb / 60.0) * 60.0, 10.0, 95.0))

        # Composite Churn Risk Score (Weighted blend)
        churn_risk_score = float(round(
            0.30 * stagnancy_risk +
            0.25 * sentiment_risk +
            0.20 * rating_risk +
            0.15 * volatility_risk +
            0.10 * size_risk,
            1
        ))

        # Classify Churn Risk Tier
        if churn_risk_score <= 25.0:
            churn_risk_tier = "LOW_CHURN_DEFENSIVE"
        elif churn_risk_score <= 50.0:
            churn_risk_tier = "MODERATE_CHURN_STABLE"
        elif churn_risk_score <= 75.0:
            churn_risk_tier = "ELEVATED_CHURN_ATTENTION"
        else:
            churn_risk_tier = "CRITICAL_CHURN_INTERVENTION"

        # Determine Primary Risk Driver
        risk_components = {
            "Release Stagnancy (Outdated Build)": stagnancy_risk,
            "Negative Review Sentiment Spikes": sentiment_risk,
            "Sub-Category Rating Degradation": rating_risk,
            "Low Engagement Volatility": volatility_risk,
            "Binary Size / APK Bloat": size_risk
        }
        primary_driver = max(risk_components, key=risk_components.get)

        # Generate Actionable Prescriptive Playbook
        playbook = []
        if stagnancy_risk >= 50.0:
            playbook.append(
                f"Release an immediate maintenance build addressing Android SDK compatibility (dormant for {days} days)."
            )
        if sentiment_risk >= 50.0:
            playbook.append(
                "Trigger automated triage on recent 1-star reviews to isolate crash clusters and billing friction."
            )
        if rating_risk >= 50.0:
            playbook.append(
                f"Rating ({rating:.1f}) trails category benchmark ({cat_rating:.1f}); introduce in-app NPS prompt after successful user tasks."
            )
        if size_risk >= 50.0:
            playbook.append(
                f"Binary size ({size_mb:.1f} MB) introduces download attrition; migrate static assets to dynamic Android App Bundles."
            )
        if len(playbook) < 3:
            playbook.append(
                "Implement personalized Day 3 and Day 7 re-engagement push notifications with relevant feature prompts."
            )
            playbook.append(
                "Deploy A/B tested onboarding tutorial to accelerate Day 1 time-to-value."
            )

        return ChurnRiskDiagnosticResult(
            app_name=app_name,
            category=category,
            rating=round(rating, 2),
            installs=installs,
            churn_risk_score=churn_risk_score,
            churn_risk_tier=churn_risk_tier,
            days_since_update=days,
            sentiment_polarity=round(mean_polarity, 3),
            stagnancy_risk=round(stagnancy_risk, 1),
            sentiment_risk=round(sentiment_risk, 1),
            rating_risk=round(rating_risk, 1),
            volatility_risk=round(volatility_risk, 1),
            size_risk=round(size_risk, 1),
            primary_churn_driver=primary_driver,
            prescriptive_playbook=playbook[:4]
        )

    # ------------------------------------------------------------------------
    # 5. Markowitz Portfolio Capital Allocation Optimizer
    # ------------------------------------------------------------------------

    def optimize_app_portfolio(
        self,
        portfolio_items: List[str],
        risk_free_rate: float = 0.03
    ) -> PortfolioAllocationResult:
        """Compute optimal Markowitz asset allocation weights maximizing Sharpe ratio.

        Parameters
        ----------
        portfolio_items : List[str]
            List of apps or categories to include in the publisher portfolio.
        risk_free_rate : float
            Baseline risk-free benchmark return (default 3%).
        """
        if not portfolio_items:
            portfolio_items = ["GAME", "FINANCE", "TOOLS", "HEALTH_AND_FITNESS", "EDUCATION"]

        cleaned_items = [item.strip() for item in portfolio_items]
        n = len(cleaned_items)

        if n == 1:
            return PortfolioAllocationResult(
                portfolio_items=cleaned_items,
                optimal_weights={cleaned_items[0]: 1.0},
                expected_annual_return=0.18,
                portfolio_volatility=0.12,
                sharpe_ratio=round((0.18 - risk_free_rate) / 0.12, 2),
                diversification_verdict="Concentrated single-asset allocation; lack of category hedging."
            )

        # Estimate expected return and volatility for each asset
        exp_returns = []
        volatilities = []

        for item in cleaned_items:
            # Check if category or app
            item_upper = item.upper()
            if item_upper in self.category_baselines or item_upper in CATEGORY_RETENTION_DEFAULTS:
                bench = CATEGORY_RETENTION_DEFAULTS.get(item_upper, CATEGORY_RETENTION_DEFAULTS["DEFAULT"])
                # Annualized expected return proportional to ARPDAU * 365 / CAC
                annual_yield = (bench["arpdau"] * 365.0) / max(bench["cac"] * 10.0, 1.0)
                exp_returns.append(float(np.clip(annual_yield * 0.12, 0.06, 0.35)))
                volatilities.append(float(np.clip(bench["alpha"] * 0.45, 0.10, 0.30)))
            else:
                # App specific
                ltv_res = self.forecast_ltv(item)
                ratio = ltv_res.ltv_cac_ratio
                exp_returns.append(float(np.clip(ratio * 0.07, 0.05, 0.38)))
                volatilities.append(0.18)

        mu = np.array(exp_returns)
        vols = np.array(volatilities)

        # Synthesize correlation matrix with realistic inter-category cross-correlations
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr_matrix[i, j] = 0.25  # moderate positive correlation across tech/mobile
                corr_matrix[j, i] = 0.25

        # Covariance matrix: Sigma = diag(vols) * Corr * diag(vols)
        sigma = np.outer(vols, vols) * corr_matrix

        # Quadratic optimization for Sharpe ratio
        def _neg_sharpe(weights: np.ndarray) -> float:
            port_ret = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_vol <= 1e-6:
                return 1e6
            return -(port_ret - risk_free_rate) / port_vol

        # Constraints: sum(w) = 1, w_i in [0.03, 0.70] for healthy diversification
        bounds = [(0.03, 0.70) for _ in range(n)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        initial_weights = np.ones(n) / n

        opt_res = minimize(
            _neg_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if opt_res.success:
            optimal_w = opt_res.x
        else:
            optimal_w = initial_weights

        optimal_w = np.clip(optimal_w, 0.0, 1.0)
        optimal_w = optimal_w / np.sum(optimal_w)

        port_ret = float(round(np.dot(optimal_w, mu), 4))
        port_vol = float(round(np.sqrt(np.dot(optimal_w.T, np.dot(sigma, optimal_w))), 4))
        sharpe = float(round((port_ret - risk_free_rate) / max(port_vol, 1e-4), 2))

        weights_dict = {
            cleaned_items[i]: float(round(optimal_w[i], 3))
            for i in range(n)
        }

        # Diversity evaluation
        herfindahl = float(np.sum(optimal_w ** 2))
        if herfindahl < 0.25:
            verdict = "Highly diversified portfolio across complementary monetization curves."
        elif herfindahl < 0.45:
            verdict = "Balanced growth-and-cash-flow portfolio with balanced risk mitigation."
        else:
            verdict = "Concentrated portfolio; vulnerable to sector-specific policy shifts or CAC spikes."

        return PortfolioAllocationResult(
            portfolio_items=cleaned_items,
            optimal_weights=weights_dict,
            expected_annual_return=port_ret,
            portfolio_volatility=port_vol,
            sharpe_ratio=sharpe,
            diversification_verdict=verdict
        )

    # ------------------------------------------------------------------------
    # 6. Report Generation
    # ------------------------------------------------------------------------

    def generate_all_reports(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Generate publication-ready CSV analytical reports for Day 13."""
        out_dir = Path(output_dir) if output_dir else _REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        created_paths = {}

        # 1. Category Retention Benchmarks
        cat_ret_rows = []
        for cat in sorted(self.category_baselines.keys()):
            res = self.estimate_retention_curve(app_or_category=cat, is_category=True)
            bench = CATEGORY_RETENTION_DEFAULTS.get(cat, CATEGORY_RETENTION_DEFAULTS["DEFAULT"])
            cat_ret_rows.append({
                "Category": cat,
                "Baseline_D1_Retention": res.milestones.day_1,
                "Day_7_Retention": res.milestones.day_7,
                "Day_14_Retention": res.milestones.day_14,
                "Day_30_Retention": res.milestones.day_30,
                "Day_90_Retention": res.milestones.day_90,
                "Day_360_Retention": res.milestones.day_360,
                "Retention_Decay_Alpha": res.decay_parameter,
                "Cumulative_User_Days_AUC": res.retention_auc,
                "Benchmark_ARPDAU": bench["arpdau"],
                "Benchmark_CAC": bench["cac"]
            })
        df_cat_ret = pd.DataFrame(cat_ret_rows)
        cat_ret_path = out_dir / "category_retention_benchmarks.csv"
        df_cat_ret.to_csv(cat_ret_path, index=False)
        created_paths["category_retention_benchmarks"] = cat_ret_path

        # 2. Sample LTV Projections
        representative_apps = [
            "Duolingo: Learn Languages Free",
            "Subway Surfers",
            "Clash of Clans",
            "Candy Crush Saga",
            "WhatsApp Messenger",
            "Instagram",
            "Facebook",
            "Roblox",
            "Uber Driver",
            "Dropbox",
            "Colorfy: Colouring Book for Adults",
            "Photo Editor & Candy Camera & Grid & ScrapBook",
            "Calorie Counter - MyFitnessPal",
            "Headspace: Meditation & Mindfulness",
            "Minecraft",
            "Grand Theft Auto: San Andreas",
            "Facetune - Ad Free",
            "Bloons TD 5"
        ]
        ltv_rows = []
        for app_name in representative_apps:
            if app_name.lower() in self._apps_lookup:
                res = self.forecast_ltv(app_name)
                ltv_rows.append({
                    "App": res.app_name,
                    "Category": res.category,
                    "Price": res.price,
                    "Installs": res.installs,
                    "ARPDAU": res.arpdau,
                    "Estimated_CAC": res.cac,
                    "LTV_30D": res.ltv_30d,
                    "LTV_90D": res.ltv_90d,
                    "LTV_180D": res.ltv_180d,
                    "LTV_360D": res.ltv_360d,
                    "LTV_CAC_Ratio": res.ltv_cac_ratio,
                    "Payback_Days": res.payback_period_days if res.payback_period_days is not None else -1,
                    "Economic_Tier": res.economic_tier
                })
        df_ltv = pd.DataFrame(ltv_rows)
        ltv_path = out_dir / "sample_ltv_projections.csv"
        df_ltv.to_csv(ltv_path, index=False)
        created_paths["sample_ltv_projections"] = ltv_path

        # 3. App Churn Risk Scorecard
        churn_rows = []
        for app_name in representative_apps:
            if app_name.lower() in self._apps_lookup:
                res = self.evaluate_churn_risk(app_name)
                churn_rows.append({
                    "App": res.app_name,
                    "Category": res.category,
                    "Rating": res.rating,
                    "Installs": res.installs,
                    "Days_Since_Update": res.days_since_update,
                    "Churn_Risk_Score": res.churn_risk_score,
                    "Churn_Risk_Tier": res.churn_risk_tier,
                    "Primary_Churn_Driver": res.primary_churn_driver,
                    "Stagnancy_Risk": res.stagnancy_risk,
                    "Sentiment_Risk": res.sentiment_risk,
                    "Rating_Risk": res.rating_risk
                })
        df_churn = pd.DataFrame(churn_rows)
        churn_path = out_dir / "app_churn_risk_scorecard.csv"
        df_churn.to_csv(churn_path, index=False)
        created_paths["app_churn_risk_scorecard"] = churn_path

        # 4. Developer Lifecycle Summary
        df_life = self.audit_developer_lifecycle()
        tier_counts = df_life["Lifecycle_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Lifecycle_Tier", "App_Count"]
        tier_counts["Percentage"] = (tier_counts["App_Count"] / len(df_life) * 100.0).round(2)
        life_path = out_dir / "developer_lifecycle_summary.csv"
        tier_counts.to_csv(life_path, index=False)
        created_paths["developer_lifecycle_summary"] = life_path

        logger.info(f"Day 13 analytical reports successfully generated in {out_dir}")
        return created_paths

    # ------------------------------------------------------------------------
    # 7. Publication Visualizations (Figures 26 & 27)
    # ------------------------------------------------------------------------

    def generate_publication_figures(self, output_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Generate high-resolution Figure 26 & Figure 27 publication plots."""
        out_dir = Path(output_dir) if output_dir else _FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        fig26_path = out_dir / "26_retention_decay_and_cohort_survival.png"
        fig27_path = out_dir / "27_churn_risk_and_developer_lifecycle.png"

        self._plot_figure_26(fig26_path)
        self._plot_figure_27(fig27_path)

        return fig26_path, fig27_path

    def _plot_figure_26(self, output_path: Path) -> None:
        """Render Figure 26: Parametric Retention Curves, Milestones, LTV Trajectory & Hazard Rates."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)

        # Panel A: Retention Decay Curves across Diverse Categories
        target_cats = ["GAME", "SOCIAL", "FINANCE", "TOOLS", "HEALTH_AND_FITNESS"]
        colors = ["#e15759", "#4e79a7", "#59a14f", "#76b7b2", "#edc948"]
        t_span = np.arange(1, 91)

        for cat, col in zip(target_cats, colors):
            res = self.estimate_retention_curve(cat, is_category=True, days=90)
            ret_vals = [res.baseline_d1 * (t ** (-res.decay_parameter)) * 100.0 for t in t_span]
            axes[0, 0].plot(t_span, ret_vals, label=f"{cat} (D1: {res.milestones.day_1*100:.1f}%)", color=col, linewidth=2.5)

        axes[0, 0].set_xlabel("Days Since App Installation (t)", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Cohort Retention Rate (%)", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("(A) Parametric Power-Law Retention Decay Curves (D1–D90)", fontsize=13, fontweight="bold", pad=12)
        axes[0, 0].legend(loc="upper right", fontsize=10)
        axes[0, 0].set_xlim(1, 90)

        # Panel B: Milestone Retention Comparison (D1, D7, D14, D30, D90)
        milestones = ["D1", "D7", "D14", "D30", "D90"]
        x_indices = np.arange(len(milestones))
        bar_width = 0.16

        for i, (cat, col) in enumerate(zip(target_cats, colors)):
            res = self.estimate_retention_curve(cat, is_category=True)
            vals = [
                res.milestones.day_1 * 100,
                res.milestones.day_7 * 100,
                res.milestones.day_14 * 100,
                res.milestones.day_30 * 100,
                res.milestones.day_90 * 100
            ]
            axes[0, 1].bar(
                x_indices + (i - 2) * bar_width,
                vals,
                width=bar_width,
                label=cat,
                color=col,
                edgecolor="black",
                alpha=0.85
            )

        axes[0, 1].set_xticks(x_indices)
        axes[0, 1].set_xticklabels(milestones, fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Retention Benchmark (%)", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("(B) Cross-Category Cohort Milestone Benchmarks", fontsize=13, fontweight="bold", pad=12)
        axes[0, 1].legend(loc="upper right", fontsize=9)

        # Panel C: Cumulative LTV Trajectory vs. CAC Breakeven
        t_ltv = np.arange(1, 361)
        sample_apps = ["Subway Surfers", "Duolingo: Learn Languages Free", "Dropbox"]
        app_colors = ["#e15759", "#59a14f", "#4e79a7"]

        for app_name, col in zip(sample_apps, app_colors):
            ltv_res = self.forecast_ltv(app_name, days=360)
            ret_curve = self.estimate_retention_curve(app_name, days=360)
            daily_rev = np.array([ret_curve.baseline_d1 * (t ** (-ret_curve.decay_parameter)) * ltv_res.arpdau for t in t_ltv])
            cum_ltv = ltv_res.price * 0.95 + np.cumsum(daily_rev / ((1.0 + 0.10/365.0) ** t_ltv))

            axes[1, 0].plot(t_ltv, cum_ltv, label=f"{app_name[:20]} (LTV: ${ltv_res.ltv_360d:.2f})", color=col, linewidth=2.5)
            # Annotate CAC breakeven if reached
            if ltv_res.payback_period_days is not None and ltv_res.payback_period_days <= 360:
                axes[1, 0].scatter(
                    [ltv_res.payback_period_days],
                    [ltv_res.cac],
                    color=col,
                    s=80,
                    zorder=5,
                    edgecolors="black"
                )

        axes[1, 0].axhline(2.10, color="gray", linestyle="--", alpha=0.7, label="Benchmark CAC ($2.10)")
        axes[1, 0].set_xlabel("Days Since Acquisition (t)", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Discounted Cumulative LTV ($)", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("(C) 360-Day Discounted LTV Trajectory & CAC Breakeven", fontsize=13, fontweight="bold", pad=12)
        axes[1, 0].legend(loc="lower right", fontsize=9)
        axes[1, 0].set_xlim(1, 360)

        # Panel D: Instantaneous Churn Hazard Rate h(t)
        # h(t) = (R(t-1) - R(t)) / R(t-1)
        t_haz = np.arange(2, 61)
        for cat, col in zip(target_cats[:3], colors[:3]):
            res = self.estimate_retention_curve(cat, is_category=True, days=60)
            r_prev = np.array([res.baseline_d1 * ((t-1) ** (-res.decay_parameter)) for t in t_haz])
            r_curr = np.array([res.baseline_d1 * (t ** (-res.decay_parameter)) for t in t_haz])
            haz = ((r_prev - r_curr) / r_prev) * 100.0
            axes[1, 1].plot(t_haz, haz, label=f"{cat} Hazard", color=col, linewidth=2.5)

        axes[1, 1].set_xlabel("Lifecycle Day (t)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Daily Instantaneous Churn Hazard (%)", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("(D) User Churn Hazard Rate (Infant Mortality Effect)", fontsize=13, fontweight="bold", pad=12)
        axes[1, 1].legend(loc="upper right", fontsize=10)
        axes[1, 1].set_xlim(2, 60)

        plt.suptitle(
            "Google Play Store: Cohort Retention Decay, Survival Milestones & LTV Trajectory",
            fontsize=16,
            fontweight="bold",
            y=0.99
        )
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure 26 saved to {output_path}")

    def _plot_figure_27(self, output_path: Path) -> None:
        """Render Figure 27: Developer Update Recency, Abandonware Quadrants & Markowitz Frontier."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)

        # Panel A: Developer Stagnancy vs. Rating Quadrant
        sample_apps = self.df_apps.dropna(subset=["Rating_Num"]).sample(
            n=min(1200, len(self.df_apps)),
            random_state=42
        )
        scatter = axes[0, 0].scatter(
            sample_apps["Days_Since_Update"],
            sample_apps["Rating_Num"],
            c=sample_apps["Days_Since_Update"],
            cmap="plasma_r",
            alpha=0.55,
            edgecolors="none",
            s=30
        )
        axes[0, 0].axvline(90, color="green", linestyle="--", alpha=0.7, label="Active Cadence (90d)")
        axes[0, 0].axvline(365, color="orange", linestyle="--", alpha=0.7, label="Stagnancy (365d)")
        axes[0, 0].axvline(730, color="red", linestyle="--", alpha=0.7, label="Abandonware (730d)")
        axes[0, 0].axhline(4.0, color="black", linestyle=":", alpha=0.5)

        axes[0, 0].set_xlabel("Days Elapsed Since Last Update", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("App Store Rating", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("(A) Developer Maintenance Quadrant (Recency vs. Rating)", fontsize=13, fontweight="bold", pad=12)
        fig.colorbar(scatter, ax=axes[0, 0], shrink=0.8, label="Days Since Update")
        axes[0, 0].legend(loc="lower left", fontsize=9)

        # Panel B: Catalog Lifecycle Tier Distribution (Donut Chart)
        df_life = self.audit_developer_lifecycle()
        tier_counts = df_life["Lifecycle_Tier"].value_counts()
        colors_donut = ["#59a14f", "#4e79a7", "#f28e2b", "#e15759"]
        wedges, texts, autotexts = axes[0, 1].pie(
            tier_counts.values,
            labels=tier_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_donut,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10, fontweight="bold")
        )
        for at in autotexts:
            at.set_color("black")
            at.set_fontsize(10)
        axes[0, 1].set_title("(B) Catalog Developer Lifecycle Distribution", fontsize=13, fontweight="bold", pad=12)

        # Panel C: Churn Risk Score Distribution across Key Categories
        top_cats = ["GAME", "FAMILY", "TOOLS", "FINANCE", "HEALTH_AND_FITNESS", "COMMUNICATION"]
        cat_churn_data = []
        for cat in top_cats:
            cat_apps = self.df_apps[self.df_apps["Category_Clean"] == cat].head(50)
            scores = [self.evaluate_churn_risk(app_row["App_Clean"]).churn_risk_score for _, app_row in cat_apps.iterrows()]
            cat_churn_data.append(scores)

        bp = axes[1, 0].boxplot(
            cat_churn_data,
            tick_labels=top_cats,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5)
        )
        box_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        axes[1, 0].set_ylabel("Composite Churn Risk Score [0–100]", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("(C) Churn Risk Distribution across Key App Verticals", fontsize=13, fontweight="bold", pad=12)
        axes[1, 0].tick_params(axis="x", rotation=25)

        # Panel D: Markowitz Portfolio Efficient Frontier
        port_res = self.optimize_app_portfolio(["GAME", "FINANCE", "TOOLS", "HEALTH_AND_FITNESS", "EDUCATION"])
        # Generate random portfolios for frontier scatter
        rng = np.random.RandomState(42)
        sim_vols = []
        sim_rets = []
        sim_sharpes = []
        rf = 0.03

        mu_vec = np.array([0.22, 0.28, 0.12, 0.19, 0.16])
        vol_vec = np.array([0.24, 0.18, 0.11, 0.19, 0.15])
        c_mat = np.eye(5) + 0.25 * (np.ones((5, 5)) - np.eye(5))
        s_mat = np.outer(vol_vec, vol_vec) * c_mat

        for _ in range(1200):
            w = rng.dirichlet(np.ones(5))
            r = np.dot(w, mu_vec)
            v = np.sqrt(np.dot(w.T, np.dot(s_mat, w)))
            sh = (r - rf) / v
            sim_rets.append(r * 100.0)
            sim_vols.append(v * 100.0)
            sim_sharpes.append(sh)

        sc = axes[1, 1].scatter(sim_vols, sim_rets, c=sim_sharpes, cmap="viridis", alpha=0.6, s=25)
        # Highlight optimal point
        opt_v = port_res.portfolio_volatility * 100.0
        opt_r = port_res.expected_annual_return * 100.0
        axes[1, 1].scatter(
            [opt_v],
            [opt_r],
            color="red",
            s=160,
            marker="*",
            edgecolors="black",
            zorder=10,
            label=f"Max Sharpe ({port_res.sharpe_ratio:.2f})"
        )

        axes[1, 1].set_xlabel("Portfolio Annual Volatility / Risk (%)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Expected Annual LTV Return (%)", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("(D) Markowitz Modern Portfolio Efficient Frontier", fontsize=13, fontweight="bold", pad=12)
        fig.colorbar(sc, ax=axes[1, 1], shrink=0.8, label="Sharpe Ratio")
        axes[1, 1].legend(loc="upper left", fontsize=10)

        plt.suptitle(
            "Google Play Store: Churn Risk, Abandonware Lifecycle & Capital Portfolio Optimization",
            fontsize=16,
            fontweight="bold",
            y=0.99
        )
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure 27 saved to {output_path}")


# ============================================================================
# CLI Execution Entry Point
# ============================================================================

def main():
    """Command-line interface for the Day 13 Retention & LTV Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Google Play Store Retention, Churn & LTV Forecasting Engine")
    parser.add_argument("--run-all", action="store_true", help="Execute complete Day 13 pipeline (reports & figures)")
    parser.add_argument("--generate-reports", action="store_true", help="Generate all CSV summary reports")
    parser.add_argument("--generate-figures", action="store_true", help="Generate publication Figures 26 & 27")
    parser.add_argument("--audit-app", type=str, help="Run churn risk & LTV audit for a specific app")
    args = parser.parse_args()

    engine = RetentionChurnLTVEngine()

    if args.audit_app:
        print(f"\n================ App Diagnostic: {args.audit_app} ================")
        churn = engine.evaluate_churn_risk(args.audit_app)
        ltv = engine.forecast_ltv(args.audit_app)
        life = engine.audit_developer_lifecycle(args.audit_app)
        print(f"Category: {churn.category}")
        print(f"Rating: {churn.rating} | Installs: {churn.installs:,}")
        print(f"Days Since Update: {churn.days_since_update} ({life.lifecycle_tier})")
        print(f"Churn Risk Score: {churn.churn_risk_score} / 100 ({churn.churn_risk_tier})")
        print(f"Primary Churn Driver: {churn.primary_churn_driver}")
        print(f"360-Day Projected LTV: ${ltv.ltv_360d:.2f} | Est CAC: ${ltv.cac:.2f} | LTV/CAC: {ltv.ltv_cac_ratio:.2f}")
        print(f"Payback Period: {ltv.payback_period_days} days ({ltv.economic_tier})")
        print("Prescriptive Action Playbook:")
        for action in churn.prescriptive_playbook:
            print(f"  * {action}")

    if args.generate_reports or args.run_all:
        print("\nGenerating Day 13 analytical reports...")
        paths = engine.generate_all_reports()
        for k, p in paths.items():
            print(f"  -> Generated {k}: {p}")

    if args.generate_figures or args.run_all:
        print("\nRendering Day 13 publication figures...")
        f26, f27 = engine.generate_publication_figures()
        print(f"  -> Rendered Figure 26: {f26}")
        print(f"  -> Rendered Figure 27: {f27}")

    if not any([args.run_all, args.generate_reports, args.generate_figures, args.audit_app]):
        parser.print_help()


if __name__ == "__main__":
    main()
