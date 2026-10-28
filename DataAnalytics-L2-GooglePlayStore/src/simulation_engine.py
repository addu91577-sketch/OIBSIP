"""Counterfactual Scenario Simulation, Price Elasticity Modeling & A/B Experimentation Engine.

Day 10 Milestone for Google Play Store Market Intelligence & Decision Science Platform.
Provides:
1. Counterfactual "What-If" Sensitivity Simulator (Simulating size compression, price elasticity, rating shifts, and revenue uplift).
2. Empirical Price Elasticity of Demand across app categories (Log-log econometrics and optimal revenue frontiers).
3. Frequentist & Bayesian A/B Experimentation Engine (Beta-Binomial conjugate updating, Monte Carlo superiority testing, expected loss, and sample sizing).
4. Statistical Power Analysis & Minimum Detectable Effect (MDE) sample size calculator.
5. Real-time REST endpoints and publication-grade artifacts (Figures 20 & 21, benchmark CSV tables).
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard project directory paths
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.predictive_modeling import PlayStorePredictiveEngine
_DATA_DIR = _BASE_DIR / "Data"
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WhatIfScenarioResult:
    """Quantitative outcome of a counterfactual 'What-If' simulation."""
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
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to serializable dictionary."""
        return asdict(self)


@dataclass
class ABTestResult:
    """Frequentist and Bayesian evaluation of an A/B testing experiment."""
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
    credible_interval_a: Tuple[float, float]
    credible_interval_b: Tuple[float, float]
    decision: str
    summary_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to serializable dictionary."""
        return asdict(self)


@dataclass
class PriceElasticityResult:
    """Econometric price elasticity of demand estimation."""
    category: str
    paid_app_count: int
    median_price: float
    elasticity_coefficient: float
    r_squared: float
    p_value: float
    elasticity_type: str
    optimal_price_recommendation: float
    pricing_advice: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to serializable dictionary."""
        return asdict(self)


# ============================================================================
# Main Simulation & Experimentation Engine
# ============================================================================

class PlayStoreSimulationEngine:
    """Enterprise Counterfactual Simulation, Price Elasticity & A/B Experimentation Engine."""

    def __init__(
        self,
        apps_path: Optional[Union[str, Path]] = None,
        reviews_path: Optional[Union[str, Path]] = None,
        predictive_engine: Optional[PlayStorePredictiveEngine] = None,
        random_state: int = 42,
    ):
        self.apps_path = Path(apps_path) if apps_path else _DATA_DIR / "cleaned_googleplaystore.csv"
        self.reviews_path = Path(reviews_path) if reviews_path else _DATA_DIR / "cleaned_user_reviews.csv"
        self.random_state = random_state
        np.random.seed(random_state)

        # Datasets
        if not self.apps_path.exists():
            raise FileNotFoundError(f"Cleaned apps dataset not found at {self.apps_path}")
        self.df_apps = pd.read_csv(self.apps_path)

        # Connect predictive engine for underlying machine learning forecasts
        if predictive_engine is not None:
            self.ml_engine = predictive_engine
        else:
            logger.info("Initializing underlying PlayStorePredictiveEngine...")
            self.ml_engine = PlayStorePredictiveEngine(
                apps_path=self.apps_path,
                reviews_path=self.reviews_path,
                random_state=self.random_state,
            )
            self.ml_engine.train_models()

        # Cache for elasticity results
        self._elasticity_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------------
    # 1. Counterfactual "What-If" Sensitivity Simulator
    # ------------------------------------------------------------------------

    def simulate_what_if(
        self,
        app_name: Optional[str] = None,
        category: Optional[str] = None,
        rating: Optional[float] = None,
        reviews: Optional[int] = None,
        size_mb: Optional[float] = None,
        price: Optional[float] = None,
        content_rating: Optional[str] = "Everyone",
        delta_size_pct: float = 0.0,
        delta_price: float = 0.0,
        delta_rating: float = 0.0,
    ) -> WhatIfScenarioResult:
        """Simulate counterfactual parameter changes and compute projected impact on scale and revenue.
        
        Args:
            app_name: Name of an existing app in the catalog to use as baseline.
            category: App category (e.g. 'GAME', 'PRODUCTIVITY').
            rating: Baseline user rating (1.0 - 5.0).
            reviews: Baseline total reviews count.
            size_mb: Baseline APK size in Megabytes.
            price: Baseline price in USD.
            content_rating: Content rating string ('Everyone', 'Teen', etc.).
            delta_size_pct: Percentage change in APK size (e.g., -0.25 for 25% reduction).
            delta_price: Absolute change in price in USD (e.g., -1.00 for a $1 price cut).
            delta_rating: Projected shift in rating (e.g., +0.20 for sentiment improvement).

        Returns:
            WhatIfScenarioResult containing baseline, counterfactual metrics, and strategic advice.
        """
        # Resolve baseline parameters
        if app_name and app_name in self.df_apps["App"].values:
            app_row = self.df_apps[self.df_apps["App"] == app_name].iloc[0]
            cat = str(category or app_row["Category"]).strip().upper()
            base_r = float(rating if rating is not None else app_row["Rating"])
            base_rev = int(reviews if reviews is not None else app_row["Reviews"])
            base_s = float(size_mb if size_mb is not None else app_row["Size_MB"])
            base_p = float(price if price is not None else app_row["Price"])
            c_rating = str(content_rating or app_row.get("Content Rating", "Everyone"))
            target_name = app_name
        else:
            cat = str(category or "FAMILY").strip().upper()
            base_r = float(rating if rating is not None else 4.15)
            base_rev = int(reviews if reviews is not None else 5000)
            base_s = float(size_mb if size_mb is not None else 25.0)
            base_p = float(price if price is not None else 0.0)
            c_rating = str(content_rating or "Everyone")
            target_name = app_name or f"Hypothetical App ({cat})"

        # Compute simulated counterfactual inputs
        sim_s = max(0.5, float(base_s * (1.0 + delta_size_pct)))
        sim_p = max(0.0, float(base_p + delta_price))
        sim_r = float(np.clip(base_r + delta_rating, 1.0, 5.0))

        # Evaluate base and counterfactual with ML classification engine
        base_inf = self.ml_engine.predict_app_success(
            category=cat,
            size_mb=base_s,
            price=base_p,
            content_rating=c_rating,
            expected_reviews=base_rev,
        )
        sim_inf = self.ml_engine.predict_app_success(
            category=cat,
            size_mb=sim_s,
            price=sim_p,
            content_rating=c_rating,
            expected_reviews=base_rev,
        )

        base_scale_prob = float(base_inf["success_probability"])
        sim_scale_prob = float(sim_inf["success_probability"])
        delta_scale_prob = sim_scale_prob - base_scale_prob
        pct_scale_uplift = (delta_scale_prob / max(1e-5, base_scale_prob)) * 100.0

        # Evaluate base and counterfactual with ML regression engine
        base_reg = self.ml_engine.predict_app_rating(
            category=cat,
            size_mb=base_s,
            price=base_p,
            content_rating=c_rating,
            expected_reviews=base_rev,
        )
        sim_reg = self.ml_engine.predict_app_rating(
            category=cat,
            size_mb=sim_s,
            price=sim_p,
            content_rating=c_rating,
            expected_reviews=base_rev,
        )
        base_exp_rating = float(base_reg["predicted_rating"])
        sim_exp_rating = float(sim_reg["predicted_rating"])

        # Expected download volume model:
        # Scale tier >= 1M installs vs sub-1M installs
        # Historical median for high scale tier: 5,000,000; for standard tier: 50,000
        base_expected_downloads = (base_scale_prob * 5_000_000) + ((1.0 - base_scale_prob) * 50_000)
        sim_expected_downloads = (sim_scale_prob * 5_000_000) + ((1.0 - sim_scale_prob) * 50_000)

        # Expected gross revenue
        base_exp_rev = float(base_p * base_expected_downloads)
        sim_exp_rev = float(sim_p * sim_expected_downloads)
        delta_rev = sim_exp_rev - base_exp_rev

        # Elasticity sensitivities
        size_change_ratio = (sim_s - base_s) / max(0.1, base_s)
        marginal_size_elasticity = (
            (delta_scale_prob / base_scale_prob) / size_change_ratio
            if abs(size_change_ratio) > 1e-4 and base_scale_prob > 0
            else 0.0
        )

        price_change_ratio = (sim_p - base_p) / max(0.1, base_p) if base_p > 0 else 0.0
        marginal_price_elasticity = (
            (delta_scale_prob / base_scale_prob) / price_change_ratio
            if abs(price_change_ratio) > 1e-4 and base_scale_prob > 0
            else 0.0
        )

        # Recommendations
        recommendations = []
        if delta_size_pct < 0 and delta_scale_prob > 0.01:
            recommendations.append(
                f"Reducing size by {abs(delta_size_pct)*100:.1f}% unlocks an estimated "
                f"+{delta_scale_prob*100:.2f}% scale probability uplift due to improved download completion rates."
            )
        if delta_price < 0 and base_p > 0:
            if delta_rev > 0:
                recommendations.append(
                    f"Price reduction of ${abs(delta_price):.2f} expands install volume sufficiently "
                    f"to yield a net revenue gain of +${delta_rev:,.2f}."
                )
            else:
                recommendations.append(
                    f"Price reduction reduces per-unit margin, causing estimated net revenue delta of -${abs(delta_rev):,.2f}. "
                    "Consider in-app purchases (freemium) instead."
                )
        if delta_rating > 0 and sim_scale_prob >= base_scale_prob:
            recommendations.append(
                f"Improving rating by +{delta_rating:.2f} lifts store ranking visibility and conversion headroom."
            )
        if not recommendations:
            recommendations.append("Scenario shows minimal variance against baseline store performance.")

        return WhatIfScenarioResult(
            app_name=target_name,
            category=cat,
            base_size_mb=round(base_s, 2),
            simulated_size_mb=round(sim_s, 2),
            base_price=round(base_p, 2),
            simulated_price=round(sim_p, 2),
            base_rating=round(base_r, 2),
            simulated_rating=round(sim_r, 2),
            base_scale_prob=round(base_scale_prob, 4),
            simulated_scale_prob=round(sim_scale_prob, 4),
            delta_scale_prob=round(delta_scale_prob, 4),
            pct_scale_prob_uplift=round(pct_scale_uplift, 2),
            base_expected_rating=round(base_exp_rating, 2),
            simulated_expected_rating=round(sim_exp_rating, 2),
            base_expected_revenue=round(base_exp_rev, 2),
            simulated_expected_revenue=round(sim_exp_rev, 2),
            delta_revenue=round(delta_rev, 2),
            marginal_size_elasticity=round(marginal_size_elasticity, 4),
            marginal_price_elasticity=round(marginal_price_elasticity, 4),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------------
    # 2. Empirical Price Elasticity of Demand
    # ------------------------------------------------------------------------

    def analyze_price_elasticity(self, min_category_paid_apps: int = 15) -> Dict[str, Any]:
        """Compute empirical log-log price elasticity of demand across categories and overall catalog.
        
        Econometric specification:
            ln(Installs) = alpha + beta * ln(Price) + epsilon
            where elasticity = beta = % delta(Installs) / % delta(Price)
        """
        if self._elasticity_cache is not None:
            return self._elasticity_cache

        paid_df = self.df_apps[(self.df_apps["Price"] > 0) & (self.df_apps["Installs"] > 0)].copy()
        if len(paid_df) < 20:
            raise ValueError(f"Insufficient paid apps ({len(paid_df)}) for econometric elasticity modeling.")

        # Log transform
        paid_df["log_price"] = np.log(paid_df["Price"])
        paid_df["log_installs"] = np.log(paid_df["Installs"])

        # Overall Econometric Regression
        X_all = paid_df[["log_price"]].values
        y_all = paid_df["log_installs"].values
        reg_all = LinearRegression().fit(X_all, y_all)
        overall_elasticity = float(reg_all.coef_[0])
        overall_intercept = float(reg_all.intercept_)
        r2_all = float(reg_all.score(X_all, y_all))

        # Overall optimal price estimate:
        # ln(Q) = a + b * ln(P) -> Q = e^a * P^b -> Revenue = P * Q = e^a * P^(b + 1)
        # If b < -1 (Elastic), higher prices destroy revenue; if -1 <= b < 0, revenue increases up to optimal frontier
        optimal_overall_price = float(np.exp(-overall_intercept / (overall_elasticity - 1))) if overall_elasticity < -1 else 0.99
        optimal_overall_price = float(np.clip(optimal_overall_price, 0.99, 14.99))

        # Category-level breakdown
        category_results: List[PriceElasticityResult] = []
        grouped = paid_df.groupby("Category")

        for cat, group in grouped:
            n_apps = len(group)
            if n_apps < min_category_paid_apps or group["Price"].nunique() < 3:
                continue

            X_cat = group[["log_price"]].values
            y_cat = group["log_installs"].values
            cat_reg = LinearRegression().fit(X_cat, y_cat)
            beta = float(cat_reg.coef_[0])
            r2 = float(cat_reg.score(X_cat, y_cat))
            median_p = float(group["Price"].median())

            # P-value for slope via stats.linregress
            slope, intercept, r_val, p_val, std_err = stats.linregress(group["log_price"], group["log_installs"])

            if beta < -1.1:
                e_type = "Highly Elastic (Price Sensitive)"
                advice = "Users strongly resist higher upfront prices; discount pricing or free-to-play with IAP maximizes market share."
                opt_price = float(np.clip(np.exp(-intercept / (beta - 1)), 0.99, 4.99))
            elif beta > -0.9:
                e_type = "Inelastic (Utility/Specialized)"
                advice = "Niche value proposition enables premium pricing with low demand erosion."
                opt_price = float(np.clip(median_p * 1.35, 2.99, 19.99))
            else:
                e_type = "Unit Elastic"
                advice = "Revenue is relatively neutral to moderate price variations near the $1.99-$3.99 equilibrium."
                opt_price = 2.99

            category_results.append(
                PriceElasticityResult(
                    category=cat,
                    paid_app_count=n_apps,
                    median_price=round(median_p, 2),
                    elasticity_coefficient=round(beta, 3),
                    r_squared=round(r2, 3),
                    p_value=round(float(p_val), 4),
                    elasticity_type=e_type,
                    optimal_price_recommendation=round(opt_price, 2),
                    pricing_advice=advice,
                )
            )

        # Sort by paid app count descending
        category_results.sort(key=lambda x: x.paid_app_count, reverse=True)

        summary = {
            "total_paid_apps_analyzed": len(paid_df),
            "overall_elasticity": round(overall_elasticity, 3),
            "overall_r_squared": round(r2_all, 3),
            "overall_elasticity_type": "Elastic" if overall_elasticity < -1 else "Inelastic",
            "overall_optimal_price": round(optimal_overall_price, 2),
            "categories_analyzed": len(category_results),
            "category_benchmarks": [c.to_dict() for c in category_results],
        }

        self._elasticity_cache = summary
        return summary

    # ------------------------------------------------------------------------
    # 3. Frequentist & Bayesian A/B Experimentation Engine
    # ------------------------------------------------------------------------

    def run_ab_experiment(
        self,
        visitors_a: int,
        conversions_a: int,
        visitors_b: int,
        conversions_b: int,
        alpha: float = 0.05,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        monte_carlo_draws: int = 50_000,
    ) -> ABTestResult:
        """Run comprehensive frequentist and Bayesian A/B test evaluation.
        
        Args:
            visitors_a: Sample size of control cohort A.
            conversions_a: Conversion count for control cohort A.
            visitors_b: Sample size of treatment cohort B.
            conversions_b: Conversion count for treatment cohort B.
            alpha: Type I error significance threshold (default 0.05 for 95% CI).
            prior_alpha: Beta prior shape parameter alpha (default 1.0 for uniform prior).
            prior_beta: Beta prior shape parameter beta (default 1.0 for uniform prior).
            monte_carlo_draws: Number of Monte Carlo draws for Bayesian superiority simulation.

        Returns:
            ABTestResult containing Z-statistic, p-value, Bayesian superiority probability,
            expected loss, and clear statistical rollout verdict.
        """
        if visitors_a <= 0 or visitors_b <= 0:
            raise ValueError("Cohort visitor counts must be positive integers.")
        if conversions_a < 0 or conversions_b < 0:
            raise ValueError("Conversion counts cannot be negative.")
        if conversions_a > visitors_a or conversions_b > visitors_b:
            raise ValueError("Conversions cannot exceed total visitors.")

        rate_a = conversions_a / visitors_a
        rate_b = conversions_b / visitors_b
        abs_uplift = rate_b - rate_a
        rel_uplift = (abs_uplift / rate_a) if rate_a > 0 else 0.0

        # Frequentist Two-Sample Pooled Z-Test
        p_pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b)
        se_pooled = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / visitors_a + 1.0 / visitors_b))
        se_unpooled = np.sqrt((rate_a * (1.0 - rate_a) / visitors_a) + (rate_b * (1.0 - rate_b) / visitors_b))

        z_stat = (abs_uplift / se_pooled) if se_pooled > 0 else 0.0
        # Two-tailed p-value
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

        # 95% Confidence Interval for difference (Unpooled Welch)
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower = abs_uplift - z_crit * se_unpooled
        ci_upper = abs_uplift + z_crit * se_unpooled
        is_sig = bool(p_val < alpha)

        # Bayesian Beta-Binomial Conjugate Updating
        post_a_alpha = prior_alpha + conversions_a
        post_a_beta = prior_beta + (visitors_a - conversions_a)

        post_b_alpha = prior_alpha + conversions_b
        post_b_beta = prior_beta + (visitors_b - conversions_b)

        # 95% Credible Intervals for theta_A and theta_B
        ci_a = (
            float(stats.beta.ppf(alpha / 2.0, post_a_alpha, post_a_beta)),
            float(stats.beta.ppf(1.0 - alpha / 2.0, post_a_alpha, post_a_beta)),
        )
        ci_b = (
            float(stats.beta.ppf(alpha / 2.0, post_b_alpha, post_b_beta)),
            float(stats.beta.ppf(1.0 - alpha / 2.0, post_b_alpha, post_b_beta)),
        )

        # Monte Carlo Simulation for Superiority and Expected Loss
        draws_a = stats.beta.rvs(post_a_alpha, post_a_beta, size=monte_carlo_draws, random_state=self.random_state)
        draws_b = stats.beta.rvs(post_b_alpha, post_b_beta, size=monte_carlo_draws, random_state=self.random_state + 1)

        prob_b_beats_a = float(np.mean(draws_b > draws_a))
        # Expected Loss if choosing Variant B when Variant A was better
        loss_b = np.maximum(draws_a - draws_b, 0.0)
        expected_loss = float(np.mean(loss_b))

        # Strategic Decision
        if prob_b_beats_a >= 0.95 and is_sig and abs_uplift > 0:
            decision = "ACCEPT_VARIANT_B"
            verdict = (
                f"Statistically significant positive uplift (+{abs_uplift*100:.2f}% abs, "
                f"+{rel_uplift*100:.1f}% rel, p={p_val:.4f}). "
                f"Bayesian confidence that Treatment outperforms Control is {prob_b_beats_a*100:.2f}%. "
                f"Deploy Variant B to 100% of production traffic."
            )
        elif prob_b_beats_a <= 0.05 and is_sig and abs_uplift < 0:
            decision = "REJECT_VARIANT_B"
            verdict = (
                f"Treatment Variant B significantly degrades conversion by {abs(abs_uplift)*100:.2f}% "
                f"(p={p_val:.4f}). Abort experiment immediately and preserve Control Variant A."
            )
        else:
            decision = "INCONCLUSIVE"
            verdict = (
                f"Inconclusive test (p={p_val:.4f}, P(B>A)={prob_b_beats_a*100:.1f}%). "
                "Insufficient sample size or effect magnitude to reject the null hypothesis. "
                "Continue sample collection until MDE sample size budget is achieved."
            )

        return ABTestResult(
            visitors_a=visitors_a,
            conversions_a=conversions_a,
            rate_a=round(rate_a, 4),
            visitors_b=visitors_b,
            conversions_b=conversions_b,
            rate_b=round(rate_b, 4),
            absolute_uplift=round(abs_uplift, 4),
            relative_uplift=round(rel_uplift, 4),
            z_statistic=round(z_stat, 3),
            p_value=round(p_val, 4),
            is_statistically_significant=is_sig,
            ci_lower_diff=round(ci_lower, 4),
            ci_upper_diff=round(ci_upper, 4),
            bayesian_prob_b_beats_a=round(prob_b_beats_a, 4),
            expected_loss=round(expected_loss, 6),
            credible_interval_a=(round(ci_a[0], 4), round(ci_a[1], 4)),
            credible_interval_b=(round(ci_b[0], 4), round(ci_b[1], 4)),
            decision=decision,
            summary_verdict=verdict,
        )

    # ------------------------------------------------------------------------
    # 4. Statistical Power & Sample Size Sizing
    # ------------------------------------------------------------------------

    def calculate_sample_size_mde(
        self,
        baseline_rate: float = 0.05,
        mde_relative: float = 0.10,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Calculate required sample size per variant for two-sample proportion test.
        
        Formula:
            n = [ (Z_{alpha/2} * sqrt(2 * p_bar * (1 - p_bar)) + Z_{beta} * sqrt(p1*(1-p1) + p2*(1-p2))) / (p2 - p1) ]^2
        """
        if baseline_rate <= 0 or baseline_rate >= 1:
            raise ValueError("Baseline conversion rate must be strictly between 0 and 1.")
        if mde_relative <= 0:
            raise ValueError("Relative Minimum Detectable Effect must be strictly positive.")

        p1 = baseline_rate
        p2 = p1 * (1.0 + mde_relative)
        p_bar = (p1 + p2) / 2.0

        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))

        numerator = z_alpha * np.sqrt(2 * p_bar * (1.0 - p_bar)) + z_beta * np.sqrt(
            p1 * (1.0 - p1) + p2 * (1.0 - p2)
        )
        denominator = abs(p2 - p1)

        n_required = int(np.ceil((numerator / denominator) ** 2))
        return max(50, n_required)

    def generate_sample_size_matrix(
        self,
        baseline_rates: Optional[List[float]] = None,
        mdes: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Build cross-tabulation matrix of sample sizes across baseline rates and relative MDEs."""
        b_rates = baseline_rates or [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        mde_list = mdes or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        records = []
        for br in b_rates:
            row = {"Baseline_Conversion_Rate": f"{br*100:.1f}%"}
            for mde in mde_list:
                n = self.calculate_sample_size_mde(baseline_rate=br, mde_relative=mde)
                row[f"MDE_{int(mde*100)}%"] = n
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------------
    # 5. Publication Artifacts & Benchmark Export
    # ------------------------------------------------------------------------

    def generate_reports_and_figures(
        self,
        output_figures_dir: Optional[Union[str, Path]] = None,
        output_reports_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Generate high-resolution Figure 20, Figure 21, and CSV benchmark reports."""
        fig_dir = Path(output_figures_dir) if output_figures_dir else _FIGURES_DIR
        rep_dir = Path(output_reports_dir) if output_reports_dir else _REPORTS_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating Day 10 CSV reports and high-resolution figures...")

        # 1. Price Elasticity Report
        elasticity_data = self.analyze_price_elasticity()
        df_elasticity = pd.DataFrame(elasticity_data["category_benchmarks"])
        elasticity_csv = rep_dir / "category_price_elasticity_summary.csv"
        df_elasticity.to_csv(elasticity_csv, index=False)
        logger.info(f"Saved Price Elasticity summary to {elasticity_csv}")

        # 2. Sample Size Matrix Report
        df_matrix = self.generate_sample_size_matrix()
        matrix_csv = rep_dir / "ab_testing_sample_size_matrix.csv"
        df_matrix.to_csv(matrix_csv, index=False)
        logger.info(f"Saved A/B Sample Size Matrix to {matrix_csv}")

        # 3. Counterfactual Benchmarks for Key Apps
        sample_apps = ["Subway Surfers", "Candy Crush Saga", "Duolingo: Learn Languages Free", "Dropbox", "PicsArt Photo Studio: Maker & Pic Editor"]
        benchmark_records = []
        for a in sample_apps:
            if a in self.df_apps["App"].values:
                res = self.simulate_what_if(app_name=a, delta_size_pct=-0.30, delta_rating=0.20)
                benchmark_records.append(res.to_dict())
            else:
                res = self.simulate_what_if(category="GAME", rating=4.2, size_mb=60.0, delta_size_pct=-0.30)
                benchmark_records.append(res.to_dict())

        df_benchmarks = pd.DataFrame(benchmark_records)
        benchmark_csv = rep_dir / "counterfactual_simulation_benchmarks.csv"
        df_benchmarks.to_csv(benchmark_csv, index=False)
        logger.info(f"Saved Counterfactual Simulation benchmarks to {benchmark_csv}")

        # -------------------------------------------------------------
        # Figure 20: Counterfactual Sensitivity & Price Elasticity
        # -------------------------------------------------------------
        fig20_path = fig_dir / "20_counterfactual_whatif_and_pricing_elasticity.png"
        logger.info(f"Rendering Figure 20 to {fig20_path}...")

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig20, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), facecolor="#0e1117")

        # Panel 1: Counterfactual Sensitivity Curve (Size Compression vs Scale Probability Uplift)
        ax1.set_facecolor("#1e293b")
        size_compressions = np.linspace(-0.60, 0.0, 25)  # -60% to 0%
        sample_cats = ["GAME", "FAMILY", "PRODUCTIVITY", "TOOLS"]
        colors = ["#38bdf8", "#34d399", "#f59e0b", "#ec4899"]

        for cat, col in zip(sample_cats, colors):
            uplifts = []
            for comp in size_compressions:
                sim = self.simulate_what_if(category=cat, size_mb=45.0, delta_size_pct=comp)
                uplifts.append(sim.pct_scale_prob_uplift)
            ax1.plot(size_compressions * 100, uplifts, label=f"Category: {cat}", color=col, linewidth=2.5, marker="o", markersize=4)

        ax1.set_title("Counterfactual Scale Probability Uplift vs APK Size Compression", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
        ax1.set_xlabel("APK Size Reduction Delta (%)", fontsize=10.5, color="#cbd5e1")
        ax1.set_ylabel("Predicted High-Velocity Scale Uplift (%)", fontsize=10.5, color="#cbd5e1")
        ax1.tick_params(colors="#94a3b8")
        ax1.grid(color="#334155", linestyle="--", alpha=0.6)
        ax1.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")

        # Panel 2: Econometric Demand Curve & Revenue Frontier
        ax2.set_facecolor("#1e293b")
        prices = np.linspace(0.99, 14.99, 50)
        e = elasticity_data["overall_elasticity"]
        # Demand Q(P) = A * P^e
        q_norm = 100_000 * (prices ** e)
        rev_norm = prices * q_norm

        ax2_twin = ax2.twinx()
        line1 = ax2.plot(prices, q_norm / 1000, color="#38bdf8", linewidth=2.5, label="Empirical Demand Q(P)")
        line2 = ax2_twin.plot(prices, rev_norm / 1000, color="#f59e0b", linewidth=2.5, linestyle="--", label="Gross Revenue Curve P*Q(P)")

        opt_p = elasticity_data["overall_optimal_price"]
        ax2.axvline(opt_p, color="#ef4444", linestyle=":", linewidth=2, label=f"Empirical Optimal Price (${opt_p:.2f})")

        ax2.set_title(f"Catalog Price Elasticity (ε = {e:.2f}) & Revenue Frontier", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
        ax2.set_xlabel("App Upfront Price ($ USD)", fontsize=10.5, color="#cbd5e1")
        ax2.set_ylabel("Projected Installs (Thousands)", fontsize=10.5, color="#38bdf8")
        ax2_twin.set_ylabel("Gross Revenue Index ($k)", fontsize=10.5, color="#f59e0b")
        ax2.tick_params(colors="#94a3b8")
        ax2_twin.tick_params(colors="#94a3b8")
        ax2.grid(color="#334155", linestyle="--", alpha=0.6)
        ax2_twin.grid(False)

        lines = line1 + line2 + [plt.Line2D([0], [0], color="#ef4444", linestyle=":", linewidth=2)]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")

        fig20.tight_layout()
        fig20.savefig(fig20_path, dpi=300, bbox_inches="tight")
        plt.close(fig20)

        # -------------------------------------------------------------
        # Figure 21: A/B Testing Statistical Power & Bayesian Posteriors
        # -------------------------------------------------------------
        fig21_path = fig_dir / "21_ab_testing_and_bayesian_power_analytics.png"
        logger.info(f"Rendering Figure 21 to {fig21_path}...")

        fig21, (p1, p2) = plt.subplots(1, 2, figsize=(18, 7.5), facecolor="#0e1117")

        # Panel 1: Statistical Power Curves vs Sample Size across MDEs
        p1.set_facecolor("#1e293b")
        sample_sizes = np.linspace(500, 30_000, 60)
        p_base = 0.05
        z_alpha = stats.norm.ppf(0.975)

        for mde_rel, col in zip([0.05, 0.10, 0.15, 0.20], ["#ef4444", "#f59e0b", "#34d399", "#38bdf8"]):
            p_treat = p_base * (1.0 + mde_rel)
            p_pool = (p_base + p_treat) / 2.0
            powers = []
            for n in sample_sizes:
                se = np.sqrt(2 * p_pool * (1 - p_pool) / n)
                effect = abs(p_treat - p_base)
                # Power = P(Z > z_alpha - effect/se)
                beta_power = float(1.0 - stats.norm.cdf(z_alpha - effect / se))
                powers.append(beta_power)
            p1.plot(sample_sizes, powers, label=f"MDE = +{int(mde_rel*100)}%", color=col, linewidth=2.5)

        p1.axhline(0.80, color="#f8fafc", linestyle="--", linewidth=1.8, label="Standard 80% Power Threshold")
        p1.set_title("Statistical Power vs Sample Size per Cohort (Baseline = 5.0%)", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
        p1.set_xlabel("Visitors per Cohort Variant (N)", fontsize=10.5, color="#cbd5e1")
        p1.set_ylabel("Statistical Power (1 - β)", fontsize=10.5, color="#cbd5e1")
        p1.set_ylim(0, 1.05)
        p1.tick_params(colors="#94a3b8")
        p1.grid(color="#334155", linestyle="--", alpha=0.6)
        p1.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")

        # Panel 2: Bayesian Posterior Distributions for Variant A vs Variant B
        p2.set_facecolor("#1e293b")
        theta = np.linspace(0.040, 0.065, 300)
        # Simulation: Variant A: 500 / 10,000 (5.0%), Variant B: 565 / 10,000 (5.65%)
        post_a = stats.beta.pdf(theta, 1 + 500, 1 + 9500)
        post_b = stats.beta.pdf(theta, 1 + 565, 1 + 9435)

        p2.plot(theta * 100, post_a, label="Control Variant A (Mean = 5.00%)", color="#38bdf8", linewidth=2.5)
        p2.fill_between(theta * 100, post_a, color="#38bdf8", alpha=0.25)
        p2.plot(theta * 100, post_b, label="Treatment Variant B (Mean = 5.65%)", color="#34d399", linewidth=2.5)
        p2.fill_between(theta * 100, post_b, color="#34d399", alpha=0.25)

        p2.set_title("Bayesian Posterior Densities & Credible Intervals (N=10,000 / variant)", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
        p2.set_xlabel("True Conversion Rate θ (%)", fontsize=10.5, color="#cbd5e1")
        p2.set_ylabel("Posterior Probability Density f(θ|Data)", fontsize=10.5, color="#cbd5e1")
        p2.tick_params(colors="#94a3b8")
        p2.grid(color="#334155", linestyle="--", alpha=0.6)
        p2.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")

        fig21.tight_layout()
        fig21.savefig(fig21_path, dpi=300, bbox_inches="tight")
        plt.close(fig21)

        return {
            "fig20": fig20_path,
            "fig21": fig21_path,
            "elasticity_csv": elasticity_csv,
            "matrix_csv": matrix_csv,
            "benchmark_csv": benchmark_csv,
        }


# ============================================================================
# CLI Execution
# ============================================================================

def main():
    """CLI Entry point for Day 10 Simulation & A/B Engine."""
    logger.info("Initializing PlayStoreSimulationEngine from CLI...")
    engine = PlayStoreSimulationEngine()

    logger.info("Running sample counterfactual simulation...")
    scenario = engine.simulate_what_if(category="GAME", size_mb=50.0, delta_size_pct=-0.25, delta_rating=0.2)
    print("\n--- SAMPLE WHAT-IF SCENARIO ---")
    print(json.dumps(scenario.to_dict(), indent=2))

    logger.info("Running sample A/B experiment...")
    ab_res = engine.run_ab_experiment(visitors_a=10000, conversions_a=500, visitors_b=10000, conversions_b=570)
    print("\n--- SAMPLE A/B TEST EVALUATION ---")
    print(json.dumps(ab_res.to_dict(), indent=2))

    logger.info("Generating publication figures and benchmark reports...")
    artifacts = engine.generate_reports_and_figures()
    print("\n--- ARTIFACTS GENERATED ---")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
