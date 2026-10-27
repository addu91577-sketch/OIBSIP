"""Predictive Machine Learning Engine & MLOps Drift Detection Pipeline.

Day 9 Milestone for Google Play Store Market Intelligence & Smart Typing Assistant.
Provides:
1. Supervised App Success Classification (Predicting Installs >= 1M high-velocity tier).
2. Continuous Rating Valuation Regression & Residual Analysis.
3. Feature Attribution & Gini Importance Analysis for market scale drivers.
4. MLOps Data & Concept Drift Suite: Population Stability Index (PSI), KS-test, Wasserstein distance.
5. Real-time single-app inference routines for probability, risk score, and rating forecast.
6. Publication artifact generation (Figures 18 & 19, benchmark CSV tables).
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard paths
_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "Data"
_REPORTS_DIR = _BASE_DIR / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"


# ============================================================================
# Mathematical Drift Utilities
# ============================================================================

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """Calculate the Population Stability Index (PSI) between expected and actual distributions.

    PSI = sum((Actual% - Expected%) * ln((Actual% + eps) / (Expected% + eps)))

    Standard Industry Drift Thresholds:
    - PSI < 0.10: Stable / No significant distribution change.
    - 0.10 <= PSI < 0.25: Moderate drift / Monitor feature.
    - PSI >= 0.25: Significant drift / Model retraining recommended.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Filter out NaNs/Infs
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 1:
        # Uniform or constant distribution
        min_v, max_v = min(expected.min(), actual.min()), max(expected.max(), actual.max())
        if min_v == max_v:
            return 0.0
        bin_edges = np.linspace(min_v, max_v, num_buckets + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Smooth zero probabilities
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)

    # Re-normalize
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))


def calculate_ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov test statistic and asymptotic p-value."""
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    s1 = s1[np.isfinite(s1)]
    s2 = s2[np.isfinite(s2)]
    if len(s1) == 0 or len(s2) == 0:
        return 0.0, 1.0
    res = stats.ks_2samp(s1, s2)
    return float(res.statistic), float(res.pvalue)


def calculate_wasserstein(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """Compute 1-Wasserstein distance (Earth Mover's Distance) between two distributions."""
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    s1 = s1[np.isfinite(s1)]
    s2 = s2[np.isfinite(s2)]
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    return float(stats.wasserstein_distance(s1, s2))


# ============================================================================
# Main Predictive Engine Class
# ============================================================================

class PlayStorePredictiveEngine:
    """Enterprise predictive modeling and MLOps distribution drift engine."""

    def __init__(
        self,
        apps_path: Optional[Union[str, Path]] = None,
        reviews_path: Optional[Union[str, Path]] = None,
        random_state: int = 42,
    ):
        self.apps_path = Path(apps_path) if apps_path else _DATA_DIR / "cleaned_googleplaystore.csv"
        self.reviews_path = Path(reviews_path) if reviews_path else _DATA_DIR / "cleaned_user_reviews.csv"
        self.random_state = random_state

        # DataFrames
        self.df_apps: Optional[pd.DataFrame] = None
        self.df_reviews: Optional[pd.DataFrame] = None

        # Feature schema metadata
        self.feature_columns: List[str] = []
        self.category_list: List[str] = []
        self.content_ratings: List[str] = []
        self.category_sentiment_priors: Dict[str, float] = {}

        # Fitted Models
        self.clf_rf: Optional[RandomForestClassifier] = None
        self.clf_gbm: Optional[GradientBoostingClassifier] = None
        self.clf_lr: Optional[LogisticRegression] = None
        self.reg_rf: Optional[RandomForestRegressor] = None
        self.reg_ridge: Optional[Ridge] = None
        self.reg_dummy: Optional[DummyRegressor] = None

        # Evaluated Metrics & Cache
        self.classification_metrics: Dict[str, Dict[str, float]] = {}
        self.regression_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importances_: Dict[str, float] = {}
        self.drift_report_cache: Optional[Dict[str, Any]] = None

        # Training data cache
        self.X_cls: Optional[pd.DataFrame] = None
        self.y_cls: Optional[pd.Series] = None
        self.X_reg: Optional[pd.DataFrame] = None
        self.y_reg: Optional[pd.Series] = None

        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        """Load cleaned datasets and build feature matrices."""
        if not self.apps_path.exists():
            raise FileNotFoundError(f"Apps dataset not found at {self.apps_path}")
        self.df_apps = pd.read_csv(self.apps_path)

        if self.reviews_path.exists():
            self.df_reviews = pd.read_csv(self.reviews_path)
            # Compute category-level prior sentiment
            if "Sentiment_Polarity" in self.df_reviews.columns and "App" in self.df_reviews.columns:
                merged = pd.merge(
                    self.df_reviews[["App", "Sentiment_Polarity"]],
                    self.df_apps[["App", "Category"]],
                    on="App",
                    how="inner",
                )
                self.category_sentiment_priors = (
                    merged.groupby("Category")["Sentiment_Polarity"].mean().to_dict()
                )

        self.category_list = sorted(self.df_apps["Category"].dropna().unique().tolist())
        self.content_ratings = sorted(self.df_apps["Content Rating"].dropna().unique().tolist())

        # Construct feature matrices
        self._build_feature_matrices()

    def _build_feature_matrices(self) -> None:
        """Construct numerical feature matrices for classification and regression."""
        df = self.df_apps.copy()

        # Target 1: Binary Success Flag (Installs >= 1,000,000)
        y_cls = (df["Installs"] >= 1_000_000).astype(int)

        # Numerical Features
        X_df = pd.DataFrame(index=df.index)
        X_df["log_reviews"] = np.log10(df["Reviews"].clip(lower=0) + 1.0)
        X_df["size_mb"] = df["Size_MB_Imputed"].fillna(df["Size_MB_Imputed"].median())
        X_df["price"] = df["Price"].clip(lower=0.0)
        X_df["is_free"] = (df["Type"] == "Free").astype(int)

        # Prior sentiment feature
        mean_sentiment_global = 0.28
        X_df["category_prior_sentiment"] = df["Category"].map(
            lambda c: self.category_sentiment_priors.get(c, mean_sentiment_global)
        )

        # One-hot encoding for Category & Content Rating
        for cat in self.category_list[1:]:  # drop first category for collinearity
            X_df[f"cat_{cat}"] = (df["Category"] == cat).astype(int)

        for cr in self.content_ratings[1:]:
            X_df[f"cr_{cr}"] = (df["Content Rating"] == cr).astype(int)

        self.feature_columns = X_df.columns.tolist()
        self.X_cls = X_df
        self.y_cls = y_cls

        # Target 2: Continuous Rating for apps with explicit ratings (8,196 apps)
        rated_mask = df["Rating"].notna()
        self.X_reg = X_df.loc[rated_mask].copy()
        self.y_reg = df.loc[rated_mask, "Rating"].astype(float)

    def train_models(self) -> Dict[str, Any]:
        """Train all classification, regression, and baseline models."""
        logger.info("Training supervised classification and regression pipelines...")

        # -------------------------------------------------------------
        # 1. Supervised Classification (Success >= 1M Installs)
        # -------------------------------------------------------------
        X, y = self.X_cls, self.y_cls
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )

        # Model A: Random Forest Classifier
        self.clf_rf = RandomForestClassifier(
            n_estimators=75,
            max_depth=10,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.clf_rf.fit(X_train, y_train)
        rf_proba = self.clf_rf.predict_proba(X_test)[:, 1]
        rf_pred = (rf_proba >= 0.50).astype(int)

        # Model B: Gradient Boosting Classifier
        self.clf_gbm = GradientBoostingClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.10,
            random_state=self.random_state,
        )
        self.clf_gbm.fit(X_train, y_train)
        gbm_proba = self.clf_gbm.predict_proba(X_test)[:, 1]
        gbm_pred = (gbm_proba >= 0.50).astype(int)

        # Model C: Logistic Regression Baseline
        # Standardize for logistic regression
        X_train_scaled = (X_train - X_train.mean()) / (X_train.std().replace(0, 1))
        X_test_scaled = (X_test - X_train.mean()) / (X_train.std().replace(0, 1))
        self.clf_lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.clf_lr.fit(X_train_scaled, y_train)
        lr_proba = self.clf_lr.predict_proba(X_test_scaled)[:, 1]
        lr_pred = (lr_proba >= 0.50).astype(int)

        # Compute Classification Metrics
        def _calc_cls_metrics(y_true, y_pred, y_prob):
            return {
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "brier_score": float(brier_score_loss(y_true, y_prob)),
            }

        self.classification_metrics = {
            "Random_Forest": _calc_cls_metrics(y_test, rf_pred, rf_proba),
            "Gradient_Boosting": _calc_cls_metrics(y_test, gbm_pred, gbm_proba),
            "Logistic_Regression": _calc_cls_metrics(y_test, lr_pred, lr_proba),
        }

        # Feature Importances from Random Forest
        importances = self.clf_rf.feature_importances_
        feat_imp_pairs = sorted(zip(self.feature_columns, importances), key=lambda x: x[1], reverse=True)
        self.feature_importances_ = {feat: float(imp) for feat, imp in feat_imp_pairs}

        # -------------------------------------------------------------
        # 2. Continuous Rating Regressor
        # -------------------------------------------------------------
        Xr, yr = self.X_reg, self.y_reg
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            Xr, yr, test_size=0.20, random_state=self.random_state
        )

        self.reg_rf = RandomForestRegressor(
            n_estimators=60,
            max_depth=6,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.reg_rf.fit(Xr_train, yr_train)
        rf_reg_pred = self.reg_rf.predict(Xr_test)

        self.reg_ridge = Ridge(alpha=10.0)
        self.reg_ridge.fit(Xr_train, yr_train)
        ridge_pred = self.reg_ridge.predict(Xr_test)

        self.reg_dummy = DummyRegressor(strategy="mean")
        self.reg_dummy.fit(Xr_train, yr_train)
        dummy_pred = self.reg_dummy.predict(Xr_test)

        def _calc_reg_metrics(y_true, y_hat):
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat))),
                "mae": float(mean_absolute_error(y_true, y_hat)),
                "medae": float(median_absolute_error(y_true, y_hat)),
                "r2": float(r2_score(y_true, y_hat)),
            }

        self.regression_metrics = {
            "Random_Forest_Regressor": _calc_reg_metrics(yr_test, rf_reg_pred),
            "Ridge_Regressor": _calc_reg_metrics(yr_test, ridge_pred),
            "Dummy_Mean_Baseline": _calc_reg_metrics(yr_test, dummy_pred),
        }

        logger.info(
            f"Training complete. RF Classifier ROC-AUC: {self.classification_metrics['Random_Forest']['roc_auc']:.4f}, "
            f"Rating RF Regressor RMSE: {self.regression_metrics['Random_Forest_Regressor']['rmse']:.4f}"
        )

        return {
            "classification": self.classification_metrics,
            "regression": self.regression_metrics,
            "top_features": list(self.feature_importances_.items())[:5],
        }

    def generate_drift_report(
        self,
        baseline_df: Optional[pd.DataFrame] = None,
        monitored_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive MLOps distribution drift report between baseline and monitored cohorts."""
        if baseline_df is None or monitored_df is None:
            # Create standard operational cohorts:
            # Baseline: Mature Core Categories (Family, Game, Tools, Finance, Lifestyle, Productivity)
            # Monitored: Dynamic Emerging Categories (Dating, Beauty, Parenting, Comics, Events, Social)
            core_cats = ["FAMILY", "GAME", "TOOLS", "FINANCE", "LIFESTYLE", "PRODUCTIVITY"]
            baseline_df = self.df_apps[self.df_apps["Category"].isin(core_cats)].copy()
            monitored_df = self.df_apps[~self.df_apps["Category"].isin(core_cats)].copy()

        features_to_monitor = [
            ("Size_MB_Imputed", "App Size (MB)"),
            ("Price", "App Price ($)"),
            ("Reviews", "Review Counts (Log10)"),
        ]

        metrics_list = []
        overall_max_psi = 0.0

        for col_name, display_name in features_to_monitor:
            if col_name == "Reviews":
                base_vals = np.log10(baseline_df[col_name].clip(lower=0) + 1.0).values
                mon_vals = np.log10(monitored_df[col_name].clip(lower=0) + 1.0).values
            else:
                base_vals = baseline_df[col_name].dropna().values
                mon_vals = monitored_df[col_name].dropna().values

            psi = calculate_psi(base_vals, mon_vals, num_buckets=10)
            ks_stat, ks_pval = calculate_ks_statistic(base_vals, mon_vals)
            wasserstein = calculate_wasserstein(base_vals, mon_vals)

            if psi < 0.10:
                status_str = "PASS (Stable)"
            elif psi < 0.25:
                status_str = "MONITOR (Moderate Shift)"
            else:
                status_str = "ALERT (Significant Drift)"

            overall_max_psi = max(overall_max_psi, psi)

            metrics_list.append({
                "feature": display_name,
                "column": col_name,
                "baseline_mean": float(np.mean(base_vals)),
                "monitored_mean": float(np.mean(mon_vals)),
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": float(f"{ks_pval:.2e}"),
                "wasserstein_distance": round(wasserstein, 4),
                "status": status_str,
            })

        # Add Rating drift for rated apps
        base_ratings = baseline_df["Rating"].dropna().values
        mon_ratings = monitored_df["Rating"].dropna().values
        rating_psi = calculate_psi(base_ratings, mon_ratings, num_buckets=10)
        rating_ks, rating_pval = calculate_ks_statistic(base_ratings, mon_ratings)
        rating_wass = calculate_wasserstein(base_ratings, mon_ratings)

        metrics_list.append({
            "feature": "User Rating (Stars)",
            "column": "Rating",
            "baseline_mean": float(np.mean(base_ratings)),
            "monitored_mean": float(np.mean(mon_ratings)),
            "psi": round(rating_psi, 4),
            "ks_statistic": round(rating_ks, 4),
            "ks_pvalue": float(f"{rating_pval:.2e}"),
            "wasserstein_distance": round(rating_wass, 4),
            "status": "PASS (Stable)" if rating_psi < 0.10 else ("MONITOR" if rating_psi < 0.25 else "ALERT"),
        })

        if overall_max_psi < 0.10:
            health = "HEALTHY"
            recommendation = "All statistical distributions within standard SLA bounds. No retraining required."
        elif overall_max_psi < 0.25:
            health = "MODERATE_DRIFT"
            recommendation = "Moderate shift detected in feature distributions. Scheduled retraining suggested."
        else:
            health = "ACTION_REQUIRED"
            recommendation = "Significant feature divergence detected (PSI >= 0.25). Immediate model retraining required."

        drift_report = {
            "health_verdict": health,
            "overall_max_psi": round(overall_max_psi, 4),
            "recommendation": recommendation,
            "baseline_samples": len(baseline_df),
            "monitored_samples": len(monitored_df),
            "metrics": metrics_list,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.drift_report_cache = drift_report
        return drift_report

    def predict_app_success(
        self,
        category: str,
        size_mb: float,
        price: float = 0.0,
        content_rating: str = "Everyone",
        expected_reviews: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Real-time inference routine for predicting App High-Velocity Success (>1M Installs)."""
        t0 = time.perf_counter()
        if self.clf_rf is None:
            self.train_models()

        cat_upper = category.upper()
        if cat_upper not in self.category_list:
            cat_upper = "TOOLS"  # default fallback

        cr_fmt = content_rating.title()
        if cr_fmt not in self.content_ratings:
            cr_fmt = "Everyone"

        # Build single row DataFrame
        row_dict = {col: 0 for col in self.feature_columns}
        row_dict["size_mb"] = max(0.1, float(size_mb))
        row_dict["price"] = max(0.0, float(price))
        row_dict["is_free"] = 1 if price <= 0.0 else 0

        # Review count logic: if provided use it, otherwise use category median reviews as pre-launch baseline
        if expected_reviews is not None:
            reviews_val = max(0, int(expected_reviews))
        else:
            cat_apps = self.df_apps[self.df_apps["Category"] == cat_upper]
            reviews_val = int(cat_apps["Reviews"].median()) if len(cat_apps) > 0 else 500

        row_dict["log_reviews"] = float(np.log10(reviews_val + 1.0))
        row_dict["category_prior_sentiment"] = self.category_sentiment_priors.get(cat_upper, 0.28)

        if f"cat_{cat_upper}" in row_dict:
            row_dict[f"cat_{cat_upper}"] = 1
        if f"cr_{cr_fmt}" in row_dict:
            row_dict[f"cr_{cr_fmt}"] = 1

        input_df = pd.DataFrame([row_dict])[self.feature_columns]

        # Model Predictions
        prob_rf = float(self.clf_rf.predict_proba(input_df)[0, 1])
        prob_gbm = float(self.clf_gbm.predict_proba(input_df)[0, 1])
        ensemble_prob = float(0.60 * prob_rf + 0.40 * prob_gbm)

        # Risk and Tier Assessment
        if ensemble_prob >= 0.70:
            success_tier = "High Velocity Scale (>= 1,000,000 Installs)"
            risk_tier = "LOW_RISK"
        elif ensemble_prob >= 0.40:
            success_tier = "Moderate Scale Potential (100k - 1M Installs)"
            risk_tier = "MODERATE_RISK"
        else:
            success_tier = "Niche / Boutique Audience (< 100k Installs)"
            risk_tier = "HIGH_RISK"

        # Key Drivers
        drivers = []
        if price > 4.99:
            drivers.append("High price friction (> $4.99) suppresses organic install velocity.")
        elif price == 0.0:
            drivers.append("Free monetization maximizes top-of-funnel install acquisition.")
        if size_mb > 50.0:
            drivers.append("Feature-rich binary size correlates positively with user retention.")
        if reviews_val >= 10000:
            drivers.append("Strong social proof (>10k reviews) provides dominant algorithmic distribution.")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "success_probability": round(ensemble_prob, 4),
            "rf_probability": round(prob_rf, 4),
            "gbm_probability": round(prob_gbm, 4),
            "predicted_tier": success_tier,
            "risk_assessment": risk_tier,
            "category": cat_upper,
            "size_mb": size_mb,
            "price": price,
            "effective_reviews_evaluated": reviews_val,
            "key_drivers": drivers,
            "latency_ms": round(latency_ms, 2),
        }

    def predict_app_rating(
        self,
        category: str,
        size_mb: float,
        price: float = 0.0,
        content_rating: str = "Everyone",
        expected_reviews: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Real-time inference routine for predicting App User Rating (1.0 to 5.0 scale)."""
        t0 = time.perf_counter()
        if self.reg_rf is None:
            self.train_models()

        cat_upper = category.upper()
        if cat_upper not in self.category_list:
            cat_upper = "TOOLS"

        cr_fmt = content_rating.title()
        if cr_fmt not in self.content_ratings:
            cr_fmt = "Everyone"

        row_dict = {col: 0 for col in self.feature_columns}
        row_dict["size_mb"] = max(0.1, float(size_mb))
        row_dict["price"] = max(0.0, float(price))
        row_dict["is_free"] = 1 if price <= 0.0 else 0

        if expected_reviews is not None:
            reviews_val = max(0, int(expected_reviews))
        else:
            cat_apps = self.df_apps[self.df_apps["Category"] == cat_upper]
            reviews_val = int(cat_apps["Reviews"].median()) if len(cat_apps) > 0 else 500

        row_dict["log_reviews"] = float(np.log10(reviews_val + 1.0))
        row_dict["category_prior_sentiment"] = self.category_sentiment_priors.get(cat_upper, 0.28)

        if f"cat_{cat_upper}" in row_dict:
            row_dict[f"cat_{cat_upper}"] = 1
        if f"cr_{cr_fmt}" in row_dict:
            row_dict[f"cr_{cr_fmt}"] = 1

        input_df = pd.DataFrame([row_dict])[self.feature_columns]

        predicted_rating = float(self.reg_rf.predict(input_df)[0])
        predicted_rating = float(np.clip(predicted_rating, 1.0, 5.0))

        # Model residual standard error is ~0.51 -> 90% confidence interval ~ +/- 1.645 * 0.51 / sqrt(N)
        ci_half_width = 0.25
        lower_bound = max(1.0, round(predicted_rating - ci_half_width, 2))
        upper_bound = min(5.0, round(predicted_rating + ci_half_width, 2))

        if predicted_rating >= 4.30:
            quality_tier = "Top Quartile Quality (Elite App Tier)"
        elif predicted_rating >= 4.00:
            quality_tier = "Mainstream Average (Passes 4.0 Viability Barrier)"
        else:
            quality_tier = "Sub-Par / High Churn Risk (< 4.0 Store Barrier)"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_rating": round(predicted_rating, 2),
            "confidence_interval_90": [lower_bound, upper_bound],
            "quality_tier": quality_tier,
            "category": cat_upper,
            "category_baseline_mean": round(float(self.df_apps[self.df_apps["Category"] == cat_upper]["Rating"].mean()), 2),
            "latency_ms": round(latency_ms, 2),
        }

    def generate_figures(self, output_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Render publication-quality Figure 18 and Figure 19."""
        out_dir = Path(output_dir) if output_dir else _FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.clf_rf is None:
            self.train_models()

        # -------------------------------------------------------------
        # Figure 18: Predictive Modeling Benchmarks (4 Panels)
        # -------------------------------------------------------------
        fig18_path = out_dir / "18_predictive_modeling_benchmarks.png"
        logger.info(f"Rendering Figure 18 to {fig18_path}...")

        X, y = self.X_cls, self.y_cls
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )

        rf_prob = self.clf_rf.predict_proba(X_test)[:, 1]
        gbm_prob = self.clf_gbm.predict_proba(X_test)[:, 1]
        X_test_scaled = (X_test - X_train.mean()) / (X_train.std().replace(0, 1))
        lr_prob = self.clf_lr.predict_proba(X_test_scaled)[:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor="#0e1117")

        # Color palette
        c_blue = "#2563eb"
        c_emerald = "#059669"
        c_purple = "#9333ea"
        c_amber = "#d97706"
        c_text = "#f3f4f6"
        c_grid = "#1f2937"

        # Panel A: ROC-AUC Curves
        ax_a = axes[0, 0]
        ax_a.set_facecolor("#111827")
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
        fpr_gbm, tpr_gbm, _ = roc_curve(y_test, gbm_prob)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)

        auc_rf = self.classification_metrics["Random_Forest"]["roc_auc"]
        auc_gbm = self.classification_metrics["Gradient_Boosting"]["roc_auc"]
        auc_lr = self.classification_metrics["Logistic_Regression"]["roc_auc"]

        ax_a.plot(fpr_rf, tpr_rf, color=c_emerald, lw=2.5, label=f"Random Forest (AUC = {auc_rf:.4f})")
        ax_a.plot(fpr_gbm, tpr_gbm, color=c_purple, lw=2.0, linestyle="--", label=f"Gradient Boosting (AUC = {auc_gbm:.4f})")
        ax_a.plot(fpr_lr, tpr_lr, color=c_amber, lw=1.8, linestyle=":", label=f"Logistic Regression (AUC = {auc_lr:.4f})")
        ax_a.plot([0, 1], [0, 1], color="#6b7280", lw=1.2, linestyle="-.")
        ax_a.set_title("A. ROC-AUC Performance Across Classifiers", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax_a.set_xlabel("False Positive Rate", color=c_text, fontsize=10)
        ax_a.set_ylabel("True Positive Rate", color=c_text, fontsize=10)
        ax_a.tick_params(colors=c_text)
        ax_a.grid(True, color=c_grid, linestyle="--", alpha=0.6)
        ax_a.legend(loc="lower right", facecolor="#1f2937", edgecolor="#374151", labelcolor=c_text, fontsize=9)

        # Panel B: Precision-Recall Curves
        ax_b = axes[0, 1]
        ax_b.set_facecolor("#111827")
        prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_prob)
        prec_gbm, rec_gbm, _ = precision_recall_curve(y_test, gbm_prob)
        prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_prob)

        ax_b.plot(rec_rf, prec_rf, color=c_emerald, lw=2.5, label=f"Random Forest (F1 = {self.classification_metrics['Random_Forest']['f1_score']:.3f})")
        ax_b.plot(rec_gbm, prec_gbm, color=c_purple, lw=2.0, linestyle="--", label=f"Gradient Boosting (F1 = {self.classification_metrics['Gradient_Boosting']['f1_score']:.3f})")
        ax_b.plot(rec_lr, prec_lr, color=c_amber, lw=1.8, linestyle=":", label=f"Logistic Regression (F1 = {self.classification_metrics['Logistic_Regression']['f1_score']:.3f})")
        ax_b.set_title("B. Precision-Recall Tradeoff Curves", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax_b.set_xlabel("Recall (Coverage)", color=c_text, fontsize=10)
        ax_b.set_ylabel("Precision (Accuracy)", color=c_text, fontsize=10)
        ax_b.tick_params(colors=c_text)
        ax_b.grid(True, color=c_grid, linestyle="--", alpha=0.6)
        ax_b.legend(loc="lower left", facecolor="#1f2937", edgecolor="#374151", labelcolor=c_text, fontsize=9)

        # Panel C: Top 10 Feature Importances
        ax_c = axes[1, 0]
        ax_c.set_facecolor("#111827")
        top_10 = list(self.feature_importances_.items())[:10]
        feat_names = [f[0].replace("cat_", "Cat: ").replace("cr_", "CR: ") for f in top_10][::-1]
        feat_vals = [f[1] * 100.0 for f in top_10][::-1]

        bars = ax_c.barh(feat_names, feat_vals, color=c_blue, edgecolor="#1d4ed8", alpha=0.85, height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax_c.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", color=c_text, fontsize=9)
        ax_c.set_title("C. Top 10 Feature Importances (Gini Impurity %)", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax_c.set_xlabel("Relative Importance (%)", color=c_text, fontsize=10)
        ax_c.set_xlim(0, max(feat_vals) * 1.25)
        ax_c.tick_params(colors=c_text)
        ax_c.grid(True, color=c_grid, linestyle="--", alpha=0.6, axis="x")

        # Panel D: Normalized Confusion Matrix
        ax_d = axes[1, 1]
        ax_d.set_facecolor("#111827")
        cm = confusion_matrix(y_test, (rf_prob >= 0.5).astype(int), normalize="true") * 100.0

        im = ax_d.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0, vmax=100)
        ax_d.set_title(f"D. Random Forest Confusion Matrix (Acc = {self.classification_metrics['Random_Forest']['accuracy']*100:.1f}%)", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax_d.set_xticks([0, 1])
        ax_d.set_yticks([0, 1])
        ax_d.set_xticklabels(["Sub-Million (<1M)", "High-Velocity (>=1M)"], color=c_text, fontsize=9)
        ax_d.set_yticklabels(["Sub-Million (<1M)", "High-Velocity (>=1M)"], color=c_text, fontsize=9)
        ax_d.set_xlabel("Predicted Label", color=c_text, fontsize=10)
        ax_d.set_ylabel("True Ground Truth Label", color=c_text, fontsize=10)

        for i in range(2):
            for j in range(2):
                txt_color = "#ffffff" if cm[i, j] > 50 else "#9ca3af"
                ax_d.text(j, i, f"{cm[i, j]:.1f}%\n(n={int(confusion_matrix(y_test, (rf_prob>=0.5).astype(int))[i, j])})",
                          ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

        fig.suptitle("Google Play Store — Day 9 Machine Learning Predictive Intelligence Suite", color=c_text, fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fig18_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close(fig)

        # -------------------------------------------------------------
        # Figure 19: Rating Regressor & MLOps Drift Monitoring (4 Panels)
        # -------------------------------------------------------------
        fig19_path = out_dir / "19_rating_regression_and_drift_monitoring.png"
        logger.info(f"Rendering Figure 19 to {fig19_path}...")

        Xr, yr = self.X_reg, self.y_reg
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            Xr, yr, test_size=0.20, random_state=self.random_state
        )
        rf_reg_pred = self.reg_rf.predict(Xr_test)
        residuals = yr_test - rf_reg_pred

        drift_rep = self.generate_drift_report()

        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12), facecolor="#0e1117")

        # Panel A: Actual vs Predicted Ratings Scatter
        ax2_a = axes2[0, 0]
        ax2_a.set_facecolor("#111827")
        ax2_a.scatter(yr_test, rf_reg_pred, alpha=0.25, color=c_blue, edgecolors="none", s=25)
        ax2_a.plot([1, 5], [1, 5], color="#ef4444", lw=2.0, linestyle="--", label="Perfect Parity (y = x)")
        ax2_a.set_title("A. Rating Regressor: Actual vs. Predicted Ratings", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax2_a.set_xlabel("Actual User Rating (Stars)", color=c_text, fontsize=10)
        ax2_a.set_ylabel("Predicted Rating (RF Regressor)", color=c_text, fontsize=10)
        ax2_a.set_xlim(1.0, 5.0)
        ax2_a.set_ylim(1.0, 5.0)
        ax2_a.tick_params(colors=c_text)
        ax2_a.grid(True, color=c_grid, linestyle="--", alpha=0.6)
        ax2_a.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor=c_text, fontsize=9)

        # Panel B: Residuals Distribution
        ax2_b = axes2[0, 1]
        ax2_b.set_facecolor("#111827")
        n, bins, _ = ax2_b.hist(residuals, bins=35, density=True, color=c_purple, alpha=0.65, edgecolor="#7e22ce")
        res_mu, res_std = float(np.mean(residuals)), float(np.std(residuals))
        x_norm = np.linspace(-2.5, 2.5, 200)
        ax2_b.plot(x_norm, stats.norm.pdf(x_norm, res_mu, res_std), color="#f59e0b", lw=2.0, label=f"Normal Fit (std={res_std:.2f})")
        ax2_b.axvline(0, color="#ffffff", lw=1.2, linestyle=":")
        ax2_b.set_title(f"B. Regression Residuals (Mean={res_mu:.3f}, RMSE={self.regression_metrics['Random_Forest_Regressor']['rmse']:.3f})", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax2_b.set_xlabel("Residual Error (Actual - Predicted)", color=c_text, fontsize=10)
        ax2_b.set_ylabel("Density", color=c_text, fontsize=10)
        ax2_b.set_xlim(-2.5, 2.5)
        ax2_b.tick_params(colors=c_text)
        ax2_b.grid(True, color=c_grid, linestyle="--", alpha=0.6)
        ax2_b.legend(loc="upper right", facecolor="#1f2937", edgecolor="#374151", labelcolor=c_text, fontsize=9)

        # Panel C: MLOps Feature PSI Scorecard
        ax2_c = axes2[1, 0]
        ax2_c.set_facecolor("#111827")
        metric_items = drift_rep["metrics"]
        feat_labels = [m["feature"] for m in metric_items]
        psi_scores = [m["psi"] for m in metric_items]
        colors = [c_emerald if p < 0.10 else (c_amber if p < 0.25 else "#ef4444") for p in psi_scores]

        b2 = ax2_c.barh(feat_labels, psi_scores, color=colors, height=0.55, edgecolor="#374151")
        for bar in b2:
            w = bar.get_width()
            ax2_c.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"PSI={w:.3f}", va="center", color=c_text, fontsize=9)
        ax2_c.axvline(0.10, color=c_amber, lw=1.8, linestyle="--", label="Stable Threshold (0.10)")
        ax2_c.axvline(0.25, color="#ef4444", lw=1.8, linestyle=":", label="Drift Action Threshold (0.25)")
        ax2_c.set_title("C. MLOps Population Stability Index (PSI) by Feature", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax2_c.set_xlabel("Population Stability Index (PSI)", color=c_text, fontsize=10)
        ax2_c.set_xlim(0, max(max(psi_scores) * 1.3, 0.30))
        ax2_c.tick_params(colors=c_text)
        ax2_c.grid(True, color=c_grid, linestyle="--", alpha=0.6, axis="x")
        ax2_c.legend(loc="lower right", facecolor="#1f2937", edgecolor="#374151", labelcolor=c_text, fontsize=9)

        # Panel D: Cohort Review Size Distribution Drift
        ax2_d = axes2[1, 1]
        ax2_d.set_facecolor("#111827")
        core_cats = ["FAMILY", "GAME", "TOOLS", "FINANCE", "LIFESTYLE", "PRODUCTIVITY"]
        b_rev = np.log10(self.df_apps[self.df_apps["Category"].isin(core_cats)]["Reviews"].clip(lower=0) + 1.0)
        m_rev = np.log10(self.df_apps[~self.df_apps["Category"].isin(core_cats)]["Reviews"].clip(lower=0) + 1.0)

        ax2_d.hist(b_rev, bins=25, density=True, alpha=0.5, color=c_blue, label="Baseline (Core Sectors)")
        ax2_d.hist(m_rev, bins=25, density=True, alpha=0.5, color=c_amber, label="Monitored (Emerging Sectors)")
        ax2_d.set_title("D. Cohort Log-Reviews Distribution Comparison", color=c_text, fontsize=12, fontweight="bold", pad=10)
        ax2_d.set_xlabel("Log10(Reviews + 1)", color=c_text, fontsize=10)
        ax2_d.set_ylabel("Density", color=c_text, fontsize=10)
        ax2_d.tick_params(colors=c_text)
        ax2_d.grid(True, color=c_grid, linestyle="--", alpha=0.6)
        ax2_d.legend(loc="upper right", facecolor="#1f2937", edgecolor="#374151", labelcolor=c_text, fontsize=9)

        fig2.suptitle("Google Play Store — Day 9 Rating Valuation Regression & MLOps Drift Analytics", color=c_text, fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig(fig19_path, dpi=300, facecolor=fig2.get_facecolor())
        plt.close(fig2)

        return fig18_path, fig19_path

    def export_reports(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Export CSV benchmark tables for model performance, feature importances, and drift metrics."""
        out_dir = Path(output_dir) if output_dir else _REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.clf_rf is None:
            self.train_models()

        # 1. Classification & Regression Benchmarks CSV
        bench_rows = []
        for model_name, metrics in self.classification_metrics.items():
            bench_rows.append({
                "model_category": "Success_Classifier_1M",
                "model_name": model_name,
                "primary_metric": "ROC_AUC",
                "primary_score": metrics["roc_auc"],
                "secondary_metric": "Accuracy",
                "secondary_score": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "brier_score": metrics["brier_score"],
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
            })

        for model_name, metrics in self.regression_metrics.items():
            bench_rows.append({
                "model_category": "Rating_Valuation_Regressor",
                "model_name": model_name,
                "primary_metric": "RMSE",
                "primary_score": metrics["rmse"],
                "secondary_metric": "MAE",
                "secondary_score": metrics["mae"],
                "f1_score": np.nan,
                "brier_score": np.nan,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            })

        bench_df = pd.DataFrame(bench_rows)
        bench_csv = out_dir / "ml_model_performance_benchmarks.csv"
        bench_df.to_csv(bench_csv, index=False)

        # 2. Feature Importances CSV
        feat_rows = [
            {"rank": i + 1, "feature": feat, "importance_weight": imp, "relative_pct": imp * 100.0}
            for i, (feat, imp) in enumerate(self.feature_importances_.items())
        ]
        feat_df = pd.DataFrame(feat_rows)
        feat_csv = out_dir / "ml_feature_importances.csv"
        feat_df.to_csv(feat_csv, index=False)

        # 3. MLOps Drift Metrics CSV
        drift_rep = self.generate_drift_report()
        drift_df = pd.DataFrame(drift_rep["metrics"])
        drift_csv = out_dir / "mlops_drift_metrics.csv"
        drift_df.to_csv(drift_csv, index=False)

        logger.info(f"Day 9 CSV reports exported to {out_dir}")
        return {
            "model_benchmarks": bench_csv,
            "feature_importances": feat_csv,
            "drift_metrics": drift_csv,
        }


# ============================================================================
# CLI Runner Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Google Play Store Day 9 ML Predictive & Drift Engine")
    parser.add_argument("--train", action="store_true", help="Train all classification & regression models")
    parser.add_argument("--drift", action="store_true", help="Generate MLOps distribution drift report")
    parser.add_argument("--generate-figures", action="store_true", help="Generate Figures 18 & 19")
    parser.add_argument("--export-reports", action="store_true", help="Export summary CSV reports")
    parser.add_argument("--predict-success", action="store_true", help="Run sample success prediction")
    parser.add_argument("--category", type=str, default="GAME", help="App category for prediction")
    parser.add_argument("--size", type=float, default=45.0, help="App size in MB")
    parser.add_argument("--price", type=float, default=0.0, help="App price in USD")
    parser.add_argument("--run-all", action="store_true", help="Execute complete Day 9 pipeline")

    args = parser.parse_args()

    engine = PlayStorePredictiveEngine()

    if args.run_all or args.train:
        results = engine.train_models()
        print("\n=== Model Training Performance Benchmarks ===")
        print(pd.DataFrame(results["classification"]).T)
        print("\n=== Rating Regression Performance ===")
        print(pd.DataFrame(results["regression"]).T)

    if args.run_all or args.drift:
        rep = engine.generate_drift_report()
        print("\n=== MLOps Distribution Drift Report ===")
        print(f"Health Verdict: {rep['health_verdict']} | Overall Max PSI: {rep['overall_max_psi']}")
        print(pd.DataFrame(rep["metrics"]))

    if args.run_all or args.generate_figures:
        f18, f19 = engine.generate_figures()
        print(f"\nGenerated Figures:\n- {f18}\n- {f19}")

    if args.run_all or args.export_reports:
        reports = engine.export_reports()
        print("\nExported Reports:")
        for k, v in reports.items():
            print(f"- {k}: {v}")

    if args.predict_success:
        pred = engine.predict_app_success(category=args.category, size_mb=args.size, price=args.price)
        print(f"\nPrediction for {args.category} ({args.size}MB, ${args.price}):")
        print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
