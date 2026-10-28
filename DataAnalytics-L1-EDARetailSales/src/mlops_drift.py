"""
Continuous MLOps Drift Monitoring & Data Integrity Guardrails.

Provides statistical drift detection algorithms:
1. Population Stability Index (PSI) for continuous and categorical distributions
2. Two-sample Kolmogorov-Smirnov (KS) test for continuous variables
3. Total Variation Distance (TVD) and Chi-Square goodness-of-fit for categorical features
4. Real-time batch drift evaluator with automated retraining alert thresholds
5. Synthetic production batch simulation for continuous testing and monitoring
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics import RetailAnalyticsEngine

EPSILON = 1e-6


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10
) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline (expected)
    and live production (actual) continuous distributions.

    PSI Interpretation:
    - PSI < 0.10: Minimal / No Drift (Stable)
    - 0.10 <= PSI < 0.25: Moderate Shift (Warning)
    - PSI >= 0.25: Significant Distribution Shift (Critical Drift / Retrain)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Handle single distinct value edge case
    if np.min(expected) == np.max(expected):
        if np.array_equal(expected, actual):
            return 0.0
        return 1.0

    # Determine quantile bins based on expected (baseline) distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Remove non-unique bin edges if continuous distribution has discrete spikes
    unique_bins = np.unique(bins)
    if len(unique_bins) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=unique_bins)
    actual_counts, _ = np.histogram(actual, bins=unique_bins)

    expected_pct = (expected_counts / len(expected)) + EPSILON
    actual_pct = (actual_counts / len(actual)) + EPSILON

    # Normalize after epsilon smoothing
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(max(0.0, psi_val), 4))


def calculate_categorical_psi(
    expected: pd.Series, actual: pd.Series
) -> Tuple[float, float]:
    """
    Calculates PSI and Total Variation Distance (TVD) for categorical variables.
    TVD = 0.5 * sum(|P(x) - Q(x)|), bounded in [0, 1].
    """
    all_categories = sorted(list(set(expected.unique()).union(set(actual.unique()))))

    expected_counts = expected.value_counts(normalize=True).to_dict()
    actual_counts = actual.value_counts(normalize=True).to_dict()

    psi_val = 0.0
    tvd_val = 0.0

    for cat in all_categories:
        e_pct = expected_counts.get(cat, 0.0) + EPSILON
        a_pct = actual_counts.get(cat, 0.0) + EPSILON
        psi_val += (a_pct - e_pct) * np.log(a_pct / e_pct)
        tvd_val += abs(a_pct - e_pct)

    tvd_val = 0.5 * tvd_val
    return float(np.round(max(0.0, psi_val), 4)), float(np.round(tvd_val, 4))


class DriftDetector:
    """
    Production MLOps drift detector monitoring retail sales features
    against the baseline golden dataset (1,000 reference records).
    """

    CONTINUOUS_FEATURES = ["Age", "Quantity", "Price per Unit", "Total Amount"]
    CATEGORICAL_FEATURES = ["Gender", "Product Category"]

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        if baseline_df is None:
            engine = RetailAnalyticsEngine()
            self.baseline_df = engine.df.copy()
        else:
            self.baseline_df = baseline_df.copy()

    def evaluate_feature_drift(
        self, feature: str, live_series: pd.Series
    ) -> Dict[str, Any]:
        """Evaluates statistical drift for an individual continuous or categorical feature."""
        baseline_series = self.baseline_df[feature]

        if feature in self.CONTINUOUS_FEATURES:
            base_arr = baseline_series.to_numpy(dtype=float)
            live_arr = live_series.to_numpy(dtype=float)

            psi = calculate_psi(base_arr, live_arr)
            ks_res = stats.ks_2samp(base_arr, live_arr)
            ks_stat = float(np.round(ks_res.statistic, 4))
            ks_p_val = float(np.round(ks_res.pvalue, 5))

            # Status determination based on Population Stability Index (PSI)
            if psi < 0.10:
                status = "STABLE"
                alert_level = "GREEN"
            elif psi < 0.25:
                status = "MODERATE_DRIFT"
                alert_level = "YELLOW"
            else:
                status = "CRITICAL_DRIFT"
                alert_level = "RED"

            return {
                "feature": feature,
                "type": "continuous",
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p_val,
                "baseline_mean": float(np.round(base_arr.mean(), 2)),
                "live_mean": float(np.round(live_arr.mean(), 2)),
                "baseline_std": float(np.round(base_arr.std(), 2)),
                "live_std": float(np.round(live_arr.std(), 2)),
                "status": status,
                "alert_level": alert_level,
            }

        elif feature in self.CATEGORICAL_FEATURES:
            psi, tvd = calculate_categorical_psi(baseline_series, live_series)

            if psi < 0.10 and tvd < 0.08:
                status = "STABLE"
                alert_level = "GREEN"
            elif psi >= 0.25 or tvd >= 0.20:
                status = "CRITICAL_DRIFT"
                alert_level = "RED"
            else:
                status = "MODERATE_DRIFT"
                alert_level = "YELLOW"

            return {
                "feature": feature,
                "type": "categorical",
                "psi": psi,
                "tvd": tvd,
                "baseline_distribution": baseline_series.value_counts(
                    normalize=True
                ).to_dict(),
                "live_distribution": live_series.value_counts(
                    normalize=True
                ).to_dict(),
                "status": status,
                "alert_level": alert_level,
            }
        else:
            raise ValueError(f"Unknown feature for drift analysis: {feature}")

    def run_drift_audit(self, live_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes an end-to-end drift audit across all features and provides
        model health diagnostics and retraining recommendations.
        """
        results: Dict[str, Any] = {}
        max_psi = 0.0
        critical_count = 0
        warning_count = 0

        for col in self.CONTINUOUS_FEATURES + self.CATEGORICAL_FEATURES:
            if col in live_df.columns:
                feat_eval = self.evaluate_feature_drift(col, live_df[col])
                results[col] = feat_eval
                max_psi = max(max_psi, feat_eval["psi"])
                if feat_eval["status"] == "CRITICAL_DRIFT":
                    critical_count += 1
                elif feat_eval["status"] == "MODERATE_DRIFT":
                    warning_count += 1

        # Global dataset verdict
        if critical_count > 0 or max_psi >= 0.25:
            global_status = "CRITICAL_DRIFT"
            retrain_required = True
            action_plan = (
                "Trigger automated pipeline retraining: Significant feature drift detected "
                "which may degrade VIP classifier (AUC) and Spend Regressor (R²)."
            )
        elif warning_count > 0 or max_psi >= 0.10:
            global_status = "MODERATE_WARNING"
            retrain_required = False
            action_plan = (
                "Monitor closely: Moderate distribution shift detected. "
                "Collect additional production logs before triggering model retraining."
            )
        else:
            global_status = "HEALTHY_STABLE"
            retrain_required = False
            action_plan = "No action required: Live distributions closely align with training baseline."

        return {
            "global_status": global_status,
            "retrain_recommended": retrain_required,
            "max_psi": float(np.round(max_psi, 4)),
            "critical_features_count": critical_count,
            "warning_features_count": warning_count,
            "sample_size": len(live_df),
            "action_plan": action_plan,
            "feature_audits": results,
        }

    def generate_synthetic_batch(
        self,
        scenario: str = "stable",
        n_samples: int = 300,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates realistic synthetic live batches to benchmark drift detection:
        - 'stable': Normal random sampling from baseline with slight noise
        - 'tech_surge': Heavy skew towards $500 Electronics and 3-4 item quantities
        - 'inflation': Catalog prices increase, driving up Total Amount
        - 'youth_pivot': Skew towards young adult (18-25) Clothing buyers
        """
        rng = np.random.default_rng(random_seed)
        df_base = self.baseline_df.sample(n=n_samples, replace=True, random_state=random_seed).copy()

        if scenario == "stable":
            # Direct non-drifted baseline subsample
            sample_n = min(n_samples, len(self.baseline_df))
            return self.baseline_df.sample(n=sample_n, replace=False, random_state=random_seed).copy()

        elif scenario == "tech_surge":
            # Significant shift towards high-end electronics
            df_surge = df_base.copy()
            df_surge["Product Category"] = rng.choice(
                ["Electronics", "Clothing", "Beauty"],
                size=n_samples,
                p=[0.75, 0.15, 0.10],
            )
            df_surge["Price per Unit"] = rng.choice(
                [300.0, 500.0, 50.0],
                size=n_samples,
                p=[0.40, 0.50, 0.10],
            )
            df_surge["Quantity"] = rng.choice([2, 3, 4], size=n_samples, p=[0.2, 0.5, 0.3])
            df_surge["Total Amount"] = df_surge["Quantity"] * df_surge["Price per Unit"]
            return df_surge

        elif scenario == "inflation":
            # Continuous upward price shift
            df_inf = df_base.copy()
            df_inf["Price per Unit"] = df_inf["Price per Unit"] * rng.uniform(1.3, 1.8, size=n_samples)
            df_inf["Total Amount"] = df_inf["Quantity"] * df_inf["Price per Unit"]
            return df_inf

        elif scenario == "youth_pivot":
            # Young adult demographic influx
            df_youth = df_base.copy()
            df_youth["Age"] = rng.integers(18, 26, size=n_samples)
            df_youth["Product Category"] = rng.choice(
                ["Clothing", "Beauty", "Electronics"],
                size=n_samples,
                p=[0.60, 0.30, 0.10],
            )
            return df_youth

        else:
            raise ValueError(f"Unknown drift scenario: {scenario}")
