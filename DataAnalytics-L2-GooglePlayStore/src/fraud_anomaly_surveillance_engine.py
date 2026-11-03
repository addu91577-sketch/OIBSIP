"""App Store Fraud, Anomaly & Review Manipulation Surveillance Engine.

Day 14 Milestone for Google Play Store Market Intelligence & Decision Science Platform.
Provides:
1. Benford's Law First-Digit Auditing & Goodness-of-Fit Engine:
   - Theoretical logarithmic distribution testing: P(d) = log10(1 + 1/d) for d in 1..9.
   - Chi-Square (chi2) test, Kolmogorov-Smirnov (KS) distance, and p-value validation.
   - Category-level Benford conformance profiling across review counts and install volumes.
2. Review Sentiment-Rating Incongruity & Bot Manipulation Mining:
   - Dissonance metric: |normalized_rating - normalized_sentiment|.
   - Identification of fake 5-star complaints (paid review farms) and 1-star brigading (review bombing).
   - Shannon Lexical Entropy H(X) = -sum(p * log2(p)) and vocabulary repetition ratios for bot detection.
3. Multivariate Outlier & Anomaly Isolation (Mahalanobis Distance with Regularization):
   - Multi-dimensional profiling across Log-Installs, Log-Reviews, Rating, Size_MB, and Price.
   - Robust regularized category covariance and chi-square anomaly quantile mapping.
   - Specific outlier reason attribution (e.g. Ghost App, Install Farm Spike, Pricing Extortion).
4. Android Dangerous Permission & Bloat Risk Profiling:
   - Category-baseline permission expectations vs actual application footprints.
   - Over-privileged permission scoring and APK binary bloat penalties.
5. Composite App Trust & Integrity Scorecard (ATIS: 0-100) & Prescriptive Compliance Playbook:
   - Holistic 4-pillar risk synthesis into Trust Tiers:
     TIER_A_VERIFIED_ORGANIC, TIER_B_NORMAL_TRUST, TIER_C_ELEVATED_SUSPICION, TIER_D_CRITICAL_FRAUD_ALERT.
   - Actionable store compliance enforcement: VERIFIED_CLEAR, ENHANCED_TELEMETRY,
     TRUST_AND_SAFETY_AUDIT, STORE_TAKEDOWN_RECOMMENDED.
6. Publication Visualizations (Figures 28 & 29) and Analytical CSV Reports.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import logging
import math
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
import scipy.stats as stats

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
# Benford's Law Theoretical Constants
# ============================================================================

BENFORD_THEORETICAL: Dict[int, float] = {
    d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)
}

# Empirical Category Baseline Permission Weights (Danger Index 0-10)
CATEGORY_PERMISSION_BENCHMARKS: Dict[str, float] = {
    "GAME": 3.5,
    "TOOLS": 6.8,
    "COMMUNICATION": 7.5,
    "SOCIAL": 7.2,
    "FINANCE": 5.8,
    "HEALTH_AND_FITNESS": 4.5,
    "MAPS_AND_NAVIGATION": 8.0,
    "PHOTOGRAPHY": 6.2,
    "SHOPPING": 4.8,
    "BUSINESS": 4.0,
    "PRODUCTIVITY": 4.2,
    "PERSONALIZATION": 5.5,
    "LIFESTYLE": 3.8,
    "ENTERTAINMENT": 3.6,
    "EDUCATION": 2.5,
    "BOOKS_AND_REFERENCE": 2.2,
    "DEFAULT": 4.0
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class BenfordTestResult:
    """Statistical evaluation of first-digit distribution against Benford's Law."""
    metric_name: str
    sample_size: int
    chi2_stat: float
    p_value: float
    ks_stat: float
    is_conformant: bool
    conformance_tier: str
    observed_dist: Dict[int, float]
    expected_dist: Dict[int, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewDissonanceResult:
    """App-level review sentiment vs star rating dissonance and bot analysis."""
    app_name: str
    review_count: int
    mean_polarity: float
    polarity_std: float
    star_rating: float
    dissonance_score: float
    incongruent_review_ratio: float
    lexical_entropy: float
    bot_suspicion_tier: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultivariateAnomalyResult:
    """Multivariate Mahalanobis outlier detection result."""
    app_name: str
    category: str
    mahalanobis_distance: float
    anomaly_score: float
    is_outlier: bool
    outlier_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppTrustScoreResult:
    """Composite App Trust & Integrity scorecard with audit triggers."""
    app_name: str
    category: str
    rating: float
    reviews: int
    installs: int
    price: float
    trust_score: float
    trust_tier: str
    benford_penalty: float
    dissonance_penalty: float
    anomaly_penalty: float
    security_penalty: float
    compliance_action: str
    audit_triggers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CategoryIntegrityBenchmark:
    """Category-level aggregation of integrity metrics."""
    category: str
    app_count: int
    mean_trust_score: float
    anomaly_rate_pct: float
    benford_chi2: float
    benford_pval: float
    integrity_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Core Fraud & Anomaly Surveillance Engine
# ============================================================================

class PlayStoreFraudSurveillanceEngine:
    """Enterprise Trust, Safety & Anomaly Detection Engine for Google Play Store."""

    def __init__(
        self,
        apps_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None,
        auto_load: bool = True
    ):
        """Initialize engine with Play Store applications and review sentiment data."""
        self._apps_df: Optional[pd.DataFrame] = None
        self._reviews_df: Optional[pd.DataFrame] = None
        self._category_models: Dict[str, Dict[str, Any]] = {}
        self._app_name_index: Dict[str, int] = {}
        self._review_app_groups: Optional[pd.core.groupby.generic.DataFrameGroupBy] = None

        if apps_df is not None:
            self._apps_df = apps_df.copy()
        if reviews_df is not None:
            self._reviews_df = reviews_df.copy()

        if auto_load and (self._apps_df is None or self._reviews_df is None):
            self._load_datasets()

        if self._apps_df is not None:
            self._build_indices()
            self._fit_multivariate_category_baselines()

    def _load_datasets(self) -> None:
        """Load cleaned Play Store datasets from disk."""
        cleaned_apps_path = _DATA_DIR / "cleaned_googleplaystore.csv"
        raw_apps_path = _DATA_DIR / "googleplaystore.csv"
        cleaned_reviews_path = _DATA_DIR / "cleaned_user_reviews.csv"
        raw_reviews_path = _DATA_DIR / "googleplaystore_user_reviews.csv"

        if self._apps_df is None:
            if cleaned_apps_path.exists():
                self._apps_df = pd.read_csv(cleaned_apps_path)
            elif raw_apps_path.exists():
                from src.data_cleaning import clean_playstore_apps
                self._apps_df = clean_playstore_apps(pd.read_csv(raw_apps_path))
            else:
                raise FileNotFoundError(f"Neither {cleaned_apps_path} nor {raw_apps_path} exists.")

        if self._reviews_df is None:
            if cleaned_reviews_path.exists():
                self._reviews_df = pd.read_csv(cleaned_reviews_path)
            elif raw_reviews_path.exists():
                from src.data_cleaning import clean_user_reviews
                self._reviews_df = clean_user_reviews(pd.read_csv(raw_reviews_path))
            else:
                logger.warning("Reviews dataset not found. Proceeding with empty reviews.")
                self._reviews_df = pd.DataFrame(columns=[
                    "App", "Translated_Review", "Sentiment", "Sentiment_Polarity", "Sentiment_Subjectivity"
                ])

    def _build_indices(self) -> None:
        """Index applications by lowercase sanitized name for fast O(1) lookups."""
        if self._apps_df is not None and not self._apps_df.empty:
            self._apps_df["App_Normalized"] = self._apps_df["App"].astype(str).str.strip().str.lower()
            self._app_name_index = {
                name: idx for idx, name in enumerate(self._apps_df["App_Normalized"])
            }

        if self._reviews_df is not None and not self._reviews_df.empty:
            self._reviews_df["App_Normalized"] = self._reviews_df["App"].astype(str).str.strip().str.lower()
            self._review_app_groups = self._reviews_df.groupby("App_Normalized")

    def _fit_multivariate_category_baselines(self) -> None:
        """Compute category-specific regularized mean vector and inverted covariance matrix."""
        df = self._apps_df.copy()
        features = ["log_installs", "log_reviews", "rating_norm", "size_norm", "log_price"]

        df["log_installs"] = np.log10(np.maximum(df["Installs"].astype(float), 1.0))
        df["log_reviews"] = np.log10(np.maximum(df["Reviews"].astype(float), 1.0))
        mean_rating = df["Rating"].mean() if not df["Rating"].isna().all() else 4.17
        df["rating_norm"] = df["Rating"].fillna(mean_rating)
        size_col = "Size_MB" if "Size_MB" in df.columns else "Size"
        size_median = df[size_col].median() if size_col in df.columns else 20.0
        df["size_norm"] = pd.to_numeric(df[size_col], errors="coerce").fillna(size_median)
        df["log_price"] = np.log10(np.maximum(df["Price"].astype(float), 0.0) + 1.0)

        global_cov = np.cov(df[features].values, rowvar=False)
        global_mean = np.mean(df[features].values, axis=0)
        # Regularize
        eye = np.eye(len(features))
        global_cov_reg = global_cov + 1e-4 * np.trace(global_cov) * eye
        global_inv_cov = np.linalg.pinv(global_cov_reg)

        self._category_models["GLOBAL"] = {
            "mean": global_mean,
            "inv_cov": global_inv_cov,
            "features": features
        }

        for cat, group in df.groupby("Category"):
            if len(group) >= 15:
                mat = group[features].values
                c_mean = np.mean(mat, axis=0)
                c_cov = np.cov(mat, rowvar=False)
                c_cov_reg = c_cov + 1e-3 * np.trace(c_cov) * eye
                c_inv_cov = np.linalg.pinv(c_cov_reg)
                self._category_models[cat] = {
                    "mean": c_mean,
                    "inv_cov": c_inv_cov,
                    "features": features
                }
            else:
                self._category_models[cat] = self._category_models["GLOBAL"]

    # ========================================================================
    # 1. Benford's Law First-Digit Auditing
    # ========================================================================

    @staticmethod
    def extract_first_digits(series: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        """Extract the leading significant digit (1-9) for a sequence of positive numbers."""
        digits = []
        for val in series:
            try:
                fval = float(val)
                if fval > 0 and not math.isnan(fval) and not math.isinf(fval):
                    s = f"{fval:.10e}".lstrip("0.")
                    for ch in s:
                        if ch in "123456789":
                            digits.append(int(ch))
                            break
            except (ValueError, TypeError):
                continue
        return np.array(digits, dtype=int)

    def test_benfords_law(
        self,
        values: Union[pd.Series, np.ndarray, List[float]],
        metric_name: str = "Values"
    ) -> BenfordTestResult:
        """Execute Chi-Square and Kolmogorov-Smirnov Goodness-of-Fit tests against Benford's Law."""
        digits = self.extract_first_digits(values)
        n = len(digits)

        if n < 10:
            # Fallback for insufficient sample size
            obs_dict = {d: 1.0 / 9.0 for d in range(1, 10)}
            return BenfordTestResult(
                metric_name=metric_name,
                sample_size=n,
                chi2_stat=0.0,
                p_value=1.0,
                ks_stat=0.0,
                is_conformant=True,
                conformance_tier="ORGANIC_BENFORD_CONFORMANT",
                observed_dist=obs_dict,
                expected_dist=BENFORD_THEORETICAL
            )

        counter = Counter(digits)
        observed_counts = np.array([counter.get(d, 0) for d in range(1, 10)], dtype=float)
        expected_probs = np.array([BENFORD_THEORETICAL[d] for d in range(1, 10)], dtype=float)
        expected_counts = expected_probs * n

        # Chi-Square statistic
        chi2_val = np.sum((observed_counts - expected_counts) ** 2 / np.maximum(expected_counts, 1e-9))
        p_val = float(1.0 - stats.chi2.cdf(chi2_val, df=8))

        # KS distance
        cum_obs = np.cumsum(observed_counts) / n
        cum_exp = np.cumsum(expected_probs)
        ks_val = float(np.max(np.abs(cum_obs - cum_exp)))

        # Tiering
        if p_val >= 0.05:
            tier = "ORGANIC_BENFORD_CONFORMANT"
            is_conf = True
        elif p_val >= 0.01:
            tier = "MODERATE_BENFORD_DIVERGENCE"
            is_conf = True
        else:
            tier = "CRITICAL_BENFORD_ANOMALY"
            is_conf = False

        obs_dict = {d: float(observed_counts[d - 1] / n) for d in range(1, 10)}

        return BenfordTestResult(
            metric_name=metric_name,
            sample_size=n,
            chi2_stat=round(float(chi2_val), 4),
            p_value=round(float(p_val), 6),
            ks_stat=round(float(ks_val), 4),
            is_conformant=is_conf,
            conformance_tier=tier,
            observed_dist=obs_dict,
            expected_dist=BENFORD_THEORETICAL
        )

    def audit_category_benford(self, category: Optional[str] = None) -> Dict[str, BenfordTestResult]:
        """Audit Benford conformance for Reviews and Installs across a category or entire catalog."""
        df = self._apps_df
        if category and category.upper() in df["Category"].values:
            subset = df[df["Category"] == category.upper()]
        else:
            subset = df

        reviews_test = self.test_benfords_law(subset["Reviews"].dropna(), metric_name=f"{category or 'All'}_Reviews")
        installs_test = self.test_benfords_law(subset["Installs"].dropna(), metric_name=f"{category or 'All'}_Installs")

        return {
            "reviews_benford": reviews_test,
            "installs_benford": installs_test
        }

    # ========================================================================
    # 2. Review Sentiment-Rating Dissonance & Bot Entropy
    # ========================================================================

    @staticmethod
    def calculate_lexical_entropy(texts: List[str]) -> Tuple[float, float]:
        """Calculate Shannon lexical entropy and word repetition ratio over token sequences."""
        tokens = []
        for t in texts:
            if isinstance(t, str):
                words = re.findall(r"\b[a-zA-Z]{2,}\b", t.lower())
                tokens.extend(words)

        if not tokens:
            return 3.5, 0.5  # Neutral default

        total = len(tokens)
        counts = Counter(tokens)
        unique = len(counts)

        # Shannon entropy H(X) = -sum(p * log2(p))
        entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
        repetition_ratio = 1.0 - (unique / total)

        return round(float(entropy), 4), round(float(repetition_ratio), 4)

    def evaluate_review_dissonance(self, app_name: str) -> ReviewDissonanceResult:
        """Analyze sentiment vs star rating alignment, incongruity, and bot entropy for an app."""
        norm_name = app_name.strip().lower()

        # Lookup app rating
        app_rating = 4.17
        if norm_name in self._app_name_index:
            row = self._apps_df.iloc[self._app_name_index[norm_name]]
            if not pd.isna(row.get("Rating")):
                app_rating = float(row["Rating"])

        # Lookup reviews
        if self._review_app_groups is not None and norm_name in self._review_app_groups.groups:
            rev_sub = self._review_app_groups.get_group(norm_name)
        else:
            # Fallback for apps without review text
            return ReviewDissonanceResult(
                app_name=app_name,
                review_count=0,
                mean_polarity=0.0,
                polarity_std=0.0,
                star_rating=round(app_rating, 2),
                dissonance_score=0.15,
                incongruent_review_ratio=0.0,
                lexical_entropy=4.2,
                bot_suspicion_tier="ORGANIC_HUMAN"
            )

        polarities = rev_sub["Sentiment_Polarity"].dropna().values
        rev_count = len(polarities)

        if rev_count == 0:
            return ReviewDissonanceResult(
                app_name=app_name,
                review_count=0,
                mean_polarity=0.0,
                polarity_std=0.0,
                star_rating=round(app_rating, 2),
                dissonance_score=0.15,
                incongruent_review_ratio=0.0,
                lexical_entropy=4.2,
                bot_suspicion_tier="ORGANIC_HUMAN"
            )

        mean_pol = float(np.mean(polarities))
        pol_std = float(np.std(polarities)) if rev_count > 1 else 0.0

        # Normalized rating [0, 1] and polarity [0, 1]
        norm_r = (app_rating - 1.0) / 4.0
        norm_p = (mean_pol + 1.0) / 2.0
        dissonance = abs(norm_r - norm_p)

        # Incongruent reviews: 5-star app with negative text OR 1-star app with positive text
        incongruent_cnt = 0
        for p in polarities:
            if app_rating >= 4.3 and p < -0.15:
                incongruent_cnt += 1
            elif app_rating <= 2.5 and p > 0.25:
                incongruent_cnt += 1

        incongruent_ratio = incongruent_cnt / rev_count

        # Entropy of review texts
        texts = rev_sub["Translated_Review"].dropna().tolist()
        entropy, rep_ratio = self.calculate_lexical_entropy(texts)

        # Suspicion tiering
        if (dissonance > 0.45 or incongruent_ratio > 0.35) and (entropy < 2.8 or pol_std < 0.05):
            bot_tier = "HIGH_BOT_BURST"
        elif dissonance > 0.35 or incongruent_ratio > 0.25 or entropy < 3.2:
            bot_tier = "SUSPECT_INCONGRUENT"
        else:
            bot_tier = "ORGANIC_HUMAN"

        return ReviewDissonanceResult(
            app_name=app_name,
            review_count=rev_count,
            mean_polarity=round(mean_pol, 4),
            polarity_std=round(pol_std, 4),
            star_rating=round(app_rating, 2),
            dissonance_score=round(dissonance, 4),
            incongruent_review_ratio=round(incongruent_ratio, 4),
            lexical_entropy=entropy,
            bot_suspicion_tier=bot_tier
        )

    # ========================================================================
    # 3. Multivariate Outlier & Mahalanobis Anomaly Detection
    # ========================================================================

    def evaluate_multivariate_anomaly(
        self,
        app_name: str,
        category: Optional[str] = None,
        installs: Optional[int] = None,
        reviews: Optional[int] = None,
        rating: Optional[float] = None,
        size_mb: Optional[float] = None,
        price: Optional[float] = None
    ) -> MultivariateAnomalyResult:
        """Compute regularized Mahalanobis distance and anomaly score for an app."""
        norm_name = app_name.strip().lower()

        # Extract values
        if norm_name in self._app_name_index and installs is None:
            row = self._apps_df.iloc[self._app_name_index[norm_name]]
            cat = str(row.get("Category", "DEFAULT"))
            inst = float(row.get("Installs", 1000))
            revs = float(row.get("Reviews", 100))
            rate = float(row.get("Rating", 4.17)) if not pd.isna(row.get("Rating")) else 4.17
            sz = float(row.get("Size_MB", 20.0)) if not pd.isna(row.get("Size_MB")) else 20.0
            pr = float(row.get("Price", 0.0))
        else:
            cat = (category or "DEFAULT").upper()
            inst = float(installs or 10000)
            revs = float(reviews or 500)
            rate = float(rating or 4.2)
            sz = float(size_mb or 25.0)
            pr = float(price or 0.0)

        # Build feature vector
        vec = np.array([
            math.log10(max(inst, 1.0)),
            math.log10(max(revs, 1.0)),
            rate,
            sz,
            math.log10(max(pr, 0.0) + 1.0)
        ])

        model = self._category_models.get(cat, self._category_models["GLOBAL"])
        mean = model["mean"]
        inv_cov = model["inv_cov"]

        diff = vec - mean
        md2 = float(diff.T @ inv_cov @ diff)
        md = math.sqrt(max(md2, 0.0))

        # Anomaly score bounded [0, 1]
        # Chi2 with df=5: 95th percentile is 11.07, sqrt is 3.32; 99th percentile is 15.09, sqrt is 3.88
        anomaly_score = min(1.0, md / 5.0)
        is_outlier = md >= 3.8

        reasons = []
        if inst > 1e6 and revs < 20:
            reasons.append("Ghost App (High Installs with Disproportionately Low Reviews)")
        if revs > 1e5 and inst < 5e4:
            reasons.append("Review Farm Spike (Reviews Exceed Plausible Install Ratio)")
        if pr > 50.0 and rate > 4.5 and revs < 50:
            reasons.append("Pricing Extortion Anomaly (Extreme Price with Fabricated Rating)")
        if sz > 150.0 and inst < 1000:
            reasons.append("Extreme Binary Bloat with Minimal Adoption")
        if md >= 4.5 and not reasons:
            reasons.append("Severe Multivariate Metric Deviation from Category Baseline")

        return MultivariateAnomalyResult(
            app_name=app_name,
            category=cat,
            mahalanobis_distance=round(md, 4),
            anomaly_score=round(anomaly_score, 4),
            is_outlier=is_outlier,
            outlier_reasons=reasons
        )

    # ========================================================================
    # 4. Android Dangerous Permission & Bloat Risk Profiling
    # ========================================================================

    def evaluate_security_risk(
        self,
        category: str,
        size_mb: float,
        price: float
    ) -> Tuple[float, List[str]]:
        """Calculate permission bloat risk penalty and flagged security concerns."""
        cat_upper = category.upper()
        base_perm = CATEGORY_PERMISSION_BENCHMARKS.get(cat_upper, CATEGORY_PERMISSION_BENCHMARKS["DEFAULT"])

        penalty = 0.0
        flags = []

        # Size bloat risk: APK > 80MB in lightweight categories
        if cat_upper in ["TOOLS", "BOOKS_AND_REFERENCE", "PRODUCTIVITY"] and size_mb > 70.0:
            penalty += 12.0
            flags.append("Abnormal APK Binary Bloat in Low-Footprint Category")
        elif size_mb > 120.0:
            penalty += 8.0
            flags.append("Heavyweight APK Footprint (>120 MB)")

        # Category permission baseline risk
        if base_perm >= 7.0:
            penalty += 10.0
            flags.append(f"High-Risk Permission Surface Category ({cat_upper})")
        elif base_perm >= 5.0:
            penalty += 5.0

        # Excessive pricing risk
        if price >= 100.0:
            penalty += 15.0
            flags.append(f"Suspicious Premium Pricing (${price:.2f})")
        elif price >= 20.0:
            penalty += 6.0

        return round(penalty, 2), flags

    # ========================================================================
    # 5. Composite App Trust & Integrity Scorecard
    # ========================================================================

    def audit_app_trust(
        self,
        app_name: str,
        custom_category: Optional[str] = None,
        custom_rating: Optional[float] = None,
        custom_reviews: Optional[int] = None,
        custom_installs: Optional[int] = None,
        custom_price: Optional[float] = None,
        custom_size: Optional[float] = None
    ) -> AppTrustScoreResult:
        """Synthesize Benford, Dissonance, Mahalanobis Anomaly, and Security into ATIS (0-100)."""
        norm_name = app_name.strip().lower()

        # Lookup or default app attributes
        if norm_name in self._app_name_index and custom_category is None:
            row = self._apps_df.iloc[self._app_name_index[norm_name]]
            cat = str(row.get("Category", "DEFAULT"))
            rat = float(row.get("Rating", 4.17)) if not pd.isna(row.get("Rating")) else 4.17
            rev = int(float(row.get("Reviews", 100)))
            inst = int(float(row.get("Installs", 1000)))
            pr = float(row.get("Price", 0.0))
            sz = float(row.get("Size_MB", 20.0)) if not pd.isna(row.get("Size_MB")) else 20.0
        else:
            cat = (custom_category or "DEFAULT").upper()
            rat = float(custom_rating or 4.2)
            rev = int(custom_reviews or 500)
            inst = int(custom_installs or 10000)
            pr = float(custom_price or 0.0)
            sz = float(custom_size or 25.0)

        # 1. Benford First-Digit Deviation Penalty
        first_d = int(str(max(rev, 1))[0])
        benford_expected_p = BENFORD_THEORETICAL.get(first_d, 0.1)
        # Digits 7, 8, 9 have lower probability (~5%); unnatural digit concentration adds small penalty
        benford_penalty = 10.0 * (0.35 - benford_expected_p) if benford_expected_p < 0.15 else 0.0

        # 2. Review Sentiment Dissonance
        diss_res = self.evaluate_review_dissonance(app_name=app_name)
        diss_penalty = diss_res.dissonance_score * 25.0
        if diss_res.bot_suspicion_tier == "HIGH_BOT_BURST":
            diss_penalty += 15.0
        elif diss_res.bot_suspicion_tier == "SUSPECT_INCONGRUENT":
            diss_penalty += 8.0

        # 3. Multivariate Anomaly
        anom_res = self.evaluate_multivariate_anomaly(
            app_name=app_name,
            category=cat,
            installs=inst,
            reviews=rev,
            rating=rat,
            size_mb=sz,
            price=pr
        )
        anom_penalty = anom_res.anomaly_score * 35.0
        if anom_res.is_outlier:
            anom_penalty += 10.0

        # 4. Security & Permission Bloat
        sec_penalty, sec_flags = self.evaluate_security_risk(category=cat, size_mb=sz, price=pr)

        # Synthesize ATIS (App Trust & Integrity Score 0-100)
        total_penalty = benford_penalty + diss_penalty + anom_penalty + sec_penalty
        trust_score = max(5.0, min(100.0, 100.0 - total_penalty))

        # Tiers & Compliance Actions
        triggers = []
        if benford_penalty > 2.0:
            triggers.append(f"Benford Low-Probability Leading Digit ({first_d})")
        if diss_res.bot_suspicion_tier != "ORGANIC_HUMAN":
            triggers.append(f"Review Incongruity / Bot Burst ({diss_res.bot_suspicion_tier})")
        triggers.extend(anom_res.outlier_reasons)
        triggers.extend(sec_flags)

        if trust_score >= 85.0:
            tier = "TIER_A_VERIFIED_ORGANIC"
            action = "VERIFIED_CLEAR"
        elif trust_score >= 70.0:
            tier = "TIER_B_NORMAL_TRUST"
            action = "ENHANCED_TELEMETRY"
        elif trust_score >= 50.0:
            tier = "TIER_C_ELEVATED_SUSPICION"
            action = "TRUST_AND_SAFETY_AUDIT"
        else:
            tier = "TIER_D_CRITICAL_FRAUD_ALERT"
            action = "STORE_TAKEDOWN_RECOMMENDED"

        if not triggers and tier != "TIER_A_VERIFIED_ORGANIC":
            triggers.append("Moderate Composite Metric Variance")

        return AppTrustScoreResult(
            app_name=app_name,
            category=cat,
            rating=round(rat, 2),
            reviews=rev,
            installs=inst,
            price=round(pr, 2),
            trust_score=round(trust_score, 1),
            trust_tier=tier,
            benford_penalty=round(benford_penalty, 2),
            dissonance_penalty=round(diss_penalty, 2),
            anomaly_penalty=round(anom_penalty, 2),
            security_penalty=round(sec_penalty, 2),
            compliance_action=action,
            audit_triggers=triggers
        )

    # ========================================================================
    # 6. Batch Catalog Auditing & Reporting
    # ========================================================================

    def audit_catalog(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Run trust & fraud surveillance audit across the entire Play Store catalog."""
        df = self._apps_df if sample_size is None else self._apps_df.head(sample_size)
        records = []

        for _, row in df.iterrows():
            app_name = str(row["App"])
            res = self.audit_app_trust(app_name=app_name)
            rec = {
                "App": res.app_name,
                "Category": res.category,
                "Rating": res.rating,
                "Reviews": res.reviews,
                "Installs": res.installs,
                "Price": res.price,
                "Trust_Score": res.trust_score,
                "Trust_Tier": res.trust_tier,
                "Benford_Penalty": res.benford_penalty,
                "Dissonance_Penalty": res.dissonance_penalty,
                "Anomaly_Penalty": res.anomaly_penalty,
                "Security_Penalty": res.security_penalty,
                "Compliance_Action": res.compliance_action,
                "Audit_Triggers": " | ".join(res.audit_triggers) if res.audit_triggers else "None"
            }
            records.append(rec)

        return pd.DataFrame(records)

    def generate_category_benchmarks(self) -> pd.DataFrame:
        """Generate category-level integrity, Benford conformity, and anomaly benchmarks."""
        catalog_audit = self.audit_catalog()
        rows = []

        for cat, group in catalog_audit.groupby("Category"):
            app_count = len(group)
            mean_trust = group["Trust_Score"].mean()
            anom_pct = (group["Trust_Tier"].isin(["TIER_C_ELEVATED_SUSPICION", "TIER_D_CRITICAL_FRAUD_ALERT"]).mean()) * 100.0

            # Run Benford test on reviews
            b_res = self.test_benfords_law(group["Reviews"], metric_name=cat)

            if mean_trust >= 80.0 and anom_pct < 10.0:
                status = "PRISTINE_INTEGRITY"
            elif mean_trust >= 70.0 and anom_pct < 20.0:
                status = "ACCEPTABLE_HEALTH"
            elif anom_pct >= 30.0:
                status = "HIGH_ANOMALY_CONCENTRATION"
            else:
                status = "MODERATE_RISK"

            rows.append({
                "Category": cat,
                "App_Count": app_count,
                "Mean_Trust_Score": round(mean_trust, 2),
                "Anomaly_Rate_Pct": round(anom_pct, 2),
                "Benford_Chi2": b_res.chi2_stat,
                "Benford_PVal": b_res.p_value,
                "Benford_Tier": b_res.conformance_tier,
                "Integrity_Status": status
            })

        return pd.DataFrame(rows).sort_values("Mean_Trust_Score", ascending=False).reset_index(drop=True)

    def get_top_suspicious_apps(self, n: int = 25) -> pd.DataFrame:
        """Retrieve top N highest-risk apps flagged for manual Trust & Safety compliance audit."""
        catalog_audit = self.audit_catalog()
        return catalog_audit.sort_values("Trust_Score", ascending=True).head(n).reset_index(drop=True)

    # ========================================================================
    # 7. Publication Visualizations (Figures 28 & 29)
    # ========================================================================

    def generate_publication_figures(self) -> Tuple[Path, Path]:
        """Generate publication-grade 300 DPI Figures 28 and 29 for executive & academic reporting."""
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        catalog_audit = self.audit_catalog()

        # --------------------------------------------------------------------
        # FIGURE 28: Benford's Law & Review Manipulation Forensic Analytics
        # --------------------------------------------------------------------
        fig28_path = _FIGURES_DIR / "28_benford_law_and_review_manipulation.png"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle(
            "Figure 28: Benford's Law Forensic Auditing & Review Sentiment Manipulation Analytics\n"
            "Google Play Store Trust & Safety Anomaly Surveillance Engine",
            fontsize=15, fontweight="bold", y=0.98
        )

        # Panel A: Benford's Law Distribution (Empirical vs Theoretical)
        ax_a = axes[0, 0]
        benford_res = self.test_benfords_law(self._apps_df["Reviews"].dropna(), metric_name="Catalog_Reviews")
        digits = list(range(1, 10))
        emp_freq = [benford_res.observed_dist.get(d, 0.0) * 100.0 for d in digits]
        theor_freq = [BENFORD_THEORETICAL[d] * 100.0 for d in digits]

        x = np.arange(len(digits))
        width = 0.35
        ax_a.bar(x - width/2, emp_freq, width, label="Empirical Reviews", color="#1f77b4", alpha=0.85)
        ax_a.bar(x + width/2, theor_freq, width, label="Benford Theoretical", color="#ff7f0e", alpha=0.85)
        ax_a.plot(x + width/2, theor_freq, "ro--", linewidth=1.5, alpha=0.7)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([str(d) for d in digits], fontweight="bold")
        ax_a.set_xlabel("First Significant Leading Digit (d)", fontweight="bold")
        ax_a.set_ylabel("Distribution Frequency (%)", fontweight="bold")
        ax_a.set_title(
            f"A. Benford First-Digit Test (chi2={benford_res.chi2_stat:.1f}, p={benford_res.p_value:.4f})\n"
            f"Conformance: {benford_res.conformance_tier}",
            fontweight="bold", fontsize=11
        )
        ax_a.legend(loc="upper right")
        ax_a.grid(True, linestyle="--", alpha=0.5)

        # Panel B: Sentiment-Rating Dissonance Matrix
        ax_b = axes[0, 1]
        sample_apps = self._apps_df.sample(min(300, len(self._apps_df)), random_state=42)
        diss_list = []
        for _, r in sample_apps.iterrows():
            d_res = self.evaluate_review_dissonance(str(r["App"]))
            diss_list.append({
                "Rating": d_res.star_rating,
                "Polarity": d_res.mean_polarity,
                "Dissonance": d_res.dissonance_score,
                "Tier": d_res.bot_suspicion_tier
            })
        df_diss = pd.DataFrame(diss_list)

        scatter_b = ax_b.scatter(
            df_diss["Rating"], df_diss["Polarity"],
            c=df_diss["Dissonance"], cmap="coolwarm", s=40, alpha=0.75, edgecolors="none"
        )
        cbar_b = plt.colorbar(scatter_b, ax=ax_b)
        cbar_b.set_label("Dissonance Score", fontweight="bold")
        ax_b.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_b.set_xlabel("App Star Rating (1.0 - 5.0)", fontweight="bold")
        ax_b.set_ylabel("Mean Review Sentiment Polarity (-1.0 to +1.0)", fontweight="bold")
        ax_b.set_title(
            "B. Review Sentiment-Rating Dissonance\n(Flagging 5-Star Paid Complaints & 1-Star Bombing)",
            fontweight="bold", fontsize=11
        )
        ax_b.grid(True, linestyle="--", alpha=0.5)

        # Panel C: Review Lexical Shannon Entropy vs App Rating
        ax_c = axes[1, 0]
        ent_list = []
        for _, r in sample_apps.iterrows():
            d_res = self.evaluate_review_dissonance(str(r["App"]))
            ent_list.append({"Rating": d_res.star_rating, "Entropy": d_res.lexical_entropy, "Tier": d_res.bot_suspicion_tier})
        df_ent = pd.DataFrame(ent_list)

        colors_map = {"ORGANIC_HUMAN": "#2ca02c", "SUSPECT_INCONGRUENT": "#ff7f0e", "HIGH_BOT_BURST": "#d62728"}
        for tier, grp in df_ent.groupby("Tier"):
            ax_c.scatter(
                grp["Rating"], grp["Entropy"],
                color=colors_map.get(tier, "#1f77b4"), label=tier, s=45, alpha=0.8
            )
        ax_c.axhline(3.0, color="red", linestyle=":", label="Bot Threshold (H < 3.0)")
        ax_c.set_xlabel("App Star Rating", fontweight="bold")
        ax_c.set_ylabel("Shannon Lexical Entropy H(X)", fontweight="bold")
        ax_c.set_title(
            "C. Lexical Shannon Entropy vs Star Rating\n(Detecting Canned Bot Reviews & Keyword Stuffing)",
            fontweight="bold", fontsize=11
        )
        ax_c.legend(loc="lower left", fontsize=9)
        ax_c.grid(True, linestyle="--", alpha=0.5)

        # Panel D: App Trust & Integrity Score Distribution Across Top Categories
        ax_d = axes[1, 1]
        top_cats = catalog_audit["Category"].value_counts().head(8).index.tolist()
        box_data = [catalog_audit[catalog_audit["Category"] == c]["Trust_Score"].values for c in top_cats]
        ax_d.boxplot(box_data, tick_labels=[c[:10] for c in top_cats], patch_artist=True,
                     boxprops=dict(facecolor="#aec7e8", color="#1f77b4"))
        ax_d.axhline(70.0, color="orange", linestyle="--", label="Suspicion Line (70)")
        ax_d.axhline(50.0, color="red", linestyle="--", label="Critical Alert Line (50)")
        ax_d.set_xlabel("Top App Categories", fontweight="bold")
        ax_d.set_ylabel("App Trust & Integrity Score (ATIS 0-100)", fontweight="bold")
        ax_d.set_title(
            "D. Trust & Integrity Score Distribution (Top 8 Categories)",
            fontweight="bold", fontsize=11
        )
        ax_d.legend(loc="lower left", fontsize=9)
        ax_d.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fig28_path, dpi=300)
        plt.close(fig)
        logger.info("Saved Figure 28 to %s", fig28_path)

        # --------------------------------------------------------------------
        # FIGURE 29: Multivariate Outliers & Strategic Trust-Churn Quadrant
        # --------------------------------------------------------------------
        fig29_path = _FIGURES_DIR / "29_multivariate_anomalies_and_risk_quadrant.png"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle(
            "Figure 29: Multivariate Outliers, Churn-Trust Quadrant & Compliance Enforcement\n"
            "Google Play Store Trust & Safety Anomaly Surveillance Engine",
            fontsize=15, fontweight="bold", y=0.98
        )

        # Panel A: Multivariate Anomaly Scatter (Log Reviews vs Log Installs)
        ax_a = axes[0, 0]
        log_i = np.log10(np.maximum(catalog_audit["Installs"], 1))
        log_r = np.log10(np.maximum(catalog_audit["Reviews"], 1))
        scatter_a = ax_a.scatter(
            log_i, log_r, c=catalog_audit["Anomaly_Penalty"],
            cmap="viridis", s=30, alpha=0.6, edgecolors="none"
        )
        cbar_a = plt.colorbar(scatter_a, ax=ax_a)
        cbar_a.set_label("Anomaly Penalty", fontweight="bold")
        ax_a.set_xlabel("Log10 Installs", fontweight="bold")
        ax_a.set_ylabel("Log10 Reviews", fontweight="bold")
        ax_a.set_title(
            "A. Multivariate Mahalanobis Anomaly Landscape\n(Log Reviews vs Log Installs)",
            fontweight="bold", fontsize=11
        )
        ax_a.grid(True, linestyle="--", alpha=0.5)

        # Panel B: Trust Score vs Churn Risk Strategic Quadrant
        ax_b = axes[0, 1]
        # Generate synthetic churn proxy for quadrant if churn file missing
        churn_path = _REPORTS_DIR / "app_churn_risk_scorecard.csv"
        if churn_path.exists():
            df_churn = pd.read_csv(churn_path)
            merged_q = pd.merge(catalog_audit, df_churn[["App", "Churn_Risk_Score"]], on="App", how="inner")
        else:
            merged_q = catalog_audit.copy()
            merged_q["Churn_Risk_Score"] = 100.0 - merged_q["Trust_Score"] * 0.7

        ax_b.scatter(
            merged_q["Churn_Risk_Score"], merged_q["Trust_Score"],
            c=merged_q["Trust_Score"], cmap="RdYlGn", s=35, alpha=0.75
        )
        ax_b.axvline(50, color="gray", linestyle="--")
        ax_b.axhline(70, color="gray", linestyle="--")
        ax_b.text(20, 85, "SAFE & HIGH RETENTION\n(Core Ecosystem Champions)", fontweight="bold", color="green", fontsize=9)
        ax_b.text(65, 85, "HIGH CHURN / HIGH TRUST\n(Engagement Deficit)", fontweight="bold", color="#d95f02", fontsize=9)
        ax_b.text(20, 45, "LOW CHURN / SUSPICIOUS\n(Sticky Fake Ecosystems)", fontweight="bold", color="#7570b3", fontsize=9)
        ax_b.text(65, 45, "CRITICAL RISK / FRAUDULENT\n(Immediate Takedown)", fontweight="bold", color="red", fontsize=9)
        ax_b.set_xlabel("Churn Risk Score (0-100)", fontweight="bold")
        ax_b.set_ylabel("Trust & Integrity Score (ATIS 0-100)", fontweight="bold")
        ax_b.set_title("B. Strategic Trust vs Churn Surveillance Quadrant", fontweight="bold", fontsize=11)
        ax_b.grid(True, linestyle="--", alpha=0.5)

        # Panel C: Suspicious App Rate by Category (%)
        ax_c = axes[1, 0]
        cat_bench = self.generate_category_benchmarks().head(10)
        ax_c.barh(
            cat_bench["Category"], cat_bench["Anomaly_Rate_Pct"],
            color="#e377c2", alpha=0.85
        )
        ax_c.set_xlabel("Anomaly / Suspicion Rate (% of apps)", fontweight="bold")
        ax_c.set_ylabel("Category", fontweight="bold")
        ax_c.set_title("C. Category Suspicion & Anomaly Concentration Rate", fontweight="bold", fontsize=11)
        ax_c.grid(True, linestyle="--", alpha=0.5)

        # Panel D: Prescriptive Compliance Enforcement Distribution
        ax_d = axes[1, 1]
        action_counts = catalog_audit["Compliance_Action"].value_counts()
        colors_action = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
        ax_d.pie(
            action_counts.values, labels=action_counts.index,
            autopct="%1.1f%%", startangle=140,
            colors=colors_action[:len(action_counts)]
        )
        ax_d.set_title(
            f"D. Automated Store Enforcement Directives\n(Catalog Size: {len(catalog_audit):,} Apps)",
            fontweight="bold", fontsize=11
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fig29_path, dpi=300)
        plt.close(fig)
        logger.info("Saved Figure 29 to %s", fig29_path)

        return fig28_path, fig29_path

    # ========================================================================
    # 8. Analytical CSV Reports
    # ========================================================================

    def save_analytical_reports(self) -> Tuple[Path, Path, Path]:
        """Generate and save the 3 Day 14 analytical CSV reports."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        catalog_audit = self.audit_catalog()
        cat_benchmarks = self.generate_category_benchmarks()
        suspicious_apps = self.get_top_suspicious_apps(n=50)

        path_scorecard = _REPORTS_DIR / "app_fraud_anomaly_scorecard.csv"
        path_benchmarks = _REPORTS_DIR / "category_integrity_benchmarks.csv"
        path_suspicious = _REPORTS_DIR / "suspicious_apps_audit_report.csv"

        catalog_audit.to_csv(path_scorecard, index=False)
        cat_benchmarks.to_csv(path_benchmarks, index=False)
        suspicious_apps.to_csv(path_suspicious, index=False)

        logger.info("Saved %s (%d rows)", path_scorecard, len(catalog_audit))
        logger.info("Saved %s (%d rows)", path_benchmarks, len(cat_benchmarks))
        logger.info("Saved %s (%d rows)", path_suspicious, len(suspicious_apps))

        return path_scorecard, path_benchmarks, path_suspicious

    def run_pipeline(self) -> None:
        """Run complete Day 14 pipeline generating figures and reports."""
        logger.info("Starting Day 14 Fraud, Anomaly & Review Surveillance Pipeline...")
        t0 = time.time()
        self.save_analytical_reports()
        self.generate_publication_figures()
        logger.info("Day 14 Pipeline completed successfully in %.2f seconds.", time.time() - t0)


# ============================================================================
# CLI Execution Entry Point
# ============================================================================

if __name__ == "__main__":
    engine = PlayStoreFraudSurveillanceEngine()
    engine.run_pipeline()
