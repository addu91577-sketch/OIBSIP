"""
Retail Machine Learning Intelligence Engine.

Provides predictive modeling pipelines:
1. VIPWhaleClassifier: Predicts VIP high-spend transaction propensity (>= $1,000)
2. SpendRegressor: Predicts transaction gross spend with confidence intervals
3. Model serialization, automated calibration, and inference schemas.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.analytics import RetailAnalyticsEngine

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VIP_MODEL_FILE = MODELS_DIR / "vip_classifier.joblib"
SPEND_MODEL_FILE = MODELS_DIR / "spend_regressor.joblib"


class RetailMLManager:
    """Manages training, cross-validation, persistence, and real-time inference for ML models."""

    def __init__(self, data_path: Optional[Path] = None):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.vip_pipeline: Optional[Pipeline] = None
        self.spend_pipeline: Optional[Pipeline] = None
        self.vip_metrics: Dict[str, Any] = {}
        self.spend_metrics: Dict[str, Any] = {}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_data(self) -> pd.DataFrame:
        """Returns clean DataFrame ready for feature extraction."""
        return self.analytics.df.copy()

    def train_vip_classifier(self) -> Dict[str, Any]:
        """
        Trains and validates the VIP Whale Propensity Classifier.
        Target: is_vip (Total Amount >= $1,000).
        """
        df = self.prepare_data()
        X = df[["Age", "Gender", "Product Category", "Quantity", "Price per Unit"]]
        y = (df["Total Amount"] >= 1000).astype(int)

        categorical_features = ["Gender", "Product Category"]
        numeric_features = ["Age", "Quantity", "Price per Unit"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
                ("num", StandardScaler(), numeric_features),
            ]
        )

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

        # Fit on full dataset
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]

        acc = float(accuracy_score(y, y_pred))
        prec = float(precision_score(y, y_pred, zero_division=0))
        rec = float(recall_score(y, y_pred))
        f1 = float(f1_score(y, y_pred))
        roc_auc = float(roc_auc_score(y, y_prob))

        # Feature importances
        preprocessor_fitted = pipeline.named_steps["preprocessor"]
        cat_features_transformed = preprocessor_fitted.named_transformers_[
            "cat"
        ].get_feature_names_out(categorical_features)
        all_features = list(cat_features_transformed) + numeric_features
        importances = pipeline.named_steps["classifier"].feature_importances_
        feature_importance_dict = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(
                zip(all_features, importances), key=lambda x: x[1], reverse=True
            )
        }

        self.vip_pipeline = pipeline
        self.vip_metrics = {
            "model_type": "RandomForestClassifier(class_weight='balanced')",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "cv_mean_roc_auc": round(float(np.mean(cv_scores)), 4),
            "cv_std_roc_auc": round(float(np.std(cv_scores)), 4),
            "feature_importances": feature_importance_dict,
        }

        joblib.dump(
            {"pipeline": self.vip_pipeline, "metrics": self.vip_metrics},
            VIP_MODEL_FILE,
        )
        return self.vip_metrics

    def train_spend_regressor(self) -> Dict[str, Any]:
        """
        Trains and validates the Transaction Spend Regressor.
        Target: Total Amount ($).
        """
        df = self.prepare_data()
        X = df[["Age", "Gender", "Product Category", "Quantity", "Price per Unit"]]
        y = df["Total Amount"]

        categorical_features = ["Gender", "Product Category"]
        numeric_features = ["Age", "Quantity", "Price per Unit"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
                ("num", StandardScaler(), numeric_features),
            ]
        )

        reg = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", reg),
            ]
        )

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        self.spend_pipeline = pipeline
        self.spend_metrics = {
            "model_type": "GradientBoostingRegressor",
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
        }

        joblib.dump(
            {"pipeline": self.spend_pipeline, "metrics": self.spend_metrics},
            SPEND_MODEL_FILE,
        )
        return self.spend_metrics

    def load_or_train_models(self) -> None:
        """Loads serialized models if present, otherwise trains them automatically."""
        if VIP_MODEL_FILE.exists():
            try:
                vip_data = joblib.load(VIP_MODEL_FILE)
                self.vip_pipeline = vip_data.get("pipeline")
                self.vip_metrics = vip_data.get("metrics", {})
            except Exception:
                self.train_vip_classifier()
        else:
            self.train_vip_classifier()

        if SPEND_MODEL_FILE.exists():
            try:
                spend_data = joblib.load(SPEND_MODEL_FILE)
                self.spend_pipeline = spend_data.get("pipeline")
                self.spend_metrics = spend_data.get("metrics", {})
            except Exception:
                self.train_spend_regressor()
        else:
            self.train_spend_regressor()

    def predict_vip(
        self,
        age: int,
        gender: str,
        product_category: str,
        quantity: int,
        price_per_unit: float,
    ) -> Dict[str, Any]:
        """Real-time VIP Whale propensity inference."""
        if self.vip_pipeline is None:
            self.load_or_train_models()

        input_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Product Category": product_category,
                    "Quantity": quantity,
                    "Price per Unit": price_per_unit,
                }
            ]
        )

        prob = float(self.vip_pipeline.predict_proba(input_df)[0, 1])
        is_vip = bool(prob >= 0.5)

        if prob >= 0.80:
            risk_tier = "VERY_HIGH"
            badge = "👑 VIP Elite Tier"
        elif prob >= 0.50:
            risk_tier = "HIGH"
            badge = "⭐ High Conversion Propensity"
        elif prob >= 0.25:
            risk_tier = "MODERATE"
            badge = "🛒 Mid-Tier Consideration"
        else:
            risk_tier = "LOW"
            badge = "📦 Standard Budget Consumer"

        calculated_spend = quantity * price_per_unit

        return {
            "is_vip": is_vip,
            "vip_probability": round(prob, 4),
            "propensity_score_pct": round(prob * 100, 2),
            "tier_badge": badge,
            "risk_tier": risk_tier,
            "calculated_basket_spend": round(float(calculated_spend), 2),
            "threshold": 1000.0,
            "inputs": {
                "age": age,
                "gender": gender,
                "product_category": product_category,
                "quantity": quantity,
                "price_per_unit": price_per_unit,
            },
        }

    def predict_spend(
        self,
        age: int,
        gender: str,
        product_category: str,
        quantity: int,
        price_per_unit: float,
    ) -> Dict[str, Any]:
        """Real-time basket expenditure regression inference."""
        if self.spend_pipeline is None:
            self.load_or_train_models()

        input_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Product Category": product_category,
                    "Quantity": quantity,
                    "Price per Unit": price_per_unit,
                }
            ]
        )

        pred_spend = float(self.spend_pipeline.predict(input_df)[0])
        mae = self.spend_metrics.get("mae", 20.0)

        deterministic_amount = quantity * price_per_unit

        return {
            "predicted_spend": round(pred_spend, 2),
            "deterministic_catalog_spend": round(float(deterministic_amount), 2),
            "expected_range_min": round(max(0.0, pred_spend - (1.96 * mae)), 2),
            "expected_range_max": round(pred_spend + (1.96 * mae), 2),
            "model_r2_score": self.spend_metrics.get("r2_score", 0.0),
            "inputs": {
                "age": age,
                "gender": gender,
                "product_category": product_category,
                "quantity": quantity,
                "price_per_unit": price_per_unit,
            },
        }
