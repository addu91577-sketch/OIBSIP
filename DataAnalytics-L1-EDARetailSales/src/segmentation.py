"""
Customer Behavioral Micro-Segmentation & RFM Intelligence Engine.

Provides:
1. Customer-level RFM (Recency, Frequency, Monetary) feature extraction
2. Unsupervised K-Means behavioral clustering on customer purchasing profiles
3. Cluster profiling, centroid analysis, and silhouette score evaluation
4. Tailored commercial activation strategies per customer segment
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.analytics import RetailAnalyticsEngine


class CustomerSegmentationEngine:
    """
    Analyzes customer purchasing dynamics using RFM heuristics
    and unsupervised K-Means clustering.
    """

    def __init__(self, data_path: Optional[Path] = None, n_clusters: int = 4, random_state: int = 42):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self._customer_df: Optional[pd.DataFrame] = None
        self._cluster_profiles: Optional[List[Dict[str, Any]]] = None
        self._metrics: Dict[str, float] = {}

    def extract_rfm_features(self) -> pd.DataFrame:
        """
        Extracts transaction-level RFM and behavioral metrics:
        - Recency: Days from transaction to latest dataset transaction date
        - Frequency: Number of orders (1 in this benchmark dataset)
        - Monetary: Total basket spend ($)
        - Units: Total quantity
        - PricePerUnit: Discrete unit price ($)
        - Age: Shopper age
        """
        df = self.analytics.df.copy()
        reference_date = pd.to_datetime(df["Date"]).max()

        df["Recency"] = (reference_date - pd.to_datetime(df["Date"])).dt.days
        df["Frequency"] = 1
        df["Monetary"] = df["Total Amount"].astype(float)

        self._customer_df = df
        return self._customer_df

    def fit_clusters(self) -> Dict[str, Any]:
        """
        Fits K-Means clustering on standardized behavioral features:
        ['Recency', 'Monetary', 'Price per Unit', 'Quantity', 'Age']
        """
        if self._customer_df is None:
            self.extract_rfm_features()

        df = self._customer_df
        features = ["Recency", "Monetary", "Price per Unit", "Quantity", "Age"]
        X = df[features].values

        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        df["Cluster"] = cluster_labels

        # Calculate cluster evaluation metrics
        sil_score = float(np.round(silhouette_score(X_scaled, cluster_labels), 4))
        db_score = float(np.round(davies_bouldin_score(X_scaled, cluster_labels), 4))
        self._metrics = {
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "inertia": float(np.round(self.kmeans.inertia_, 2)),
            "n_clusters": self.n_clusters,
        }

        # Build cluster archetypes & profiles
        self._build_cluster_profiles()
        return self._metrics

    def _build_cluster_profiles(self):
        """Constructs rich profile descriptors and actionable strategies for each cluster."""
        df = self._customer_df
        profiles = []

        total_store_revenue = df["Monetary"].sum()
        total_customers = len(df)

        for cluster_id in range(self.n_clusters):
            c_df = df[df["Cluster"] == cluster_id]
            size = len(c_df)
            rev = float(c_df["Monetary"].sum())
            rev_share = float(np.round((rev / total_store_revenue) * 100, 2))
            cust_share = float(np.round((size / total_customers) * 100, 2))

            mean_spend = float(np.round(c_df["Monetary"].mean(), 2))
            mean_recency = float(np.round(c_df["Recency"].mean(), 1))
            mean_qty = float(np.round(c_df["Quantity"].mean(), 2))
            mean_price = float(np.round(c_df["Price per Unit"].mean(), 2))
            mean_age = float(np.round(c_df["Age"].mean(), 1))
            top_category = str(c_df["Product Category"].mode().iloc[0])
            gender_split = c_df["Gender"].value_counts(normalize=True).to_dict()

            # Dynamic archetype assignment based on spend and price tier
            if mean_spend >= 900:
                archetype = "💎 High-Rollers & Tech Whales"
                description = "Elite high-ticket buyers purchasing $300–$500 flagship items in multi-unit baskets."
                strategy = "Exclusive VIP concierge, early hardware drops, white-glove warranty perks."
            elif mean_spend >= 400:
                archetype = "👗 Core Lifestyle & Apparel Shoppers"
                description = "Mid-to-high ticket buyers consistently spending across Clothing and Electronics."
                strategy = "Cross-category bundle promotions, seasonal lookbooks, tiered spend threshold rewards."
            elif mean_price >= 200:
                archetype = "✨ High-Margin Premium Specialists"
                description = "Selective buyers targeting single premium luxury items ($300–$500) with moderate basket count."
                strategy = "Volume escalation bundles (buy 1 luxury item, get 2nd at 15% off) to increase units."
            else:
                archetype = "🏷️ Casual Budget Explorers"
                description = "Price-sensitive shoppers purchasing entry-tier items ($25–$50) with lower total spend."
                strategy = "Impulse add-on merchandising, low-cost essentials bundles, free shipping thresholds."

            profiles.append({
                "cluster_id": cluster_id,
                "archetype": archetype,
                "customer_count": size,
                "customer_share_pct": cust_share,
                "total_revenue": rev,
                "revenue_share_pct": rev_share,
                "mean_monetary": mean_spend,
                "mean_recency_days": mean_recency,
                "mean_quantity": mean_qty,
                "mean_unit_price": mean_price,
                "mean_age": mean_age,
                "top_category": top_category,
                "female_pct": float(np.round(gender_split.get("Female", 0.0) * 100, 1)),
                "male_pct": float(np.round(gender_split.get("Male", 0.0) * 100, 1)),
                "description": description,
                "recommended_strategy": strategy,
            })

        # Sort profiles by total revenue descending
        profiles.sort(key=lambda x: x["total_revenue"], reverse=True)
        self._cluster_profiles = profiles

    def get_summary(self) -> Dict[str, Any]:
        """Returns the complete segmentation summary report."""
        if self._cluster_profiles is None:
            self.fit_clusters()

        return {
            "metrics": self._metrics,
            "profiles": self._cluster_profiles,
            "total_customers": len(self._customer_df),
            "total_revenue": float(self._customer_df["Monetary"].sum()),
        }

    @property
    def clustered_df(self) -> pd.DataFrame:
        """Returns customer DataFrame enriched with RFM and cluster columns."""
        if self._customer_df is None:
            self.fit_clusters()
        return self._customer_df
