"""
Unit and integration tests for customer RFM behavioral micro-segmentation.
"""

import pytest
from src.segmentation import CustomerSegmentationEngine


def test_rfm_feature_extraction():
    """Verifies that RFM metrics are computed correctly across the dataset."""
    engine = CustomerSegmentationEngine()
    df = engine.extract_rfm_features()

    assert "Recency" in df.columns
    assert "Frequency" in df.columns
    assert "Monetary" in df.columns
    assert df["Recency"].min() >= 0
    assert df["Recency"].max() <= 365
    assert (df["Frequency"] == 1).all()
    assert df["Monetary"].sum() == 456000.0


def test_kmeans_clustering_shape_and_fit():
    """Verifies K-Means fit produces non-empty clusters for all clusters."""
    engine = CustomerSegmentationEngine(n_clusters=4, random_state=42)
    metrics = engine.fit_clusters()

    assert metrics["n_clusters"] == 4
    assert metrics["silhouette_score"] > 0.15
    assert metrics["davies_bouldin_score"] > 0.0

    clustered = engine.clustered_df
    assert "Cluster" in clustered.columns
    unique_clusters = clustered["Cluster"].unique()
    assert len(unique_clusters) == 4
    for c in range(4):
        assert (clustered["Cluster"] == c).sum() > 20


def test_cluster_profiles_completeness():
    """Verifies cluster profile descriptors, revenue sums, and strategy recommendations."""
    engine = CustomerSegmentationEngine(n_clusters=4, random_state=42)
    summary = engine.get_summary()

    profiles = summary["profiles"]
    assert len(profiles) == 4

    total_share = sum(p["revenue_share_pct"] for p in profiles)
    assert total_share == pytest.approx(100.0, abs=0.5)

    total_customers = sum(p["customer_count"] for p in profiles)
    assert total_customers == 1000

    for p in profiles:
        assert "archetype" in p
        assert "recommended_strategy" in p
        assert len(p["recommended_strategy"]) > 10
        assert p["mean_monetary"] > 0
