"""Unit and regression tests for Day 4 Pricing, Sentiment & Interactive Dashboard Module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pricing_sentiment_analysis import (
    compute_pricing_distribution,
    compute_category_monetization,
    get_top_revenue_apps,
    compute_sentiment_scores,
    compute_sentiment_metrics,
    compute_category_sentiment_summary,
    compute_market_opportunity_matrix,
    generate_developer_recommendations,
    generate_all_day4_reports
)


@pytest.fixture(scope="module")
def playstore_df():
    """Load cleaned Play Store apps dataset."""
    data_path = Path(__file__).resolve().parent.parent / "Data" / "cleaned_googleplaystore.csv"
    assert data_path.exists(), f"Cleaned apps dataset missing at {data_path}"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def reviews_df():
    """Load cleaned user reviews dataset."""
    data_path = Path(__file__).resolve().parent.parent / "Data" / "cleaned_user_reviews.csv"
    assert data_path.exists(), f"Cleaned reviews dataset missing at {data_path}"
    return pd.read_csv(data_path)


def test_compute_pricing_distribution(playstore_df):
    """Verify descriptive statistics and tier segmentation for app pricing."""
    stats = compute_pricing_distribution(playstore_df)

    assert stats["total_apps"] == len(playstore_df)
    assert stats["free_apps"] + stats["paid_apps"] == stats["total_apps"]
    assert stats["free_apps"] == 8905
    assert stats["paid_apps"] == 754
    assert abs(stats["free_percentage"] - 92.19) < 0.1
    assert abs(stats["paid_percentage"] - 7.81) < 0.1

    # Pricing quantiles
    assert stats["paid_min"] == 0.99
    assert stats["paid_median"] == 2.99
    assert stats["paid_max"] == 400.0

    # Tiers check
    tier_sum = sum(t["count"] for t in stats["tier_breakdown"].values())
    assert tier_sum == stats["paid_apps"]

    # Luxury / Outlier apps (>= $30)
    assert stats["outlier_count_gte_30"] >= 15
    assert len(stats["outliers"]) == stats["outlier_count_gte_30"]


def test_compute_category_monetization(playstore_df):
    """Verify category-level revenue calculations and paid catalog metrics."""
    monetization = compute_category_monetization(playstore_df)

    assert len(monetization) == 33
    required_cols = [
        "Category", "Total_Apps", "Free_Apps", "Paid_Apps", "Paid_Share_Pct",
        "Total_Installs", "Estimated_Revenue", "Revenue_Share_Pct"
    ]
    for col in required_cols:
        assert col in monetization.columns

    # Verify total estimated revenue exceeds $200 Million
    total_rev = monetization["Estimated_Revenue"].sum()
    assert total_rev > 250_000_000.0

    # Top revenue categories should lead with FAMILY and GAME
    top_categories = monetization.head(5)["Category"].tolist()
    assert "FAMILY" in top_categories
    assert "GAME" in top_categories

    # Check revenue share percentages sum close to 100%
    assert abs(monetization["Revenue_Share_Pct"].sum() - 100.0) < 0.5


def test_get_top_revenue_apps(playstore_df):
    """Verify top grossing paid app identification and sorting."""
    top_apps = get_top_revenue_apps(playstore_df, n=15)

    assert len(top_apps) == 15
    assert (top_apps["Estimated_Revenue"] > 0).all()

    # Verify sorting order (strictly descending)
    revenues = top_apps["Estimated_Revenue"].tolist()
    assert revenues == sorted(revenues, reverse=True)

    # Minecraft must be present in top revenue apps
    app_names = top_apps["App"].tolist()
    assert any("Minecraft" in name for name in app_names)


def test_compute_sentiment_scores(reviews_df):
    """Verify VADER and TextBlob sentiment scoring integrity."""
    # Test on a small slice for speed
    sample_reviews = reviews_df.head(100)
    scored = compute_sentiment_scores(sample_reviews)

    assert "VADER_Compound" in scored.columns
    assert "VADER_Sentiment" in scored.columns
    assert "TextBlob_Polarity" in scored.columns

    # Check bounds
    assert (scored["VADER_Compound"] >= -1.0).all() and (scored["VADER_Compound"] <= 1.0).all()
    assert set(scored["VADER_Sentiment"].unique()).issubset({"Positive", "Neutral", "Negative"})


def test_compute_sentiment_metrics(reviews_df):
    """Verify multi-engine sentiment comparative metrics."""
    sample_reviews = reviews_df.head(200)
    scored = compute_sentiment_scores(sample_reviews)
    metrics = compute_sentiment_metrics(scored)

    assert metrics["total_reviews"] == 200
    assert "agreement_percentage" in metrics
    assert 50.0 <= metrics["agreement_percentage"] <= 90.0
    assert "vader_compound_mean" in metrics
    assert "textblob_polarity_mean" in metrics


def test_compute_category_sentiment_summary(playstore_df, reviews_df):
    """Verify category-level sentiment aggregation."""
    # Score a representative subset of reviews
    scored = compute_sentiment_scores(reviews_df.head(1000))
    summary = compute_category_sentiment_summary(scored, playstore_df)

    assert len(summary) > 0
    assert "Category" in summary.columns
    assert "Mean_VADER_Compound" in summary.columns
    assert "VADER_Positive_Pct" in summary.columns
    assert "VADER_Negative_Pct" in summary.columns

    # Compound scores should fall within [-1, 1]
    assert (summary["Mean_VADER_Compound"] >= -1.0).all()
    assert (summary["Mean_VADER_Compound"] <= 1.0).all()


def test_compute_market_opportunity_matrix(playstore_df, reviews_df):
    """Verify Market Opportunity Matrix synthesis and strategic quadrants."""
    monetization = compute_category_monetization(playstore_df)
    scored = compute_sentiment_scores(reviews_df.head(1000))
    sentiment = compute_category_sentiment_summary(scored, playstore_df)

    matrix = compute_market_opportunity_matrix(playstore_df, monetization, sentiment)

    assert len(matrix) == 33
    assert "Opportunity_Score" in matrix.columns
    assert "Strategic_Quadrant" in matrix.columns

    # Scores must be bounded [0, 100]
    assert (matrix["Opportunity_Score"] >= 0.0).all()
    assert (matrix["Opportunity_Score"] <= 100.0).all()

    # Scores must be sorted descending
    scores = matrix["Opportunity_Score"].tolist()
    assert scores == sorted(scores, reverse=True)

    # Valid quadrants
    valid_quadrants = {
        "Quadrant I: High Satisfaction / High Delight",
        "Quadrant II: High Rating / Latent Review Friction",
        "Quadrant III: High Review Sentiment / Structural Gaps",
        "Quadrant IV: Red Ocean / High User Friction"
    }
    assert set(matrix["Strategic_Quadrant"].unique()).issubset(valid_quadrants)


def test_generate_developer_recommendations(playstore_df, reviews_df):
    """Verify 3 actionable data-backed developer recommendations."""
    monetization = compute_category_monetization(playstore_df)
    scored = compute_sentiment_scores(reviews_df.head(1000))
    sentiment = compute_category_sentiment_summary(scored, playstore_df)
    matrix = compute_market_opportunity_matrix(playstore_df, monetization, sentiment)

    recs = generate_developer_recommendations(matrix)

    assert len(recs) == 3
    for rec in recs:
        assert "recommendation_id" in rec
        assert "title" in rec
        assert "category_targets" in rec
        assert "empirical_rationale" in rec
        assert "actionable_playbook" in rec


def test_generate_all_day4_reports(tmp_path, playstore_df, reviews_df):
    """Verify end-to-end Day 4 pipeline artifact generation."""
    data_dir = tmp_path / "Data"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    reports_dir.mkdir()

    # Save representative sample with matching apps and paid apps
    sample_reviews = reviews_df.head(200)
    reviewed_app_names = sample_reviews["App"].unique()
    matched_apps = playstore_df[playstore_df["App"].isin(reviewed_app_names)]
    paid_sample = playstore_df[playstore_df["Type"] == "Paid"].head(50)
    sample_apps = pd.concat([matched_apps, paid_sample, playstore_df.head(100)]).drop_duplicates(subset=["App"])

    sample_apps.to_csv(data_dir / "cleaned_googleplaystore.csv", index=False)
    sample_reviews.to_csv(data_dir / "cleaned_user_reviews.csv", index=False)

    results = generate_all_day4_reports(data_dir=str(data_dir), reports_dir=str(reports_dir))

    for file_path in results["generated_files"]:
        p = Path(file_path)
        assert p.exists(), f"Expected artifact {p.name} does not exist"
        assert p.stat().st_size > 0, f"Artifact {p.name} is empty"
