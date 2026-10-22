"""Unit and validation tests for Google Play Store Category, Ratings, and Size Analysis module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.category_ratings_analysis import (
    compute_category_distribution,
    compute_rating_statistics,
    compute_category_ratings,
    compute_size_installs_correlation,
    compute_stratified_correlations,
    generate_category_ratings_reports
)


@pytest.fixture(scope="module")
def playstore_df():
    """Load the cleaned Play Store dataset for testing."""
    data_path = Path(__file__).parent.parent / "Data" / "cleaned_googleplaystore.csv"
    assert data_path.exists(), f"Cleaned dataset missing at {data_path}"
    return pd.read_csv(data_path)


def test_compute_category_distribution(playstore_df):
    """Test category volume distribution calculations."""
    cat_dist = compute_category_distribution(playstore_df)

    assert isinstance(cat_dist, pd.DataFrame)
    assert set(["Category", "App_Count", "Percentage", "Cumulative_Percentage"]).issubset(cat_dist.columns)
    assert len(cat_dist) == 33  # 33 distinct categories

    # Sum of counts must equal total apps
    assert cat_dist["App_Count"].sum() == len(playstore_df)
    # Sum of percentages should be ~100%
    assert np.isclose(cat_dist["Percentage"].sum(), 100.0, atol=1e-3)
    assert np.isclose(cat_dist["Cumulative_Percentage"].iloc[-1], 100.0, atol=1e-3)

    # Top category is FAMILY
    assert cat_dist.iloc[0]["Category"] == "FAMILY"
    assert cat_dist.iloc[0]["App_Count"] == 1874

    # Bottom category is BEAUTY
    assert cat_dist.iloc[-1]["Category"] == "BEAUTY"
    assert cat_dist.iloc[-1]["App_Count"] == 53


def test_compute_rating_statistics(playstore_df):
    """Test overall rating descriptive statistics."""
    stats = compute_rating_statistics(playstore_df)

    assert stats["total_apps"] == 9659
    assert stats["rated_apps"] == 8196
    assert stats["unrated_apps"] == 1463
    assert np.isclose(stats["rated_percentage"], 84.85, atol=0.1)

    # Mean and median should reflect left-skewed distribution
    assert 4.10 <= stats["mean"] <= 4.25
    assert stats["median"] == 4.30
    assert stats["skewness"] < -1.0  # Strongly left-skewed
    assert stats["min"] >= 1.0
    assert stats["max"] <= 5.0
    assert np.isclose(stats["iqr"], stats["q75"] - stats["q25"])


def test_compute_category_ratings(playstore_df):
    """Test category-level rating metrics aggregation."""
    cat_ratings = compute_category_ratings(playstore_df)

    assert len(cat_ratings) == 33
    required_cols = [
        "Category", "Rated_Count", "Mean_Rating", "Median_Rating",
        "Std_Rating", "IQR_Rating", "Total_Apps", "Coverage_Pct"
    ]
    for col in required_cols:
        assert col in cat_ratings.columns

    # Verify ratings are within [1.0, 5.0]
    assert (cat_ratings["Mean_Rating"] >= 1.0).all()
    assert (cat_ratings["Mean_Rating"] <= 5.0).all()

    # EVENTS has the highest mean rating, DATING has the lowest
    assert cat_ratings.iloc[0]["Category"] == "EVENTS"
    assert cat_ratings.iloc[-1]["Category"] == "DATING"
    assert cat_ratings.iloc[-1]["Mean_Rating"] < 4.0


def test_compute_size_installs_correlation(playstore_df):
    """Test correlation metrics between app size and installs."""
    # Test with explicit size
    corr = compute_size_installs_correlation(playstore_df, use_imputed=False)

    assert corr["sample_size"] == 8431
    assert -1.0 <= corr["pearson_r"] <= 1.0
    assert -1.0 <= corr["spearman_rho"] <= 1.0
    assert 0.0 <= corr["pearson_p"] <= 1.0
    assert 0.0 <= corr["spearman_p"] <= 1.0

    # Statistically significant positive rank correlation
    assert corr["spearman_rho"] > 0.25
    assert corr["spearman_p"] < 1e-50
    assert corr["pearson_log_r"] > 0.25

    # Test with imputed size
    corr_imp = compute_size_installs_correlation(playstore_df, use_imputed=True)
    assert corr_imp["sample_size"] == len(playstore_df)
    assert corr_imp["spearman_rho"] > 0.20


def test_compute_stratified_correlations(playstore_df):
    """Test category-stratified size vs installs correlations."""
    strat = compute_stratified_correlations(playstore_df, min_apps=30)

    assert isinstance(strat, pd.DataFrame)
    assert len(strat) > 0
    assert "Spearman_rho" in strat.columns
    assert "Pearson_r" in strat.columns

    # Check that highest correlation categories include SHOPPING, ART_AND_DESIGN, or PHOTOGRAPHY
    top_categories = strat.head(3)["Category"].tolist()
    assert any(c in top_categories for c in ["SHOPPING", "ART_AND_DESIGN", "PHOTOGRAPHY"])

    # Sample sizes must all be >= 30
    assert (strat["Sample_Size"] >= 30).all()


def test_generate_category_ratings_reports(tmp_path):
    """Test that end-to-end report and figure generation writes all artifacts."""
    clean_csv_path = Path(__file__).parent.parent / "Data" / "cleaned_googleplaystore.csv"
    reports_dir = tmp_path / "reports" / "figures"
    results = generate_category_ratings_reports(
        data_path=str(clean_csv_path),
        output_dir=str(reports_dir)
    )

    assert results is not None
    # Check that figure images were saved
    expected_figures = [
        "01_category_distribution.png",
        "02_rating_distributions.png",
        "03_size_vs_installs_scatter.png",
        "04_category_size_installs_correlations.png"
    ]
    for fig_name in expected_figures:
        assert (reports_dir / fig_name).exists(), f"Missing figure {fig_name}"

    # Check summary CSVs
    expected_csvs = [
        "category_distribution_summary.csv",
        "category_ratings_summary.csv",
        "stratified_correlations_summary.csv"
    ]
    for csv_name in expected_csvs:
        assert (reports_dir.parent / csv_name).exists(), f"Missing summary CSV {csv_name}"
