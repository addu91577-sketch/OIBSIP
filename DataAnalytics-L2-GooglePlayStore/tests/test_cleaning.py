"""Unit and validation tests for Google Play Store Data Cleaning pipeline.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import (
    parse_size_to_mb,
    clean_playstore_apps,
    clean_user_reviews,
    execute_pipeline
)


def test_parse_size_to_mb():
    """Test unit conversions and edge cases for app sizes."""
    assert parse_size_to_mb("19M") == 19.0
    assert parse_size_to_mb("10.5m") == 10.5
    assert np.isclose(parse_size_to_mb("1024k"), 1.0)
    assert np.isclose(parse_size_to_mb("512K"), 0.5)
    assert np.isnan(parse_size_to_mb("Varies with device"))
    assert np.isnan(parse_size_to_mb(None))
    assert np.isnan(parse_size_to_mb(""))


def test_clean_playstore_apps():
    """Test data cleaning on sample dirty records."""
    dirty_data = pd.DataFrame([
        {
            "App": "Test App 1",
            "Category": "FAMILY",
            "Rating": 4.5,
            "Reviews": "1,000",
            "Size": "25M",
            "Installs": "100,000+",
            "Type": "Free",
            "Price": "0",
            "Content Rating": "Everyone",
            "Genres": "Casual",
            "Last Updated": "January 1, 2018",
            "Current Ver": "1.0",
            "Android Ver": "4.0 and up",
        },
        {
            # Corrupted row simulation (row 10472)
            "App": "Corrupted App",
            "Category": "1.9",
            "Rating": 19.0,
            "Reviews": "3.0M",
            "Size": "1,000+",
            "Installs": "Free",
            "Type": "0",
            "Price": "Everyone",
            "Content Rating": None,
            "Genres": "February 11, 2018",
            "Last Updated": "1.0.19",
            "Current Ver": "4.0 and up",
            "Android Ver": None,
        },
        {
            # Duplicate app with higher review count
            "App": "Test App 1",
            "Category": "FAMILY",
            "Rating": 4.6,
            "Reviews": "5,000",
            "Size": "25M",
            "Installs": "500,000+",
            "Type": "Free",
            "Price": "0",
            "Content Rating": "Everyone",
            "Genres": "Casual",
            "Last Updated": "June 1, 2018",
            "Current Ver": "1.2",
            "Android Ver": "4.0 and up",
        },
        {
            # Paid app
            "App": "Paid App 1",
            "Category": "GAME",
            "Rating": 4.8,
            "Reviews": "250",
            "Size": "500k",
            "Installs": "1,000+",
            "Type": None,
            "Price": "$4.99",
            "Content Rating": "Teen",
            "Genres": "Action",
            "Last Updated": "July 15, 2018",
            "Current Ver": "2.0",
            "Android Ver": "4.1 and up",
        },
    ])

    clean_df = clean_playstore_apps(dirty_data, deduplicate=True)

    # 1. Corrupted row filtered out
    assert "Corrupted App" not in clean_df["App"].values

    # 2. Duplicate resolved: only 1 instance of Test App 1, with Reviews == 5000
    test_app = clean_df[clean_df["App"] == "Test App 1"].iloc[0]
    assert test_app["Reviews"] == 5000
    assert test_app["Installs"] == 500000

    # 3. Type inference
    paid_app = clean_df[clean_df["App"] == "Paid App 1"].iloc[0]
    assert paid_app["Type"] == "Paid"
    assert paid_app["Price"] == 4.99
    assert np.isclose(paid_app["Size_MB"], 500 / 1024.0)

    # 4. Data types
    assert pd.api.types.is_integer_dtype(clean_df["Reviews"])
    assert pd.api.types.is_integer_dtype(clean_df["Installs"])
    assert pd.api.types.is_float_dtype(clean_df["Price"])
    assert pd.api.types.is_float_dtype(clean_df["Size_MB"])


def test_clean_user_reviews():
    """Test user reviews cleaning, null removal, and deduplication."""
    sample_reviews = pd.DataFrame([
        {
            "App": "App A",
            "Translated_Review": "Great app, highly recommended!",
            "Sentiment": "Positive",
            "Sentiment_Polarity": 0.8,
            "Sentiment_Subjectivity": 0.6,
        },
        {
            "App": "App A",
            "Translated_Review": "Great app, highly recommended!",  # Exact duplicate
            "Sentiment": "Positive",
            "Sentiment_Polarity": 0.8,
            "Sentiment_Subjectivity": 0.6,
        },
        {
            "App": "App B",
            "Translated_Review": None,  # Missing review
            "Sentiment": None,
            "Sentiment_Polarity": np.nan,
            "Sentiment_Subjectivity": np.nan,
        },
        {
            "App": "App C",
            "Translated_Review": "Terrible bugs and crashes.",
            "Sentiment": None,  # Missing label, should be imputed
            "Sentiment_Polarity": -0.6,
            "Sentiment_Subjectivity": 0.9,
        },
    ])

    clean_rev = clean_user_reviews(sample_reviews)

    assert len(clean_rev) == 2
    assert "App B" not in clean_rev["App"].values
    imputed_sentiment = clean_rev[clean_rev["App"] == "App C"].iloc[0]["Sentiment"]
    assert imputed_sentiment == "Negative"


def test_saved_cleaned_datasets_exist():
    """Validate that cleaned datasets generated on disk adhere to schemas."""
    apps_path = Path("Data/cleaned_googleplaystore.csv")
    revs_path = Path("Data/cleaned_user_reviews.csv")

    assert apps_path.exists(), "Cleaned apps CSV must exist"
    assert revs_path.exists(), "Cleaned reviews CSV must exist"

    df_apps = pd.read_csv(apps_path)
    df_revs = pd.read_csv(revs_path)

    assert len(df_apps) == 9659
    assert df_apps["App"].nunique() == 9659
    assert df_apps["Rating"].max() <= 5.0
    assert df_apps["Rating"].min() >= 1.0 or df_apps["Rating"].isna().any()
    assert (df_apps["Installs"] >= 0).all()
    assert (df_apps["Price"] >= 0.0).all()

    assert len(df_revs) == 29692
    assert not df_revs["Translated_Review"].isna().any()


if __name__ == "__main__":
    print("Running tests via pytest / manual runner...")
    test_parse_size_to_mb()
    test_clean_playstore_apps()
    test_clean_user_reviews()
    test_saved_cleaned_datasets_exist()
    print("All tests passed successfully!")
