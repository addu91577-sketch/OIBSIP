"""Data cleaning module for Google Play Store Apps and User Reviews datasets.

Provides modular and production-ready cleaning functions to handle:
- Corrupted shifted records (e.g., row 10472 where Category was omitted)
- Type conversion for Reviews, Installs, Price, and Size
- Handling 'Varies with device' and unit variations (k/M)
- Deduplication strategy prioritizing the most up-to-date scraped instances
- User review text cleaning, missing translation handling, and sentiment validation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def parse_size_to_mb(val: Union[str, float, int]) -> float:
    """Convert raw Play Store app size strings to numeric megabytes (MB).

    Examples:
        - "19M" -> 19.0
        - "201k" -> ~0.196
        - "Varies with device" -> np.nan
        - np.nan -> np.nan
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()
    if val_str.lower() in ("varies with device", "", "none", "nan"):
        return np.nan

    try:
        if val_str.endswith("M") or val_str.endswith("m"):
            return float(val_str[:-1])
        elif val_str.endswith("k") or val_str.endswith("K"):
            return float(val_str[:-1]) / 1024.0
        elif val_str.replace(".", "", 1).isdigit():
            return float(val_str)
    except (ValueError, TypeError):
        return np.nan

    return np.nan


def clean_playstore_apps(
    filepath_or_df: Union[str, Path, pd.DataFrame],
    deduplicate: bool = True
) -> pd.DataFrame:
    """Clean the Google Play Store apps dataset.

    Performs the following cleaning pipeline:
    1. Loads dataset if path is provided.
    2. Filters out corrupted shifted rows (such as row 10472 where Category is numeric).
    3. Cleans and casts 'Reviews' to int64.
    4. Cleans 'Installs' (removes '+' and ',') and casts to int64.
    5. Cleans 'Price' (removes '$') and casts to float64.
    6. Parses 'Size' into 'Size_MB' and computes category-median imputed 'Size_MB_Imputed'.
    7. Fills missing 'Type' based on 'Price' (Price > 0 -> 'Paid', else 'Free').
    8. Filters out invalid 'Rating' outside [1.0, 5.0].
    9. Converts 'Last Updated' to standard datetime format.
    10. Deduplicates apps by 'App' name, keeping the record with highest reviews.

    Args:
        filepath_or_df: Path to CSV file or existing DataFrame.
        deduplicate: Whether to deduplicate apps keeping max reviews.

    Returns:
        pd.DataFrame: Cleaned Play Store apps dataset.
    """
    if isinstance(filepath_or_df, (str, Path)):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df.copy()

    # Step 1: Filter out known corrupted rows (Category cannot be numeric or '1.9')
    # Row 10472 in Kaggle dataset has Category='1.9' and shifts all columns left
    numeric_category_mask = df["Category"].astype(str).str.replace(".", "", regex=False).str.isdigit()
    df = df[~numeric_category_mask].copy()

    # Step 2: Clean 'Reviews'
    reviews_clean = (
        df["Reviews"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Reviews"] = pd.to_numeric(reviews_clean, errors="coerce").fillna(0).astype("int64")

    # Step 3: Clean 'Installs' (e.g. '10,000+' -> 10000)
    installs_clean = (
        df["Installs"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Installs"] = pd.to_numeric(installs_clean, errors="coerce").fillna(0).astype("int64")

    # Step 4: Clean 'Price' (e.g. '$4.99' -> 4.99, '0' -> 0.0)
    price_clean = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df["Price"] = pd.to_numeric(price_clean, errors="coerce").fillna(0.0).astype(float)

    # Step 5: Clean 'Size' to numeric Size_MB
    df["Size_MB"] = df["Size"].apply(parse_size_to_mb)
    # Impute missing sizes with the median of their respective Category
    category_medians = df.groupby("Category")["Size_MB"].transform("median")
    df["Size_MB_Imputed"] = df["Size_MB"].fillna(category_medians).round(2)

    # Step 6: Fix 'Type' (Free / Paid)
    derived_type = np.where(df["Price"] > 0, "Paid", "Free")
    df["Type"] = np.where(df["Type"].isna() | (~df["Type"].isin(["Free", "Paid"])), derived_type, df["Type"])

    # Step 7: Validate 'Rating' boundaries (1.0 to 5.0)
    df.loc[(df["Rating"] < 1.0) | (df["Rating"] > 5.0), "Rating"] = np.nan

    # Step 8: Standardize 'Last Updated' to datetime
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")

    # Step 9: Deduplication Strategy
    if deduplicate:
        # Sort by Reviews descending so highest review count record is retained
        df = df.sort_values(by=["Reviews", "Last Updated"], ascending=[False, False])
        df = df.drop_duplicates(subset=["App"], keep="first")
        df = df.sort_index().reset_index(drop=True)

    return df


def clean_user_reviews(
    filepath_or_df: Union[str, Path, pd.DataFrame]
) -> pd.DataFrame:
    """Clean the Google Play Store user reviews dataset.

    Performs the following cleaning pipeline:
    1. Loads dataset if path is provided.
    2. Drops rows with missing 'Translated_Review'.
    3. Strips whitespace and drops duplicate review records.
    4. Validates 'Sentiment_Polarity' to range [-1.0, 1.0].
    5. Validates 'Sentiment_Subjectivity' to range [0.0, 1.0].
    6. Fills any missing Sentiment categorization using polarity thresholds.

    Args:
        filepath_or_df: Path to reviews CSV file or existing DataFrame.

    Returns:
        pd.DataFrame: Cleaned user reviews dataset.
    """
    if isinstance(filepath_or_df, (str, Path)):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df.copy()

    # Step 1: Drop missing reviews
    df = df.dropna(subset=["Translated_Review"]).copy()

    # Step 2: Strip whitespace
    df["Translated_Review"] = df["Translated_Review"].astype(str).str.strip()

    # Step 3: Remove duplicate reviews
    df = df.drop_duplicates(subset=["App", "Translated_Review"]).reset_index(drop=True)

    # Step 4: Ensure numeric polarity & subjectivity
    df["Sentiment_Polarity"] = pd.to_numeric(df["Sentiment_Polarity"], errors="coerce")
    df["Sentiment_Subjectivity"] = pd.to_numeric(df["Sentiment_Subjectivity"], errors="coerce")

    # Step 5: Fill or validate Sentiment labels
    def resolve_sentiment(row: pd.Series) -> str:
        sentiment = str(row.get("Sentiment", ""))
        if sentiment in ("Positive", "Negative", "Neutral"):
            return sentiment
        pol = row.get("Sentiment_Polarity", np.nan)
        if pd.isna(pol):
            return "Neutral"
        if pol > 0.05:
            return "Positive"
        elif pol < -0.05:
            return "Negative"
        return "Neutral"

    df["Sentiment"] = df.apply(resolve_sentiment, axis=1)

    return df


def generate_cleaning_audit(
    raw_apps: pd.DataFrame,
    clean_apps: pd.DataFrame,
    raw_reviews: pd.DataFrame,
    clean_reviews: pd.DataFrame
) -> Dict[str, Union[int, float, Dict]]:
    """Generate comprehensive audit metrics before and after cleaning."""
    return {
        "apps_raw_count": len(raw_apps),
        "apps_clean_count": len(clean_apps),
        "apps_duplicates_removed": len(raw_apps) - len(clean_apps),
        "apps_nulls_before": int(raw_apps.isnull().sum().sum()),
        "apps_nulls_after": int(clean_apps.isnull().sum().sum()),
        "reviews_raw_count": len(raw_reviews),
        "reviews_clean_count": len(clean_reviews),
        "reviews_removed_null_and_dup": len(raw_reviews) - len(clean_reviews),
        "reviews_sentiment_distribution": clean_reviews["Sentiment"].value_counts().to_dict(),
    }


def execute_pipeline(
    raw_apps_path: Union[str, Path] = "Data/googleplaystore.csv",
    raw_reviews_path: Union[str, Path] = "Data/googleplaystore_user_reviews.csv",
    output_dir: Union[str, Path] = "Data"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Execute complete end-to-end data cleaning pipeline and save artifacts."""
    raw_apps = pd.read_csv(raw_apps_path)
    raw_reviews = pd.read_csv(raw_reviews_path)

    clean_apps = clean_playstore_apps(raw_apps)
    clean_reviews = clean_user_reviews(raw_reviews)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    clean_apps_file = out_path / "cleaned_googleplaystore.csv"
    clean_reviews_file = out_path / "cleaned_user_reviews.csv"

    clean_apps.to_csv(clean_apps_file, index=False)
    clean_reviews.to_csv(clean_reviews_file, index=False)

    audit = generate_cleaning_audit(raw_apps, clean_apps, raw_reviews, clean_reviews)
    return clean_apps, clean_reviews, audit


if __name__ == "__main__":
    print("Executing Google Play Store Data Cleaning Pipeline...")
    apps, reviews, audit = execute_pipeline()
    print("Data Cleaning Complete!")
    print(f"Apps: {audit['apps_raw_count']} raw -> {audit['apps_clean_count']} clean unique apps.")
    print(f"Reviews: {audit['reviews_raw_count']} raw -> {audit['reviews_clean_count']} clean unique reviews.")
    print(f"Reviews sentiment distribution: {audit['reviews_sentiment_distribution']}")
