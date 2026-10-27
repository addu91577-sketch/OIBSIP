"""
Unit tests validating raw dataset schema, data hygiene, and mathematical integrity.
"""

from pathlib import Path
import pandas as pd
import pytest


def test_dataset_file_exists():
    """Dataset file must exist at expected location."""
    path = Path("Data/retail_sales_dataset.csv")
    assert path.exists(), f"Dataset not found at {path}"


def test_dataset_dimensions(analytics_engine):
    """Dataset must strictly contain 1,000 transactions and 9 core attributes."""
    df = analytics_engine.df
    assert len(df) == 1000, f"Expected 1,000 rows, found {len(df)}"
    expected_cols = [
        "Transaction ID",
        "Date",
        "Customer ID",
        "Gender",
        "Age",
        "Product Category",
        "Quantity",
        "Price per Unit",
        "Total Amount",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_zero_null_values(analytics_engine):
    """Hygiene check: dataset must contain zero missing or null values."""
    df = analytics_engine.df
    null_counts = df.isnull().sum().sum()
    assert null_counts == 0, f"Dataset contains {null_counts} null values"


def test_unique_transaction_ids(analytics_engine):
    """Transaction ID must be strictly unique with zero duplicates."""
    df = analytics_engine.df
    assert df["Transaction ID"].nunique() == 1000
    assert df.duplicated().sum() == 0


def test_age_cohort_boundaries(analytics_engine):
    """Age values must fall strictly within the validated 18 to 64 window."""
    df = analytics_engine.df
    assert df["Age"].min() >= 18
    assert df["Age"].max() <= 64


def test_discrete_catalog_price_tiers(analytics_engine):
    """Price per Unit must strictly belong to the 5 discrete pricing tiers."""
    df = analytics_engine.df
    unique_prices = set(df["Price per Unit"].unique())
    assert unique_prices == {25, 30, 50, 300, 500}


def test_basket_quantity_boundaries(analytics_engine):
    """Basket quantity must be strictly bounded between 1 and 4 units."""
    df = analytics_engine.df
    assert df["Quantity"].min() == 1
    assert df["Quantity"].max() == 4


def test_total_amount_deterministic_invariant(analytics_engine):
    """Mathematical invariant: Total Amount == Quantity * Price per Unit across all rows."""
    df = analytics_engine.df
    computed = df["Quantity"] * df["Price per Unit"]
    mismatches = (df["Total Amount"] != computed).sum()
    assert mismatches == 0, f"Found {mismatches} mathematical invariant violations"
