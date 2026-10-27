"""
Unit tests validating analytical aggregations, statistical profiles, and Pareto concentration calculations.
"""

import pytest


def test_executive_kpi_overview_exactness(analytics_engine):
    """Verifies baseline storewide financial and operational metrics."""
    kpis = analytics_engine.get_kpi_overview()
    assert kpis["total_revenue"] == 456000.0
    assert kpis["total_orders"] == 1000
    assert kpis["total_units_sold"] == 2514
    assert kpis["average_order_value"] == 456.00
    assert kpis["median_spend"] == 135.00
    assert kpis["average_units_per_order"] == 2.51
    assert kpis["vip_orders_count"] == 202
    assert kpis["vip_revenue_share_pct"] == 62.89


def test_descriptive_statistics_profiling(analytics_engine):
    """Verifies descriptive statistics across all numerical columns."""
    stats = analytics_engine.get_descriptive_statistics()
    assert set(stats.keys()) == {"Age", "Quantity", "Price per Unit", "Total Amount"}

    # Total Amount checks
    ta = stats["Total Amount"]
    assert ta["mean"] == 456.00
    assert ta["median"] == 135.00
    assert ta["min"] == 25.0
    assert ta["max"] == 2000.0
    assert ta["skewness"] > 1.0  # Right-skewed distribution

    # Age checks
    age = stats["Age"]
    assert 41.0 <= age["mean"] <= 42.0
    assert abs(age["skewness"]) < 0.1  # Balanced distribution


def test_monthly_trends_and_seasonality(analytics_engine):
    """Verifies monthly aggregation and peak/trough identification."""
    monthly = analytics_engine.get_monthly_trends()
    assert len(monthly) == 13  # 12 months in 2023 + 1 cutoff month in 2024

    # May 2023 must be highest revenue month
    may_record = next(r for r in monthly if r["YearMonth"] == "2023-05")
    assert may_record["total_revenue"] == 53150.0
    assert may_record["orders"] == 105

    # September 2023 must be lowest full-month revenue
    sep_record = next(r for r in monthly if r["YearMonth"] == "2023-09")
    assert sep_record["total_revenue"] == 23620.0
    assert sep_record["orders"] == 65


def test_day_of_week_velocity_and_weekend_lift(analytics_engine):
    """Verifies weekly velocity and weekend technology lift."""
    dow = analytics_engine.get_day_of_week_dynamics()
    daily_trends = {r["DayOfWeek"]: r for r in dow["daily_trends"]}

    # Saturday must have highest revenue
    assert daily_trends["Saturday"]["total_revenue"] == 78815.0
    # Thursday must have lowest revenue
    assert daily_trends["Thursday"]["total_revenue"] == 53835.0

    # Electronics must display positive weekend lift
    lifts = {r["category"]: r for r in dow["category_weekend_lifts"]}
    assert lifts["Electronics"]["weekend_lift_pct"] > 20.0  # +23.0%
    assert lifts["Beauty"]["weekend_lift_pct"] < 0.0  # Negative lift (weekday frontloading)


def test_demographics_gender_and_age_cohorts(analytics_engine):
    """Verifies gender parity and cohort distributions."""
    demo = analytics_engine.get_demographics_breakdown()
    gender_map = {r["Gender"]: r for r in demo["gender_breakdown"]}

    # Gender symmetry
    assert gender_map["Female"]["orders"] == 510
    assert gender_map["Male"]["orders"] == 490
    assert abs(gender_map["Female"]["aov"] - gender_map["Male"]["aov"]) < 2.0  # ~$1.12 variance

    # Cohort count
    assert len(demo["age_cohorts"]) == 4


def test_empirical_gini_coefficient_precision(analytics_engine):
    """Empirical Gini coefficient must be ~0.6091."""
    gini = analytics_engine.calculate_gini()
    assert 0.608 <= gini <= 0.610


def test_pareto_spend_tier_distribution(analytics_engine):
    """Verifies spend tier segmentation sums and revenue polarization."""
    pareto = analytics_engine.get_pareto_concentration()
    tiers = {r["SpendTier"]: r for r in pareto["spend_tier_polarization"]}

    # Whale tier
    assert tiers["VIP / Whale"]["orders"] == 202
    assert tiers["VIP / Whale"]["total_revenue"] == 286800.0
    assert tiers["VIP / Whale"]["revenue_share_pct"] == 62.89

    # Deciles
    deciles = pareto["decile_revenue_breakdown"]
    assert len(deciles) == 10
    total_dec_rev = sum(d["revenue"] for d in deciles)
    assert total_dec_rev == 456000.0
