"""
Unit tests for Geospatial Gravity, Huff Catchment & Store Cannibalization Engine.
"""

import pytest
from src.geospatial_gravity import GeospatialGravityEngine


@pytest.fixture
def gravity_engine():
    return GeospatialGravityEngine()


def test_haversine_distance_calculation():
    # Distance between Manhattan (40.7580, -73.9855) and Jersey City (40.7178, -74.0431)
    dist = GeospatialGravityEngine.haversine_distance_km(40.7580, -73.9855, 40.7178, -74.0431)
    assert 5.0 < dist < 12.0


def test_network_catalog(gravity_engine):
    cat = gravity_engine.get_network_catalog()
    assert cat["total_active_stores"] == 3
    assert cat["total_demand_tracts"] == 10
    assert cat["total_catchment_population"] > 400000
    assert cat["total_market_retail_expenditure_millions"] > 100.0


def test_compute_huff_probabilities_and_shares(gravity_engine):
    res = gravity_engine.compute_huff_market_shares(beta=1.75, lambda_param=1.0)
    shares = res["store_market_shares"]
    assert "STORE_A_MANHATTAN" in shares
    assert "STORE_B_WHITE_PLAINS" in shares
    assert "STORE_C_JERSEY_CITY" in shares

    # Combined brand market share across metropolitan area
    total_pct = sum(v["market_share_pct"] for v in shares.values())
    assert 30.0 <= total_pct <= 99.0

    for s_info in shares.values():
        assert s_info["captured_market_spend_millions"] > 0
        assert s_info["effective_catchment_population"] > 0


def test_compute_huff_validation_errors(gravity_engine):
    with pytest.raises(ValueError, match="Distance decay exponent"):
        gravity_engine.compute_huff_market_shares(beta=0.0)

    with pytest.raises(ValueError, match="Attractiveness elasticity"):
        gravity_engine.compute_huff_market_shares(lambda_param=-0.5)


def test_calculate_catchment_trade_areas(gravity_engine):
    catchment = gravity_engine.calculate_catchment_trade_areas("STORE_A_MANHATTAN")
    assert catchment["store_id"] == "STORE_A_MANHATTAN"
    assert "primary_trade_area_60pct" in catchment
    assert "secondary_trade_area_80pct" in catchment
    assert "tertiary_trade_area_95pct" in catchment

    pri = catchment["primary_trade_area_60pct"]
    assert pri["tract_count"] >= 1
    assert pri["total_spend_millions"] > 0
    assert pri["avg_distance_km"] > 0


def test_calculate_catchment_invalid_store(gravity_engine):
    with pytest.raises(ValueError, match="not found"):
        gravity_engine.calculate_catchment_trade_areas("STORE_NONEXISTENT")


def test_analyze_store_cannibalization(gravity_engine):
    sim = gravity_engine.analyze_store_cannibalization(
        new_store_name="Brooklyn Heights Expansion",
        new_store_lat=40.6960,
        new_store_lon=-73.9933,
        new_store_sqft=30000,
        new_store_attractiveness=85.0,
        beta=1.75,
    )
    fin = sim["financial_impact_summary"]
    assert fin["new_store_gross_projected_millions"] > 0
    assert fin["sister_stores_cannibalized_millions"] >= 0
    assert fin["net_incremental_expansion_revenue_millions"] > 0
    assert 0 <= fin["cannibalization_ratio_pct"] <= 100

    # Ensure pairwise breakdown contains all 3 original stores
    breakdown = sim["pairwise_cannibalization_breakdown"]
    assert len(breakdown) == 3
    assert "STORE_A_MANHATTAN" in breakdown


def test_analyze_store_cannibalization_validation(gravity_engine):
    with pytest.raises(ValueError, match="Square footage must be positive"):
        gravity_engine.analyze_store_cannibalization(
            new_store_name="Test",
            new_store_lat=40.0,
            new_store_lon=-73.0,
            new_store_sqft=0,
            new_store_attractiveness=80.0,
        )


def test_evaluate_expansion_candidates(gravity_engine):
    eval_res = gravity_engine.evaluate_expansion_candidates()
    ranked = eval_res["ranked_expansion_sites"]
    assert len(ranked) == 3
    assert ranked[0]["rank"] == 1
    # Check descending order of net incremental revenue
    for i in range(len(ranked) - 1):
        assert ranked[i]["net_incremental_revenue_millions"] >= ranked[i + 1]["net_incremental_revenue_millions"]
