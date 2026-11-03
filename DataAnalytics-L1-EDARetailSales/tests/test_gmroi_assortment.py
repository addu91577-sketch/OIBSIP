"""
Unit tests for the GMROI & Merchandise Assortment Rationalization Engine.
"""

import pytest
from src.gmroi_assortment import GMROIAssortmentEngine


@pytest.fixture(scope="module")
def gmroi_engine():
    return GMROIAssortmentEngine()


def test_calculate_category_gmroi(gmroi_engine):
    res = gmroi_engine.calculate_category_gmroi()
    assert res["store_total_revenue"] == pytest.approx(456000.0, rel=1e-2)
    assert res["store_total_gross_profit"] > 0
    assert res["store_blended_gross_margin_pct"] > 0
    assert res["store_total_inventory_investment"] > 0
    assert res["store_blended_gmroi"] > 0

    cats = res["categories"]
    assert len(cats) == 3
    cat_names = [c["category"] for c in cats]
    assert "Beauty" in cat_names
    assert "Clothing" in cat_names
    assert "Electronics" in cat_names

    for c in cats:
        assert c["annual_revenue"] > 0
        assert c["units_sold"] > 0
        assert c["gmroi_ratio"] > 0
        assert c["commercial_rating"] in [
            "Elite / High Capital Productivity",
            "Healthy / Above Benchmark",
            "Acceptable / Standard",
            "Capital Dilutive / Underperforming",
        ]


def test_compute_margin_turn_matrix(gmroi_engine):
    matrix = gmroi_engine.compute_margin_turn_matrix()
    assert "thresholds" in matrix
    assert "quadrant_summary" in matrix
    assert "classified_skus" in matrix

    quads = matrix["quadrant_summary"]
    for q in ["Stars", "High-Yield Specialists", "Traffic Drivers", "Dead Stock / Capital Traps"]:
        assert q in quads
        assert quads[q]["count"] >= 0

    skus = matrix["classified_skus"]
    assert len(skus) > 0
    for s in skus:
        assert s["category"] in ["Beauty", "Clothing", "Electronics"]
        assert s["price_tier"] in [25.0, 30.0, 50.0, 300.0, 500.0]
        assert s["gmroi"] > 0
        assert len(s["merchandising_action"]) > 0


def test_evaluate_assortment_rationalization(gmroi_engine):
    diag = gmroi_engine.evaluate_assortment_rationalization()
    assert diag["total_skus_evaluated"] > 0
    assert diag["rationalization_candidates_count"] >= 0
    assert diag["total_capital_at_risk"] >= 0

    for cand in diag["candidates"]:
        assert cand["capital_drag_ratio"] > 0
        assert cand["inventory_capital_tied_up"] > 0
        assert len(cand["recommended_action"]) > 0


def test_simulate_working_capital_reallocation(gmroi_engine):
    sim = gmroi_engine.simulate_working_capital_reallocation(
        reallocation_pct=0.30,
        source_quadrant="Dead Stock / Capital Traps",
        target_quadrant="Stars",
    )
    assert sim["source_quadrant"] == "Dead Stock / Capital Traps"
    assert sim["target_quadrant"] == "Stars"
    assert sim["reallocation_pct"] == 30.0
    assert sim["capital_reallocated_dollars"] > 0
    assert sim["projected_store_gmroi"] >= sim["baseline_store_gmroi"]


def test_simulate_working_capital_reallocation_invalid(gmroi_engine):
    with pytest.raises(ValueError):
        gmroi_engine.simulate_working_capital_reallocation(reallocation_pct=0.99)

    with pytest.raises(ValueError):
        gmroi_engine.simulate_working_capital_reallocation(source_quadrant="NonExistentQuadrant")
