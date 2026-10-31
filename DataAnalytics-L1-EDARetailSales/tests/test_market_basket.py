"""
Unit tests validating market basket association rules, cross-merchandising affinity,
and demographic cross-selling propensity.
"""

import pytest
from src.market_basket import MarketBasketAffinityEngine


@pytest.fixture(scope="module")
def mb_engine():
    return MarketBasketAffinityEngine()


def test_category_summary(mb_engine):
    """Verifies baseline category volume, revenue, and penetration rates."""
    summary = mb_engine.get_category_summary()
    assert set(summary.keys()) == {"Beauty", "Clothing", "Electronics"}

    total_pct = sum(c["penetration_rate"] for c in summary.values())
    assert total_pct == pytest.approx(1.0, rel=1e-3)

    for cat_data in summary.values():
        assert cat_data["transaction_count"] > 0
        assert cat_data["total_revenue"] > 0.0
        assert cat_data["avg_price_per_unit"] > 0.0


def test_build_cooccurrence_and_affinity_matrix(mb_engine):
    """Verifies pairwise association rule metrics (Support, Confidence, Lift, Leverage, Chi2)."""
    matrix = mb_engine.build_cooccurrence_and_affinity_matrix()
    assert matrix["total_shopping_sessions"] > 0
    assert len(matrix["rules"]) == 6  # 3 categories * 2 directional pairs each

    for rule in matrix["rules"]:
        assert 0.0 <= rule["support_antecedent"] <= 1.0
        assert 0.0 <= rule["support_consequent"] <= 1.0
        assert 0.0 <= rule["support_joint"] <= 1.0
        assert 0.0 <= rule["confidence"] <= 1.0
        assert rule["lift"] > 0.0
        assert rule["chi2_statistic"] >= 0.0
        assert 0.0 <= rule["p_value"] <= 1.0
        assert rule["affinity_nature"] in [
            "Positive Synergy",
            "Negative Cannibalization",
            "Neutral Independent",
        ]


def test_mine_association_rules(mb_engine):
    """Verifies filtering and ranking of association rules by threshold criteria."""
    rules = mb_engine.mine_association_rules(min_support=0.01, min_confidence=0.10, min_lift=0.50)
    assert len(rules) > 0
    for r in rules:
        assert r["support_joint"] >= 0.01
        assert r["confidence"] >= 0.10
        assert r["lift"] >= 0.50

    # Must be sorted by lift descending
    lifts = [r["lift"] for r in rules]
    assert lifts == sorted(lifts, reverse=True)


def test_recommend_merchandising_combos(mb_engine):
    """Verifies bundle generation, margin retention, and savings calculation."""
    combos = mb_engine.recommend_merchandising_combos("Electronics", discount_rate=0.15)
    assert len(combos) == 2  # Clothing and Beauty pairings
    for combo in combos:
        assert combo["base_category"] == "Electronics"
        assert combo["discounted_bundle_price"] < combo["original_bundle_price"]
        assert combo["customer_savings"] > 0.0
        assert combo["promotional_discount_pct"] == 15.0
        assert combo["estimated_retained_margin_pct"] > 0.0
        assert combo["synergy_score"] > 0.0


def test_recommend_merchandising_combos_invalid_category(mb_engine):
    """Requesting combos for non-existent category must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown target category"):
        mb_engine.recommend_merchandising_combos("Books")


def test_recommend_merchandising_combos_invalid_discount(mb_engine):
    """Discount rate out of [0.01, 0.40] must raise ValueError."""
    with pytest.raises(ValueError, match="Discount rate must be between"):
        mb_engine.recommend_merchandising_combos("Clothing", discount_rate=0.50)


def test_demographic_affinity_breakdown(mb_engine):
    """Verifies cross-category demographic penetration across standard age cohorts."""
    demo = mb_engine.demographic_affinity_breakdown()
    assert demo["cohort_count"] >= 3
    cohorts = demo["cohort_breakdown"]

    for c_name, c_info in cohorts.items():
        assert c_info["total_transactions"] > 0
        assert c_info["average_basket_spend"] > 0.0
        assert c_info["dominant_category"] in ["Beauty", "Clothing", "Electronics"]
        assert 0.0 <= c_info["multi_category_affinity_index"] <= 1.0
