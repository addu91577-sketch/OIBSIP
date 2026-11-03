"""
Unit tests for the Omnichannel Multi-Touch Attribution & Marketing ROI Engine.
"""

import pytest
import numpy as np
from src.attribution_marketing import AttributionMarketingEngine


@pytest.fixture(scope="module")
def attribution_engine():
    return AttributionMarketingEngine()


def test_journeys_generation_and_shape(attribution_engine):
    df = attribution_engine._journeys_df
    assert df is not None
    assert len(df) == 1000
    assert "channels" in df.columns
    assert "days_before" in df.columns
    assert "total_amount" in df.columns

    # Verify each row has valid channels
    for _, row in df.head(50).iterrows():
        assert len(row["channels"]) >= 1
        assert len(row["channels"]) == len(row["days_before"])
        assert row["days_before"][-1] == 0  # Converting touchpoint is day 0


def test_heuristic_attribution_summary(attribution_engine):
    summary = attribution_engine.get_heuristic_attribution_summary()
    assert summary["total_transactions"] == 1000
    assert summary["total_revenue"] == pytest.approx(456000.0, rel=1e-2)
    models = summary["models"]

    expected_models = ["First-Touch", "Last-Touch", "Linear", "Time-Decay", "Position-Based"]
    for m in expected_models:
        assert m in models
        channels_data = models[m]
        assert len(channels_data) == 5

        # Check conservation of orders and revenue
        tot_orders = sum([c["attributed_orders"] for c in channels_data])
        tot_rev = sum([c["attributed_revenue"] for c in channels_data])
        assert tot_orders == pytest.approx(1000.0, abs=1.0)
        assert tot_rev == pytest.approx(456000.0, rel=1e-2)

        for c in channels_data:
            assert c["attributed_orders"] >= 0
            assert c["attributed_revenue"] >= 0
            assert c["roas"] >= 0
            assert c["cpa"] >= 0


def test_markov_attribution(attribution_engine):
    res = attribution_engine.compute_markov_attribution()
    assert res["baseline_conversion_probability"] > 0.0
    assert res["channels_evaluated"] == 5

    attrs = res["channel_attribution"]
    assert len(attrs) == 5

    total_share = sum([a["attributed_share_pct"] for a in attrs])
    assert total_share == pytest.approx(100.0, abs=0.5)

    total_rev = sum([a["attributed_revenue"] for a in attrs])
    assert total_rev == pytest.approx(456000.0, rel=1e-2)

    for a in attrs:
        assert a["removal_effect"] >= 0.0
        assert a["roas"] > 0.0
        assert a["cpa"] > 0.0


def test_shapley_attribution(attribution_engine):
    res = attribution_engine.compute_shapley_attribution()
    assert res["total_attributed_revenue"] == pytest.approx(456000.0, rel=1e-2)
    attrs = res["shapley_attribution"]
    assert len(attrs) == 5

    tot_rev = sum([a["attributed_revenue"] for a in attrs])
    assert tot_rev == pytest.approx(456000.0, rel=1e-2)

    tot_share = sum([a["share_pct"] for a in attrs])
    assert tot_share == pytest.approx(100.0, abs=0.5)


def test_optimize_marketing_budget(attribution_engine):
    res = attribution_engine.optimize_marketing_budget(total_budget=80000.0, risk_aversion=0.5)
    assert res["total_budget"] == 80000.0
    assert res["projected_optimal_revenue"] > 0.0
    assert res["projected_blended_roas"] > 0.0
    assert len(res["channel_allocations"]) == 5

    alloc_sum = sum([a["optimal_spend"] for a in res["channel_allocations"]])
    assert alloc_sum == pytest.approx(80000.0, abs=1.0)

    # Check bounds (each channel receives at least 5% and at most 45%)
    for a in res["channel_allocations"]:
        assert a["optimal_spend"] >= 80000.0 * 0.049
        assert a["optimal_spend"] <= 80000.0 * 0.451
        assert a["action"] in ["Increase Spend", "Decrease Spend", "Maintain"]
