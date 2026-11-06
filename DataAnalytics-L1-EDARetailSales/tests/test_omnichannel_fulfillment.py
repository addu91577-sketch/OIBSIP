"""
Unit tests for Omnichannel Fulfillment & Micro-Fulfillment Unit Economics Engine.
"""

import pytest
from src.omnichannel_fulfillment import OmnichannelFulfillmentEngine


@pytest.fixture
def fulfillment_engine():
    return OmnichannelFulfillmentEngine()


def test_network_topology_structure(fulfillment_engine):
    topo = fulfillment_engine.get_network_topology()
    assert "total_nodes" in topo
    assert topo["total_nodes"] == 6
    assert len(topo["distribution_centers"]) == 3
    assert len(topo["retail_store_nodes"]) == 3
    assert 1 in topo["carrier_zones_rate_card"]
    assert 5 in topo["carrier_zones_rate_card"]
    assert "split_shipment_penalty_usd" in topo


def test_route_order_optimal_bopis_selection(fulfillment_engine):
    # Testing standard routing for Electronics in Zone 1
    res = fulfillment_engine.route_order(
        category="Electronics",
        quantity=1,
        price_per_unit=300.0,
        destination_zone=1,
        channel_preference="BOPIS",
    )
    opt = res["optimal_fulfillment"]
    assert opt["channel"] == "BOPIS"
    assert opt["freight_cost"] == 0.0
    assert opt["pick_pack_cost"] == 1.50
    assert opt["total_fulfillment_cost"] == 1.50
    assert opt["net_margin_contribution"] > 0
    assert res["esg_sustainability"]["carbon_intensity_rating"] == "Green"


def test_route_order_ship_from_store_and_expedited(fulfillment_engine):
    res = fulfillment_engine.route_order(
        category="Clothing",
        quantity=2,
        price_per_unit=50.0,
        destination_zone=2,
        channel_preference="SHIP_FROM_STORE",
        expedited=True,
    )
    opt = res["optimal_fulfillment"]
    assert opt["channel"] == "SHIP_FROM_STORE"
    assert opt["freight_cost"] > 0
    assert opt["transit_days"] in [1, 2]
    assert len(res["all_evaluated_channels"]) == 4


def test_route_order_validation_errors(fulfillment_engine):
    with pytest.raises(ValueError, match="Invalid category"):
        fulfillment_engine.route_order("Automotive", 1, 100.0)

    with pytest.raises(ValueError, match="Invalid delivery zone"):
        fulfillment_engine.route_order("Beauty", 1, 50.0, destination_zone=9)

    with pytest.raises(ValueError, match="Quantity must be strictly positive"):
        fulfillment_engine.route_order("Beauty", 0, 50.0)

    with pytest.raises(ValueError, match="Price per unit must be strictly positive"):
        fulfillment_engine.route_order("Beauty", 1, -10.0)


def test_simulate_network_fulfillment_demand(fulfillment_engine):
    sim = fulfillment_engine.simulate_network_fulfillment_demand(
        bopis_adoption_rate=0.20,
        sfs_allocation_rate=0.30,
        expedited_share_rate=0.10,
    )
    assert sim["simulation_parameters"]["total_orders_simulated"] == 1000
    fin = sim["financial_summary"]
    assert fin["total_fulfillment_expenditure_usd"] > 0
    assert fin["average_cost_per_order_usd"] > 0
    assert fin["cost_savings_vs_naive_central_dc_usd"] > 0

    assert "BOPIS" in sim["channel_volume_breakdown"]
    assert "SHIP_FROM_STORE" in sim["channel_volume_breakdown"]
    assert sim["operational_metrics"]["total_carbon_emissions_tonnes_co2e"] > 0


def test_simulate_network_fulfillment_validation(fulfillment_engine):
    with pytest.raises(ValueError, match="bopis_adoption_rate"):
        fulfillment_engine.simulate_network_fulfillment_demand(bopis_adoption_rate=-0.1)

    with pytest.raises(ValueError, match="Combined BOPIS and SFS allocation"):
        fulfillment_engine.simulate_network_fulfillment_demand(bopis_adoption_rate=0.6, sfs_allocation_rate=0.5)


def test_optimize_delivery_promise_tradeoff(fulfillment_engine):
    res = fulfillment_engine.optimize_delivery_promise_tradeoff(
        order_value=250.0,
        category="Electronics",
        destination_zone=3,
    )
    tiers = res["sla_tiers"]
    assert len(tiers) == 3
    tier_names = [t["tier_name"] for t in tiers]
    assert "Expedited Next-Day" in tier_names
    assert "Standard Ground (Default)" in tier_names
    assert "Eco-Consolidated Green Saver" in tier_names
    # Expedited should be more expensive than standard
    exp = [t for t in tiers if "Expedited" in t["tier_name"]][0]
    std = [t for t in tiers if "Standard" in t["tier_name"]][0]
    assert exp["fulfillment_cost_usd"] > std["fulfillment_cost_usd"]


def test_calculate_esg_carbon_footprint(fulfillment_engine):
    esg = fulfillment_engine.calculate_esg_carbon_footprint()
    assert esg["storewide_scope3_carbon_kg"] > 0
    assert esg["carbon_intensity_kg_per_dollar_revenue"] > 0
    assert len(esg["green_logistics_initiatives"]) == 3
