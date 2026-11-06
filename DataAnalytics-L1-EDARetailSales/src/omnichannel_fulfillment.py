"""
Omnichannel Order Fulfillment Routing & Micro-Fulfillment Unit Economics Optimizer.

Provides:
1. Multi-Node Distributed Order Fulfillment Routing & Split-Shipment Optimizer:
   - Central Distribution Centers (CDC), Regional DCs (RDC North, RDC South), and Retail Store Nodes (SFS).
   - Multi-channel cost objective function minimizing pick-pack, carrier freight, split penalties, and holding drag.
   - Evaluates Ship-from-Store (SFS) vs Central DC Direct vs Buy-Online-Pick-Up-in-Store (BOPIS / Click & Collect).
2. Dynamic Delivery Promise SLA vs Margin Preservation Frontier:
   - Evaluates Standard Ground vs Expedited (Next-Day) vs Eco-Consolidated fulfillment.
   - Determines margin retention across basket price tiers and shipping subsidy leakage.
3. Scope 3 Last-Mile Carbon Footprint & ESG Logistics Auditing:
   - Carbon emission accounting (g CO2e / tonne-km) across carrier fleet types (Ground Van, Freight, Local EV).
   - Benchmarks green fulfillment savings from localized store fulfillment and BOPIS customer pickup.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class OmnichannelFulfillmentEngine:
    """
    Enterprise retail omnichannel logistics, distributed fulfillment routing,
    and micro-fulfillment unit economics optimization engine.
    """

    # Fulfillment Network Topology Nodes
    FULFILLMENT_NODES: Dict[str, Dict[str, Any]] = {
        "CDC_CHICAGO": {
            "name": "Central Distribution Center (Chicago)",
            "type": "CDC",
            "capacity_units_per_day": 50000,
            "pick_pack_cost": 2.20,
            "holding_cost_per_day": 0.15,
            "carbon_intensity_factor": 1.0,
            "inventory_depth_pct": 1.00,
        },
        "RDC_NORTH": {
            "name": "Regional DC North (New York)",
            "type": "RDC",
            "capacity_units_per_day": 20000,
            "pick_pack_cost": 2.80,
            "holding_cost_per_day": 0.22,
            "carbon_intensity_factor": 0.85,
            "inventory_depth_pct": 0.70,
        },
        "RDC_SOUTH": {
            "name": "Regional DC South (Atlanta)",
            "type": "RDC",
            "capacity_units_per_day": 18000,
            "pick_pack_cost": 2.60,
            "holding_cost_per_day": 0.18,
            "carbon_intensity_factor": 0.88,
            "inventory_depth_pct": 0.65,
        },
        "STORE_FLAGSHIP": {
            "name": "Store 101 - Flagship (Manhattan)",
            "type": "STORE",
            "capacity_units_per_day": 1500,
            "pick_pack_cost": 4.50,
            "holding_cost_per_day": 0.45,
            "carbon_intensity_factor": 0.50,
            "inventory_depth_pct": 0.40,
        },
        "STORE_SUBURB": {
            "name": "Store 102 - Suburban Hub (White Plains)",
            "type": "STORE",
            "capacity_units_per_day": 2200,
            "pick_pack_cost": 3.80,
            "holding_cost_per_day": 0.32,
            "carbon_intensity_factor": 0.55,
            "inventory_depth_pct": 0.45,
        },
        "STORE_REGIONAL": {
            "name": "Store 103 - Regional Mall (Garden State)",
            "type": "STORE",
            "capacity_units_per_day": 3000,
            "pick_pack_cost": 3.60,
            "holding_cost_per_day": 0.30,
            "carbon_intensity_factor": 0.52,
            "inventory_depth_pct": 0.50,
        },
    }

    # Last-Mile Delivery Zone Rate Card (Carrier Ground Contract)
    DELIVERY_ZONES: Dict[int, Dict[str, Any]] = {
        1: {"name": "Zone 1 (0-50 miles)", "base_rate": 4.80, "weight_rate_per_kg": 0.65, "transit_days": 1, "carbon_kg": 0.35},
        2: {"name": "Zone 2 (51-150 miles)", "base_rate": 6.20, "weight_rate_per_kg": 0.85, "transit_days": 2, "carbon_kg": 0.65},
        3: {"name": "Zone 3 (151-400 miles)", "base_rate": 8.50, "weight_rate_per_kg": 1.15, "transit_days": 3, "carbon_kg": 1.25},
        4: {"name": "Zone 4 (401-1000 miles)", "base_rate": 11.80, "weight_rate_per_kg": 1.55, "transit_days": 4, "carbon_kg": 2.40},
        5: {"name": "Zone 5 (1000+ miles)", "base_rate": 15.50, "weight_rate_per_kg": 2.05, "transit_days": 5, "carbon_kg": 4.10},
    }

    # Typical package physical weights per product category (kg)
    CATEGORY_WEIGHTS: Dict[str, float] = {
        "Beauty": 0.45,
        "Clothing": 0.85,
        "Electronics": 2.60,
    }

    # Gross margin baselines
    CATEGORY_GROSS_MARGIN: Dict[str, float] = {
        "Beauty": 0.52,
        "Clothing": 0.46,
        "Electronics": 0.28,
    }

    SPLIT_SHIPMENT_PENALTY: float = 4.25  # Extra box, packing material, redundant carrier dispatch fee

    def __init__(self, data_path: Optional[Path] = None, seed: int = 42):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.rng = np.random.default_rng(seed)

    def get_network_topology(self) -> Dict[str, Any]:
        """Returns the complete multi-echelon retail fulfillment network topology and rate card."""
        return {
            "total_nodes": len(self.FULFILLMENT_NODES),
            "distribution_centers": [k for k, v in self.FULFILLMENT_NODES.items() if v["type"] in ["CDC", "RDC"]],
            "retail_store_nodes": [k for k, v in self.FULFILLMENT_NODES.items() if v["type"] == "STORE"],
            "nodes_catalog": self.FULFILLMENT_NODES,
            "carrier_zones_rate_card": self.DELIVERY_ZONES,
            "category_unit_weights_kg": self.CATEGORY_WEIGHTS,
            "split_shipment_penalty_usd": self.SPLIT_SHIPMENT_PENALTY,
        }

    def route_order(
        self,
        category: str,
        quantity: int,
        price_per_unit: float,
        destination_zone: int = 2,
        channel_preference: Optional[str] = None,
        expedited: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculates optimal node assignment, routing costs, delivery SLA, margin preservation,
        and carbon footprint for an individual customer order.
        """
        if category not in self.CATEGORY_WEIGHTS:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {list(self.CATEGORY_WEIGHTS.keys())}")
        if destination_zone not in self.DELIVERY_ZONES:
            raise ValueError(f"Invalid delivery zone '{destination_zone}'. Must be an integer between 1 and 5.")
        if quantity <= 0:
            raise ValueError("Quantity must be strictly positive.")
        if price_per_unit <= 0:
            raise ValueError("Price per unit must be strictly positive.")

        order_revenue = float(quantity * price_per_unit)
        base_margin_pct = self.CATEGORY_GROSS_MARGIN.get(category, 0.40)
        unfulfilled_gross_margin = float(order_revenue * base_margin_pct)
        total_weight_kg = float(self.CATEGORY_WEIGHTS[category] * quantity)

        zone_info = self.DELIVERY_ZONES[destination_zone]

        # Evaluate candidate fulfillment options: CDC Direct, Regional DC, SFS (Ship-from-store), BOPIS
        candidate_routes = []

        # 1. BOPIS (Buy Online, Pick Up In Store) - customer drives to store
        bopis_pick_cost = 1.50
        bopis_shipping_cost = 0.0
        bopis_split = False
        bopis_carbon = 0.08  # Micro store-staging footprint
        bopis_transit_days = 0
        bopis_total_cost = bopis_pick_cost + bopis_shipping_cost
        candidate_routes.append({
            "channel": "BOPIS",
            "fulfilling_node": "STORE_FLAGSHIP" if destination_zone == 1 else "STORE_SUBURB",
            "pick_pack_cost": bopis_pick_cost,
            "freight_cost": bopis_shipping_cost,
            "split_penalty": 0.0,
            "total_fulfillment_cost": round(bopis_total_cost, 2),
            "carbon_kg": round(bopis_carbon, 2),
            "transit_days": bopis_transit_days,
            "net_margin_contribution": round(unfulfilled_gross_margin - bopis_total_cost, 2),
            "net_margin_pct": round(((unfulfilled_gross_margin - bopis_total_cost) / order_revenue) * 100, 2),
            "split_shipment": bopis_split,
        })

        # 2. SFS (Ship-from-Store) - Local store dispatch
        sfs_node = "STORE_FLAGSHIP" if destination_zone <= 2 else "STORE_REGIONAL"
        sfs_pick_cost = self.FULFILLMENT_NODES[sfs_node]["pick_pack_cost"]
        sfs_base_rate = 5.20 if destination_zone <= 2 else zone_info["base_rate"] + 1.20
        sfs_freight = sfs_base_rate + (total_weight_kg * 0.75)
        if expedited:
            sfs_freight *= 1.65
        sfs_carbon = zone_info["carbon_kg"] * 0.60
        sfs_total = sfs_pick_cost + sfs_freight
        candidate_routes.append({
            "channel": "SHIP_FROM_STORE",
            "fulfilling_node": sfs_node,
            "pick_pack_cost": round(sfs_pick_cost, 2),
            "freight_cost": round(sfs_freight, 2),
            "split_penalty": 0.0,
            "total_fulfillment_cost": round(sfs_total, 2),
            "carbon_kg": round(sfs_carbon, 2),
            "transit_days": 1 if destination_zone <= 2 else 2,
            "net_margin_contribution": round(unfulfilled_gross_margin - sfs_total, 2),
            "net_margin_pct": round(((unfulfilled_gross_margin - sfs_total) / order_revenue) * 100, 2),
            "split_shipment": False,
        })

        # 3. CDC DIRECT (Central Distribution Center)
        cdc_node = "CDC_CHICAGO"
        cdc_pick_cost = self.FULFILLMENT_NODES[cdc_node]["pick_pack_cost"]
        cdc_freight = zone_info["base_rate"] + (total_weight_kg * zone_info["weight_rate_per_kg"])
        if expedited:
            cdc_freight *= 1.85
        cdc_carbon = zone_info["carbon_kg"] * 1.15
        cdc_total = cdc_pick_cost + cdc_freight
        candidate_routes.append({
            "channel": "CDC_DIRECT",
            "fulfilling_node": cdc_node,
            "pick_pack_cost": round(cdc_pick_cost, 2),
            "freight_cost": round(cdc_freight, 2),
            "split_penalty": 0.0,
            "total_fulfillment_cost": round(cdc_total, 2),
            "carbon_kg": round(cdc_carbon, 2),
            "transit_days": 1 if expedited else zone_info["transit_days"],
            "net_margin_contribution": round(unfulfilled_gross_margin - cdc_total, 2),
            "net_margin_pct": round(((unfulfilled_gross_margin - cdc_total) / order_revenue) * 100, 2),
            "split_shipment": False,
        })

        # 4. Regional DC (RDC)
        rdc_node = "RDC_NORTH" if destination_zone <= 3 else "RDC_SOUTH"
        rdc_pick_cost = self.FULFILLMENT_NODES[rdc_node]["pick_pack_cost"]
        rdc_zone_adj = max(1, destination_zone - 1)
        rdc_freight = self.DELIVERY_ZONES[rdc_zone_adj]["base_rate"] + (total_weight_kg * self.DELIVERY_ZONES[rdc_zone_adj]["weight_rate_per_kg"])
        if expedited:
            rdc_freight *= 1.70
        rdc_carbon = self.DELIVERY_ZONES[rdc_zone_adj]["carbon_kg"] * 0.90
        # If high quantity and store node, potential split penalty
        rdc_split = False
        rdc_split_cost = 0.0
        if quantity >= 4 and rdc_node == "RDC_SOUTH":
            rdc_split = True
            rdc_split_cost = self.SPLIT_SHIPMENT_PENALTY
        rdc_total = rdc_pick_cost + rdc_freight + rdc_split_cost
        candidate_routes.append({
            "channel": "REGIONAL_DC",
            "fulfilling_node": rdc_node,
            "pick_pack_cost": round(rdc_pick_cost, 2),
            "freight_cost": round(rdc_freight, 2),
            "split_penalty": round(rdc_split_cost, 2),
            "total_fulfillment_cost": round(rdc_total, 2),
            "carbon_kg": round(rdc_carbon, 2),
            "transit_days": 1 if expedited else self.DELIVERY_ZONES[rdc_zone_adj]["transit_days"],
            "net_margin_contribution": round(unfulfilled_gross_margin - rdc_total, 2),
            "net_margin_pct": round(((unfulfilled_gross_margin - rdc_total) / order_revenue) * 100, 2),
            "split_shipment": rdc_split,
        })

        # Filter by channel preference if requested
        if channel_preference:
            preferred = [r for r in candidate_routes if r["channel"].lower() == channel_preference.lower()]
            if not preferred:
                raise ValueError(f"Unknown or unavailable channel preference '{channel_preference}'.")
            selected_route = preferred[0]
        else:
            # Optimal algorithm selects maximum net margin contribution
            selected_route = max(candidate_routes, key=lambda x: x["net_margin_contribution"])

        return {
            "order_summary": {
                "category": category,
                "quantity": quantity,
                "price_per_unit": price_per_unit,
                "gross_revenue": round(order_revenue, 2),
                "unfulfilled_gross_margin": round(unfulfilled_gross_margin, 2),
                "total_weight_kg": round(total_weight_kg, 2),
                "destination_zone": destination_zone,
                "expedited_sla": expedited,
            },
            "optimal_fulfillment": selected_route,
            "all_evaluated_channels": candidate_routes,
            "cost_breakdown": {
                "pick_pack_usd": selected_route["pick_pack_cost"],
                "last_mile_freight_usd": selected_route["freight_cost"],
                "split_shipment_penalty_usd": selected_route["split_penalty"],
                "total_fulfillment_expenditure_usd": selected_route["total_fulfillment_cost"],
            },
            "esg_sustainability": {
                "carbon_emissions_kg_co2": selected_route["carbon_kg"],
                "carbon_intensity_rating": "Green" if selected_route["carbon_kg"] < 0.5 else ("Moderate" if selected_route["carbon_kg"] < 1.5 else "High"),
            },
        }

    def simulate_network_fulfillment_demand(
        self,
        bopis_adoption_rate: float = 0.25,
        sfs_allocation_rate: float = 0.35,
        expedited_share_rate: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Simulates end-to-end fulfillment routing over historical dataset transactions,
        benchmarking total network expenditure, split rates, and Scope 3 carbon output.
        """
        if not (0.0 <= bopis_adoption_rate <= 1.0):
            raise ValueError("bopis_adoption_rate must be between 0.0 and 1.0.")
        if not (0.0 <= sfs_allocation_rate <= 1.0):
            raise ValueError("sfs_allocation_rate must be between 0.0 and 1.0.")
        if (bopis_adoption_rate + sfs_allocation_rate) > 1.0:
            raise ValueError("Combined BOPIS and SFS allocation cannot exceed 1.0.")

        total_orders = len(self.df)
        total_revenue = float(self.df["Total Amount"].sum())

        channel_counts = {"BOPIS": 0, "SHIP_FROM_STORE": 0, "REGIONAL_DC": 0, "CDC_DIRECT": 0}
        channel_costs = {"BOPIS": 0.0, "SHIP_FROM_STORE": 0.0, "REGIONAL_DC": 0.0, "CDC_DIRECT": 0.0}
        channel_carbon = {"BOPIS": 0.0, "SHIP_FROM_STORE": 0.0, "REGIONAL_DC": 0.0, "CDC_DIRECT": 0.0}
        splits_count = 0

        # Deterministic simulation across transactions
        for idx, row in self.df.iterrows():
            cat = str(row["Product Category"])
            qty = int(row["Quantity"])
            price = float(row["Price per Unit"])

            # Map transaction ID hash to deterministic destination zone (1 to 5)
            t_id = int(str(row["Transaction ID"]).replace("T", "").replace("CUST", "").replace("-", "") or idx)
            zone = int((t_id % 5) + 1)

            # Assign channel based on rates
            u = (t_id * 17) % 100 / 100.0
            if u < bopis_adoption_rate:
                chan = "BOPIS"
            elif u < (bopis_adoption_rate + sfs_allocation_rate):
                chan = "SHIP_FROM_STORE"
            elif u < 0.85:
                chan = "REGIONAL_DC"
            else:
                chan = "CDC_DIRECT"

            is_expedited = ((t_id * 23) % 100 / 100.0) < expedited_share_rate

            route_res = self.route_order(
                category=cat,
                quantity=qty,
                price_per_unit=price,
                destination_zone=zone,
                channel_preference=chan,
                expedited=is_expedited,
            )

            opt = route_res["optimal_fulfillment"]
            c_name = opt["channel"]
            if c_name not in channel_counts:
                c_name = "CDC_DIRECT"
            channel_counts[c_name] += 1
            channel_costs[c_name] += opt["total_fulfillment_cost"]
            channel_carbon[c_name] += opt["carbon_kg"]
            if opt["split_shipment"]:
                splits_count += 1

        total_fulfillment_cost = sum(channel_costs.values())
        total_carbon_kg = sum(channel_carbon.values())
        avg_cost_per_order = total_fulfillment_cost / max(1, total_orders)
        cost_as_pct_revenue = (total_fulfillment_cost / max(1.0, total_revenue)) * 100.0

        # Benchmark vs naive all-CDC central expedited/ground fulfillment
        naive_all_cdc_cost = total_orders * 16.50
        naive_all_cdc_carbon = total_orders * 2.85
        cost_savings_vs_cdc = naive_all_cdc_cost - total_fulfillment_cost
        carbon_avoided_kg = naive_all_cdc_carbon - total_carbon_kg

        return {
            "simulation_parameters": {
                "total_orders_simulated": total_orders,
                "bopis_adoption_rate": bopis_adoption_rate,
                "sfs_allocation_rate": sfs_allocation_rate,
                "expedited_share_rate": expedited_share_rate,
            },
            "financial_summary": {
                "total_fulfillment_expenditure_usd": round(total_fulfillment_cost, 2),
                "average_cost_per_order_usd": round(avg_cost_per_order, 2),
                "fulfillment_cost_pct_revenue": round(cost_as_pct_revenue, 2),
                "cost_savings_vs_naive_central_dc_usd": round(cost_savings_vs_cdc, 2),
                "savings_pct_vs_central": round((cost_savings_vs_cdc / naive_all_cdc_cost) * 100, 2),
            },
            "channel_volume_breakdown": {
                k: {"orders": v, "share_pct": round((v / total_orders) * 100, 2), "total_cost_usd": round(channel_costs[k], 2)}
                for k, v in channel_counts.items()
            },
            "operational_metrics": {
                "split_shipment_orders": splits_count,
                "split_shipment_rate_pct": round((splits_count / max(1, total_orders)) * 100, 2),
                "total_carbon_emissions_tonnes_co2e": round(total_carbon_kg / 1000.0, 3),
                "carbon_emissions_avoided_tonnes_co2e": round(carbon_avoided_kg / 1000.0, 3),
            },
        }

    def optimize_delivery_promise_tradeoff(
        self,
        order_value: float = 300.0,
        category: str = "Electronics",
        destination_zone: int = 3,
    ) -> Dict[str, Any]:
        """
        Analyzes the delivery promise SLA frontier (Expedited Next-Day vs Standard Ground vs Eco-Consolidated)
        comparing speed, shipping subsidy cost, carbon emission, and gross margin preservation.
        """
        if order_value <= 0:
            raise ValueError("Order value must be positive.")
        if category not in self.CATEGORY_WEIGHTS:
            raise ValueError(f"Category '{category}' is invalid.")

        qty = 1
        unit_price = order_value

        std_route = self.route_order(category, qty, unit_price, destination_zone=destination_zone, channel_preference="CDC_DIRECT", expedited=False)
        exp_route = self.route_order(category, qty, unit_price, destination_zone=destination_zone, channel_preference="CDC_DIRECT", expedited=True)

        std_cost = std_route["optimal_fulfillment"]["total_fulfillment_cost"]
        exp_cost = exp_route["optimal_fulfillment"]["total_fulfillment_cost"]
        eco_cost = round(std_cost * 0.82, 2)  # Eco-consolidated discount

        std_margin = std_route["optimal_fulfillment"]["net_margin_contribution"]
        exp_margin = exp_route["optimal_fulfillment"]["net_margin_contribution"]
        eco_margin = round(std_route["order_summary"]["unfulfilled_gross_margin"] - eco_cost, 2)

        return {
            "order_value": order_value,
            "category": category,
            "destination_zone": destination_zone,
            "sla_tiers": [
                {
                    "tier_name": "Expedited Next-Day",
                    "transit_days": 1,
                    "fulfillment_cost_usd": exp_cost,
                    "net_margin_usd": exp_margin,
                    "margin_erosion_usd": round(std_margin - exp_margin, 2),
                    "carbon_kg": exp_route["optimal_fulfillment"]["carbon_kg"],
                    "free_shipping_breakeven_order_value": round(exp_cost / 0.15, 2),
                },
                {
                    "tier_name": "Standard Ground (Default)",
                    "transit_days": self.DELIVERY_ZONES[destination_zone]["transit_days"],
                    "fulfillment_cost_usd": std_cost,
                    "net_margin_usd": std_margin,
                    "margin_erosion_usd": 0.0,
                    "carbon_kg": std_route["optimal_fulfillment"]["carbon_kg"],
                    "free_shipping_breakeven_order_value": round(std_cost / 0.15, 2),
                },
                {
                    "tier_name": "Eco-Consolidated Green Saver",
                    "transit_days": self.DELIVERY_ZONES[destination_zone]["transit_days"] + 2,
                    "fulfillment_cost_usd": eco_cost,
                    "net_margin_usd": eco_margin,
                    "margin_uplift_vs_standard_usd": round(eco_margin - std_margin, 2),
                    "carbon_kg": round(std_route["optimal_fulfillment"]["carbon_kg"] * 0.70, 2),
                    "customer_incentive_credit_usd": 2.00,
                },
            ],
            "strategic_recommendation": (
                "Subsidize Standard Ground for baskets >= $150. Require $12 surcharge on Expedited Next-Day to protect gross margins."
            ),
        }

    def calculate_esg_carbon_footprint(self) -> Dict[str, Any]:
        """Audits Scope 3 last-mile transportation emissions and benchmarks ESG green fulfillment savings."""
        sim_res = self.simulate_network_fulfillment_demand()
        total_co2_kg = sim_res["operational_metrics"]["total_carbon_emissions_tonnes_co2e"] * 1000.0

        return {
            "storewide_scope3_carbon_kg": round(total_co2_kg, 2),
            "carbon_intensity_kg_per_dollar_revenue": round(total_co2_kg / float(self.df["Total Amount"].sum()), 4),
            "carbon_by_channel": {
                k: {"share_pct": v["share_pct"], "co2_kg": round(v["share_pct"] * total_co2_kg / 100.0, 2)}
                for k, v in sim_res["channel_volume_breakdown"].items()
            },
            "green_logistics_initiatives": [
                {"initiative": "BOPIS Click & Collect Promotion", "potential_co2_reduction_pct": 24.5, "annual_co2_savings_kg": round(total_co2_kg * 0.245, 1)},
                {"initiative": "EV Local Fleet Deployment for SFS", "potential_co2_reduction_pct": 18.0, "annual_co2_savings_kg": round(total_co2_kg * 0.18, 1)},
                {"initiative": "Eco-Consolidated Multi-Order Batching", "potential_co2_reduction_pct": 14.2, "annual_co2_savings_kg": round(total_co2_kg * 0.142, 1)},
            ],
        }
